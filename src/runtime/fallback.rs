//! Generic CPU fallback utilities for GPU backends
//!
#![allow(unreachable_code, dead_code)] // dispatch_dtype! macro uses early returns in all branches; some fallbacks only used by wgpu
//!
//! This module provides shared CPU fallback implementations for operations
//! that are not yet implemented natively on GPU backends (CUDA, WGPU).
//!
//! # When Fallback Triggers
//!
//! Fallback automatically occurs when:
//! 1. **Operation not implemented**: The GPU backend lacks a native kernel for the operation
//! 2. **Feature combination unsupported**: e.g., sparse operations on GPU without sparse feature
//! 3. **Explicit fallback**: Some operations intentionally use CPU for correctness/simplicity
//!
//! Examples:
//! - Sparse matrix operations before native GPU kernels are implemented
//! - Advanced reductions (e.g., median, percentile) that are complex on GPU
//! - Operations on data types not supported by GPU hardware (e.g., i128)
//!
//! # Performance Impact
//!
//! Fallback incurs significant overhead:
//!
//! | Operation Component | Cost | Notes |
//! |---------------------|------|-------|
//! | GPU→CPU transfer | ~10-100 µs/MB | PCIe bandwidth limited (~12 GB/s) |
//! | CPU computation | Varies | Typically 10-100x slower than GPU |
//! | CPU→GPU transfer | ~10-100 µs/MB | Same PCIe bottleneck |
//!
//! **Example**: 1M element addition
//! - GPU native: ~50 µs
//! - CPU fallback: ~8 MB transfer (2×4 MB) + ~100 µs compute = ~280 µs total
//! - **5-6x slower** for this small operation, worse for larger tensors
//!
//! **When fallback is acceptable**:
//! - Development/prototyping (correctness over performance)
//! - Infrequent operations (e.g., model initialization)
//! - Small tensors where transfer overhead dominates anyway
//! - Operations bottlenecked by algorithm complexity, not memory bandwidth
//!
//! **When to avoid fallback**:
//! - Training loops (every forward/backward pass)
//! - Large batch processing
//! - Real-time inference
//!
//! # Numerical Equivalence Guarantees
//!
//! Fallback operations are **numerically equivalent** to native CPU operations:
//!
//! ## Exact Equivalence
//!
//! Integer operations produce **bit-for-bit identical** results:
//! ```ignore
//! let cpu_result = cpu_client.add(&a, &b);
//! let gpu_fallback_result = gpu_client.add(&a, &b); // Falls back to CPU
//! assert_eq!(cpu_result.to_vec::<i32>(), gpu_fallback_result.to_vec::<i32>());
//! ```
//!
//! ## Floating-Point Equivalence
//!
//! Floating-point operations match **within machine epsilon**:
//! - Same algorithm used (CPU backend)
//! - Same accumulation order (deterministic)
//! - Same rounding behavior (IEEE 754)
//!
//! Differences only arise from:
//! 1. **Compiler differences**: Rare, typically ~1 ULP (unit of least precision)
//! 2. **Library versions**: E.g., different BLAS implementations
//! 3. **Non-associativity**: Should not occur (same code path)
//!
//! **Testing**: Backend parity tests verify GPU fallback matches CPU:
//! ```ignore
//! #[test]
//! fn test_fallback_parity() {
//!     let cpu = cpu_client.matmul(&a, &b);
//!     let gpu = gpu_client.matmul(&a, &b); // May use fallback
//!     assert_allclose(&cpu, &gpu, rtol=1e-6, atol=1e-7);
//! }
//! ```
//!
//! # Design
//!
//! The fallback pattern:
//! 1. Copy tensor data from GPU to CPU
//! 2. Execute operation using CPU backend
//! 3. Copy result back to GPU
//!
//! This is intentionally simple and correct - performance optimization
//! via native GPU kernels is a separate phase.
//!
//! # Usage
//!
//! GPU backends use these helpers to implement `TensorOps`:
//!
//! ```ignore
//! use numr::runtime::fallback::{CpuFallbackContext, binary_op_fallback};
//!
//! impl TensorOps<CudaRuntime> for CudaClient {
//!     fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
//!         binary_op_fallback(
//!             a, b,
//!             |cpu_a, cpu_b, cpu_client| cpu_client.add(cpu_a, cpu_b),
//!             &self.device,
//!             "add",
//!         )
//!     }
//! }
//! ```

use crate::dispatch_dtype;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::{
    ActivationOps, BinaryOp, BinaryOps, CompareOp, CompareOps, ConditionalOps, MatmulOps, ReduceOp,
    ReduceOps, ScalarOps, UnaryOp, UnaryOps, broadcast_shape, reduce_output_shape,
};
use crate::runtime::{Device, Runtime, cpu};
use crate::tensor::Tensor;

// ============================================================================
// CPU Fallback Context
// ============================================================================

/// CPU fallback context for operations not yet implemented natively on GPU.
///
/// This struct holds the CPU device and client needed for fallback operations.
/// Using a struct avoids repeated boilerplate in every fallback function.
///
/// # Thread Safety
///
/// Each call creates a new context. The CPU runtime is stateless and
/// thread-safe, so this is fine for concurrent use.
pub struct CpuFallbackContext {
    /// CPU device (lightweight, just an ID)
    pub device: cpu::CpuDevice,
    /// CPU client (contains allocator reference)
    pub client: cpu::CpuClient,
}

impl CpuFallbackContext {
    /// Create a new CPU fallback context.
    #[inline]
    pub fn new() -> Self {
        let device = cpu::CpuDevice::new();
        let client = cpu::CpuRuntime::default_client(&device);
        Self { device, client }
    }

    /// Create a CPU tensor from GPU tensor data.
    ///
    /// This copies the tensor data from GPU memory to CPU memory.
    #[inline]
    pub fn tensor_from_gpu<T: Element, R: Runtime<DType = DType>>(
        &self,
        tensor: &Tensor<R>,
    ) -> Result<Tensor<cpu::CpuRuntime>> {
        let data: Vec<T> = tensor.to_vec();
        Tensor::<cpu::CpuRuntime>::try_from_slice(&data, tensor.shape(), &self.device)
    }
}

impl Default for CpuFallbackContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that two tensors have matching dtypes for binary operations.
#[inline]
pub fn validate_binary_dtypes<R: Runtime<DType = DType>>(
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Compute broadcast shape for binary operations.
#[inline]
pub fn compute_broadcast_shape<R: Runtime<DType = DType>>(
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

// ============================================================================
// Generic Fallback Operations
// ============================================================================

/// Perform a binary operation using CPU fallback.
///
/// # Type Parameters
///
/// * `R` - The GPU runtime (CudaRuntime, WgpuRuntime)
/// * `D` - The device type for the GPU runtime
///
/// # Arguments
///
/// * `a` - First input tensor (on GPU)
/// * `b` - Second input tensor (on GPU)
/// * `op` - Binary operation to perform
/// * `device` - GPU device to create output tensor on
/// * `op_name` - Operation name for error messages
pub fn binary_op_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    op: BinaryOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b)?;

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add(&a_cpu, &b_cpu)?,
            BinaryOp::Sub => cpu.client.sub(&a_cpu, &b_cpu)?,
            BinaryOp::Mul => cpu.client.mul(&a_cpu, &b_cpu)?,
            BinaryOp::Div => cpu.client.div(&a_cpu, &b_cpu)?,
            BinaryOp::Pow => cpu.client.pow(&a_cpu, &b_cpu)?,
            BinaryOp::Max => cpu.client.maximum(&a_cpu, &b_cpu)?,
            BinaryOp::Min => cpu.client.minimum(&a_cpu, &b_cpu)?,
            BinaryOp::Atan2 => cpu.client.atan2(&a_cpu, &b_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform a unary operation using CPU fallback.
pub fn unary_op_fallback<R, D>(
    a: &Tensor<R>,
    op: UnaryOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;

        let result_cpu = match op {
            UnaryOp::Neg => cpu.client.neg(&a_cpu)?,
            UnaryOp::Abs => cpu.client.abs(&a_cpu)?,
            UnaryOp::Sqrt => cpu.client.sqrt(&a_cpu)?,
            UnaryOp::Rsqrt => cpu.client.rsqrt(&a_cpu)?,
            UnaryOp::Cbrt => cpu.client.cbrt(&a_cpu)?,
            UnaryOp::Exp => cpu.client.exp(&a_cpu)?,
            UnaryOp::Exp2 => cpu.client.exp2(&a_cpu)?,
            UnaryOp::Expm1 => cpu.client.expm1(&a_cpu)?,
            UnaryOp::Log => cpu.client.log(&a_cpu)?,
            UnaryOp::Log2 => cpu.client.log2(&a_cpu)?,
            UnaryOp::Log10 => cpu.client.log10(&a_cpu)?,
            UnaryOp::Log1p => cpu.client.log1p(&a_cpu)?,
            UnaryOp::Sin => cpu.client.sin(&a_cpu)?,
            UnaryOp::Cos => cpu.client.cos(&a_cpu)?,
            UnaryOp::Tan => cpu.client.tan(&a_cpu)?,
            UnaryOp::Asin => cpu.client.asin(&a_cpu)?,
            UnaryOp::Acos => cpu.client.acos(&a_cpu)?,
            UnaryOp::Atan => cpu.client.atan(&a_cpu)?,
            UnaryOp::Sinh => cpu.client.sinh(&a_cpu)?,
            UnaryOp::Cosh => cpu.client.cosh(&a_cpu)?,
            UnaryOp::Tanh => cpu.client.tanh(&a_cpu)?,
            UnaryOp::Asinh => cpu.client.asinh(&a_cpu)?,
            UnaryOp::Acosh => cpu.client.acosh(&a_cpu)?,
            UnaryOp::Atanh => cpu.client.atanh(&a_cpu)?,
            UnaryOp::Recip => cpu.client.recip(&a_cpu)?,
            UnaryOp::Square => cpu.client.square(&a_cpu)?,
            UnaryOp::Floor => cpu.client.floor(&a_cpu)?,
            UnaryOp::Ceil => cpu.client.ceil(&a_cpu)?,
            UnaryOp::Round => cpu.client.round(&a_cpu)?,
            UnaryOp::Trunc => cpu.client.trunc(&a_cpu)?,
            UnaryOp::Sign => cpu.client.sign(&a_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform a scalar operation using CPU fallback.
pub fn scalar_op_fallback<R, D>(
    a: &Tensor<R>,
    op: BinaryOp,
    scalar: f64,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;

        let result_cpu = match op {
            BinaryOp::Add => cpu.client.add_scalar(&a_cpu, scalar)?,
            BinaryOp::Sub => cpu.client.sub_scalar(&a_cpu, scalar)?,
            BinaryOp::Mul => cpu.client.mul_scalar(&a_cpu, scalar)?,
            BinaryOp::Div => cpu.client.div_scalar(&a_cpu, scalar)?,
            BinaryOp::Pow => cpu.client.pow_scalar(&a_cpu, scalar)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform a reduce operation using CPU fallback.
pub fn reduce_op_fallback<R, D>(
    a: &Tensor<R>,
    op: ReduceOp,
    dims: &[usize],
    keepdim: bool,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let out_shape = reduce_output_shape(a.shape(), dims, keepdim);
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;

        let result_cpu = match op {
            ReduceOp::Sum => cpu.client.sum(&a_cpu, dims, keepdim)?,
            ReduceOp::Mean => cpu.client.mean(&a_cpu, dims, keepdim)?,
            ReduceOp::Max => cpu.client.max(&a_cpu, dims, keepdim)?,
            ReduceOp::Min => cpu.client.min(&a_cpu, dims, keepdim)?,
            _ => return Err(Error::UnsupportedDType { dtype, op: op_name }),
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform an activation operation using CPU fallback.
///
/// This is a generic helper for activation functions (relu, sigmoid)
/// that share the same pattern: copy to CPU, apply function, copy back.
pub fn activation_fallback<R, D, F>(
    a: &Tensor<R>,
    device: &D,
    op_name: &'static str,
    op_fn: F,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
    F: Fn(&cpu::CpuClient, &Tensor<cpu::CpuRuntime>) -> Result<Tensor<cpu::CpuRuntime>>,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;
        let result_cpu = op_fn(&cpu.client, &a_cpu)?;
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform softmax operation using CPU fallback.
pub fn softmax_fallback<R, D>(
    a: &Tensor<R>,
    dim: isize,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = a.dtype();
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;
        let result_cpu = cpu.client.softmax(&a_cpu, dim)?;
        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, a.shape(), device));
    }, op_name);

    unreachable!()
}

/// Perform matmul operation using CPU fallback.
pub fn matmul_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    out_shape: &[usize],
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b)?;

        let result_cpu = cpu.client.matmul(&a_cpu, &b_cpu)?;
        let result_data: Vec<T> = result_cpu.to_vec();

        return Ok(Tensor::<R>::from_slice(&result_data, out_shape, device));
    }, op_name);

    unreachable!()
}

/// Perform a compare operation using CPU fallback.
pub fn compare_op_fallback<R, D>(
    a: &Tensor<R>,
    b: &Tensor<R>,
    op: CompareOp,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;
    let cpu = CpuFallbackContext::new();

    dispatch_dtype!(dtype, T => {
        let a_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a)?;
        let b_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b)?;

        let result_cpu = match op {
            CompareOp::Eq => cpu.client.eq(&a_cpu, &b_cpu)?,
            CompareOp::Ne => cpu.client.ne(&a_cpu, &b_cpu)?,
            CompareOp::Lt => cpu.client.lt(&a_cpu, &b_cpu)?,
            CompareOp::Le => cpu.client.le(&a_cpu, &b_cpu)?,
            CompareOp::Gt => cpu.client.gt(&a_cpu, &b_cpu)?,
            CompareOp::Ge => cpu.client.ge(&a_cpu, &b_cpu)?,
        };

        let result_data: Vec<T> = result_cpu.to_vec();
        return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
    }, op_name);

    unreachable!()
}

/// Compute broadcast shape for ternary operations (where_cond).
///
/// Returns the broadcasted shape of all three tensors.
#[inline]
pub fn compute_ternary_broadcast_shape<R: Runtime<DType = DType>>(
    cond: &Tensor<R>,
    x: &Tensor<R>,
    y: &Tensor<R>,
) -> Result<Vec<usize>> {
    // First compute broadcast of x and y
    let xy_shape = broadcast_shape(x.shape(), y.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: x.shape().to_vec(),
        rhs: y.shape().to_vec(),
    })?;

    // Then broadcast cond with the x-y broadcast result
    broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| Error::BroadcastError {
        lhs: cond.shape().to_vec(),
        rhs: xy_shape,
    })
}

/// Perform a where_cond (ternary conditional select) operation using CPU fallback.
///
/// This supports full broadcasting across cond, x, and y tensors.
///
/// # Arguments
///
/// * `cond` - Condition tensor (U8/boolean) on GPU
/// * `x` - "True" values tensor on GPU
/// * `y` - "False" values tensor on GPU
/// * `device` - GPU device to create output tensor on
/// * `op_name` - Operation name for error messages
pub fn where_cond_fallback<R, D>(
    cond: &Tensor<R>,
    x: &Tensor<R>,
    y: &Tensor<R>,
    device: &D,
    op_name: &'static str,
) -> Result<Tensor<R>>
where
    R: Runtime<Device = D, DType = DType>,
    D: Device + Clone,
{
    // Validate dtypes (x and y must match, cond can be any dtype - non-zero = true)
    let dtype = validate_binary_dtypes(x, y)?;
    let cond_dtype = cond.dtype();

    let out_shape = compute_ternary_broadcast_shape(cond, x, y)?;
    let cpu = CpuFallbackContext::new();

    // Double dispatch: condition dtype and value dtype
    dispatch_dtype!(cond_dtype, C => {
        dispatch_dtype!(dtype, T => {
            // Copy all three tensors to CPU
            let cond_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<C, R>(cond)?;
            let x_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(x)?;
            let y_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(y)?;

            // Execute where_cond on CPU (CPU impl now handles any cond dtype)
            let result_cpu = cpu.client.where_cond(&cond_cpu, &x_cpu, &y_cpu)?;

            // Copy result back to GPU
            let result_data: Vec<T> = result_cpu.to_vec();
            return Ok(Tensor::<R>::from_slice(&result_data, &out_shape, device));
        }, op_name);
    }, op_name);

    unreachable!()
}

// ============================================================================
// Sparse Matrix Fallback Helpers
// ============================================================================

#[cfg(feature = "sparse")]
/// CSC element-wise operation fallback (GPU → CPU → GPU)
#[allow(private_interfaces)]
pub fn csc_elementwise_fallback<T: Element, R: Runtime<DType = DType>, F, FA, FB>(
    a_col_ptrs: &Tensor<R>,
    a_row_indices: &Tensor<R>,
    a_values: &Tensor<R>,
    b_col_ptrs: &Tensor<R>,
    b_row_indices: &Tensor<R>,
    b_values: &Tensor<R>,
    shape: [usize; 2],
    strategy: super::cpu::sparse::MergeStrategy,
    semantics: super::cpu::sparse::OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R::Device: Device + Clone,
    F: Fn(T, T) -> T + Copy,
    FA: Fn(T) -> T + Copy,
    FB: Fn(T) -> T + Copy,
{
    let device = a_values.device();
    let cpu = CpuFallbackContext::new();

    // Copy to CPU
    let a_col_ptrs_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<i64, R>(a_col_ptrs)?;
    let a_row_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(a_row_indices)?;
    let a_values_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a_values)?;
    let b_col_ptrs_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<i64, R>(b_col_ptrs)?;
    let b_row_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(b_row_indices)?;
    let b_values_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b_values)?;

    // Execute on CPU using the merge_csc_impl from cpu/sparse.rs
    let (result_col_ptrs_cpu, result_row_indices_cpu, result_values_cpu) =
        super::cpu::sparse::merge_csc_impl(
            &a_col_ptrs_cpu,
            &a_row_indices_cpu,
            &a_values_cpu,
            &b_col_ptrs_cpu,
            &b_row_indices_cpu,
            &b_values_cpu,
            shape,
            strategy,
            semantics,
            op,
            only_a_op,
            only_b_op,
        )?;

    // Copy back to GPU
    let col_ptrs_data: Vec<i64> = result_col_ptrs_cpu.to_vec();
    let row_indices_data: Vec<i64> = result_row_indices_cpu.to_vec();
    let values_data: Vec<T> = result_values_cpu.to_vec();

    let result_col_ptrs =
        Tensor::<R>::from_slice(&col_ptrs_data, result_col_ptrs_cpu.shape(), device);
    let result_row_indices =
        Tensor::<R>::from_slice(&row_indices_data, result_row_indices_cpu.shape(), device);
    let result_values = Tensor::<R>::from_slice(&values_data, result_values_cpu.shape(), device);

    Ok((result_col_ptrs, result_row_indices, result_values))
}

#[cfg(feature = "sparse")]
/// COO element-wise operation fallback (GPU → CPU → GPU)
#[allow(private_interfaces)]
pub fn coo_elementwise_fallback<T: Element, R: Runtime<DType = DType>, F, FA, FB>(
    a_row_indices: &Tensor<R>,
    a_col_indices: &Tensor<R>,
    a_values: &Tensor<R>,
    b_row_indices: &Tensor<R>,
    b_col_indices: &Tensor<R>,
    b_values: &Tensor<R>,
    semantics: super::cpu::sparse::OperationSemantics,
    op: F,
    only_a_op: FA,
    only_b_op: FB,
) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>
where
    R::Device: Device + Clone,
    F: Fn(T, T) -> T + Copy,
    FA: Fn(T) -> T + Copy,
    FB: Fn(T) -> T + Copy,
{
    let device = a_values.device();
    let cpu = CpuFallbackContext::new();

    // Copy to CPU
    let a_row_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(a_row_indices)?;
    let a_col_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(a_col_indices)?;
    let a_values_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(a_values)?;
    let b_row_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(b_row_indices)?;
    let b_col_indices_cpu: Tensor<cpu::CpuRuntime> =
        cpu.tensor_from_gpu::<i64, R>(b_col_indices)?;
    let b_values_cpu: Tensor<cpu::CpuRuntime> = cpu.tensor_from_gpu::<T, R>(b_values)?;

    // Execute on CPU using the merge_coo_impl from cpu/sparse.rs
    let (result_row_indices_cpu, result_col_indices_cpu, result_values_cpu) =
        super::cpu::sparse::merge_coo_impl(
            &a_row_indices_cpu,
            &a_col_indices_cpu,
            &a_values_cpu,
            &b_row_indices_cpu,
            &b_col_indices_cpu,
            &b_values_cpu,
            semantics,
            op,
            only_a_op,
            only_b_op,
        )?;

    // Copy back to GPU
    let row_indices_data: Vec<i64> = result_row_indices_cpu.to_vec();
    let col_indices_data: Vec<i64> = result_col_indices_cpu.to_vec();
    let values_data: Vec<T> = result_values_cpu.to_vec();

    let result_row_indices =
        Tensor::<R>::from_slice(&row_indices_data, result_row_indices_cpu.shape(), device);
    let result_col_indices =
        Tensor::<R>::from_slice(&col_indices_data, result_col_indices_cpu.shape(), device);
    let result_values = Tensor::<R>::from_slice(&values_data, result_values_cpu.shape(), device);

    Ok((result_row_indices, result_col_indices, result_values))
}
