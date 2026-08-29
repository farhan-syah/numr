//! Scalar operation CUDA kernel launchers
//!
//! Provides launchers for element-wise tensor-scalar operations
//! (add_scalar, mul_scalar, etc.).

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Macro to generate scalar operation launcher functions.
///
/// This eliminates code duplication across dtype-specific launchers.
/// Each launcher follows the same pattern:
/// 1. Load the scalar kernel module
/// 2. Get the kernel function for the specific dtype
/// 3. Configure launch parameters
/// 4. Launch with (input_ptr, scalar, output_ptr, count) arguments
macro_rules! define_scalar_launcher {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident, $scalar_ty:ty, $dtype:expr, $dtype_name:literal
    ) => {
        $(#[$meta])*
        $vis unsafe fn $name(
            context: &Arc<CudaContext>,
            stream: &CudaStream,
            device_index: usize,
            op: &str,
            a_ptr: u64,
            scalar: $scalar_ty,
            out_ptr: u64,
            numel: usize,
        ) -> Result<()> {
            unsafe {
                let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
                let func_name = kernel_name(op, $dtype);
                let func = get_kernel_function(&module, &func_name)?;

                let grid = elementwise_launch_config(numel);
                let block = (BLOCK_SIZE, 1, 1);
                let n = numel as u32;

                let cfg = launch_config(grid, block, 0);
                let mut builder = stream.launch_builder(&func);
                builder.arg(&a_ptr);
                builder.arg(&scalar);
                builder.arg(&out_ptr);
                builder.arg(&n);

                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA scalar kernel '{}' ({}) launch failed: {:?}",
                        op, $dtype_name, e
                    ))
                })?;

                Ok(())
            }
        }
    };
}

define_scalar_launcher!(
    /// Launch a scalar operation kernel for f32.
    ///
    /// Performs element-wise operation: `output[i] = op(input[i], scalar)`
    ///
    /// # Supported Operations
    ///
    /// - `add_scalar`: Add scalar to each element
    /// - `sub_scalar`: Subtract scalar from each element
    /// - `mul_scalar`: Multiply each element by scalar
    /// - `div_scalar`: Divide each element by scalar
    /// - `pow_scalar`: Raise each element to scalar power
    ///
    /// # Safety
    ///
    /// - All pointers must be valid device memory
    /// - Tensors must have at least `numel` elements
    pub fn launch_scalar_op_f32, f32, DType::F32, "f32"
);

define_scalar_launcher!(
    /// Launch a scalar operation kernel for f64.
    ///
    /// See [`launch_scalar_op_f32`] for documentation.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch_scalar_op_f32`.
    pub fn launch_scalar_op_f64, f64, DType::F64, "f64"
);

define_scalar_launcher!(
    /// Launch a scalar operation kernel for i32.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch_scalar_op_f32`.
    pub fn launch_scalar_op_i32, i32, DType::I32, "i32"
);

define_scalar_launcher!(
    /// Launch a scalar operation kernel for i64.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch_scalar_op_f32`.
    pub fn launch_scalar_op_i64, i64, DType::I64, "i64"
);

/// Launch a scalar operation kernel for f16/bf16/fp8 (uses f32 scalar value).
///
/// This launcher handles multiple half-precision types that all use f32 scalars:
/// - F16 (IEEE 754 half precision, requires "f16" feature)
/// - BF16 (bfloat16, requires "f16" feature)
/// - FP8E4M3 / FP8E5M2 (8-bit floating point, always available)
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
pub unsafe fn launch_scalar_op_half(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    scalar: f32,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel '{}' ({:?}) launch failed: {:?}",
                op, dtype, e
            ))
        })?;

        Ok(())
    }
}

define_scalar_launcher!(
    /// Launch a scalar operation kernel for Complex64 (uses f32 scalar value).
    ///
    /// Scalar operations on complex numbers:
    /// - `add_scalar`: (a+bi) + s = (a+s) + bi
    /// - `sub_scalar`: (a+bi) - s = (a-s) + bi
    /// - `mul_scalar`: s(a+bi) = sa + sbi
    /// - `div_scalar`: (a+bi)/s = (a/s) + (b/s)i
    /// - `pow_scalar`: z^s using polar form
    ///
    /// # Safety
    ///
    /// Same requirements as `launch_scalar_op_f32`.
    pub fn launch_scalar_op_c64, f32, DType::Complex64, "c64"
);

define_scalar_launcher!(
    /// Launch a scalar operation kernel for Complex128 (uses f64 scalar value).
    ///
    /// See [`launch_scalar_op_c64`] for operation semantics.
    ///
    /// # Safety
    ///
    /// Same requirements as `launch_scalar_op_f64`.
    pub fn launch_scalar_op_c128, f64, DType::Complex128, "c128"
);

/// Launch `pow_scalar` for I32 or I64.
///
/// Separate from [`launch_scalar_op_i32`] because the integer `pow_scalar`
/// kernels take the exponent as `f64`, not as the element type: a fractional or
/// negative exponent must reach the kernel unrounded so it can take the same
/// double path CPU takes.
///
/// # Safety
///
/// Same requirements as `launch_scalar_op_f32`.
pub unsafe fn launch_pow_scalar_int(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    scalar: f64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SCALAR_MODULE)?;
        let func_name = kernel_name("pow_scalar", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&scalar);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA scalar kernel 'pow_scalar' ({:?}) launch failed: {:?}",
                dtype, e
            ))
        })?;

        Ok(())
    }
}
