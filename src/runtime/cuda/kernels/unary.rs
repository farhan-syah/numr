//! Unary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise unary operations (neg, abs, sqrt, exp, etc.)
//! on a single tensor.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config, launch_unary_kernel,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a unary operation kernel.
///
/// Performs element-wise operation: `output[i] = op(input[i])`
///
/// # Supported Operations
///
/// - `neg`: Negation (-x)
/// - `abs`: Absolute value
/// - `sqrt`: Square root
/// - `exp`: Exponential (e^x)
/// - `log`: Natural logarithm
/// - `sin`, `cos`, `tan`: Trigonometric functions
/// - `tanh`: Hyperbolic tangent
/// - `recip`: Reciprocal (1/x)
/// - `square`: Square (x*x)
/// - `floor`, `ceil`, `trunc`: Rounding functions
/// - `round`: Round to nearest, ties away from zero
/// - `round_ties_even`: Round to nearest, ties to even (IEEE roundTiesToEven)
/// - `sign`: Sign function (-1, 0, or 1)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "neg", "sqrt")
/// * `dtype` - Data type of the tensor
/// * `a_ptr` - Device pointer to input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_unary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::UNARY_MODULE,
            op,
            dtype,
            a_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch an isnan kernel.
///
/// Performs element-wise NaN check: `output[i] = isnan(input[i]) ? 1 : 0`
/// Output is always U8 (boolean), regardless of input dtype.
///
/// # Supported Input Dtypes
///
/// - F32, F64, F16, BF16, FP8E4M3, FP8E5M2
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` elements
/// - Output tensor must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `input_dtype` - Data type of the input tensor
/// * `a_ptr` - Device pointer to input tensor
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_isnan_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::UNARY_MODULE)?;
        let func_name = kernel_name("isnan", input_dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA isnan kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch an isinf kernel.
///
/// Performs element-wise infinity check: `output[i] = isinf(input[i]) ? 1 : 0`
/// Output is always U8 (boolean), regardless of input dtype.
///
/// # Supported Input Dtypes
///
/// - F32, F64, F16, BF16, FP8E4M3, FP8E5M2
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` elements
/// - Output tensor must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `input_dtype` - Data type of the input tensor
/// * `a_ptr` - Device pointer to input tensor
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_isinf_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::UNARY_MODULE)?;
        let func_name = kernel_name("isinf", input_dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA isinf kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a logical_not kernel.
///
/// Performs element-wise logical NOT: `output[i] = !input[i]`
/// Both input and output are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_not_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::UNARY_MODULE)?;
        let func_name = "logical_not_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_not kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
