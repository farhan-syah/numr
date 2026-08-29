//! Generic launchers for element-wise unary and binary kernels.
//!
//! These cover the one-input and two-input patterns shared by unary, binary,
//! compare, and activation kernels: one thread per element, no shared memory.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::{BLOCK_SIZE, elementwise_launch_config, launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_name;

/// Launch an element-wise unary kernel (one input, one output).
///
/// This handles the common pattern for operations like neg, abs, sqrt, exp, etc.
///
/// # Safety
///
/// `input_ptr` and `output_ptr` must be valid device memory pointers with at least
/// `numel` elements of the appropriate dtype.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "unary", "activation")
/// * `op` - Operation name (e.g., "neg", "relu")
/// * `dtype` - Data type of the tensors
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_unary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}

/// Launch an element-wise binary kernel (two inputs, one output).
///
/// This handles the common pattern for operations like add, sub, mul, div, etc.
///
/// # Safety
///
/// All pointers must be valid device memory with at least `numel` elements.
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `module_name` - PTX module name (e.g., "binary", "compare")
/// * `op` - Operation name (e.g., "add", "eq")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    module_name: &'static str,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(op, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel '{}' launch failed: {:?}",
                module_name, op, e
            ))
        })?;

        Ok(())
    }
}
