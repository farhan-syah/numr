//! Cumulative operation CUDA kernel launchers
//!
//! Provides launchers for cumulative operations:
//! - `cumsum` - Cumulative sum along a dimension
//! - `cumprod` - Cumulative product along a dimension
//! - `logsumexp` - Numerically stable log-sum-exp reduction

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, cumulative_module, elementwise_launch_config, get_kernel_function,
    get_or_load_module, kernel_name, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Cumulative Sum
// ============================================================================

/// Launch a cumsum kernel for contiguous data.
///
/// Each thread processes one outer segment sequentially.
///
/// # Safety
///
/// - `input_ptr` must be valid device memory with at least `scan_size * outer_size` elements
/// - `output_ptr` must be valid device memory with at least `scan_size * outer_size` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of tensors
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `scan_size` - Number of elements to scan per segment
/// * `outer_size` - Number of independent scans
pub unsafe fn launch_cumsum(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    scan_size: usize,
    outer_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("cumsum", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(outer_size)?;
    let block = (BLOCK_SIZE, 1, 1);
    let scan_size_u32 = scan_size as u32;
    let outer_size_u32 = outer_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&scan_size_u32);
        builder.arg(&outer_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA cumsum kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a strided cumsum kernel.
///
/// For cumsum along non-last dimensions.
///
/// # Safety
///
/// Same as `launch_cumsum`.
///
/// # Arguments
///
/// * `inner_size` - Stride between consecutive elements in scan dimension
pub unsafe fn launch_cumsum_strided(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("cumsum_strided", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let total_inner = outer_size * inner_size;
    let grid = elementwise_launch_config(total_inner)?;
    let block = (BLOCK_SIZE, 1, 1);
    let scan_size_u32 = scan_size as u32;
    let outer_size_u32 = outer_size as u32;
    let inner_size_u32 = inner_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&scan_size_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&inner_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA cumsum_strided kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Cumulative Product
// ============================================================================

/// Launch a cumprod kernel for contiguous data.
///
/// Each thread processes one outer segment sequentially.
///
/// # Safety
///
/// - `input_ptr` must be valid device memory with at least `scan_size * outer_size` elements
/// - `output_ptr` must be valid device memory with at least `scan_size * outer_size` elements
pub unsafe fn launch_cumprod(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    scan_size: usize,
    outer_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("cumprod", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(outer_size)?;
    let block = (BLOCK_SIZE, 1, 1);
    let scan_size_u32 = scan_size as u32;
    let outer_size_u32 = outer_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&scan_size_u32);
        builder.arg(&outer_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA cumprod kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a strided cumprod kernel.
///
/// For cumprod along non-last dimensions.
///
/// # Safety
///
/// Same as `launch_cumprod`.
pub unsafe fn launch_cumprod_strided(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("cumprod_strided", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let total_inner = outer_size * inner_size;
    let grid = elementwise_launch_config(total_inner)?;
    let block = (BLOCK_SIZE, 1, 1);
    let scan_size_u32 = scan_size as u32;
    let outer_size_u32 = outer_size as u32;
    let inner_size_u32 = inner_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&scan_size_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&inner_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA cumprod_strided kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Log-Sum-Exp (Reduction)
// ============================================================================

/// Launch a logsumexp kernel for contiguous data.
///
/// Computes log(sum(exp(x))) in a numerically stable way.
/// This is a reduction operation (not a scan).
///
/// # Safety
///
/// - `input_ptr` must be valid device memory with at least `reduce_size * outer_size` elements
/// - `output_ptr` must be valid device memory with at least `outer_size` elements
///
/// # Arguments
///
/// * `reduce_size` - Number of elements to reduce per segment
/// * `outer_size` - Number of independent reductions
pub unsafe fn launch_logsumexp(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    reduce_size: usize,
    outer_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("logsumexp", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(outer_size)?;
    let block = (BLOCK_SIZE, 1, 1);
    let reduce_size_u32 = reduce_size as u32;
    let outer_size_u32 = outer_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&reduce_size_u32);
        builder.arg(&outer_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA logsumexp kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a strided logsumexp kernel.
///
/// For logsumexp along non-last dimensions.
///
/// # Safety
///
/// - `input_ptr` must be valid device memory
/// - `output_ptr` must be valid device memory with at least `outer_size * inner_size` elements
///
/// # Arguments
///
/// * `reduce_size` - Number of elements to reduce per segment
/// * `outer_size` - Number of outer dimensions
/// * `inner_size` - Number of inner dimensions
pub unsafe fn launch_logsumexp_strided(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    reduce_size: usize,
    outer_size: usize,
    inner_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, cumulative_module(dtype))?;
    let func_name = kernel_name("logsumexp_strided", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let total_inner = outer_size * inner_size;
    let grid = elementwise_launch_config(total_inner)?;
    let block = (BLOCK_SIZE, 1, 1);
    let reduce_size_u32 = reduce_size as u32;
    let outer_size_u32 = outer_size as u32;
    let inner_size_u32 = inner_size as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&reduce_size_u32);
        builder.arg(&outer_size_u32);
        builder.arg(&inner_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA logsumexp_strided kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}
