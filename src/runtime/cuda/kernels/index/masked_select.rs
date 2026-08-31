//! Masked select kernel launchers, plus the mask count and prefix sum the
//! select depends on, in both the same-shape and broadcast-mask forms.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch masked_count kernel to count true elements in mask.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
pub unsafe fn launch_masked_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_kernel")?;

        let grid = elementwise_launch_config(n)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_count kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch masked_prefix_sum kernel to compute prefix sum of mask.
///
/// # Safety
///
/// - mask_ptr must be valid device memory with n u8 elements
/// - prefix_sum_ptr must be valid device memory with n u32 elements
pub unsafe fn launch_masked_prefix_sum(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_kernel")?;

        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch masked_select kernel.
///
/// Selects elements from input where mask is true, using precomputed prefix sum.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum
/// - output must have space for at least count_true elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let func_name = format!(
        "masked_select_{}",
        index_dtype_suffix(dtype, "masked_select")?
    );

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_select kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

// ============================================================================
// Broadcast mask
// ============================================================================

/// Launch broadcast masked_count kernel.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - count_ptr must be valid device memory with 1 u32 element (initialized to 0)
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_count_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    count_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_count_broadcast_kernel")?;

        let grid = elementwise_launch_config(n)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&count_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_count_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_prefix_sum kernel.
///
/// # Safety
///
/// - mask_ptr must be valid device memory
/// - prefix_sum_ptr must be valid device memory with n u32 elements
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_prefix_sum_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    mask_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "masked_prefix_sum_broadcast_kernel")?;

        let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&mask_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_prefix_sum_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_select kernel.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - prefix_sum must be precomputed via launch_masked_prefix_sum_broadcast
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_select_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    prefix_sum_ptr: u64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let func_name = format!(
        "masked_select_broadcast_{}",
        index_dtype_suffix(dtype, "masked_select_broadcast")?
    );

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n)?;
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        builder.arg(&prefix_sum_ptr);
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_select_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
