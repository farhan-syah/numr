//! Index select and index bounds validation kernel launchers

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

/// Launch index_select kernel.
///
/// Selects elements along a dimension using a 1D index tensor.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - indices must be a 1D tensor of i64 values
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_select(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    index_len: usize,
) -> Result<()> {
    let total = outer_size * index_len * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!(
            "index_select_{}",
            index_dtype_suffix(dtype, "index_select")?
        );
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;
        let index_len_u32 = index_len as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&index_len_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA index_select kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Puts values at specified indices along a dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - indices must be a 1D tensor of i64 values
/// - output must already contain a copy of the input tensor
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_index_put(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    indices_ptr: u64,
    src_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
    index_len: usize,
) -> Result<()> {
    let total = outer_size * index_len * inner_size;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("index_put_{}", index_dtype_suffix(dtype, "index_put")?);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let outer_u32 = outer_size as u32;
        let dim_u32 = dim_size as u32;
        let inner_u32 = inner_size as u32;
        let index_len_u32 = index_len as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer_u32);
        builder.arg(&dim_u32);
        builder.arg(&inner_u32);
        builder.arg(&index_len_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA index_put kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch index bounds validation kernel.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Returns the count of out-of-bounds indices in error_count buffer.
///
/// # Safety
///
/// - indices_ptr must be valid device memory with index_len i64 elements
/// - error_count_ptr must be valid device memory with 1 u32 element (initialized to 0)
pub unsafe fn launch_validate_indices(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    indices_ptr: u64,
    error_count_ptr: u64,
    index_len: usize,
    dim_size: usize,
) -> Result<()> {
    if index_len == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, "validate_indices_kernel")?;

        let grid = elementwise_launch_config(index_len);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let index_len_u32 = index_len as u32;
        let dim_size_u32 = dim_size as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&indices_ptr);
        builder.arg(&error_count_ptr);
        builder.arg(&index_len_u32);
        builder.arg(&dim_size_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA validate_indices kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
