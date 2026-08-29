//! Scatter and copy kernel launchers.

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

/// Maximum number of tensor dimensions supported by the scatter kernel.
/// Must match INDEX_MAX_DIMS in index_ops.cuh.
const MAX_DIMS: usize = 8;

/// Launch scatter kernel.
///
/// Scatters values from src to output at positions specified by indices.
/// `output[i][indices[i][j][k]][k] = src[i][j][k]` (when dim=1)
///
/// Shape and stride arrays are passed as individual scalar kernel arguments (not
/// device pointers) so this launcher is safe for CUDA graph capture/replay.
///
/// # Errors
///
/// Returns `Error::BackendLimitation` if `ndim > MAX_DIMS`.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Output must be pre-initialized (typically a copy of input)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_scatter(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    src_ptr: u64,
    output_ptr: u64,
    ndim: usize,
    dim: usize,
    output_shape: &[u32],
    output_strides: &[u32],
    src_shape: &[u32],
    src_strides: &[u32],
    src_total: usize,
) -> Result<()> {
    if src_total == 0 {
        return Ok(());
    }

    if ndim > MAX_DIMS {
        return Err(Error::BackendLimitation {
            backend: "CUDA",
            operation: "scatter",
            reason: format!(
                "tensor has {} dimensions but scatter kernel supports at most {}",
                ndim, MAX_DIMS
            ),
        });
    }

    // Build zero-padded stack arrays
    let mut output_shape_args = [0u32; MAX_DIMS];
    let mut output_strides_args = [0u32; MAX_DIMS];
    let mut src_shape_args = [0u32; MAX_DIMS];
    let mut src_strides_args = [0u32; MAX_DIMS];

    for i in 0..ndim {
        output_shape_args[i] = output_shape[i];
        output_strides_args[i] = output_strides[i];
        src_shape_args[i] = src_shape[i];
        src_strides_args[i] = src_strides[i];
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("scatter_{}", index_dtype_suffix(dtype, "scatter")?);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(src_total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let dim_u32 = dim as u32;
        let src_total_u32 = src_total as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&indices_ptr);
        builder.arg(&src_ptr);
        builder.arg(&output_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&dim_u32);
        // Pass output_shape as 8 individual u32 args
        for i in 0..MAX_DIMS {
            builder.arg(&output_shape_args[i]);
        }
        // Pass output_strides as 8 individual u32 args
        for i in 0..MAX_DIMS {
            builder.arg(&output_strides_args[i]);
        }
        // Pass src_shape as 8 individual u32 args
        for i in 0..MAX_DIMS {
            builder.arg(&src_shape_args[i]);
        }
        // Pass src_strides as 8 individual u32 args
        for i in 0..MAX_DIMS {
            builder.arg(&src_strides_args[i]);
        }
        builder.arg(&src_total_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA scatter kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Launch copy kernel for scatter initialization.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - dst must have space for n elements
pub unsafe fn launch_copy(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func_name = format!("copy_{}", index_dtype_suffix(dtype, "copy")?);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&n_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA copy kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}
