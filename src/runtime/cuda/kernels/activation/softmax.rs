//! Softmax CUDA kernel launchers (forward + backward)
//!
//! Kernel source: softmax.cu

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_name, kernel_names, launch_config,
    softmax_launch_config,
};

/// Launch softmax over the last dimension.
///
/// Uses shared memory for parallel reduction of max and sum values.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * dim_size` elements
/// - `output_ptr` must have `outer_size * dim_size` elements
pub unsafe fn launch_softmax(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
        let outer = outer_size as u32;
        let dim = dim_size as u32;

        let shared_mem = if dtype == DType::F64 {
            shared_mem * 2
        } else {
            shared_mem
        };

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&dim);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA softmax kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

/// Minimum device waves the flat `softmax_dim` grid must cover before it takes
/// the full block width.
const SOFTMAX_DIM_MIN_WAVES: usize = 2;

/// Block width for the flattened `(outer, inner)` launch.
///
/// The work here is `outer * inner` threads, independent of `dim_size`, so a
/// shape with a small `inner` yields few threads however long the reduction is.
/// At the full block width those pack into one or two blocks and occupy a
/// fraction of the device, which is slower than the wider grid a narrower block
/// produces. Stay a warp wide at minimum so consecutive threads still cover
/// consecutive `inner` indices and coalesce.
fn softmax_dim_block_width(device_index: usize, total: usize) -> u32 {
    let profile = CudaDevice::new(device_index).profile();
    let compute_units = profile.compute_units as usize;
    let target_blocks = compute_units.saturating_mul(SOFTMAX_DIM_MIN_WAVES);
    if target_blocks == 0 {
        return BLOCK_SIZE;
    }
    let warp_size = profile.lane_width.max(1) as usize;
    let per_block = total.div_ceil(target_blocks).max(warp_size);
    (per_block.next_power_of_two() as u32).min(BLOCK_SIZE)
}

/// Grid and block for the flattened `(outer, inner)` launch shared by
/// `softmax_dim` and `softmax_bwd_dim`: one thread per pair, so each warp
/// covers consecutive `inner` indices and coalesces.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `outer_size * inner_size` needs
/// more than `u32::MAX` blocks. The kernels this config drives decode a flat
/// `tid < outer_size * inner_size` index with no `y`/`z` grid component, so
/// truncating a too-large grid count through `as u32` would silently launch
/// too few threads and skip elements instead of failing loudly.
fn softmax_dim_grid(
    device_index: usize,
    outer_size: usize,
    inner_size: usize,
) -> Result<((u32, u32, u32), u32)> {
    let total = outer_size.saturating_mul(inner_size);
    let block_x = softmax_dim_block_width(device_index, total);
    let grid_x = total.div_ceil(block_x as usize).max(1);
    if grid_x > u32::MAX as usize {
        return Err(Error::InvalidArgument {
            arg: "outer_size * inner_size",
            reason: format!(
                "{total} elements need a 1-D grid of {grid_x} blocks, exceeding the \
                 CUDA max grid extent of {}",
                u32::MAX
            ),
        });
    }
    Ok(((grid_x as u32, 1, 1), block_x))
}

/// Launch softmax over a non-last dimension.
///
/// For shape `[A, B, C]` with softmax over dim=1: outer=A, dim=B, inner=C.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have `outer_size * dim_size * inner_size` elements
pub unsafe fn launch_softmax_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block_x) = softmax_dim_grid(device_index, outer_size, inner_size)?;
        let outer = outer_size as u32;
        let dim = dim_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block_x, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&dim);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA softmax_dim kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch softmax backward kernel (last dimension).
///
/// Computes: d_input = output * (grad - sum(grad * output))
///
/// # Safety
/// - All pointers must be valid device memory of `outer_size * dim_size` elements
pub unsafe fn launch_softmax_bwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    output_ptr: u64,
    d_input_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_bwd", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
        let outer = outer_size as u32;
        let dim = dim_size as u32;

        let shared_mem = if dtype == DType::F64 {
            shared_mem * 2
        } else {
            shared_mem
        };

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&output_ptr);
        builder.arg(&d_input_ptr);
        builder.arg(&outer);
        builder.arg(&dim);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA softmax_bwd kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch fused softmax-with-bias over the last dimension.
///
/// Computes `softmax(a + bias, last_dim)` in a single kernel pass.
/// The bias must have `dim_size` elements (the last-dim size); it is applied
/// element-wise by position, broadcasting over all outer dimensions of `a`.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` and `output_ptr` must have `outer_size * dim_size` elements
/// - `bias_ptr` must have at least `dim_size` elements (bias cycles by `dim_size`)
pub unsafe fn launch_softmax_with_bias(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    bias_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    dim_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_bias", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size, shared_mem) = softmax_launch_config(outer_size, dim_size);
        let outer = outer_size as u32;
        let dim = dim_size as u32;

        let shared_mem = if dtype == DType::F64 {
            shared_mem * 2
        } else {
            shared_mem
        };

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&dim);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA softmax_with_bias kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

/// Launch softmax backward kernel (non-last dimension).
///
/// # Safety
/// - All pointers must be valid device memory
pub unsafe fn launch_softmax_bwd_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    output_ptr: u64,
    d_input_ptr: u64,
    outer_size: usize,
    dim_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::SOFTMAX_MODULE)?;
        let func_name = kernel_name("softmax_bwd_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block_x) = softmax_dim_grid(device_index, outer_size, inner_size)?;
        let outer = outer_size as u32;
        let dim = dim_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block_x, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&grad_ptr);
        builder.arg(&output_ptr);
        builder.arg(&d_input_ptr);
        builder.arg(&outer);
        builder.arg(&dim);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA softmax_bwd_dim kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
