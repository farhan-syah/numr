//! im2col kernel launcher.
//!
//! Gathers the conv1d receptive fields into the `[N, C_in*K, L_out]` column
//! buffer the GEMM formulation of conv1d contracts over.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::launch_dims::launch_config;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// CUDA caps the y and z grid dimensions at 65535 blocks. Both axes carry a
/// grid-stride loop in the kernel, so the extents are clamped rather than
/// rejected.
const CUDA_MAX_GRID_YZ: usize = 65535;

/// Widest im2col block along the output axis.
const IM2COL_BLOCK_MAX: u32 = 256;

/// Whether this dtype has an `im2col1d_*` kernel.
///
/// The float widths conv1d accepts, minus FP8: FP8 matmul accumulates in F32
/// and has its own conv1d kernel, so it stays on the direct path.
#[inline]
pub fn im2col_has_kernel(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
}

/// Launch the conv1d im2col kernel.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor `(N, C_in, L)`
/// * `col_ptr` - Column buffer `(N, C_in*K, L_out)`
/// * `padding` - Resolved LEFT padding
///
/// # Safety
///
/// Both pointers must be valid device allocations of the sizes implied by the
/// shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_im2col1d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    col_ptr: u64,
    batch: usize,
    c_in: usize,
    length: usize,
    kernel_size: usize,
    output_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<()> {
    let rows = c_in * kernel_size;
    if batch == 0 || rows == 0 || output_length == 0 {
        return Ok(());
    }

    if !im2col_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "im2col1d",
        });
    }

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::IM2COL_MODULE)?;
        let func = get_kernel_function(&module, &kernel_name("im2col1d", dtype))?;

        // Threads walk consecutive output positions, so a short row gets a
        // narrow block instead of leaving most lanes idle.
        let block_x = (output_length as u32)
            .next_multiple_of(32)
            .clamp(32, IM2COL_BLOCK_MAX);
        let grid = (
            (output_length as u32).div_ceil(block_x),
            rows.min(CUDA_MAX_GRID_YZ) as u32,
            batch.min(CUDA_MAX_GRID_YZ) as u32,
        );
        let cfg = launch_config(grid, (block_x, 1, 1), 0);

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let length_u32 = length as u32;
        let kernel_size_u32 = kernel_size as u32;
        let output_length_u32 = output_length as u32;
        let stride_u32 = stride as u32;
        let padding_u32 = padding as u32;
        let dilation_u32 = dilation as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&col_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&length_u32);
        builder.arg(&kernel_size_u32);
        builder.arg(&output_length_u32);
        builder.arg(&stride_u32);
        builder.arg(&padding_u32);
        builder.arg(&dilation_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA im2col1d kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}
