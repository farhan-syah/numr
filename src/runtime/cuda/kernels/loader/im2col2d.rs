//! im2col2d kernel launcher.
//!
//! Gathers the conv2d receptive fields into the `[N, C_in*Kh*Kw, H_out*W_out]`
//! column buffer the GEMM formulation of conv2d contracts over.

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

/// Widest im2col2d block along the flattened spatial axis.
const IM2COL2D_BLOCK_MAX: u32 = 256;

/// Whether this dtype has an `im2col2d_*` kernel.
///
/// The float widths conv2d accepts, minus FP8: FP8 matmul accumulates in F32
/// and has its own conv2d kernel, so it stays on the direct path.
#[inline]
pub fn im2col2d_has_kernel(dtype: DType) -> bool {
    matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
}

/// Launch the conv2d im2col kernel.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor `(N, C_in, H, W)`
/// * `col_ptr` - Column buffer `(N, C_in*Kh*Kw, H_out*W_out)`
/// * `pad_top`, `pad_left` - Resolved TOP/LEFT padding
///
/// # Safety
///
/// Both pointers must be valid device allocations of the sizes implied by the
/// shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_im2col2d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    col_ptr: u64,
    batch: usize,
    c_in: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_top: usize,
    pad_left: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<()> {
    let rows = c_in * kernel_h * kernel_w;
    let spatial = output_h * output_w;
    if batch == 0 || rows == 0 || spatial == 0 {
        return Ok(());
    }

    if !im2col2d_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "im2col2d",
        });
    }

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::IM2COL2D_MODULE)?;
        let func = get_kernel_function(&module, &kernel_name("im2col2d", dtype))?;

        // Threads walk consecutive flattened output positions, so a short
        // spatial extent gets a narrow block instead of leaving most lanes idle.
        let block_x = (spatial as u32)
            .next_multiple_of(32)
            .clamp(32, IM2COL2D_BLOCK_MAX);
        let grid = (
            (spatial as u32).div_ceil(block_x),
            rows.min(CUDA_MAX_GRID_YZ) as u32,
            batch.min(CUDA_MAX_GRID_YZ) as u32,
        );
        let cfg = launch_config(grid, (block_x, 1, 1), 0);

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let height_u32 = height as u32;
        let width_u32 = width as u32;
        let kernel_h_u32 = kernel_h as u32;
        let kernel_w_u32 = kernel_w as u32;
        let output_h_u32 = output_h as u32;
        let output_w_u32 = output_w as u32;
        let stride_h_u32 = stride_h as u32;
        let stride_w_u32 = stride_w as u32;
        let pad_top_u32 = pad_top as u32;
        let pad_left_u32 = pad_left as u32;
        let dilation_h_u32 = dilation_h as u32;
        let dilation_w_u32 = dilation_w as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&col_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&height_u32);
        builder.arg(&width_u32);
        builder.arg(&kernel_h_u32);
        builder.arg(&kernel_w_u32);
        builder.arg(&output_h_u32);
        builder.arg(&output_w_u32);
        builder.arg(&stride_h_u32);
        builder.arg(&stride_w_u32);
        builder.arg(&pad_top_u32);
        builder.arg(&pad_left_u32);
        builder.arg(&dilation_h_u32);
        builder.arg(&dilation_w_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA im2col2d kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}
