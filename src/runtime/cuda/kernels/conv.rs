//! Convolution CUDA kernel launchers
//!
//! Provides launchers for convolution operations: conv1d, conv2d, depthwise_conv2d.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for convolution operations
pub const CONV_MODULE: &str = "conv";

/// Candidate conv1d block widths (threads per block along the output axis),
/// narrowest first. `output_length` is as small as ~26 at some hot shapes, so a
/// fixed 256-wide block would leave most lanes idle; the launcher picks the
/// narrowest width that still covers the row. The minimum is a full warp, which
/// the kernel relies on to keep each warp on a single output-channel slot.
const CONV1D_BLOCK_CANDIDATES: [u32; 4] = [32, 64, 128, 256];

/// Fallback conv1d block width when `output_length` exceeds every candidate.
const CONV1D_BLOCK_MAX: u32 = 256;

/// Target threads per conv1d block. A narrow row is padded out along the
/// output-channel axis (`blockDim.y`) instead of being launched as a 32-thread
/// block, because an SM holds at most 16 blocks and 32-thread blocks would cap
/// it at 16 of its 48 warp slots.
const CONV1D_BLOCK_THREADS: u32 = 128;

/// Output channels each thread of `conv1d_oc4_*` accumulates.
const CONV1D_OC_BLOCK: usize = 4;

/// CUDA caps the y and z grid dimensions at 65535 blocks.
const CUDA_MAX_GRID_YZ: usize = 65535;

// ============================================================================
// Conv1d
// ============================================================================

/// Launch conv1d kernel.
///
/// Performs 1D convolution with optional groups support.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C_in, L)
/// * `weight_ptr` - Weight tensor (C_out, C_in/groups, K)
/// * `bias_ptr` - Optional bias tensor (C_out,)
/// * `output_ptr` - Output tensor (N, C_out, L_out)
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_conv1d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    c_in: usize,
    length: usize,
    c_out: usize,
    kernel_size: usize,
    output_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let total = batch * c_out * output_length;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, CONV_MODULE)?;

        // Register-block over output channels when a group holds at least
        // CONV1D_OC_BLOCK of them. Depthwise (c_out_per_group == 1) and other
        // narrow-group shapes fall back to the scalar kernel, which is the
        // untiled kernel's work exactly.
        let c_out_per_group = c_out.checked_div(groups).unwrap_or(0);
        let oc_blocked =
            !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) && c_out_per_group >= CONV1D_OC_BLOCK;

        let base = if oc_blocked { "conv1d_oc4" } else { "conv1d" };
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // The FP8 conv1d kernel keeps its own legacy flat launch over a linear
        // index; only the macro-generated float kernels take the (ox, slot,
        // batch) grid that removes the per-thread integer division.
        let three_d_grid = !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2);

        let cfg = if three_d_grid {
            let block_x = CONV1D_BLOCK_CANDIDATES
                .into_iter()
                .find(|&w| output_length <= w as usize)
                .unwrap_or(CONV1D_BLOCK_MAX);
            let block_y = (CONV1D_BLOCK_THREADS / block_x).max(1);

            // One slot per output channel, or per chunk of CONV1D_OC_BLOCK
            // channels when the register-blocked kernel runs. Chunking is per
            // group so the channels a thread blocks over share a c_in range.
            let slots = if oc_blocked {
                groups * c_out_per_group.div_ceil(CONV1D_OC_BLOCK)
            } else {
                c_out
            };
            let grid_y = slots.div_ceil(block_y as usize);

            if grid_y > CUDA_MAX_GRID_YZ || batch > CUDA_MAX_GRID_YZ {
                return Err(Error::Internal(format!(
                    "CUDA conv1d: grid y={} z={} exceed the grid limit of {}",
                    grid_y, batch, CUDA_MAX_GRID_YZ
                )));
            }

            let grid = (
                (output_length as u32).div_ceil(block_x),
                grid_y as u32,
                batch as u32,
            );
            launch_config(grid, (block_x, block_y, 1), 0)
        } else {
            let grid = elementwise_launch_config(total);
            launch_config(grid, (BLOCK_SIZE, 1, 1), 0)
        };

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let length_u32 = length as u32;
        let c_out_u32 = c_out as u32;
        let kernel_size_u32 = kernel_size as u32;
        let output_length_u32 = output_length as u32;
        let stride_u32 = stride as u32;
        let padding_u32 = padding as u32;
        let dilation_u32 = dilation as u32;
        let groups_u32 = groups as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&length_u32);
        builder.arg(&c_out_u32);
        builder.arg(&kernel_size_u32);
        builder.arg(&output_length_u32);
        builder.arg(&stride_u32);
        builder.arg(&padding_u32);
        builder.arg(&dilation_u32);
        builder.arg(&groups_u32);
        builder.arg(&has_bias_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA conv1d kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// ConvTranspose1d
// ============================================================================

/// Launch conv_transpose1d kernel.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C_in, L)
/// * `weight_ptr` - Weight tensor (C_in, C_out/groups, K) — input channels lead
/// * `bias_ptr` - Optional bias tensor (C_out,)
/// * `output_ptr` - Output tensor (N, C_out, L_out)
/// * `padding` - Resolved LEFT padding; for transposed conv this TRIMS the output
///
/// # Safety
///
/// Caller must ensure all pointers are valid device allocations of the sizes
/// implied by the shape arguments.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_conv_transpose1d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    c_in: usize,
    length: usize,
    c_out: usize,
    kernel_size: usize,
    output_length: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<()> {
    let total = batch * c_out * output_length;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, CONV_MODULE)?;
        let func_name = kernel_name("conv_transpose1d", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let length_u32 = length as u32;
        let c_out_u32 = c_out as u32;
        let kernel_size_u32 = kernel_size as u32;
        let output_length_u32 = output_length as u32;
        let stride_u32 = stride as u32;
        let padding_u32 = padding as u32;
        let dilation_u32 = dilation as u32;
        let groups_u32 = groups as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&length_u32);
        builder.arg(&c_out_u32);
        builder.arg(&kernel_size_u32);
        builder.arg(&output_length_u32);
        builder.arg(&stride_u32);
        builder.arg(&padding_u32);
        builder.arg(&dilation_u32);
        builder.arg(&groups_u32);
        builder.arg(&has_bias_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA conv_transpose1d kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

// ============================================================================
// Conv2d
// ============================================================================

/// Launch conv2d kernel.
///
/// Performs 2D convolution with optional groups support.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C_in, H, W)
/// * `weight_ptr` - Weight tensor (C_out, C_in/groups, K_h, K_w)
/// * `bias_ptr` - Optional bias tensor (C_out,)
/// * `output_ptr` - Output tensor (N, C_out, H_out, W_out)
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_conv2d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    c_in: usize,
    height: usize,
    width: usize,
    c_out: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    groups: usize,
) -> Result<()> {
    let total = batch * c_out * output_h * output_w;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, CONV_MODULE)?;
        let func_name = kernel_name("conv2d", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let batch_u32 = batch as u32;
        let c_in_u32 = c_in as u32;
        let height_u32 = height as u32;
        let width_u32 = width as u32;
        let c_out_u32 = c_out as u32;
        let kernel_h_u32 = kernel_h as u32;
        let kernel_w_u32 = kernel_w as u32;
        let output_h_u32 = output_h as u32;
        let output_w_u32 = output_w as u32;
        let stride_h_u32 = stride_h as u32;
        let stride_w_u32 = stride_w as u32;
        let pad_h_u32 = pad_h as u32;
        let pad_w_u32 = pad_w as u32;
        let dilation_h_u32 = dilation_h as u32;
        let dilation_w_u32 = dilation_w as u32;
        let groups_u32 = groups as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&c_in_u32);
        builder.arg(&height_u32);
        builder.arg(&width_u32);
        builder.arg(&c_out_u32);
        builder.arg(&kernel_h_u32);
        builder.arg(&kernel_w_u32);
        builder.arg(&output_h_u32);
        builder.arg(&output_w_u32);
        builder.arg(&stride_h_u32);
        builder.arg(&stride_w_u32);
        builder.arg(&pad_h_u32);
        builder.arg(&pad_w_u32);
        builder.arg(&dilation_h_u32);
        builder.arg(&dilation_w_u32);
        builder.arg(&groups_u32);
        builder.arg(&has_bias_u32);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA conv2d kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}

// ============================================================================
// Depthwise Conv2d
// ============================================================================

/// Launch depthwise_conv2d kernel.
///
/// Performs depthwise 2D convolution where each channel is convolved independently.
///
/// # Arguments
///
/// * `input_ptr` - Input tensor (N, C, H, W)
/// * `weight_ptr` - Weight tensor (C, 1, K_h, K_w)
/// * `bias_ptr` - Optional bias tensor (C,)
/// * `output_ptr` - Output tensor (N, C, H_out, W_out)
///
/// # Safety
///
/// All pointers must be valid device memory with sufficient size.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_depthwise_conv2d(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    weight_ptr: u64,
    bias_ptr: Option<u64>,
    output_ptr: u64,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    output_h: usize,
    output_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    dilation_h: usize,
    dilation_w: usize,
) -> Result<()> {
    let total = batch * channels * output_h * output_w;
    if total == 0 {
        return Ok(());
    }

    unsafe {
        let module = get_or_load_module(context, device_index, CONV_MODULE)?;
        let func_name = kernel_name("depthwise_conv2d", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(total);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let batch_u32 = batch as u32;
        let channels_u32 = channels as u32;
        let height_u32 = height as u32;
        let width_u32 = width as u32;
        let kernel_h_u32 = kernel_h as u32;
        let kernel_w_u32 = kernel_w as u32;
        let output_h_u32 = output_h as u32;
        let output_w_u32 = output_w as u32;
        let stride_h_u32 = stride_h as u32;
        let stride_w_u32 = stride_w as u32;
        let pad_h_u32 = pad_h as u32;
        let pad_w_u32 = pad_w as u32;
        let dilation_h_u32 = dilation_h as u32;
        let dilation_w_u32 = dilation_w as u32;
        let has_bias_u32: u32 = if bias_ptr.is_some() { 1 } else { 0 };
        let bias_ptr_val = bias_ptr.unwrap_or(0);

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&weight_ptr);
        builder.arg(&bias_ptr_val);
        builder.arg(&output_ptr);
        builder.arg(&batch_u32);
        builder.arg(&channels_u32);
        builder.arg(&height_u32);
        builder.arg(&width_u32);
        builder.arg(&kernel_h_u32);
        builder.arg(&kernel_w_u32);
        builder.arg(&output_h_u32);
        builder.arg(&output_w_u32);
        builder.arg(&stride_h_u32);
        builder.arg(&stride_w_u32);
        builder.arg(&pad_h_u32);
        builder.arg(&pad_w_u32);
        builder.arg(&dilation_h_u32);
        builder.arg(&dilation_w_u32);
        builder.arg(&has_bias_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA depthwise_conv2d kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}
