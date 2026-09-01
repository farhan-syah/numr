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
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

/// Module name for convolution operations
pub const CONV_MODULE: &str = "conv";

/// Candidate conv1d block widths (threads per block along the output axis),
/// narrowest first. `output_length` is as small as ~26 at some hot shapes, so a
/// fixed 256-wide block would leave most lanes idle; the launcher picks the
/// narrowest width that still covers the row.
///
/// A full warp is the floor for `conv1d_oc4` only, which needs each warp on one
/// output-channel slot so its four stores stay coalesced. The scalar kernel
/// indexes `ox`, `oc` and `batch` independently and carries no such assumption,
/// so it takes [`CONV1D_BLOCK_NARROW`] below a warp instead of leaving most of
/// the warp idle.
const CONV1D_BLOCK_CANDIDATES: [u32; 4] = [32, 64, 128, 256];

/// Sub-warp block widths for the scalar kernel, narrowest first. Decode-shaped
/// convolutions run at `output_length` of 1, where a 32-wide block retires 31
/// lanes at the bounds check before they do any work. Threads freed here go to
/// the output-channel axis, which always has work.
const CONV1D_BLOCK_NARROW: [u32; 5] = [1, 2, 4, 8, 16];

/// Fallback conv1d block width when `output_length` exceeds every candidate.
const CONV1D_BLOCK_MAX: u32 = 256;

/// Target threads per block, shared by conv1d and depthwise_conv2d's
/// row/position-indexed kernels. conv1d pads a narrow row out along the second
/// grid axis (`blockDim.y`, output channels) instead of launching a 32-thread
/// block, because an SM holds at most 16 blocks and 32-thread blocks would cap
/// it at 16 of its 48 warp slots. `depthwise_conv2d_ox` reaches the same count
/// with a flat block: it folds the output-row axis into x, so it needs no
/// second block axis to pad with.
const CONV_BLOCK_THREADS: u32 = 128;

/// Output channels each thread of `conv1d_oc4_*` accumulates.
const CONV1D_OC_BLOCK: usize = 4;

/// Consecutive output positions each thread of `conv1d_ox_*` accumulates.
/// Must match `CONV1D_OX_BLOCK` in `conv1d_ox.cu`.
const CONV1D_OX_BLOCK: usize = 4;

/// Module name for the position-blocked conv1d kernel (`conv1d_ox.cu` compiles
/// to its own fatbin, separate from [`CONV_MODULE`]).
const CONV1D_OX_MODULE: &str = "conv1d_ox";

/// Minimum `output_length` before `conv1d_ox` is preferred over the scalar
/// `conv1d` kernel for depthwise/narrow-group shapes (the oc4 kernel is
/// unavailable there). Guarantees the row covers at least one full
/// [`CONV1D_OX_BLOCK`]-wide chunk plus a remainder.
const CONV1D_OX_MIN_OUTPUT_LENGTH: usize = 2 * CONV1D_OX_BLOCK;

/// Device waves the position-blocked grid must still reach after blocking.
///
/// `output_length` alone does not gate this kernel. Blocking divides the thread
/// count by [`CONV1D_OX_BLOCK`], so a shape with few channels can have a long
/// row and still be left with too few threads to fill the device: a narrow,
/// low-channel-count shape can regress badly versus the untiled kernel once
/// blocking leaves only a handful of warps to hide memory latency. One wave is
/// `compute_units * CONV_BLOCK_THREADS`; two is the smallest count that
/// rejects that shape while keeping every depthwise case that gains.
const CONV1D_OX_MIN_WAVES: usize = 2;

/// Consecutive output columns each thread of `depthwise_conv2d_ox_*`
/// accumulates. Must match `DEPTHWISE_CONV2D_OX_BLOCK` in
/// `depthwise_conv2d_ox.cu`.
const DEPTHWISE_CONV2D_OX_BLOCK: usize = 4;

/// Module name for the column-blocked depthwise conv2d kernel
/// (`depthwise_conv2d_ox.cu` compiles to its own fatbin, separate from
/// [`CONV_MODULE`]).
const DEPTHWISE_CONV2D_OX_MODULE: &str = "depthwise_conv2d_ox";

/// Minimum `output_w` before `depthwise_conv2d_ox` is preferred over the flat
/// kernel. Guarantees the row covers at least one full
/// [`DEPTHWISE_CONV2D_OX_BLOCK`]-wide chunk plus a remainder.
const DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH: usize = 2 * DEPTHWISE_CONV2D_OX_BLOCK;

/// Device waves the column-blocked depthwise grid must still reach after
/// blocking. Same reasoning as [`CONV1D_OX_MIN_WAVES`]: blocking divides the
/// thread count by [`DEPTHWISE_CONV2D_OX_BLOCK`], so a wide row is not on its
/// own enough to keep the device fed once channels and rows are few.
const DEPTHWISE_CONV2D_OX_MIN_WAVES: usize = 2;

/// CUDA caps the y and z grid dimensions at 65535 blocks.
const CUDA_MAX_GRID_YZ: usize = 65535;

/// Block width (threads along the output-position axis) for the kernels that
/// index position, channel and batch independently, so may go below a warp.
/// `x_extent` is the number of thread slots the axis needs, already divided by
/// the per-thread blocking factor where one applies.
fn position_block_width(x_extent: usize) -> u32 {
    CONV1D_BLOCK_NARROW
        .into_iter()
        .find(|&w| x_extent <= w as usize)
        .unwrap_or_else(|| {
            CONV1D_BLOCK_CANDIDATES
                .into_iter()
                .find(|&w| x_extent <= w as usize)
                .unwrap_or(CONV1D_BLOCK_MAX)
        })
}

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
        // Register-block over output channels when a group holds at least
        // CONV1D_OC_BLOCK of them. Depthwise (c_out_per_group == 1) and other
        // narrow-group shapes fall back to conv1d_ox (position-blocked) when
        // the row is wide enough, else the untiled scalar kernel.
        let c_out_per_group = c_out.checked_div(groups).unwrap_or(0);
        let c_in_per_group = c_in.checked_div(groups).unwrap_or(0);
        let is_fp8 = matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2);
        let oc_blocked = !is_fp8 && c_out_per_group >= CONV1D_OC_BLOCK;
        // Threads the position-blocked grid would launch, and the wave it must
        // fill. `compute_units` is 0 on an unknown profile, which makes the
        // wave test trivially true and leaves the row-length gate deciding.
        let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
        let ox_threads = batch
            .saturating_mul(c_out)
            .saturating_mul(output_length.div_ceil(CONV1D_OX_BLOCK));
        let ox_wave = compute_units
            .saturating_mul(CONV_BLOCK_THREADS as usize)
            .saturating_mul(CONV1D_OX_MIN_WAVES);
        let ox_blocked = !is_fp8
            && !oc_blocked
            && output_length >= CONV1D_OX_MIN_OUTPUT_LENGTH
            && ox_threads >= ox_wave;

        let base = if oc_blocked {
            "conv1d_oc4"
        } else if ox_blocked {
            "conv1d_ox"
        } else {
            "conv1d"
        };
        let module_name = if ox_blocked {
            CONV1D_OX_MODULE
        } else {
            CONV_MODULE
        };
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        // The FP8 conv1d kernel keeps its own legacy flat launch over a linear
        // index; only the macro-generated float kernels take the (ox, slot,
        // batch) grid that removes the per-thread integer division.
        let three_d_grid = !is_fp8;

        let cfg = if three_d_grid {
            // conv1d_ox packs CONV1D_OX_BLOCK output positions per thread, so
            // the x axis walks blocks-of-CONV1D_OX_BLOCK, not raw positions.
            let x_extent = if ox_blocked {
                output_length.div_ceil(CONV1D_OX_BLOCK)
            } else {
                output_length
            };

            // oc4 keeps the warp floor; the scalar and ox kernels may go
            // narrower.
            let block_x = if oc_blocked {
                CONV1D_BLOCK_CANDIDATES
                    .into_iter()
                    .find(|&w| x_extent <= w as usize)
                    .unwrap_or(CONV1D_BLOCK_MAX)
            } else {
                position_block_width(x_extent)
            };
            let block_y = (CONV_BLOCK_THREADS / block_x).max(1);

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
                (x_extent as u32).div_ceil(block_x),
                grid_y as u32,
                batch as u32,
            );
            launch_config(grid, (block_x, block_y, 1), 0)
        } else {
            let grid = elementwise_launch_config(total)?;
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
        let c_in_per_group_u32 = c_in_per_group as u32;
        let c_out_per_group_u32 = c_out_per_group as u32;
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
        // The FP8 conv1d kernel keeps its own legacy flat signature (see
        // `three_d_grid` above) and was never given these two host-computed
        // params, so only the macro-generated kernels receive them.
        if three_d_grid {
            builder.arg(&c_in_per_group_u32);
            builder.arg(&c_out_per_group_u32);
        }
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

        let grid = elementwise_launch_config(total)?;
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

        let grid = elementwise_launch_config(total)?;
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
        // Register-block over output columns when the row is wide enough and
        // the blocked grid still fills the device. FP8 keeps its own legacy
        // flat kernel, as it does for conv1d.
        let is_fp8 = matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2);
        // `compute_units` is 0 on an unknown profile, which makes the wave
        // test trivially true and leaves the row-width gate deciding.
        let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
        let x_extent = output_w.div_ceil(DEPTHWISE_CONV2D_OX_BLOCK);
        // The blocked kernel folds (oy, column-block) onto the x axis and keeps
        // channel and batch on y and z.
        let x_work = output_h.saturating_mul(x_extent);
        let ox_threads = batch.saturating_mul(channels).saturating_mul(x_work);
        let ox_wave = compute_units
            .saturating_mul(CONV_BLOCK_THREADS as usize)
            .saturating_mul(DEPTHWISE_CONV2D_OX_MIN_WAVES);

        let ox_blocked = !is_fp8
            && output_w >= DEPTHWISE_CONV2D_OX_MIN_OUTPUT_WIDTH
            && ox_threads >= ox_wave
            // A shape that overflows the channel y axis or the batch z axis
            // falls back to the flat kernel instead of failing. The x axis
            // carries the folded work and is bounded far higher, so it needs
            // no gate.
            && channels <= CUDA_MAX_GRID_YZ
            && batch <= CUDA_MAX_GRID_YZ;

        let base = if ox_blocked {
            "depthwise_conv2d_ox"
        } else {
            "depthwise_conv2d"
        };
        let module_name = if ox_blocked {
            DEPTHWISE_CONV2D_OX_MODULE
        } else {
            CONV_MODULE
        };
        let module = get_or_load_module(context, device_index, module_name)?;
        let func_name = kernel_name(base, dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let cfg = if ox_blocked {
            let grid = (
                (x_work as u32).div_ceil(CONV_BLOCK_THREADS),
                channels as u32,
                batch as u32,
            );
            launch_config(grid, (CONV_BLOCK_THREADS, 1, 1), 0)
        } else {
            let grid = elementwise_launch_config(total)?;
            launch_config(grid, (BLOCK_SIZE, 1, 1), 0)
        };

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

#[cfg(test)]
mod tests {
    use super::{CONV1D_OC_BLOCK, CONV1D_OX_BLOCK, DEPTHWISE_CONV2D_OX_BLOCK};

    /// Blocking factors that appear in BOTH the kernel source and this
    /// launcher. The launcher sizes the grid from them and the kernel decides
    /// how much work a thread does, so a mismatch does not fail to build — it
    /// silently leaves outputs uncomputed. Parse the kernel and check.
    fn kernel_define(source: &str, name: &str) -> usize {
        let needle = format!("#define {name} ");
        let line = source
            .lines()
            .find(|l| l.starts_with(&needle))
            .unwrap_or_else(|| panic!("{name} is not defined in the kernel source"));
        line[needle.len()..]
            .trim()
            .trim_end_matches('u')
            .parse()
            .unwrap_or_else(|e| panic!("{name} is not a plain integer literal: {e}"))
    }

    #[test]
    fn oc_block_matches_the_kernel() {
        let source = include_str!("conv.cu");
        assert_eq!(
            kernel_define(source, "CONV1D_OC_BLOCK"),
            CONV1D_OC_BLOCK,
            "conv.cu and conv.rs disagree on the oc4 blocking factor"
        );
    }

    #[test]
    fn ox_block_matches_the_kernel() {
        let source = include_str!("conv1d_ox.cu");
        assert_eq!(
            kernel_define(source, "CONV1D_OX_BLOCK"),
            CONV1D_OX_BLOCK,
            "conv1d_ox.cu and conv.rs disagree on the position blocking factor"
        );
    }

    #[test]
    fn depthwise_ox_block_matches_the_kernel() {
        let source = include_str!("depthwise_conv2d_ox.cu");
        assert_eq!(
            kernel_define(source, "DEPTHWISE_CONV2D_OX_BLOCK"),
            DEPTHWISE_CONV2D_OX_BLOCK,
            "depthwise_conv2d_ox.cu and conv.rs disagree on the column blocking factor"
        );
    }
}
