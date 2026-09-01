//! im2col + GEMM formulation of CUDA conv2d.
//!
//! The direct conv2d kernel gives one thread per output element and re-reads
//! the whole input once per output channel. Gathering the receptive fields
//! into a column buffer turns the same work into one batched GEMM, which
//! reaches the tuned matmul kernels instead.
//!
//! # Layout
//!
//! `im2col2d` writes `col` as `[N, C_in*Kh*Kw, H_out*W_out]`, contraction axis
//! first, matching conv1d's im2col and conv_transpose1d's gather. The
//! flattened spatial axis stays innermost (contiguous) on both the input read
//! and the column write, so writes coalesce. The weight
//! `[C_out, C_in, Kh, Kw]` is already row-major in exactly that order — this
//! path is restricted to `groups == 1`, so there is no group axis to reorder
//! around — and reshapes with NO COPY to `[C_out, C_in*Kh*Kw]`. Batched over
//! `N` the GEMM yields `[N, C_out, H_out*W_out]`, which reshapes to
//! `[N, C_out, H_out, W_out]`. No permute anywhere, unlike conv_transpose1d's
//! gather, which does need one because input channels lead in that op's
//! weight layout.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv2dParams;
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::cuda::kernels::{im2col2d_has_kernel, launch_im2col2d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction length (`c_in * kh * kw`) worth routing through the
/// GEMM.
///
/// Smallest contraction (`c_in * kh * kw`) routed through im2col + GEMM.
///
/// Measured, and MUCH lower than the 1-D paths' floors — do not sync them.
/// conv2d's GEMM is thousands of columns wide (`h_out * w_out`) where the 1-D
/// paths have hundreds, so it amortizes the column buffer far better and pays
/// off at a far shorter contraction.
///
/// Swept with the spatial size held fixed: contraction 9 loses to the direct
/// kernel, 27 wins, and the margin grows to several-fold by 576. 27 is the
/// lowest contraction measured to win.
///
/// The sweep varied `c_in` only. A much smaller spatial extent gives the GEMM
/// fewer columns and would move this crossover up; re-sweep before trusting it
/// on small images.
const MIN_CONTRACTION: usize = 27;

/// Smallest number of output channels. Below this the GEMM has too few rows
/// to reuse a loaded column tile, which is the whole gain over the direct
/// kernel.
///
/// PROVISIONAL, awaiting measurement.
const MIN_C_OUT: usize = 4;

/// Largest column buffer, in elements. im2col trades memory for arithmetic
/// intensity; past this the extra allocation and traffic outweigh the GEMM.
///
/// PROVISIONAL, awaiting measurement. Kept at the same cap as the 1-D paths;
/// conv2d's column buffer grows faster with shape (an extra spatial factor),
/// so this may need to come down once swept.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Whether conv2d takes the im2col + GEMM path instead of the direct kernel.
///
/// The principle matches conv1d's im2col gate: the GEMM wins when it is well
/// shaped — a long contraction and enough output channels to reuse each
/// column tile. Grouped, narrow-channel and shallow shapes keep the direct
/// kernel, where the column buffer would cost more than it saves.
pub fn use_conv2d_im2col(params: &Conv2dParams, dtype: DType) -> bool {
    if !im2col2d_has_kernel(dtype) {
        return false;
    }

    // Grouped convolution splits the GEMM into one small problem per group.
    // The direct kernel loses no work to grouping, so grouped shapes (and
    // depthwise, which is always grouped) stay on it. Matches conv1d's
    // im2col and conv_transpose1d's gather.
    if params.groups != 1 {
        return false;
    }

    let contraction = params.c_in * params.kernel_h * params.kernel_w;
    let spatial = params.output_h * params.output_w;

    let col_elements = params
        .batch
        .checked_mul(contraction)
        .and_then(|v| v.checked_mul(spatial));

    match col_elements {
        Some(n) if n <= MAX_COL_ELEMENTS => {}
        _ => return false,
    }

    contraction >= MIN_CONTRACTION && params.c_out >= MIN_C_OUT
}

/// Run conv2d as im2col followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, `params` must come
/// from `validate_conv2d`, and `groups` must be 1.
pub fn conv2d_im2col(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv2dParams,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let contraction = params.c_in * params.kernel_h * params.kernel_w;
    let spatial = params.output_h * params.output_w;

    let col =
        Tensor::<CudaRuntime>::empty(&[params.batch, contraction, spatial], dtype, &client.device)?;

    unsafe {
        launch_im2col2d(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            input.ptr(),
            col.ptr(),
            params.batch,
            params.c_in,
            params.height,
            params.width,
            params.kernel_h,
            params.kernel_w,
            params.output_h,
            params.output_w,
            params.stride_h,
            params.stride_w,
            params.pad_top,
            params.pad_left,
            params.dilation_h,
            params.dilation_w,
        )?;
    }

    // The weight is already `[C_out, C_in*Kh*Kw]` row-major, no copy needed.
    let weight_gemm = weight.reshape(&[1, params.c_out, contraction])?;

    let out = client.matmul(&weight_gemm, &col)?;
    let out = out.reshape(&[params.batch, params.c_out, params.output_h, params.output_w])?;

    // The fused `matmul_bias` adds one value per GEMM COLUMN; here a column is
    // an output position and the bias is per output CHANNEL, which is a GEMM
    // row. The bias is broadcast over the channel axis instead.
    match bias {
        Some(b) => {
            let b_channels = b.reshape(&[1, params.c_out, 1, 1])?;
            client.add(&out, &b_channels)
        }
        None => Ok(out),
    }
}
