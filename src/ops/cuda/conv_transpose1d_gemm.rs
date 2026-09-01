//! Gather + GEMM formulation of CUDA conv_transpose1d.
//!
//! The direct conv_transpose1d kernel gives one thread per output element and
//! re-reads the whole input once per output channel. Gathering the contributing
//! input samples into a column buffer turns the same work into one batched
//! GEMM, which reaches the tuned matmul kernels instead.
//!
//! # Gather, not scatter
//!
//! The textbook column form of transposed convolution is col2im: a scatter-add
//! whose overlapping writes need atomics, which numr does not use. The gather
//! form avoids them. For a fixed output position, each tap is fed by at most
//! one input sample, so every column element is written once by one thread.
//! See `col_transpose1d.cu` for the index relation, which matches
//! `runtime/cpu/kernels/conv_transpose.rs` tap for tap.
//!
//! # Layout
//!
//! The gather writes `col` as `[N, C_in*K, L_out]`, contraction axis first, so
//! the spatial axis stays the innermost (contiguous) one on both the input read
//! and the column write. The weight is `[C_in, C_out, K]` — input channels lead
//! for this op — so it needs a real permute to `[C_out, C_in*K]`, done once
//! outside the kernel. Batched over `N` the GEMM yields `[N, C_out, L_out]`
//! directly, with no final permute.
//!
//! # `output_padding`
//!
//! It lengthens the output only. The extra positions gather no input sample,
//! so their column entries are zero and the GEMM writes zeros (plus bias) with
//! no special case anywhere.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::cuda::kernels::{col_transpose1d_has_kernel, launch_col_transpose1d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction length worth routing through the GEMM.
///
/// Below this the column buffer costs more to write than the GEMM saves, and
/// the K loop is too short for the tiled kernels to amortise their prologue.
///
/// Smallest contraction (`c_in * kernel_size`) routed through the gather + GEMM.
///
/// Measured, and NOT the same crossover conv1d has — do not sync the two. The
/// gather here reads a strided, stride-and-divisibility-filtered input, and the
/// weight needs a real permute copy before the GEMM, so this path carries more
/// fixed cost than conv1d's im2col and needs a longer contraction to pay it off.
///
/// Swept on a fixed geometry with only the kernel size varying: 2048 and 4096
/// both lose to the direct kernel, 8192 wins, and the win grows sharply beyond.
const MIN_CONTRACTION: usize = 8192;

/// Smallest number of output channels. Below this the GEMM has too few rows to
/// reuse a loaded column tile, which is the whole gain over the direct kernel.
///
/// PROVISIONAL, awaiting measurement.
const MIN_C_OUT: usize = 4;

/// Largest column buffer, in elements. The gather trades memory for arithmetic
/// intensity; past this the extra allocation and traffic outweigh the GEMM.
///
/// PROVISIONAL, awaiting measurement.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Whether conv_transpose1d takes the gather + GEMM path instead of the direct
/// kernel.
///
/// The principle matches conv1d's im2col gate: the GEMM wins when it is well
/// shaped — a long contraction and enough output channels to reuse each column
/// tile. Narrow-channel and shallow shapes keep the direct kernel, where the
/// column buffer would cost more than it saves.
pub fn use_conv_transpose1d_gemm(params: &ConvTranspose1dParams, dtype: DType) -> bool {
    if !col_transpose1d_has_kernel(dtype) {
        return false;
    }

    // Grouped transposed convolution splits the GEMM into one small problem per
    // group and would need a per-group weight permute. The direct kernel loses
    // no work to grouping, so grouped shapes stay on it.
    if params.groups != 1 {
        return false;
    }

    let col_elements = params
        .batch
        .checked_mul(params.c_in)
        .and_then(|v| v.checked_mul(params.kernel_size))
        .and_then(|v| v.checked_mul(params.output_length));

    match col_elements {
        Some(n) if n <= MAX_COL_ELEMENTS => {}
        _ => return false,
    }

    let contraction = params.c_in * params.kernel_size;
    contraction >= MIN_CONTRACTION && params.c_out >= MIN_C_OUT
}

/// Run conv_transpose1d as a column gather followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, `params` must come
/// from `validate_conv_transpose1d`, and `groups` must be 1.
pub fn conv_transpose1d_gemm(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &ConvTranspose1dParams,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let contraction = params.c_in * params.kernel_size;

    let col = Tensor::<CudaRuntime>::empty(
        &[params.batch, contraction, params.output_length],
        dtype,
        &client.device,
    )?;

    unsafe {
        launch_col_transpose1d(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            input.ptr(),
            col.ptr(),
            params.batch,
            params.c_in,
            params.length,
            params.kernel_size,
            params.output_length,
            params.stride,
            params.pad_left,
            params.dilation,
        )?;
    }

    // Column row `ic*K + k` must meet weight element `[ic, oc, k]`, so the
    // weight's leading two axes swap. Unlike conv1d's im2col this is a real
    // copy, because input channels lead in this op's weight layout.
    let weight_gemm =
        weight
            .transpose(0, 1)?
            .contiguous()?
            .reshape(&[1, params.c_out, contraction])?;

    let out = client.matmul(&weight_gemm, &col)?;

    // The fused `matmul_bias` adds one value per GEMM COLUMN; here a column is
    // an output position and the bias is per output CHANNEL, which is a GEMM
    // row. The bias is broadcast over the channel axis instead.
    match bias {
        Some(b) => {
            let b_channels = b.reshape(&[1, params.c_out, 1])?;
            client.add(&out, &b_channels)
        }
        None => Ok(out),
    }
}
