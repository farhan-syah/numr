//! im2col + GEMM formulation of CUDA conv1d.
//!
//! The direct conv1d kernel re-reads the whole input once per output channel.
//! Gathering the receptive fields into a column buffer turns the same work into
//! one GEMM, which reaches the tuned matmul kernels instead.
//!
//! # Layout
//!
//! `im2col` writes `col` as `[N, C_in*K, L_out]`, contraction axis first. Split
//! row-major that is `[N, groups, (C_in/groups)*K, L_out]`, and the weight
//! `[C_out, C_in/groups, K]` reshapes with no copy to
//! `[1, groups, C_out/groups, (C_in/groups)*K]` because output channels are
//! ordered group-major. Batched over `(N, groups)` the GEMM yields
//! `[N, groups, C_out/groups, L_out]`, which reshapes to `[N, C_out, L_out]`.
//! No transpose and no final permute.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::conv_common::Conv1dParams;
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::cuda::kernels::{im2col_has_kernel, launch_im2col1d};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

/// Smallest contraction length worth routing through the GEMM.
///
/// Below this the column buffer costs more to write than the GEMM saves, and
/// the K loop is too short for the tiled kernels to amortise their prologue.
const MIN_CONTRACTION: usize = 64;

/// Smallest number of output channels per group. Below this the GEMM has too
/// few rows to reuse a loaded column tile, which is the whole gain over the
/// direct kernel. Depthwise convolution sits at one and always stays direct.
const MIN_C_OUT_PER_GROUP: usize = 4;

/// Shortest `output_length` routed through im2col.
///
/// Measured, and the previous value of 16 was not: on a large-channel family
/// im2col was faster than the direct kernel at EVERY output length swept, from
/// 1 up to 300. The old threshold sent the short end of that range to the
/// slower path for no reason.
///
/// Kept above 1 because the sweep covered only that family. The opposite
/// corner — `c_out` at its floor of MIN_C_OUT_PER_GROUP with contraction at
/// MIN_CONTRACTION — makes a degenerate GEMM whose col buffer may not pay for
/// itself, and no measurement covers it. Gating on the GEMM's output element
/// count rather than on length alone would express that better; that is a
/// follow-up, not a tuned number.
const MIN_OUTPUT_LENGTH: usize = 4;

/// Largest column buffer, in elements. im2col trades memory for arithmetic
/// intensity; past this the extra allocation and traffic outweigh the GEMM.
const MAX_COL_ELEMENTS: usize = 1 << 26;

/// Whether conv1d takes the im2col + GEMM path instead of the direct kernel.
///
/// The principle: im2col wins when the resulting GEMM is well shaped — a long
/// contraction, enough rows per group to reuse each column tile, and enough
/// columns to fill a tile. Depthwise, narrow-channel and very short outputs
/// keep the direct kernel, where a batch of tiny GEMMs plus the column buffer
/// would cost more than it saves.
pub fn use_conv1d_im2col(params: &Conv1dParams, dtype: DType) -> bool {
    if !im2col_has_kernel(dtype) || params.groups == 0 {
        return false;
    }

    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let contraction = c_in_per_group * params.kernel_size;

    let col_elements = params
        .batch
        .checked_mul(params.c_in)
        .and_then(|v| v.checked_mul(params.kernel_size))
        .and_then(|v| v.checked_mul(params.output_length));

    match col_elements {
        Some(n) if n <= MAX_COL_ELEMENTS => {}
        _ => return false,
    }

    // Grouped convolution splits the GEMM into one small problem per group, each
    // still only `output_length` wide. That shape is far off the tiled kernel's
    // best case, and the direct kernel — which loses no work to grouping — wins
    // by a wide margin. Measured, not assumed: re-check with `benches/conv.rs`
    // before widening this.
    if params.groups != 1 {
        return false;
    }

    contraction >= MIN_CONTRACTION
        && c_out_per_group >= MIN_C_OUT_PER_GROUP
        && params.output_length >= MIN_OUTPUT_LENGTH
}

/// Run conv1d as im2col followed by a batched GEMM.
///
/// `input`, `weight` and `bias` must already be contiguous, and `params` must
/// come from `validate_conv1d`.
pub fn conv1d_im2col(
    client: &CudaClient,
    input: &Tensor<CudaRuntime>,
    weight: &Tensor<CudaRuntime>,
    bias: Option<&Tensor<CudaRuntime>>,
    params: &Conv1dParams,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = input.dtype();
    let rows = params.c_in * params.kernel_size;

    let col = Tensor::<CudaRuntime>::empty(
        &[params.batch, rows, params.output_length],
        dtype,
        &client.device,
    )?;

    unsafe {
        launch_im2col1d(
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
            params.pad_left, // symmetric padding, same convention as the direct kernel
            params.dilation,
        )?;
    }

    let c_in_per_group = params.c_in / params.groups;
    let c_out_per_group = params.c_out / params.groups;
    let contraction = c_in_per_group * params.kernel_size;

    // Both reshapes are stride-preserving views of contiguous data.
    let col_grouped = col.reshape(&[
        params.batch,
        params.groups,
        contraction,
        params.output_length,
    ])?;
    let weight_grouped = weight.reshape(&[1, params.groups, c_out_per_group, contraction])?;

    let out = client.matmul(&weight_grouped, &col_grouped)?;
    let out = out.reshape(&[params.batch, params.c_out, params.output_length])?;

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
