//! Batch-dimension normalization for WebGPU batched matmul kernels.
//!
//! The batched kernels take a single batch count and read both operands with the
//! same batch stride, so they only handle operands that already share the output's
//! batch shape. Broadcast batch dims, a mismatched batch rank, or more than one
//! batch dim all need the operands expanded to the output's batch shape and
//! flattened to 3D first.
//!
//! The expansion stays on device: a zero-stride broadcast view followed by a
//! device-side copy.

use super::super::super::WgpuRuntime;
use crate::error::Result;
use crate::tensor::Tensor;

/// Operands flattened to `[batch, m, k]` and `[batch, k, n]`, when the batched
/// kernels cannot take them as they are.
///
/// Returns `None` when both operands already match the output's batch shape, so
/// shapes the kernels already handle keep their existing cost.
pub(crate) fn flatten_batched_operands(
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    out_shape: &[usize],
) -> Result<Option<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)>> {
    if out_shape.len() <= 2 {
        return Ok(None);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let out_batch = &out_shape[..out_shape.len() - 2];
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];

    let a_target: Vec<usize> = out_batch.iter().copied().chain([m, k]).collect();
    let b_target: Vec<usize> = out_batch.iter().copied().chain([k, n]).collect();

    // A single batch dim shared by both operands is exactly what the kernels take.
    if out_shape.len() == 3 && a_shape == a_target.as_slice() && b_shape == b_target.as_slice() {
        return Ok(None);
    }

    let batch: usize = out_batch.iter().product();
    let a3 = a
        .broadcast_to(&a_target)?
        .contiguous()?
        .reshape(&[batch, m, k])?;
    let b3 = b
        .broadcast_to(&b_target)?
        .contiguous()?
        .reshape(&[batch, k, n])?;

    Ok(Some((a3, b3)))
}
