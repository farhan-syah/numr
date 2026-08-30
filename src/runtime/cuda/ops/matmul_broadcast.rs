//! Batch-dimension broadcasting for CUDA batched matmul kernels.
//!
//! The batched kernels locate an operand's data with `batch % operand_batch_count`.
//! Wrapping by a count only matches per-dimension broadcasting when the operand's
//! broadcast dims are its leading ones: `[1, h]` against `[b, h]` cycles correctly,
//! but `[b, 1]` against `[b, h]` needs the index to advance every `h` outputs, so
//! wrapping silently reads a different batch.
//!
//! Operands the kernels cannot index are expanded on device before launch. Operands
//! they already handle are passed through untouched, so working shapes keep their
//! existing cost.

use super::super::CudaRuntime;
use crate::error::Result;
use crate::ops::matmul::matmul_batch_indices;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Batch count an operand contributes, i.e. the product of all but its last two dims.
///
/// Floored at 1 deliberately: this feeds `batch % batch_count(..)` below, where a 0
/// is a division-by-zero panic. A genuinely zero batch dim leaves a zero-element
/// output, which matmul returns before any indexing happens.
pub(crate) fn batch_count(shape: &[usize]) -> usize {
    shape
        .iter()
        .take(shape.len().saturating_sub(2))
        .product::<usize>()
        .max(1)
}

/// Whether `batch % batch_count(operand)` lands on the batch that per-dimension
/// broadcasting calls for, for every output batch.
///
/// Two shape-only cases settle it without building index tables, which covers
/// every call that does not actually broadcast:
/// - batch dims already equal to the output's, so nothing broadcasts;
/// - a single batch, where every output maps to index 0 and `i % 1` is always 0.
///
/// Anything else falls through to the authoritative check against
/// [`matmul_batch_indices`], rather than re-deriving the indexing rules here.
fn modulo_indexing_is_correct(operand_shape: &[usize], out_shape: &[usize], is_lhs: bool) -> bool {
    let operand_batch = &operand_shape[..operand_shape.len().saturating_sub(2)];
    let out_batch = &out_shape[..out_shape.len().saturating_sub(2)];
    if operand_batch == out_batch || batch_count(operand_shape) == 1 {
        return true;
    }

    let (a_idx, b_idx) = if is_lhs {
        matmul_batch_indices(operand_shape, out_shape, out_shape)
    } else {
        matmul_batch_indices(out_shape, operand_shape, out_shape)
    };
    let indices = if is_lhs { a_idx } else { b_idx };
    let count = batch_count(operand_shape);
    indices.iter().enumerate().all(|(i, &idx)| i % count == idx)
}

/// Expand an operand whose batch dims the kernels cannot index, or return it as-is.
///
/// The expansion stays on device: a zero-stride broadcast view followed by a
/// device-side copy.
fn materialize_batch_broadcast(
    operand: &Tensor<CudaRuntime>,
    out_shape: &[usize],
    is_lhs: bool,
) -> Result<Option<Tensor<CudaRuntime>>> {
    if modulo_indexing_is_correct(operand.shape(), out_shape, is_lhs) {
        return Ok(None);
    }

    let shape = operand.shape();
    let mut target: Vec<usize> = out_shape[..out_shape.len().saturating_sub(2)].to_vec();
    target.extend_from_slice(&shape[shape.len().saturating_sub(2)..]);

    Ok(Some(operand.broadcast_to(&target)?.contiguous()?))
}

/// Operands and batch counts ready to hand to a batched kernel launch.
pub(crate) struct BatchedOperands {
    pub a: Tensor<CudaRuntime>,
    pub b: Tensor<CudaRuntime>,
    pub a_batch: usize,
    pub b_batch: usize,
}

/// Resolve both operands for a batched kernel launch.
///
/// Every batched launch must take its pointers and its batch counts from the same
/// tensors: a count derived from an expanded operand paired with a pointer to the
/// original, unexpanded buffer makes the kernel read past the end of it. Returning
/// both together from one place is what keeps them in step.
///
/// Callers needing contiguous buffers must call [`BatchedOperands::contiguous`]
/// rather than making the originals contiguous beforehand.
pub(crate) fn resolve_batched_operands(
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
    out_shape: &[usize],
) -> Result<BatchedOperands> {
    let a = materialize_batch_broadcast(a, out_shape, true)?.unwrap_or_else(|| a.clone());
    let b = materialize_batch_broadcast(b, out_shape, false)?.unwrap_or_else(|| b.clone());
    let a_batch = batch_count(a.shape());
    let b_batch = batch_count(b.shape());

    Ok(BatchedOperands {
        a,
        b,
        a_batch,
        b_batch,
    })
}

impl BatchedOperands {
    /// Contiguous copies of both resolved operands.
    pub(crate) fn contiguous(&self) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        Ok((ensure_contiguous(&self.a)?, ensure_contiguous(&self.b)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modulo_indexing_correct_for_leading_broadcast() {
        // `[1, 4]` against out `[2, 4]` cycles, which wrapping handles.
        assert!(modulo_indexing_is_correct(
            &[1, 4, 2, 3],
            &[2, 4, 3, 3],
            false
        ));
    }

    #[test]
    fn test_modulo_indexing_wrong_for_middle_broadcast() {
        // `[2, 1]` against out `[2, 4]` must advance every 4 outputs, which
        // wrapping by a count of 2 cannot express.
        assert!(!modulo_indexing_is_correct(
            &[2, 1, 2, 1],
            &[2, 4, 3, 1],
            false
        ));
    }

    #[test]
    fn test_modulo_indexing_correct_for_equal_batches() {
        assert!(modulo_indexing_is_correct(
            &[2, 4, 3, 2],
            &[2, 4, 3, 3],
            true
        ));
    }

    #[test]
    fn test_modulo_indexing_correct_for_unbatched() {
        assert!(modulo_indexing_is_correct(&[3, 2], &[3, 4], true));
    }

    /// The shape-only fast paths must agree with the authoritative index check.
    #[test]
    fn test_fast_paths_agree_with_index_check() {
        let cases: &[(&[usize], &[usize], bool)] = &[
            (&[2, 4, 3, 2], &[2, 4, 3, 3], true),
            (&[1, 4, 2, 3], &[2, 4, 3, 3], false),
            (&[2, 1, 2, 1], &[2, 4, 3, 1], false),
            (&[4, 2, 3], &[2, 4, 3, 3], false),
            (&[2, 1, 3, 2], &[2, 4, 3, 3], true),
            (&[3, 2], &[3, 4], true),
        ];
        for &(operand, out, is_lhs) in cases {
            let (a_idx, b_idx) = if is_lhs {
                matmul_batch_indices(operand, out, out)
            } else {
                matmul_batch_indices(out, operand, out)
            };
            let indices = if is_lhs { a_idx } else { b_idx };
            let count = batch_count(operand);
            let expected = indices.iter().enumerate().all(|(i, &idx)| i % count == idx);
            assert_eq!(
                modulo_indexing_is_correct(operand, out, is_lhs),
                expected,
                "operand={operand:?} out={out:?} is_lhs={is_lhs}"
            );
        }
    }
}
