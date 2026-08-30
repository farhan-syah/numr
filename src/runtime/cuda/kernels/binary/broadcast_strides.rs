//! Broadcast-stride and magic-divisor helpers for binary broadcast kernels.
//!
//! Computes per-dimension strides and fast-division constants that
//! `launch_broadcast_binary_op` (in `launchers`) packs into kernel arguments,
//! plus the fast-trailing-broadcast precondition check.

use crate::error::{Error, Result};

/// Compute broadcast strides for a tensor shape relative to the output shape.
///
/// For each dimension in the output shape:
/// - If the input dimension matches, use the original stride
/// - If the input dimension is 1 (broadcast), use stride 0
/// - If the input doesn't have this dimension (prepended), use stride 0
///
/// Errors when the input rank exceeds the output rank: right-aligning is then
/// impossible, and the offset that maps input dims onto output dims would
/// underflow. Every caller derives `output_shape` from a broadcast, so this
/// cannot happen today, but the helper does not rely on that.
pub(crate) fn compute_broadcast_strides(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<u32>> {
    let mut strides = vec![0u32; output_shape.len()];
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    if input_ndim > output_ndim {
        return Err(Error::InvalidArgument {
            arg: "input_shape",
            reason: format!(
                "broadcast strides: input rank {input_ndim} exceeds output rank {output_ndim}"
            ),
        });
    }

    // Compute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Map input dimensions to output dimensions (right-aligned)
    let offset = output_ndim - input_ndim;
    for i in 0..output_ndim {
        if i < offset {
            // Dimension doesn't exist in input, broadcast with stride 0
            strides[i] = 0;
        } else {
            let input_idx = i - offset;
            if input_shape[input_idx] == 1 {
                // Broadcasting dimension, stride 0
                strides[i] = 0;
            } else {
                // Normal dimension, use input stride
                strides[i] = input_strides[input_idx] as u32;
            }
        }
    }

    Ok(strides)
}

/// Maximum number of dimensions supported by the inline broadcast kernel.
///
/// Must match `MAX_BROADCAST_DIMS` in `binary.cu`.
pub const MAX_BROADCAST_DIMS: usize = 8;

/// Compute magic-number fast-division constants for divisor `d`.
///
/// Returns `(magic, shift)` encoding. The CUDA kernel must use:
///   if (magic == 0) { q = remaining >> shift; }   // d==1 (shift=0) or power-of-2 (shift=k)
///   else            { q = __umulhi(remaining, magic) >> shift; }  // general case
/// Then: coord = remaining - q * shape[d]; remaining = q;
///
/// - d == 0: (0, 0) — unused dim, kernel skips via ndim guard
/// - d == 1: (0, 0) — q = remaining >> 0 = remaining; coord = remaining - remaining = 0 ✓
/// - d == 2^k: (0, k) — q = remaining >> k (exact); coord = remaining - q*d ✓
/// - d general: __umulhi(x, magic) >> shift == floor(x/d) for all x in [0, 2^32) ✓
pub fn compute_magic_divisor(d: u32) -> (u32, u32) {
    if d <= 1 {
        // d==0: unused sentinel. d==1: q = remaining >> 0 = remaining; coord = 0.
        return (0u32, 0u32);
    }
    if d.is_power_of_two() {
        let shift = d.trailing_zeros();
        return (0u32, shift);
    }
    // General case d >= 3, not power-of-2:
    // magic = ceil(2^(32+p) / d), shift = p = floor(log2(d))
    // Guarantees: __umulhi(x, magic) >> p == floor(x/d) for all x in [0, 2^32).
    let p = 31u32 - d.leading_zeros();
    let numerator: u64 = 1u64 << (32 + p);
    let magic_full = (numerator + (d as u64) - 1) / (d as u64);
    // For non-power-of-2 d>=3, magic_full always fits in u32.
    debug_assert!(magic_full <= 0xFFFF_FFFFu64, "magic overflow for d={d}");
    (magic_full as u32, p)
}

/// Check whether `a` and `b` satisfy the fast trailing-broadcast preconditions:
/// - `a` must be contiguous with the same shape as `out_shape` (a_strides == natural strides)
/// - `b` must be a contiguous trailing-broadcast of `out_shape`: all leading dims of `b`
///   that differ from `out_shape` must be 1, and the remaining trailing dims must match.
///   The b_numel (product of b's non-broadcast dims) must be a contiguous suffix of out_shape.
///
/// Returns `Some(b_numel)` if the fast path applies, `None` otherwise.
pub fn detect_fast_trailing_broadcast(
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Option<usize> {
    // a must exactly match out_shape (no broadcasting on a side)
    if a_shape != out_shape {
        return None;
    }

    // b must be a trailing suffix of out_shape.
    // Aligned right: b_shape right-pads with 1s if shorter.
    // For each position, b must either be 1 (broadcast) or equal to out.
    // The non-1 dimensions of b must form a contiguous SUFFIX of out_shape.
    let ndim = out_shape.len();
    let b_ndim = b_shape.len();
    let offset = ndim.saturating_sub(b_ndim);

    // Find where b's non-trivial (non-1) dimensions start
    let mut b_start = b_ndim; // index in b_shape where first non-1 dim is
    for i in 0..b_ndim {
        if b_shape[i] != 1 {
            b_start = i;
            break;
        }
    }

    // All dims in b from b_start onward must match out_shape
    for i in b_start..b_ndim {
        let out_i = offset + i;
        if b_shape[i] != out_shape[out_i] {
            return None;
        }
    }

    // All dims in b before b_start must be 1 (already guaranteed by construction)
    // and all corresponding out dims before offset+b_start must be non-trivial
    // (but that's fine, a covers them linearly).

    // b_numel = product of b's non-1 suffix
    let b_numel: usize = b_shape[b_start..].iter().product();
    if b_numel == 0 {
        return None;
    }

    Some(b_numel)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_strides_right_aligns_lower_rank_input() {
        // Shape [3, 1] against output [2, 3, 4]: the prepended dim broadcasts,
        // the size-3 dim keeps its input stride, the size-1 dim broadcasts.
        let strides = compute_broadcast_strides(&[3, 1], &[2, 3, 4])
            .expect("input rank 2 fits output rank 3");
        assert_eq!(strides, vec![0, 1, 0]);
    }

    #[test]
    fn broadcast_strides_rejects_input_rank_above_output_rank() {
        // Previously underflowed `output_ndim - input_ndim` and then indexed
        // `input_shape` out of bounds. It must be an error, not a panic.
        let err = compute_broadcast_strides(&[2, 3, 4], &[3, 4])
            .expect_err("input rank 3 cannot right-align onto output rank 2");
        match err {
            Error::InvalidArgument { arg, reason } => {
                assert_eq!(arg, "input_shape");
                assert!(reason.contains("input rank 3"), "reason: {reason}");
                assert!(reason.contains("output rank 2"), "reason: {reason}");
            }
            other => panic!("expected InvalidArgument, got {other:?}"),
        }
    }
}
