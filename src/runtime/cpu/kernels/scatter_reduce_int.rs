//! Integer scatter-with-reduction, accumulated wide.
//!
//! Split from the generic [`super::index::scatter_reduce_kernel`] because an
//! integer reduction cannot keep its running total in the element type. A
//! scatter that sums two `i32`s near the type's limit overflows — a panic in
//! debug, a wrapped total in release — and `mean` then divides a total that is
//! already wrong. `mean([2_000_000_000, 2_000_000_000])` as I32 is
//! 2_000_000_000, a value the output dtype represents perfectly, and only a
//! wide accumulator can report it.
//!
//! The convention is the one documented in [`super::wide_acc`]: accumulators
//! saturate, elementwise ops wrap. So the total runs in `i128` with saturating
//! add and multiply, `mean` divides that total exactly once, and the narrow
//! back to `T` clamps to the output dtype's range.
//!
//! The CUDA counterpart is `scatter_reduce_int_impl` in
//! src/runtime/cuda/kernels/scatter_reduce.cu, which does the same arithmetic
//! in `Numr128`.

use super::wide_acc::WideAcc;
use crate::dtype::Element;
use crate::ops::ScatterReduceOp;

/// Scatter `src` into a copy of `dst` along `dim`, reducing with `op`.
///
/// `out` receives the result. Out-of-range indices are skipped, matching the
/// CPU float path and the CUDA kernel.
///
/// # Safety
///
/// - `dst` and `out` must point to `shape.iter().product()` elements
/// - `indices` and `src` must point to `index_shape.iter().product()` elements
/// - `T::DTYPE` must be an integer dtype
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_reduce_int_kernel<T: Element>(
    dst: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
    op: ScatterReduceOp,
    include_self: bool,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    let dst_numel: usize = shape.iter().product();

    // The accumulator seeds from the destination when include_self is set, and
    // from the reduction's identity otherwise. Max and Min seed from the
    // element type's own bounds rather than from infinity: `T::from_f64` would
    // saturate infinity to exactly those bounds anyway.
    let identity: i128 = match op {
        ScatterReduceOp::Sum | ScatterReduceOp::Mean => 0,
        ScatterReduceOp::Prod => 1,
        ScatterReduceOp::Max => T::from_f64(f64::NEG_INFINITY).to_i128(),
        ScatterReduceOp::Min => T::from_f64(f64::INFINITY).to_i128(),
    };

    let mut acc: Vec<i128> = Vec::with_capacity(dst_numel);
    for i in 0..dst_numel {
        acc.push(if include_self {
            (*dst.add(i)).to_i128()
        } else {
            identity
        });
    }

    // Mean's denominator. include_self makes the destination's own value one of
    // the averaged contributions.
    let mut counts: Vec<u64> = vec![u64::from(include_self); dst_numel];

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total: usize = index_shape.iter().product();

    for src_idx in 0..total {
        let index_val = *indices.add(src_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            continue;
        }

        // The destination position replaces the source's coordinate along `dim`
        // with the index value, keeping every other coordinate.
        let mut remaining = src_idx;
        let mut dst_offset = 0;
        for d in 0..ndim {
            let coord = remaining / idx_strides[d];
            remaining %= idx_strides[d];
            dst_offset += if d == dim {
                index_val as usize * out_strides[d]
            } else {
                coord * out_strides[d]
            };
        }

        let value = (*src.add(src_idx)).to_i128();
        acc[dst_offset] = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => acc[dst_offset].wide_add(value),
            ScatterReduceOp::Prod => acc[dst_offset].wide_mul(value),
            // Comparison is exact in i128, so no accumulator rule applies here.
            ScatterReduceOp::Max => acc[dst_offset].max(value),
            ScatterReduceOp::Min => acc[dst_offset].min(value),
        };
        counts[dst_offset] += 1;
    }

    for i in 0..dst_numel {
        // Divide once, at the end. A destination nobody scattered into keeps
        // its accumulator: count 0 means the value is the seed, not a mean.
        let value = if op == ScatterReduceOp::Mean && counts[i] > 0 {
            acc[i] / counts[i] as i128
        } else {
            acc[i]
        };
        *out.add(i) = T::from_i128_saturating(value);
    }
}
