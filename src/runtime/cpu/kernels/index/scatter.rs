//! Scatter kernels: write source values into a tensor at index-tensor
//! positions, either overwriting or reducing.
//!
//! Integer dtypes reduce through `scatter_reduce_int_kernel`, which accumulates
//! wide so a running total the output dtype cannot hold does not wrap.

use super::super::scatter_reduce_int::scatter_reduce_int_kernel;
use crate::dtype::Element;
use crate::ops::ScatterReduceOp;

/// Scatter values into a tensor at positions specified by an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][index[i][j][k]][k] = src[i][j][k]`
///
/// First copies `a` to `out`, then scatters `src` values.
///
/// # Arguments
/// * `a` - Base tensor to scatter into
/// * `indices` - Index tensor pointer (i64 values)
/// * `src` - Source values to scatter
/// * `out` - Output pointer (must be separate from a)
/// * `shape` - Shape of input/output tensor
/// * `index_shape` - Shape of index/src tensors
/// * `dim` - Dimension along which to scatter
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `out` must not alias with `a`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    let a_numel: usize = shape.iter().product();

    // First, copy a to out
    std::ptr::copy_nonoverlapping(a, out, a_numel);

    // Compute strides for output tensor (row-major)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/src tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Scatter src values to out at index positions
    for src_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = src_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(src_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - skip
            continue;
        }

        // Compute destination position: replace multi_idx[dim] with index_val
        let mut dst_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            dst_offset += coord * out_strides[d];
        }

        *out.add(dst_offset) = *src.add(src_idx);
    }
}

/// Scatter values with reduction into a destination tensor.
///
/// # Arguments
/// * `dst` - Destination tensor data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `src` - Source values to scatter
/// * `out` - Output pointer
/// * `counts` - Optional count buffer for Mean reduction (must be pre-zeroed).
///   Unused for integer dtypes, which count inside `scatter_reduce_int_kernel`.
/// * `shape` - Shape of destination tensor
/// * `index_shape` - Shape of index/src tensors
/// * `dim` - Dimension along which to scatter
/// * `op` - Reduction operation to apply
/// * `include_self` - Whether to include dst values in reduction
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `counts` must be valid if op == Mean and include_self == false
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn scatter_reduce_kernel<T: Element>(
    dst: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    counts: *mut u32,
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

    // An integer reduction cannot keep its running total in the element type,
    // so it takes the wide-accumulator kernel instead. See
    // super::scatter_reduce_int.
    if T::DTYPE.is_int() {
        scatter_reduce_int_kernel::<T>(
            dst,
            indices,
            src,
            out,
            shape,
            index_shape,
            dim,
            op,
            include_self,
        );
        return;
    }

    let dst_numel: usize = shape.iter().product();

    // Initialize output based on operation and include_self
    if include_self {
        // Copy dst to out
        std::ptr::copy_nonoverlapping(dst, out, dst_numel);
        // Initialize counts to 1 for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            let counts_slice = std::slice::from_raw_parts_mut(counts, dst_numel);
            for c in counts_slice.iter_mut() {
                *c = 1;
            }
        }
    } else {
        // Initialize based on reduction operation
        let out_slice = std::slice::from_raw_parts_mut(out, dst_numel);
        match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => {
                for elem in out_slice.iter_mut() {
                    *elem = T::zero();
                }
            }
            ScatterReduceOp::Prod => {
                for elem in out_slice.iter_mut() {
                    *elem = T::one();
                }
            }
            ScatterReduceOp::Max => {
                // Use negative infinity for Max initialization
                for elem in out_slice.iter_mut() {
                    *elem = T::from_f64(f64::NEG_INFINITY);
                }
            }
            ScatterReduceOp::Min => {
                // Use positive infinity for Min initialization
                for elem in out_slice.iter_mut() {
                    *elem = T::from_f64(f64::INFINITY);
                }
            }
        }
        // Initialize counts to 0 for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            let counts_slice = std::slice::from_raw_parts_mut(counts, dst_numel);
            for c in counts_slice.iter_mut() {
                *c = 0;
            }
        }
    }

    // Compute strides for output tensor (row-major)
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        out_strides[i] = out_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/src tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Scatter with reduction
    for src_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = src_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(src_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - skip
            continue;
        }

        // Compute destination position: replace multi_idx[dim] with index_val
        let mut dst_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            dst_offset += coord * out_strides[d];
        }

        let src_val = *src.add(src_idx);
        let dst_val = *out.add(dst_offset);

        // Apply reduction operation
        let new_val = match op {
            ScatterReduceOp::Sum | ScatterReduceOp::Mean => dst_val + src_val,
            ScatterReduceOp::Prod => dst_val * src_val,
            ScatterReduceOp::Max => {
                if src_val.to_f64() > dst_val.to_f64() {
                    src_val
                } else {
                    dst_val
                }
            }
            ScatterReduceOp::Min => {
                if src_val.to_f64() < dst_val.to_f64() {
                    src_val
                } else {
                    dst_val
                }
            }
        };

        *out.add(dst_offset) = new_val;

        // Update count for Mean
        if op == ScatterReduceOp::Mean && !counts.is_null() {
            *counts.add(dst_offset) += 1;
        }
    }

    // Finalize Mean: divide by count
    if op == ScatterReduceOp::Mean && !counts.is_null() {
        let out_slice = std::slice::from_raw_parts_mut(out, dst_numel);
        let counts_slice = std::slice::from_raw_parts(counts, dst_numel);
        for (elem, &count) in out_slice.iter_mut().zip(counts_slice.iter()) {
            if count > 0 {
                *elem = T::from_f64(elem.to_f64() / count as f64);
            }
        }
    }
}
