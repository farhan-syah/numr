//! Selection kernels driven by a 1-D index vector or a contiguous range.
//!
//! `index_select_kernel` reads, `index_put_kernel` writes at the same index
//! positions, and `slice_assign_kernel` overwrites a contiguous slice.

use crate::dtype::Element;

/// Select elements along a dimension using a 1D index tensor.
///
/// Simpler than gather - the index tensor is 1D and applies uniformly
/// to all positions in the specified dimension.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - 1D index tensor pointer (i64 values), length = index_len
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `dim` - Dimension along which to select
/// * `index_len` - Length of the 1D index tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_select_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    dim: usize,
    index_len: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute sizes: outer * dim_size * inner.
    // Never clamp these with `.max(1)`: an empty slice already products to 1, so a
    // clamp only fires on a genuinely zero dim and then indexes past the allocation.
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();

    // For each outer position
    for outer in 0..outer_size {
        // For each selected index
        for (sel_idx, &idx_ptr) in std::slice::from_raw_parts(indices, index_len)
            .iter()
            .enumerate()
        {
            let idx = idx_ptr as usize;
            if idx >= dim_size {
                // Out of bounds - fill with zeros
                for inner in 0..inner_size {
                    let out_offset = outer * index_len * inner_size + sel_idx * inner_size + inner;
                    *out.add(out_offset) = T::zero();
                }
                continue;
            }

            // Copy the entire inner slice
            for inner in 0..inner_size {
                let src_offset = outer * dim_size * inner_size + idx * inner_size + inner;
                let out_offset = outer * index_len * inner_size + sel_idx * inner_size + inner;
                *out.add(out_offset) = *a.add(src_offset);
            }
        }
    }
}

/// Put values at specified indices along a dimension.
///
/// Copies input `a` to output, then overwrites positions specified by `indices`
/// with values from `src`.
///
/// # Arguments
/// * `a` - Input tensor data pointer
/// * `indices` - 1D index tensor pointer (i64)
/// * `src` - Source values to put at indexed positions
/// * `out` - Output data pointer (must be same size as input)
/// * `shape` - Shape of input tensor `a`
/// * `dim` - Dimension along which to put values
/// * `index_len` - Length of the 1D index tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn index_put_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    src: *const T,
    out: *mut T,
    shape: &[usize],
    dim: usize,
    index_len: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute sizes: outer * dim_size * inner.
    // Never clamp these with `.max(1)`: an empty slice already products to 1, so a
    // clamp only fires on a genuinely zero dim and then indexes past the allocation.
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();

    // First, copy all of a to out
    let total_size: usize = shape.iter().product();
    std::ptr::copy_nonoverlapping(a, out, total_size);

    // Now overwrite the indexed positions with src values
    for outer in 0..outer_size {
        for (sel_idx, &idx_ptr) in std::slice::from_raw_parts(indices, index_len)
            .iter()
            .enumerate()
        {
            let idx = idx_ptr as usize;
            if idx >= dim_size {
                // Out of bounds - skip
                continue;
            }

            // Overwrite the entire inner slice at this index
            for inner in 0..inner_size {
                let out_offset = outer * dim_size * inner_size + idx * inner_size + inner;
                let src_offset = outer * index_len * inner_size + sel_idx * inner_size + inner;
                *out.add(out_offset) = *src.add(src_offset);
            }
        }
    }
}

/// Slice assign kernel: copies src into a slice of dst along a dimension.
///
/// dst is first fully copied to output, then src overwrites the slice region.
///
/// # Safety
///
/// All pointers must be valid with the correct element counts.
pub unsafe fn slice_assign_kernel<T: Copy>(
    dst: *const T,
    src: *const T,
    out: *mut T,
    outer_size: usize,
    dst_dim_size: usize,
    src_dim_size: usize,
    inner_size: usize,
    start: usize,
) {
    let dst_total = outer_size * dst_dim_size * inner_size;

    // Copy entire dst to output
    std::ptr::copy_nonoverlapping(dst, out, dst_total);

    // Overwrite the slice region with src
    for o in 0..outer_size {
        for s in 0..src_dim_size {
            let src_offset = o * src_dim_size * inner_size + s * inner_size;
            let dst_offset = o * dst_dim_size * inner_size + (start + s) * inner_size;
            std::ptr::copy_nonoverlapping(src.add(src_offset), out.add(dst_offset), inner_size);
        }
    }
}
