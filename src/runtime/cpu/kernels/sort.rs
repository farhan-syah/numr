//! Sorting and search operation kernels

use crate::dtype::Element;
use std::cmp::Ordering;

/// Sort tensor along a dimension, returning sorted values and indices.
///
/// # Arguments
/// * `a` - Input data pointer (outer_size * sort_size * inner_size elements)
/// * `out_values` - Output pointer for sorted values (same shape as input)
/// * `out_indices` - Output pointer for indices (I64), (same shape as input)
/// * `outer_size` - Product of dimensions before the sort dimension
/// * `sort_size` - Size of the dimension being sorted
/// * `inner_size` - Product of dimensions after the sort dimension
/// * `descending` - If true, sort in descending order
///
/// # Safety
/// - `a` must point to `outer_size * sort_size * inner_size` elements
/// - `out_values` must point to `outer_size * sort_size * inner_size` elements
/// - `out_indices` must point to `outer_size * sort_size * inner_size` i64 elements
#[inline]
pub unsafe fn sort_kernel<T: Element + PartialOrd>(
    a: *const T,
    out_values: *mut T,
    out_indices: *mut i64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) {
    if sort_size == 0 {
        return;
    }

    // Working buffer for (value, original_index) pairs
    let mut pairs: Vec<(T, i64)> = Vec::with_capacity(sort_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Gather slice along sort dimension
            pairs.clear();
            for i in 0..sort_size {
                let idx = outer * sort_size * inner_size + i * inner_size + inner;
                pairs.push((*a.add(idx), i as i64));
            }

            // Stable sort by value
            if descending {
                pairs.sort_by(|a, b| b.0.sort_cmp(&a.0));
            } else {
                pairs.sort_by(|a, b| a.0.sort_cmp(&b.0));
            }

            // Scatter results back
            for (i, (val, orig_idx)) in pairs.iter().enumerate() {
                let out_idx = outer * sort_size * inner_size + i * inner_size + inner;
                *out_values.add(out_idx) = *val;
                *out_indices.add(out_idx) = *orig_idx;
            }
        }
    }
}

/// Sort tensor along a dimension, returning only sorted values (no indices).
///
/// # Arguments
/// * `a` - Input data pointer (outer_size * sort_size * inner_size elements)
/// * `out` - Output pointer for sorted values (same shape as input)
/// * `outer_size` - Product of dimensions before the sort dimension
/// * `sort_size` - Size of the dimension being sorted
/// * `inner_size` - Product of dimensions after the sort dimension
/// * `descending` - If true, sort in descending order
///
/// # Safety
/// - `a` must point to `outer_size * sort_size * inner_size` elements
/// - `out` must point to `outer_size * sort_size * inner_size` elements
#[inline]
pub unsafe fn sort_values_kernel<T: Element + PartialOrd>(
    a: *const T,
    out: *mut T,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) {
    if sort_size == 0 {
        return;
    }

    // Working buffer
    let mut values: Vec<T> = Vec::with_capacity(sort_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Gather slice along sort dimension
            values.clear();
            for i in 0..sort_size {
                let idx = outer * sort_size * inner_size + i * inner_size + inner;
                values.push(*a.add(idx));
            }

            // Sort values
            if descending {
                values.sort_by(|a, b| b.sort_cmp(a));
            } else {
                values.sort_by(|a, b| a.sort_cmp(b));
            }

            // Scatter results back
            for (i, val) in values.iter().enumerate() {
                let out_idx = outer * sort_size * inner_size + i * inner_size + inner;
                *out.add(out_idx) = *val;
            }
        }
    }
}

/// Argsort: return indices that would sort the tensor along a dimension.
///
/// # Arguments
/// * `a` - Input data pointer (outer_size * sort_size * inner_size elements)
/// * `out` - Output pointer for indices (I64), (same shape as input)
/// * `outer_size` - Product of dimensions before the sort dimension
/// * `sort_size` - Size of the dimension being sorted
/// * `inner_size` - Product of dimensions after the sort dimension
/// * `descending` - If true, return indices for descending order
///
/// # Safety
/// - `a` must point to `outer_size * sort_size * inner_size` elements
/// - `out` must point to `outer_size * sort_size * inner_size` i64 elements
#[inline]
pub unsafe fn argsort_kernel<T: Element + PartialOrd>(
    a: *const T,
    out: *mut i64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) {
    if sort_size == 0 {
        return;
    }

    // Working buffer for (value, original_index) pairs
    let mut pairs: Vec<(T, i64)> = Vec::with_capacity(sort_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Gather slice along sort dimension
            pairs.clear();
            for i in 0..sort_size {
                let idx = outer * sort_size * inner_size + i * inner_size + inner;
                pairs.push((*a.add(idx), i as i64));
            }

            // Stable sort by value
            if descending {
                pairs.sort_by(|a, b| b.0.sort_cmp(&a.0));
            } else {
                pairs.sort_by(|a, b| a.0.sort_cmp(&b.0));
            }

            // Scatter indices
            for (i, (_, orig_idx)) in pairs.iter().enumerate() {
                let out_idx = outer * sort_size * inner_size + i * inner_size + inner;
                *out.add(out_idx) = *orig_idx;
            }
        }
    }
}

/// Top-K: return top K values and their indices along a dimension.
///
/// # Arguments
/// * `a` - Input data pointer (outer_size * sort_size * inner_size elements)
/// * `out_values` - Output pointer for top-k values (outer_size * k * inner_size elements)
/// * `out_indices` - Output pointer for top-k indices (I64) (outer_size * k * inner_size elements)
/// * `outer_size` - Product of dimensions before the sort dimension
/// * `sort_size` - Size of the dimension being reduced (source dimension)
/// * `inner_size` - Product of dimensions after the sort dimension
/// * `k` - Number of top elements to return
/// * `largest` - If true, return largest elements; if false, return smallest
/// * `sorted` - If true, return in sorted order
///
/// # Safety
/// - `a` must point to `outer_size * sort_size * inner_size` elements
/// - `out_values` must point to `outer_size * k * inner_size` elements
/// - `out_indices` must point to `outer_size * k * inner_size` i64 elements
/// - `k` must be <= `sort_size`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn topk_kernel<T: Element + PartialOrd>(
    a: *const T,
    out_values: *mut T,
    out_indices: *mut i64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    k: usize,
    largest: bool,
    sorted: bool,
) {
    if sort_size == 0 || k == 0 {
        return;
    }

    let k = k.min(sort_size);

    // Working buffer for (value, original_index) pairs
    let mut pairs: Vec<(T, i64)> = Vec::with_capacity(sort_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Gather slice along sort dimension
            pairs.clear();
            for i in 0..sort_size {
                let idx = outer * sort_size * inner_size + i * inner_size + inner;
                pairs.push((*a.add(idx), i as i64));
            }

            // Sort to find top-k
            if largest {
                pairs.sort_by(|a, b| b.0.sort_cmp(&a.0));
            } else {
                pairs.sort_by(|a, b| a.0.sort_cmp(&b.0));
            }

            // Take first k elements
            let topk = &pairs[..k];

            // If not sorted, we could preserve original order, but for simplicity
            // we always return in sorted order (sorted=true behavior)
            // For sorted=false, we'd need to re-sort by original index
            let output: Vec<(T, i64)> = if sorted {
                topk.to_vec()
            } else {
                // Re-sort by original index to preserve input order
                let mut temp = topk.to_vec();
                temp.sort_by_key(|(_, idx)| *idx);
                temp
            };

            // Write output
            for (i, (val, orig_idx)) in output.iter().enumerate() {
                let out_idx = outer * k * inner_size + i * inner_size + inner;
                *out_values.add(out_idx) = *val;
                *out_indices.add(out_idx) = *orig_idx;
            }
        }
    }
}

/// Count unique elements in a sorted 1D array.
///
/// # Safety
/// - `a` must point to `len` elements in sorted order
#[inline]
pub unsafe fn count_unique_kernel<T: Element + PartialEq>(a: *const T, len: usize) -> usize {
    if len == 0 {
        return 0;
    }

    let mut count = 1;
    let mut prev = *a;

    for i in 1..len {
        let curr = *a.add(i);
        if curr != prev {
            count += 1;
            prev = curr;
        }
    }

    count
}

/// Extract unique elements from a sorted 1D array.
///
/// # Arguments
/// * `a` - Input pointer (sorted, len elements)
/// * `out` - Output pointer (unique_count elements)
/// * `len` - Length of input
/// * `unique_count` - Number of unique elements (from count_unique_kernel)
///
/// # Safety
/// - `a` must point to `len` elements in sorted order
/// - `out` must point to `unique_count` elements
#[inline]
pub unsafe fn extract_unique_kernel<T: Element + PartialEq>(
    a: *const T,
    out: *mut T,
    len: usize,
    _unique_count: usize,
) {
    if len == 0 {
        return;
    }

    let mut out_idx = 0;
    let mut prev = *a;
    *out.add(out_idx) = prev;
    out_idx += 1;

    for i in 1..len {
        let curr = *a.add(i);
        if curr != prev {
            *out.add(out_idx) = curr;
            out_idx += 1;
            prev = curr;
        }
    }
}

/// Extract unique elements with inverse indices and counts.
///
/// # Arguments
/// * `a` - Original input pointer (unsorted, numel elements)
/// * `sorted` - Sorted input pointer (numel elements)
/// * `sort_indices` - Indices that sort original -> sorted (numel i64 elements)
/// * `out_unique` - Output pointer for unique values (unique_count elements)
/// * `out_inverse` - Output pointer for inverse indices (numel i64 elements)
/// * `out_counts` - Output pointer for counts (unique_count i64 elements)
/// * `numel` - Number of elements in input
/// * `unique_count` - Number of unique elements
///
/// # Safety
/// - All pointers must be valid for the specified sizes
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn unique_with_counts_kernel<T: Element + PartialEq>(
    _a: *const T,
    sorted: *const T,
    sort_indices: *const i64,
    out_unique: *mut T,
    out_inverse: *mut i64,
    out_counts: *mut i64,
    numel: usize,
    _unique_count: usize,
) {
    if numel == 0 {
        return;
    }

    let mut unique_idx: i64 = 0;
    let mut prev = *sorted;
    *out_unique.add(0) = prev;
    *out_counts.add(0) = 1;

    // First element's inverse
    let orig_idx = *sort_indices.add(0) as usize;
    *out_inverse.add(orig_idx) = 0;

    for i in 1..numel {
        let curr = *sorted.add(i);
        let orig_idx = *sort_indices.add(i) as usize;

        if curr != prev {
            unique_idx += 1;
            *out_unique.add(unique_idx as usize) = curr;
            *out_counts.add(unique_idx as usize) = 1;
            prev = curr;
        } else {
            *out_counts.add(unique_idx as usize) += 1;
        }

        *out_inverse.add(orig_idx) = unique_idx;
    }
}

/// Count nonzero elements.
///
/// # Safety
/// - `a` must point to `numel` elements
#[inline]
pub unsafe fn count_nonzero_kernel<T: Element>(a: *const T, numel: usize) -> usize {
    let a_slice = std::slice::from_raw_parts(a, numel);
    a_slice.iter().filter(|&&x| x.to_f64() != 0.0).count()
}

/// Extract indices of nonzero elements.
///
/// # Arguments
/// * `a` - Input data pointer
/// * `out` - Output pointer for flattened indices (I64, nnz elements)
/// * `numel` - Number of elements in input
///
/// # Safety
/// - `a` must point to `numel` elements
/// - `out` must have enough space for all nonzero indices
#[inline]
pub unsafe fn nonzero_flat_kernel<T: Element>(a: *const T, out: *mut i64, numel: usize) {
    let a_slice = std::slice::from_raw_parts(a, numel);
    let mut out_idx = 0;

    for (i, &val) in a_slice.iter().enumerate() {
        if val.to_f64() != 0.0 {
            *out.add(out_idx) = i as i64;
            out_idx += 1;
        }
    }
}

/// Convert flat indices to multi-dimensional indices.
///
/// # Arguments
/// * `flat_indices` - Input flat indices (I64, nnz elements)
/// * `out` - Output multi-indices (I64, nnz * ndim elements, row-major)
/// * `nnz` - Number of nonzero elements
/// * `shape` - Shape of original tensor
///
/// # Safety
/// - `flat_indices` must point to `nnz` i64 elements
/// - `out` must point to `nnz * ndim` i64 elements
#[inline]
pub unsafe fn flat_to_multi_index_kernel(
    flat_indices: *const i64,
    out: *mut i64,
    nnz: usize,
    shape: &[usize],
) {
    let ndim = shape.len();
    if ndim == 0 || nnz == 0 {
        return;
    }

    // Precompute strides
    let mut strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    for i in 0..nnz {
        let mut remaining = *flat_indices.add(i) as usize;
        for d in 0..ndim {
            let coord = remaining / strides[d];
            remaining %= strides[d];
            *out.add(i * ndim + d) = coord as i64;
        }
    }
}

/// Binary search to find insertion points (searchsorted).
///
/// For each value in `values`, finds the index in `sorted_seq` where the value
/// should be inserted to maintain sorted order.
///
/// # Arguments
/// * `sorted_seq` - Sorted sequence (seq_len elements)
/// * `values` - Values to search for (num_values elements)
/// * `out` - Output indices (I64, num_values elements)
/// * `seq_len` - Length of sorted sequence
/// * `num_values` - Number of values to search
/// * `right` - If true, find rightmost insertion point; if false, leftmost
///
/// # Safety
/// - `sorted_seq` must point to `seq_len` elements in sorted order (or be non-null if seq_len == 0)
/// - `values` must point to `num_values` elements
/// - `out` must point to `num_values` i64 elements
#[inline]
pub unsafe fn searchsorted_kernel<T: Element + PartialOrd>(
    sorted_seq: *const T,
    values: *const T,
    out: *mut i64,
    seq_len: usize,
    num_values: usize,
    right: bool,
) {
    // Handle empty sequence - all values insert at position 0
    if seq_len == 0 {
        for i in 0..num_values {
            *out.add(i) = 0;
        }
        return;
    }

    let seq_slice = std::slice::from_raw_parts(sorted_seq, seq_len);

    for i in 0..num_values {
        let val = *values.add(i);

        // Binary search under the same total order the sequence was sorted by
        let idx = if right {
            // Find rightmost position where we can insert val
            seq_slice.partition_point(|x| x.sort_cmp(&val) != Ordering::Greater)
        } else {
            // Find leftmost position where we can insert val
            seq_slice.partition_point(|x| x.sort_cmp(&val) == Ordering::Less)
        };

        *out.add(i) = idx as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_ascending() {
        let a = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let mut values = [0.0f32; 5];
        let mut indices = [0i64; 5];

        unsafe {
            sort_kernel(
                a.as_ptr(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                1,
                5,
                1,
                false,
            );
        }

        assert_eq!(values, [1.0, 1.0, 3.0, 4.0, 5.0]);
        // Indices should point to original positions
        assert_eq!(indices, [1, 3, 0, 2, 4]);
    }

    #[test]
    fn test_sort_descending() {
        let a = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let mut values = [0.0f32; 5];
        let mut indices = [0i64; 5];

        unsafe {
            sort_kernel(
                a.as_ptr(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                1,
                5,
                1,
                true,
            );
        }

        assert_eq!(values, [5.0, 4.0, 3.0, 1.0, 1.0]);
        assert_eq!(indices, [4, 2, 0, 1, 3]);
    }

    #[test]
    fn test_argsort() {
        let a = [3.0f32, 1.0, 4.0];
        let mut indices = [0i64; 3];

        unsafe {
            argsort_kernel(a.as_ptr(), indices.as_mut_ptr(), 1, 3, 1, false);
        }

        assert_eq!(indices, [1, 0, 2]); // sorted would be [1.0, 3.0, 4.0]
    }

    #[test]
    fn test_topk() {
        let a = [3.0f32, 1.0, 4.0, 1.0, 5.0];
        let mut values = [0.0f32; 2];
        let mut indices = [0i64; 2];

        unsafe {
            topk_kernel(
                a.as_ptr(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                1,
                5,
                1,
                2,
                true, // largest
                true, // sorted
            );
        }

        assert_eq!(values, [5.0, 4.0]);
        assert_eq!(indices, [4, 2]);
    }

    #[test]
    fn test_count_unique() {
        let sorted = [1.0f32, 1.0, 2.0, 3.0, 3.0, 3.0];

        let count = unsafe { count_unique_kernel(sorted.as_ptr(), 6) };
        assert_eq!(count, 3);
    }

    #[test]
    fn test_searchsorted() {
        let sorted_seq = [1.0f32, 3.0, 5.0, 7.0];
        let values = [2.0f32, 4.0, 6.0, 0.0, 8.0];
        let mut out = [0i64; 5];

        unsafe {
            searchsorted_kernel(
                sorted_seq.as_ptr(),
                values.as_ptr(),
                out.as_mut_ptr(),
                4,
                5,
                false, // left
            );
        }

        // 2 -> 1 (between 1 and 3)
        // 4 -> 2 (between 3 and 5)
        // 6 -> 3 (between 5 and 7)
        // 0 -> 0 (before 1)
        // 8 -> 4 (after 7)
        assert_eq!(out, [1, 2, 3, 0, 4]);
    }

    #[test]
    fn test_count_nonzero() {
        let a = [0.0f32, 1.0, 0.0, 2.0, 3.0, 0.0];
        let count = unsafe { count_nonzero_kernel(a.as_ptr(), 6) };
        assert_eq!(count, 3);
    }

    /// A single NaN used to leave the whole slice unsorted (issue #9).
    #[test]
    fn test_sort_nan_last_ascending() {
        let a = [3.0f32, f32::NAN, 1.0, 2.0, f32::NAN, 0.0];
        let mut values = [0.0f32; 6];

        unsafe {
            sort_values_kernel(a.as_ptr(), values.as_mut_ptr(), 1, 6, 1, false);
        }

        assert_eq!(&values[..4], &[0.0, 1.0, 2.0, 3.0]);
        assert!(values[4].is_nan() && values[5].is_nan());
    }

    #[test]
    fn test_sort_nan_first_descending() {
        let a = [3.0f32, f32::NAN, 1.0, 2.0, f32::NAN, 0.0];
        let mut values = [0.0f32; 6];

        unsafe {
            sort_values_kernel(a.as_ptr(), values.as_mut_ptr(), 1, 6, 1, true);
        }

        assert!(values[0].is_nan() && values[1].is_nan());
        assert_eq!(&values[2..], &[3.0, 2.0, 1.0, 0.0]);
    }

    #[test]
    fn test_sort_nan_beats_infinity() {
        let a = [f32::NAN, f32::INFINITY, f32::NEG_INFINITY];
        let mut values = [0.0f32; 3];

        unsafe {
            sort_values_kernel(a.as_ptr(), values.as_mut_ptr(), 1, 3, 1, false);
        }

        assert_eq!(&values[..2], &[f32::NEG_INFINITY, f32::INFINITY]);
        assert!(values[2].is_nan());
    }

    #[test]
    fn test_sort_all_nan() {
        let a = [f32::NAN; 4];
        let mut values = [0.0f32; 4];
        let mut indices = [0i64; 4];

        unsafe {
            sort_kernel(
                a.as_ptr(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                1,
                4,
                1,
                false,
            );
        }

        assert!(values.iter().all(|v| v.is_nan()));
        // NaNs tie, so the stable sort preserves input order.
        assert_eq!(indices, [0, 1, 2, 3]);
    }

    #[test]
    fn test_argsort_nan_indices() {
        let a = [1.0f32, f32::NAN, 0.0];
        let mut out = [0i64; 3];

        unsafe {
            argsort_kernel(a.as_ptr(), out.as_mut_ptr(), 1, 3, 1, false);
        }

        assert_eq!(out, [2, 0, 1]);
    }

    /// `-0.0` and `+0.0` tie, so ties break by input order in both directions.
    #[test]
    fn test_signed_zero_tie_is_stable() {
        let a = [0.0f32, -0.0, 0.0, -0.0];

        for descending in [false, true] {
            let mut out = [0i64; 4];
            unsafe {
                argsort_kernel(a.as_ptr(), out.as_mut_ptr(), 1, 4, 1, descending);
            }
            assert_eq!(out, [0, 1, 2, 3], "descending={descending}");
        }
    }

    #[test]
    fn test_topk_largest_ranks_nan_first() {
        let a = [1.0f32, f32::NAN, 5.0, 2.0];
        let mut values = [0.0f32; 2];
        let mut indices = [0i64; 2];

        unsafe {
            topk_kernel(
                a.as_ptr(),
                values.as_mut_ptr(),
                indices.as_mut_ptr(),
                1,
                4,
                1,
                2,
                true,
                true,
            );
        }

        assert!(values[0].is_nan());
        assert_eq!(values[1], 5.0);
        assert_eq!(indices, [1, 2]);
    }

    #[test]
    fn test_searchsorted_nan_goes_to_end() {
        let seq = [1.0f32, 2.0, 3.0, f32::NAN];
        let values = [f32::NAN, 2.5];
        let mut out = [0i64; 2];

        unsafe {
            searchsorted_kernel(seq.as_ptr(), values.as_ptr(), out.as_mut_ptr(), 4, 2, false);
        }

        // NaN inserts at the head of the trailing NaN run, 2.5 between 2 and 3.
        assert_eq!(out, [3, 2]);
    }
}
