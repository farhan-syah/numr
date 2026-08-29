//! Gather kernels: read elements out of a tensor at index-tensor positions.
//!
//! Three index shapes: one index per output element along a dimension
//! (`gather_kernel`), N-dimensional coordinates in the last index dimension
//! (`gather_nd_kernel`), and paired row/column vectors (`gather_2d_kernel`).

use crate::dtype::Element;

/// Gather elements along a dimension using an index tensor.
///
/// For a 3D tensor with dim=1:
/// `out[i][j][k] = input[i][index[i][j][k]][k]`
///
/// # Arguments
/// * `a` - Input data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `out` - Output pointer
/// * `shape` - Shape of input tensor
/// * `index_shape` - Shape of index tensor (same as output shape)
/// * `dim` - Dimension along which to gather
///
/// # Safety
/// - All pointers must be valid for the specified shapes
/// - `indices` must contain valid indices within bounds of `shape[dim]`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_kernel<T: Element>(
    a: *const T,
    indices: *const i64,
    out: *mut T,
    shape: &[usize],
    index_shape: &[usize],
    dim: usize,
) {
    let ndim = shape.len();
    if ndim == 0 {
        return;
    }

    // Compute strides for input tensor (row-major)
    let mut a_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        a_strides[i] = a_strides[i + 1] * shape[i + 1];
    }

    // Compute strides for index/output tensor (row-major)
    let mut idx_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        idx_strides[i] = idx_strides[i + 1] * index_shape[i + 1];
    }

    let total = index_shape.iter().product::<usize>();

    // Iterate over all output positions
    for out_idx in 0..total {
        // Convert linear index to multi-dimensional indices
        let mut remaining = out_idx;
        let mut multi_idx = vec![0usize; ndim];
        for d in 0..ndim {
            multi_idx[d] = remaining / idx_strides[d];
            remaining %= idx_strides[d];
        }

        // Get the index value from the indices tensor
        let index_val = *indices.add(out_idx);
        if index_val < 0 || index_val as usize >= shape[dim] {
            // Out of bounds - set to zero (could also panic)
            *out.add(out_idx) = T::zero();
            continue;
        }

        // Compute source position: replace multi_idx[dim] with index_val
        let mut src_offset = 0;
        for d in 0..ndim {
            let coord = if d == dim {
                index_val as usize
            } else {
                multi_idx[d]
            };
            src_offset += coord * a_strides[d];
        }

        *out.add(out_idx) = *a.add(src_offset);
    }
}

/// Gather elements using N-dimensional indices.
///
/// The last dimension of `indices` contains coordinates into `input`.
///
/// # Arguments
/// * `input` - Input data pointer
/// * `indices` - Index tensor pointer (i64 values)
/// * `out` - Output pointer
/// * `input_shape` - Shape of input tensor
/// * `indices_shape` - Shape of indices tensor
/// * `out_shape` - Shape of output tensor
///
/// # Safety
/// - All pointers must be valid for the specified shapes
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gather_nd_kernel<T: Element>(
    input: *const T,
    indices: *const i64,
    out: *mut T,
    input_shape: &[usize],
    indices_shape: &[usize],
    out_shape: &[usize],
) {
    if input_shape.is_empty() || indices_shape.is_empty() {
        return;
    }

    let input_ndim = input_shape.len();
    let indices_ndim = indices_shape.len();
    let index_depth = indices_shape[indices_ndim - 1]; // M: number of coordinates

    // Compute input strides
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim - 1).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Compute indices strides
    let mut indices_strides = vec![1usize; indices_ndim];
    for i in (0..indices_ndim - 1).rev() {
        indices_strides[i] = indices_strides[i + 1] * indices_shape[i + 1];
    }

    // Compute output strides
    let out_ndim = out_shape.len();
    let mut out_strides = vec![1usize; out_ndim.max(1)];
    for i in (0..out_ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    // Number of index vectors (product of indices.shape[:-1])
    let num_indices: usize = indices_shape[..indices_ndim - 1]
        .iter()
        .product::<usize>()
        .max(1);

    // Size of trailing dimensions from input (after the indexed dimensions)
    let trailing_size: usize = if index_depth < input_ndim {
        input_shape[index_depth..].iter().product()
    } else {
        1
    };

    // For each index vector
    for idx_vec in 0..num_indices {
        // Compute offset into indices tensor for this index vector
        let indices_offset = idx_vec * index_depth;

        // Read the index coordinates
        let mut input_offset = 0usize;
        let mut valid = true;
        for d in 0..index_depth {
            let coord = *indices.add(indices_offset + d);
            if coord < 0 || coord as usize >= input_shape[d] {
                valid = false;
                break;
            }
            input_offset += (coord as usize) * input_strides[d];
        }

        // Compute output offset
        let out_offset = idx_vec * trailing_size;

        if !valid {
            // Out of bounds - fill with zeros
            for i in 0..trailing_size {
                *out.add(out_offset + i) = T::zero();
            }
        } else {
            // Copy trailing elements
            for i in 0..trailing_size {
                *out.add(out_offset + i) = *input.add(input_offset + i);
            }
        }
    }
}

/// Gather elements from a 2D matrix using row and column index vectors.
///
/// For each index i, extracts `input[rows[i], cols[i]]`.
///
/// # Arguments
/// * `input` - 2D input data pointer (row-major layout)
/// * `rows` - Row index pointer (i64 values)
/// * `cols` - Column index pointer (i64 values)
/// * `out` - Output pointer
/// * `nrows` - Number of rows in input
/// * `ncols` - Number of columns in input
/// * `num_indices` - Number of (row, col) pairs to gather
///
/// # Safety
/// - All pointers must be valid for the specified sizes
/// - Indices must be within bounds of input dimensions
///
/// # Returns
/// * `true` if all indices were valid, `false` if any out-of-bounds
#[inline]
pub unsafe fn gather_2d_kernel<T: Element>(
    input: *const T,
    rows: *const i64,
    cols: *const i64,
    out: *mut T,
    nrows: usize,
    ncols: usize,
    num_indices: usize,
) -> bool {
    if num_indices == 0 {
        return true;
    }

    let rows_slice = std::slice::from_raw_parts(rows, num_indices);
    let cols_slice = std::slice::from_raw_parts(cols, num_indices);

    for i in 0..num_indices {
        let r = rows_slice[i];
        let c = cols_slice[i];

        // Bounds checking
        if r < 0 || r as usize >= nrows || c < 0 || c as usize >= ncols {
            return false;
        }

        // Row-major indexing: input[r, c] = input[r * ncols + c]
        let input_offset = (r as usize) * ncols + (c as usize);
        *out.add(i) = *input.add(input_offset);
    }

    true
}
