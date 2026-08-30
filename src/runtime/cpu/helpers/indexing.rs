//! Indexing operation helpers for CPU tensors

use super::super::kernels;
#[cfg(target_arch = "x86_64")]
use super::super::kernels::simd::index as simd_index;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScatterReduceOp;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Normalize index tensor to I64 for CPU indexing kernels.
///
/// CPU indexing kernels consume i64 indices. This helper accepts both I32 and I64
/// public API inputs and materializes an I64 tensor when needed.
fn normalize_indices_to_i64(
    client: &CpuClient,
    indices: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    match indices.dtype() {
        DType::I64 => Ok(indices.clone()),
        DType::I32 => {
            let idx_i32: Vec<i32> = indices.to_vec();
            let idx_i64: Vec<i64> = idx_i32.into_iter().map(i64::from).collect();
            Tensor::<CpuRuntime>::from_slice(&idx_i64, indices.shape(), &client.device)
        }
        other => Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: other,
        }),
    }
}

/// Gather elements from a 2D matrix using row and column index vectors.
///
/// For each index i, extracts `input[rows[i], cols[i]]`.
pub fn gather_2d_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    rows: &Tensor<CpuRuntime>,
    cols: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    let shape = input.shape();

    // Validate input is 2D
    if shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: shape.to_vec(),
        });
    }

    let nrows = shape[0];
    let ncols = shape[1];

    let rows_i64 = normalize_indices_to_i64(client, rows)?;
    let cols_i64 = normalize_indices_to_i64(client, cols)?;

    // Validate rows and cols are 1D and have same length
    if rows_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![rows_i64.numel()],
            got: rows_i64.shape().to_vec(),
        });
    }

    if cols_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![cols_i64.numel()],
            got: cols_i64.shape().to_vec(),
        });
    }

    let num_indices = rows_i64.numel();
    if cols_i64.numel() != num_indices {
        return Err(Error::ShapeMismatch {
            expected: vec![num_indices],
            got: cols_i64.shape().to_vec(),
        });
    }

    // Make all inputs contiguous
    let input_contig = ensure_contiguous(input)?;
    let rows_contig = ensure_contiguous(&rows_i64)?;
    let cols_contig = ensure_contiguous(&cols_i64)?;

    // Allocate output
    let out = Tensor::<CpuRuntime>::empty(&[num_indices], dtype, &client.device)?;

    let input_ptr = input_contig.ptr();
    let rows_ptr = rows_contig.ptr();
    let cols_ptr = cols_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        let success = unsafe {
            kernels::gather_2d_kernel::<T>(
                input_ptr as *const T,
                rows_ptr as *const i64,
                cols_ptr as *const i64,
                out_ptr as *mut T,
                nrows,
                ncols,
                num_indices,
            )
        };
        if !success {
            return Err(Error::IndexOutOfBounds {
                index: 0, // Generic - actual invalid index is in rows/cols
                size: nrows.max(ncols),
            });
        }
    }, "gather_2d");

    Ok(out)
}

/// Gather elements along a dimension using an index tensor.
pub fn gather_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Validate index dimensions
    if index_i64.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Output shape is same as index shape
    let out_shape = index_i64.shape().to_vec();

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    let a_ptr = a_contig.ptr();
    let index_ptr = index_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::gather_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                &out_shape,
                dim,
            );
        }
    }, "gather");

    Ok(out)
}

/// Scatter values into a tensor at positions specified by an index tensor.
pub fn scatter_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate shapes
    if index_i64.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    let a_ptr = a_contig.ptr();
    let index_ptr = index_contig.ptr();
    let src_ptr = src_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scatter_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                shape,
                index_i64.shape(),
                dim,
            );
        }
    }, "scatter");

    Ok(out)
}

/// Select elements along a dimension using a 1D index tensor.
pub fn index_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Index must be 1D
    if index_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index_i64.numel()],
            got: index_i64.shape().to_vec(),
        });
    }

    let index_len = index_i64.shape()[0];

    // Output shape: replace dimension `dim` with index length
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;

    // Validate all indices are within bounds (before calling unsafe kernel)
    let dim_size = shape[dim];
    let index_data =
        unsafe { std::slice::from_raw_parts(index_contig.ptr() as *const i64, index_len) };
    for &idx in index_data.iter() {
        // Negative indices are not supported - must be in [0, dim_size)
        if idx < 0 || idx as usize >= dim_size {
            return Err(Error::IndexOutOfBounds {
                index: if idx < 0 { 0 } else { idx as usize },
                size: dim_size,
            });
        }
    }

    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    let a_ptr = a_contig.ptr();
    let index_ptr = index_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::index_select_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                out_ptr as *mut T,
                shape,
                dim,
                index_len,
            );
        }
    }, "index_select");

    Ok(out)
}

/// Put values at specified indices along a dimension.
pub fn index_put_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    // Index must be 1D
    if index_i64.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![index_i64.numel()],
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate src dtype matches
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let index_len = index_i64.shape()[0];

    // Validate src shape: must match a's shape except at dim where it equals index_len
    let mut expected_src_shape = shape.to_vec();
    expected_src_shape[dim] = index_len;
    if src.shape() != expected_src_shape {
        return Err(Error::ShapeMismatch {
            expected: expected_src_shape,
            got: src.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;

    // Validate all indices are within bounds (before calling unsafe kernel)
    let dim_size = shape[dim];
    let index_data =
        unsafe { std::slice::from_raw_parts(index_contig.ptr() as *const i64, index_len) };
    for &idx in index_data.iter() {
        // Negative indices are not supported - must be in [0, dim_size)
        if idx < 0 || idx as usize >= dim_size {
            return Err(Error::IndexOutOfBounds {
                index: if idx < 0 { 0 } else { idx as usize },
                size: dim_size,
            });
        }
    }

    // Clone a's data for output
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    let a_ptr = a_contig.ptr();
    let index_ptr = index_contig.ptr();
    let src_ptr = src_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::index_put_kernel::<T>(
                a_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                shape,
                dim,
                index_len,
            );
        }
    }, "index_put");

    Ok(out)
}

/// Select elements where mask is true, returning a flattened 1D tensor.
pub fn masked_select_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(&mask_broadcast)?;

    let numel = a.numel();
    let a_ptr = a_contig.ptr();
    let mask_ptr = mask_contig.ptr();

    // Use SIMD for f32/f64 on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        // Count true elements using SIMD
        let count = unsafe { simd_index::masked_count(mask_ptr as *const u8, numel) };

        // Allocate output with correct size
        let out = Tensor::<CpuRuntime>::empty(&[count], dtype, &client.device)?;
        let out_ptr = out.ptr();

        match dtype {
            DType::F32 => {
                unsafe {
                    simd_index::masked_select_f32(
                        a_ptr as *const f32,
                        mask_ptr as *const u8,
                        out_ptr as *mut f32,
                        numel,
                    );
                }
                return Ok(out);
            }
            DType::F64 => {
                unsafe {
                    simd_index::masked_select_f64(
                        a_ptr as *const f64,
                        mask_ptr as *const u8,
                        out_ptr as *mut f64,
                        numel,
                    );
                }
                return Ok(out);
            }
            _ => {
                // Fall through to scalar for other types
                dispatch_dtype!(dtype, T => {
                    unsafe {
                        kernels::masked_select_kernel::<T>(
                            a_ptr as *const T,
                            mask_ptr as *const u8,
                            out_ptr as *mut T,
                            numel,
                        );
                    }
                }, "masked_select");
                return Ok(out);
            }
        }
    }

    // Scalar fallback for non-x86_64
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Count true elements first
        let count = unsafe { kernels::index::masked_count_kernel(mask_ptr as *const u8, numel) };

        // Allocate output with correct size
        let out = Tensor::<CpuRuntime>::empty(&[count], dtype, &client.device)?;
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::masked_select_kernel::<T>(
                    a_ptr as *const T,
                    mask_ptr as *const u8,
                    out_ptr as *mut T,
                    numel,
                );
            }
        }, "masked_select");

        Ok(out)
    }
}

/// Fill elements where mask is true with a scalar value.
pub fn masked_fill_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    mask: &Tensor<CpuRuntime>,
    value: f64,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();

    // Validate mask dtype
    if mask.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to input shape
    let mask_broadcast = mask.broadcast_to(a.shape())?;

    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(&mask_broadcast)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let numel = a.numel();
    let a_ptr = a_contig.ptr();
    let mask_ptr = mask_contig.ptr();
    let out_ptr = out.ptr();

    // Use SIMD for f32/f64 on x86_64
    #[cfg(target_arch = "x86_64")]
    {
        match dtype {
            DType::F32 => {
                unsafe {
                    simd_index::masked_fill_f32(
                        a_ptr as *const f32,
                        mask_ptr as *const u8,
                        out_ptr as *mut f32,
                        numel,
                        value as f32,
                    );
                }
                return Ok(out);
            }
            DType::F64 => {
                unsafe {
                    simd_index::masked_fill_f64(
                        a_ptr as *const f64,
                        mask_ptr as *const u8,
                        out_ptr as *mut f64,
                        numel,
                        value,
                    );
                }
                return Ok(out);
            }
            _ => {} // Fall through to scalar for other types
        }
    }

    // Scalar fallback for non-x86_64 or non-f32/f64 types
    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::masked_fill_kernel::<T>(
                a_ptr as *const T,
                mask_ptr as *const u8,
                out_ptr as *mut T,
                numel,
                value,
            );
        }
    }, "masked_fill");

    Ok(out)
}

/// Look up embeddings from an embedding table using indices.
///
/// # Algorithm
/// For each index i in the flattened indices tensor:
///   output[i, :] = embeddings[indices[i], :]
///
/// Output shape: indices.shape() + [embedding_dim]
pub fn embedding_lookup_impl(
    client: &CpuClient,
    embeddings: &Tensor<CpuRuntime>,
    indices: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    let indices_i64 = normalize_indices_to_i64(client, indices)?;

    let vocab_size = emb_shape[0];
    let embedding_dim = emb_shape[1];
    let num_indices = indices_i64.numel();

    // Output shape: indices.shape() + [embedding_dim]
    let mut out_shape = indices_i64.shape().to_vec();
    out_shape.push(embedding_dim);

    let emb_contig = ensure_contiguous(embeddings)?;
    let idx_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    let emb_ptr = emb_contig.ptr();
    let idx_ptr = idx_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::embedding_lookup_kernel::<T>(
                emb_ptr as *const T,
                idx_ptr as *const i64,
                out_ptr as *mut T,
                num_indices,
                vocab_size,
                embedding_dim,
            );
        }
    }, "embedding_lookup");

    Ok(out)
}

/// Scatter values with reduction into a destination tensor.
#[allow(clippy::too_many_arguments)]
pub fn scatter_reduce_impl(
    client: &CpuClient,
    dst: &Tensor<CpuRuntime>,
    dim: usize,
    index: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
    op: ScatterReduceOp,
    include_self: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = dst.dtype();
    let shape = dst.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let index_i64 = normalize_indices_to_i64(client, index)?;

    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Validate that index and src have same shape
    if index_i64.shape() != src.shape() {
        return Err(Error::ShapeMismatch {
            expected: src.shape().to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    // Validate that index has same number of dimensions as dst
    if index_i64.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: index_i64.shape().to_vec(),
        });
    }

    let dst_contig = ensure_contiguous(dst)?;
    let index_contig = ensure_contiguous(&index_i64)?;
    let src_contig = ensure_contiguous(src)?;
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    // Allocate counts buffer for Mean operation
    let dst_numel: usize = shape.iter().product();
    let counts_buffer: Vec<u32> = vec![0; dst_numel];

    let dst_ptr = dst_contig.ptr();
    let index_ptr = index_contig.ptr();
    let src_ptr = src_contig.ptr();
    let out_ptr = out.ptr();
    let counts_ptr = if op == ScatterReduceOp::Mean {
        counts_buffer.as_ptr() as *mut u32
    } else {
        std::ptr::null_mut()
    };

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::scatter_reduce_kernel::<T>(
                dst_ptr as *const T,
                index_ptr as *const i64,
                src_ptr as *const T,
                out_ptr as *mut T,
                counts_ptr,
                shape,
                index_i64.shape(),
                dim,
                op,
                include_self,
            );
        }
    }, "scatter_reduce");

    Ok(out)
}

/// Gather elements using N-dimensional indices.
pub fn gather_nd_impl(
    client: &CpuClient,
    input: &Tensor<CpuRuntime>,
    indices: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();
    let input_shape = input.shape();
    let indices_i64 = normalize_indices_to_i64(client, indices)?;
    let indices_shape = indices_i64.shape();

    // Indices must have at least 1 dimension
    if indices_shape.is_empty() {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: indices_shape.to_vec(),
        });
    }

    // Last dimension of indices is the number of coordinates (M)
    let indices_ndim = indices_shape.len();
    let index_depth = indices_shape[indices_ndim - 1]; // M

    // M must not exceed input dimensions
    if index_depth > input_shape.len() {
        return Err(Error::InvalidDimension {
            dim: index_depth as isize,
            ndim: input_shape.len(),
        });
    }

    // Compute output shape:
    // indices.shape[:-1] + input.shape[M:]
    let mut out_shape: Vec<usize> = indices_shape[..indices_ndim - 1].to_vec();
    out_shape.extend_from_slice(&input_shape[index_depth..]);

    // Handle scalar output case
    if out_shape.is_empty() {
        out_shape.push(1);
    }

    let input_contig = ensure_contiguous(input)?;
    let indices_contig = ensure_contiguous(&indices_i64)?;
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    let input_ptr = input_contig.ptr();
    let indices_ptr = indices_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::gather_nd_kernel::<T>(
                input_ptr as *const T,
                indices_ptr as *const i64,
                out_ptr as *mut T,
                input_shape,
                indices_shape,
                &out_shape,
            );
        }
    }, "gather_nd");

    Ok(out)
}

/// Slice assign implementation: copies src into a slice of dst along dim starting at start.
pub fn slice_assign_impl(
    client: &CpuClient,
    dst: &Tensor<CpuRuntime>,
    src: &Tensor<CpuRuntime>,
    dim: usize,
    start: usize,
) -> Result<Tensor<CpuRuntime>> {
    let ndim = dst.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // Validate shapes match except at dim
    if src.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: dst.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && src.shape()[d] != dst.shape()[d] {
            return Err(Error::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
    }

    let src_dim_size = src.shape()[dim];
    let dst_dim_size = dst.shape()[dim];
    if start + src_dim_size > dst_dim_size {
        return Err(Error::InvalidArgument {
            arg: "start",
            reason: format!(
                "start ({}) + src dim size ({}) exceeds dst dim size ({})",
                start, src_dim_size, dst_dim_size
            ),
        });
    }

    let dtype = dst.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    // Compute outer/inner sizes
    let outer_size: usize = dst.shape()[..dim].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };
    let inner_size: usize = dst.shape()[dim + 1..].iter().product();
    let inner_size = if inner_size == 0 { 1 } else { inner_size };

    let dst_c = ensure_contiguous(dst)?;
    let src_c = ensure_contiguous(src)?;
    let out = Tensor::<CpuRuntime>::empty(dst.shape(), dtype, &client.device)?;

    // Nothing to assign. The kernel copies with `copy_nonoverlapping`, and two
    // zero-byte allocations share one dangling address, which that call rejects
    // as overlapping even for a count of 0.
    if out.numel() == 0 {
        return Ok(out);
    }

    let dst_ptr = dst_c.ptr();
    let src_ptr = src_c.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::slice_assign_kernel::<T>(
                dst_ptr as *const T,
                src_ptr as *const T,
                out_ptr as *mut T,
                outer_size,
                dst_dim_size,
                src_dim_size,
                inner_size,
                start,
            );
        }
    }, "slice_assign");

    Ok(out)
}
