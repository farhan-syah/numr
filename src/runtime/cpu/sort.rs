//! Sorting and search operations for CPU runtime

use super::helpers::{dispatch_dtype, ensure_contiguous};
use super::{CpuClient, CpuRuntime, kernels};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IndexingOps, compute_reduce_strides};
use crate::runtime::normalize_dim;
use crate::tensor::Tensor;

/// Sort tensor along a dimension (values only)
pub fn sort_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
    descending: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Ok(a.clone());
    }

    let dim_idx = normalize_dim(dim, ndim)?;
    let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::sort_values_kernel::<T>(
                a_ptr as *const T,
                out_ptr as *mut T,
                outer_size,
                sort_size,
                inner_size,
                descending,
            );
        }
    }, "sort");

    Ok(out)
}

/// Sort tensor along a dimension, returning both values and indices
pub fn sort_with_indices_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
    descending: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        let indices = Tensor::<CpuRuntime>::zeros(shape, DType::I64, &client.device)?;
        return Ok((a.clone(), indices));
    }

    let dim_idx = normalize_dim(dim, ndim)?;
    let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
    let a_contig = ensure_contiguous(a)?;
    let out_values = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;
    let out_indices = Tensor::<CpuRuntime>::empty(shape, DType::I64, &client.device)?;

    let a_ptr = a_contig.ptr();
    let values_ptr = out_values.ptr();
    let indices_ptr = out_indices.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::sort_kernel::<T>(
                a_ptr as *const T,
                values_ptr as *mut T,
                indices_ptr as *mut i64,
                outer_size,
                sort_size,
                inner_size,
                descending,
            );
        }
    }, "sort_with_indices");

    Ok((out_values, out_indices))
}

/// Return indices that would sort the tensor
pub fn argsort_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
    descending: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Tensor::<CpuRuntime>::zeros(shape, DType::I64, &client.device);
    }

    let dim_idx = normalize_dim(dim, ndim)?;
    let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(shape, DType::I64, &client.device)?;

    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::argsort_kernel::<T>(
                a_ptr as *const T,
                out_ptr as *mut i64,
                outer_size,
                sort_size,
                inner_size,
                descending,
            );
        }
    }, "argsort");

    Ok(out)
}

/// Return top K values and indices
pub fn topk_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    k: usize,
    dim: isize,
    largest: bool,
    sorted: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        if k > 1 {
            return Err(Error::InvalidArgument {
                arg: "k",
                reason: "k cannot be greater than 1 for scalar tensors".to_string(),
            });
        }
        let indices = Tensor::<CpuRuntime>::zeros(shape, DType::I64, &client.device)?;
        return Ok((a.clone(), indices));
    }

    let dim_idx = normalize_dim(dim, ndim)?;
    let dim_size = shape[dim_idx];

    if k > dim_size {
        return Err(Error::InvalidArgument {
            arg: "k",
            reason: format!(
                "k ({}) cannot be greater than dimension size ({})",
                k, dim_size
            ),
        });
    }
    if k == 0 {
        let mut out_shape = shape.to_vec();
        out_shape[dim_idx] = 0;
        let out_values = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
        let out_indices = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &client.device)?;
        return Ok((out_values, out_indices));
    }

    let (outer_size, sort_size, inner_size) = compute_reduce_strides(shape, dim_idx);
    let a_contig = ensure_contiguous(a)?;

    let mut out_shape = shape.to_vec();
    out_shape[dim_idx] = k;

    let out_values = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
    let out_indices = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &client.device)?;

    let a_ptr = a_contig.ptr();
    let values_ptr = out_values.ptr();
    let indices_ptr = out_indices.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::topk_kernel::<T>(
                a_ptr as *const T,
                values_ptr as *mut T,
                indices_ptr as *mut i64,
                outer_size,
                sort_size,
                inner_size,
                k,
                largest,
                sorted,
            );
        }
    }, "topk");

    Ok((out_values, out_indices))
}

/// Return unique elements
pub fn unique_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    _sorted: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let numel = a.numel();

    if numel == 0 {
        return Tensor::<CpuRuntime>::empty(&[0], dtype, &client.device);
    }

    let a_flat = a.reshape(&[numel])?;
    let a_contig = ensure_contiguous(&a_flat)?;

    // Sort first
    let sorted_tensor = sort_impl(client, &a_contig, 0, false)?;
    let sorted_ptr = sorted_tensor.ptr();

    // Count unique
    let unique_count = dispatch_dtype!(dtype, T => {
        unsafe { kernels::count_unique_kernel::<T>(sorted_ptr as *const T, numel) }
    }, "unique_count");

    // Extract unique
    let out = Tensor::<CpuRuntime>::empty(&[unique_count], dtype, &client.device)?;
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::extract_unique_kernel::<T>(
                sorted_ptr as *const T,
                out_ptr as *mut T,
                numel,
                unique_count,
            );
        }
    }, "unique_extract");

    Ok(out)
}

/// Return unique elements with inverse indices and counts
pub fn unique_with_counts_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let dtype = a.dtype();
    let numel = a.numel();

    if numel == 0 {
        let unique = Tensor::<CpuRuntime>::empty(&[0], dtype, &client.device)?;
        let inverse = Tensor::<CpuRuntime>::empty(&[0], DType::I64, &client.device)?;
        let counts = Tensor::<CpuRuntime>::empty(&[0], DType::I64, &client.device)?;
        return Ok((unique, inverse, counts));
    }

    let a_flat = a.reshape(&[numel])?;
    let a_contig = ensure_contiguous(&a_flat)?;

    // Get sort indices
    let sort_indices = argsort_impl(client, &a_contig, 0, false)?;

    // Gather sorted data
    let sorted_tensor = client.gather(&a_contig, 0, &sort_indices)?;
    let sorted_ptr = sorted_tensor.ptr();

    // Count unique
    let unique_count = dispatch_dtype!(dtype, T => {
        unsafe { kernels::count_unique_kernel::<T>(sorted_ptr as *const T, numel) }
    }, "unique_count");

    // Extract all
    let out_unique = Tensor::<CpuRuntime>::empty(&[unique_count], dtype, &client.device)?;
    let out_inverse = Tensor::<CpuRuntime>::empty(&[numel], DType::I64, &client.device)?;
    let out_counts = Tensor::<CpuRuntime>::empty(&[unique_count], DType::I64, &client.device)?;

    let a_ptr = a_contig.ptr();
    let sort_indices_ptr = sort_indices.ptr();
    let unique_ptr = out_unique.ptr();
    let inverse_ptr = out_inverse.ptr();
    let counts_ptr = out_counts.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::unique_with_counts_kernel::<T>(
                a_ptr as *const T,
                sorted_ptr as *const T,
                sort_indices_ptr as *const i64,
                unique_ptr as *mut T,
                inverse_ptr as *mut i64,
                counts_ptr as *mut i64,
                numel,
                unique_count,
            );
        }
    }, "unique_with_counts");

    Ok((out_unique, out_inverse, out_counts))
}

/// Return indices of nonzero elements
pub fn nonzero_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();
    let numel = a.numel();

    if numel == 0 {
        return Tensor::<CpuRuntime>::empty(&[0, ndim], DType::I64, &client.device);
    }

    let a_contig = ensure_contiguous(a)?;
    let a_ptr = a_contig.ptr();

    // Count nonzero
    let nnz = dispatch_dtype!(dtype, T => {
        unsafe { kernels::count_nonzero_kernel::<T>(a_ptr as *const T, numel) }
    }, "nonzero_count");

    if nnz == 0 {
        return Tensor::<CpuRuntime>::empty(&[0, ndim], DType::I64, &client.device);
    }

    if ndim == 0 {
        return Tensor::<CpuRuntime>::empty(&[1, 0], DType::I64, &client.device);
    }

    // Get flat indices
    let flat_indices = Tensor::<CpuRuntime>::empty(&[nnz], DType::I64, &client.device)?;
    let flat_ptr = flat_indices.ptr() as *mut i64;

    dispatch_dtype!(dtype, T => {
        unsafe { kernels::nonzero_flat_kernel::<T>(a_ptr as *const T, flat_ptr, numel); }
    }, "nonzero_flat");

    // Convert to multi-index
    let out = Tensor::<CpuRuntime>::empty(&[nnz, ndim], DType::I64, &client.device)?;
    let out_ptr = out.ptr() as *mut i64;

    unsafe {
        kernels::flat_to_multi_index_kernel(flat_ptr, out_ptr, nnz, shape);
    }

    Ok(out)
}

/// Binary search for insertion points
pub fn searchsorted_impl(
    client: &CpuClient,
    sorted_sequence: &Tensor<CpuRuntime>,
    values: &Tensor<CpuRuntime>,
    right: bool,
) -> Result<Tensor<CpuRuntime>> {
    if sorted_sequence.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![sorted_sequence.numel()],
            got: sorted_sequence.shape().to_vec(),
        });
    }

    if sorted_sequence.dtype() != values.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: sorted_sequence.dtype(),
            rhs: values.dtype(),
        });
    }

    let dtype = sorted_sequence.dtype();
    let seq_len = sorted_sequence.numel();
    let num_values = values.numel();

    if num_values == 0 {
        return Tensor::<CpuRuntime>::empty(values.shape(), DType::I64, &client.device);
    }

    let seq_contig = ensure_contiguous(sorted_sequence)?;
    let values_contig = ensure_contiguous(values)?;
    let out = Tensor::<CpuRuntime>::empty(values.shape(), DType::I64, &client.device)?;

    let seq_ptr = seq_contig.ptr();
    let values_ptr = values_contig.ptr();
    let out_ptr = out.ptr() as *mut i64;

    dispatch_dtype!(dtype, T => {
        unsafe {
            kernels::searchsorted_kernel::<T>(
                seq_ptr as *const T,
                values_ptr as *const T,
                out_ptr,
                seq_len,
                num_values,
                right,
            );
        }
    }, "searchsorted");

    Ok(out)
}
