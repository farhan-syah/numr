//! Cumulative operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::{Error, Result};
use crate::ops::reduce_output_shape;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Normalize a dimension index, allowing negative indexing
#[inline]
fn normalize_dim(ndim: usize, dim: isize) -> Option<usize> {
    if dim >= 0 {
        let d = dim as usize;
        if d < ndim { Some(d) } else { None }
    } else {
        let d = dim + ndim as isize;
        if d >= 0 { Some(d as usize) } else { None }
    }
}

/// Cumulative sum along a dimension
pub fn cumsum_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    let dim_idx = normalize_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    // Handle empty tensor
    if a.numel() == 0 {
        return Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);
    }

    // Make contiguous for simplicity
    let a_contig = ensure_contiguous(a)?;

    // Output has same shape as input
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    // Compute sizes for the scan
    let scan_size = shape[dim_idx];
    let outer_size: usize = shape[..dim_idx].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim_idx + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: scan dimension is last (or tensor is 1D)
                kernels::cumsum_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::cumsum_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                    inner_size,
                );
            }
        }
    }, "cumsum");

    Ok(out)
}

/// Cumulative product along a dimension
pub fn cumprod_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: isize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    let dim_idx = normalize_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

    // Handle empty tensor
    if a.numel() == 0 {
        return Tensor::<CpuRuntime>::empty(shape, dtype, &client.device);
    }

    // Make contiguous for simplicity
    let a_contig = ensure_contiguous(a)?;

    // Output has same shape as input
    let out = Tensor::<CpuRuntime>::empty(shape, dtype, &client.device)?;

    // Compute sizes for the scan
    let scan_size = shape[dim_idx];
    let outer_size: usize = shape[..dim_idx].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim_idx + 1..].iter().product();
    let inner_size = inner_size.max(1);

    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: scan dimension is last (or tensor is 1D)
                kernels::cumprod_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::cumprod_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    scan_size,
                    outer_size,
                    inner_size,
                );
            }
        }
    }, "cumprod");

    Ok(out)
}

/// Log-sum-exp along specified dimensions (numerically stable)
///
/// Only supports floating-point dtypes (F32, F64, F16, BF16).
pub fn logsumexp_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Logsumexp only makes sense for floating-point types
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp",
        });
    }

    // Validate dimensions
    for &d in dims {
        if d >= ndim {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim,
            });
        }
    }

    // For single last-dimension reduction on contiguous tensor, use fast path
    if dims.len() == 1 && dims[0] == ndim - 1 && a.is_contiguous() {
        let reduce_size = shape[ndim - 1];
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = outer_size.max(1);

        let out_shape = reduce_output_shape(shape, dims, keepdim);
        let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

        let a_ptr = a.ptr();
        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::logsumexp_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            }
        }, "logsumexp");

        return Ok(out);
    }

    // For empty dims, return copy
    if dims.is_empty() {
        return Ok(a.clone());
    }

    // General case: reduce one dimension at a time
    let a_contig = ensure_contiguous(a)?;

    let mut sorted_dims: Vec<usize> = dims.to_vec();
    sorted_dims.sort_unstable();
    sorted_dims.reverse();

    let mut current = a_contig;
    for &dim in &sorted_dims {
        current = logsumexp_single_dim(client, &current, dim, keepdim)?;
    }

    Ok(current)
}

/// Log-sum-exp along a single dimension
fn logsumexp_single_dim(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = shape[dim + 1..].iter().product();
    let inner_size = inner_size.max(1);

    // Output shape: remove the reduced dimension (or keep as 1)
    let out_shape: Vec<usize> = if keepdim {
        shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if i == dim { 1 } else { s })
            .collect()
    } else {
        shape
            .iter()
            .enumerate()
            .filter_map(|(i, &s)| if i != dim { Some(s) } else { None })
            .collect()
    };

    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;

    let a_ptr = a.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            if inner_size == 1 {
                // Fast path: reduce dimension is last
                kernels::logsumexp_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                );
            } else {
                // General path: strided access
                kernels::logsumexp_strided_kernel(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    reduce_size,
                    outer_size,
                    inner_size,
                    inner_size, // in_stride
                    inner_size, // out_stride
                );
            }
        }
    }, "logsumexp");

    Ok(out)
}
