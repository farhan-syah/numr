//! Quantile, percentile, and median operations for CPU runtime.

use super::super::helpers::{dispatch_dtype, ensure_contiguous};
use super::super::sort::sort_impl;
use super::super::{CpuClient, CpuRuntime};
use super::{Interpolation, quantile_kernel};
use crate::error::{Error, Result};
use crate::ops::{compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::normalize_dim;
use crate::tensor::Tensor;

/// Compute quantile along a dimension.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `q` - Quantile to compute, must be in [0.0, 1.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
/// * `interpolation` - Interpolation method ("linear", "lower", "higher", "nearest", "midpoint")
///
/// # Returns
///
/// Tensor containing the quantile values.
///
/// # Errors
///
/// - `InvalidArgument` if q is not in [0, 1]
/// - `InvalidArgument` if interpolation method is invalid
/// - `InvalidDimension` if dim is out of bounds
pub fn quantile_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<CpuRuntime>> {
    // Validate q is in [0, 1]
    if !(0.0..=1.0).contains(&q) {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("Quantile q must be in [0, 1], got {}", q),
        });
    }

    let interp = Interpolation::parse(interpolation)?;
    let dtype = a.dtype();

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            return Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
        }

        let flat = a.reshape(&[numel])?;
        return quantile_impl(client, &flat, q, Some(0), keepdim, interpolation);
    }

    let dim_val = dim.unwrap();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Ok(a.clone());
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        return Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device);
    }

    // Sort along the dimension
    let sorted = sort_impl(client, a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let out = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    let sorted_contig = ensure_contiguous(&sorted)?;
    let sorted_ptr = sorted_contig.ptr();
    let out_ptr = out.ptr();

    // Dispatch to typed kernel
    dispatch_dtype!(dtype, T => {
        unsafe {
            quantile_kernel::<T>(
                sorted_ptr as *const T,
                out_ptr as *mut T,
                outer_size,
                reduce_size,
                inner_size,
                q,
                interp,
            );
        }
    }, "quantile");

    Ok(out)
}

/// Compute percentile (quantile * 100) along a dimension.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `p` - Percentile to compute, must be in [0.0, 100.0]
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the percentile values.
pub fn percentile_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    if !(0.0..=100.0).contains(&p) {
        return Err(Error::InvalidArgument {
            arg: "p",
            reason: format!("Percentile p must be in [0, 100], got {}", p),
        });
    }

    quantile_impl(client, a, p / 100.0, dim, keepdim, "linear")
}

/// Compute median (50th percentile) along a dimension.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor
/// * `dim` - Dimension to reduce along (None = flatten first)
/// * `keepdim` - Whether to keep the reduced dimension
///
/// # Returns
///
/// Tensor containing the median values.
pub fn median_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CpuRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}
