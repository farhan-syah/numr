//! Mode (most frequent value) operations for CPU runtime.

use super::super::helpers::ensure_contiguous;
use super::super::sort::sort_impl;
use super::super::{CpuClient, CpuRuntime};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{TypeConversionOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::common::statistics_common::compute_mode_strided;
use crate::runtime::normalize_dim;
use crate::tensor::Tensor;

/// Compute mode (most frequent value) along a dimension.
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
/// Tuple of (mode_values, mode_counts) tensors.
pub fn mode_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
    let dtype = a.dtype();

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            let values = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
            let counts = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &client.device)?;
            return Ok((values, counts));
        }

        let flat = a.reshape(&[numel])?;
        return mode_impl(client, &flat, Some(0), keepdim);
    }

    let dim_val = dim.unwrap();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        // Scalar input: mode is itself with count 1
        let counts = Tensor::<CpuRuntime>::full_scalar(&[], DType::I64, 1.0, &client.device)?;
        return Ok((a.clone(), counts));
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        let values = Tensor::<CpuRuntime>::empty(&out_shape, dtype, &client.device)?;
        let counts = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &client.device)?;
        return Ok((values, counts));
    }

    // Sort along the dimension
    let sorted = sort_impl(client, a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    let sorted_contig = ensure_contiguous(&sorted)?;

    // Use shared implementation to compute mode
    // Match on dtype explicitly for type-safe to_vec() conversion
    match dtype {
        DType::F32 => {
            let sorted_data: Vec<f32> = sorted_contig.to_vec();
            let (mode_values, mode_counts) =
                compute_mode_strided(&sorted_data, outer_size, reduce_size, inner_size);
            Ok((
                Tensor::<CpuRuntime>::from_slice(&mode_values, &out_shape, &client.device)?,
                Tensor::<CpuRuntime>::from_slice(&mode_counts, &out_shape, &client.device)?,
            ))
        }
        DType::F64 => {
            let sorted_data: Vec<f64> = sorted_contig.to_vec();
            let (mode_values, mode_counts) =
                compute_mode_strided(&sorted_data, outer_size, reduce_size, inner_size);
            Ok((
                Tensor::<CpuRuntime>::from_slice(&mode_values, &out_shape, &client.device)?,
                Tensor::<CpuRuntime>::from_slice(&mode_counts, &out_shape, &client.device)?,
            ))
        }
        DType::I32 => {
            let sorted_data: Vec<i32> = sorted_contig.to_vec();
            let (mode_values, mode_counts) =
                compute_mode_strided(&sorted_data, outer_size, reduce_size, inner_size);
            Ok((
                Tensor::<CpuRuntime>::from_slice(&mode_values, &out_shape, &client.device)?,
                Tensor::<CpuRuntime>::from_slice(&mode_counts, &out_shape, &client.device)?,
            ))
        }
        DType::I64 => {
            let sorted_data: Vec<i64> = sorted_contig.to_vec();
            let (mode_values, mode_counts) =
                compute_mode_strided(&sorted_data, outer_size, reduce_size, inner_size);
            Ok((
                Tensor::<CpuRuntime>::from_slice(&mode_values, &out_shape, &client.device)?,
                Tensor::<CpuRuntime>::from_slice(&mode_counts, &out_shape, &client.device)?,
            ))
        }
        DType::U32 => {
            let sorted_data: Vec<u32> = sorted_contig.to_vec();
            let (mode_values, mode_counts) =
                compute_mode_strided(&sorted_data, outer_size, reduce_size, inner_size);
            Ok((
                Tensor::<CpuRuntime>::from_slice(&mode_values, &out_shape, &client.device)?,
                Tensor::<CpuRuntime>::from_slice(&mode_counts, &out_shape, &client.device)?,
            ))
        }
        _ => {
            // For other dtypes (F16, BF16, etc.), cast to F32, compute, cast back
            let a_f32 = client.cast(a, DType::F32)?;
            let (values_f32, counts) = mode_impl(client, &a_f32, dim, keepdim)?;
            let values = client.cast(&values_f32, dtype)?;
            Ok((values, counts))
        }
    }
}
