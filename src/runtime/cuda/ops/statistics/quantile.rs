//! Quantile, percentile, and median operations for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, IndexingOps, ScalarOps, SortingOps, TypeConversionOps};
use crate::runtime::common::statistics_common::Interpolation;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::normalize_dim;
use crate::tensor::Tensor;

/// Compute quantile along a dimension entirely on GPU.
///
/// # Algorithm
///
/// 1. Sort along dimension (GPU)
/// 2. Compute floor and ceiling indices based on quantile (CPU calculation)
/// 3. Index into sorted tensor at floor/ceil positions (GPU)
/// 4. Interpolate between floor and ceil values using scalar operations (GPU)
/// 5. All heavy lifting stays on GPU - only scalar indices are computed on CPU
pub fn quantile_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<CudaRuntime>> {
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
            return Tensor::<CudaRuntime>::try_empty(&out_shape, dtype, &client.device);
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
        return Tensor::<CudaRuntime>::try_empty(&out_shape, dtype, &client.device);
    }

    // Sort along dimension using CUDA sort (GPU)
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape
    use crate::ops::reduce_dim_output_shape;
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);

    // Check for empty output
    let out_numel = out_shape.iter().product::<usize>();
    if out_numel == 0 {
        return Tensor::<CudaRuntime>::try_empty(&out_shape, dtype, &client.device);
    }

    // Calculate quantile indices (small computation, OK on CPU)
    let (floor_idx, ceil_idx, frac) =
        crate::runtime::common::statistics_common::compute_quantile_indices(q, dim_size);

    // index_select requires at least 1D indices, so use [1] for scalar output
    let is_scalar_output = out_shape.is_empty();
    let work_shape = if is_scalar_output {
        vec![1]
    } else {
        out_shape.clone()
    };

    // Create index tensors for gather
    let floor_indices = Tensor::<CudaRuntime>::try_full_scalar(
        &work_shape,
        dtype,
        floor_idx as f64,
        &client.device,
    )?;
    let ceil_indices = Tensor::<CudaRuntime>::try_full_scalar(
        &work_shape,
        dtype,
        ceil_idx as f64,
        &client.device,
    )?;

    // Cast indices to I64 for gather
    let floor_indices_i64 = client.cast(&floor_indices, DType::I64)?;
    let ceil_indices_i64 = client.cast(&ceil_indices, DType::I64)?;

    // Gather values at floor and ceiling indices along the dimension
    let floor_values = client.index_select(&sorted, dim_idx, &floor_indices_i64)?;
    let ceil_values = client.index_select(&sorted, dim_idx, &ceil_indices_i64)?;

    // Interpolate on GPU based on interpolation method
    let result = match interp {
        Interpolation::Linear => {
            // result = floor_val + frac * (ceil_val - floor_val)
            if frac.abs() < f64::EPSILON {
                // No interpolation needed
                floor_values
            } else if (frac - 1.0).abs() < f64::EPSILON {
                // Use ceiling value
                ceil_values
            } else {
                // Linear interpolation on GPU
                let diff = client.sub(&ceil_values, &floor_values)?;
                let weighted_diff = client.mul_scalar(&diff, frac)?;
                client.add(&floor_values, &weighted_diff)?
            }
        }
        Interpolation::Lower => floor_values,
        Interpolation::Higher => ceil_values,
        Interpolation::Midpoint => {
            // (floor + ceil) / 2
            let sum = client.add(&floor_values, &ceil_values)?;
            client.mul_scalar(&sum, 0.5)?
        }
        Interpolation::Nearest => {
            // Choose nearest: if frac >= 0.5, use ceil, else floor
            if frac >= 0.5 {
                ceil_values
            } else {
                floor_values
            }
        }
    };

    // Reshape to target output shape (handles scalar case)
    if is_scalar_output {
        result.reshape(&out_shape)
    } else {
        Ok(result)
    }
}

/// Compute percentile (quantile * 100) along a dimension.
pub fn percentile_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    if !(0.0..=100.0).contains(&p) {
        return Err(Error::InvalidArgument {
            arg: "p",
            reason: format!("Percentile p must be in [0, 100], got {}", p),
        });
    }

    quantile_impl(client, a, p / 100.0, dim, keepdim, "linear")
}

/// Compute median (50th percentile) along a dimension.
pub fn median_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}
