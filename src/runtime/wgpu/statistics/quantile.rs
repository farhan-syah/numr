//! Quantile, percentile, and median operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, IndexingOps, ScalarOps, SortingOps, TypeConversionOps, reduce_dim_output_shape,
};
use crate::runtime::common::statistics_common::Interpolation;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, normalize_dim};
use crate::tensor::Tensor;

/// Compute quantile along a dimension using GPU operations.
///
/// # Algorithm
///
/// 1. Sort data along the specified dimension (GPU)
/// 2. Compute floor and ceil indices for quantile position
/// 3. Gather values at both indices using index_select on the sorted dimension
/// 4. Perform linear interpolation between floor and ceil values (GPU)
/// 5. Return interpolated result
///
/// # GPU-Only Operations
///
/// All computation stays on GPU - no data transfers. Uses GPU indexing operations
/// (index_select) to gather the floor and ceil values, then interpolates on GPU.
pub fn quantile_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    q: f64,
    dim: Option<isize>,
    keepdim: bool,
    interpolation: &str,
) -> Result<Tensor<WgpuRuntime>> {
    // Validate q is in [0, 1]
    if !(0.0..=1.0).contains(&q) {
        return Err(Error::InvalidArgument {
            arg: "q",
            reason: format!("Quantile q must be in [0, 1], got {}", q),
        });
    }

    let _interp = Interpolation::parse(interpolation)?;
    let dtype = a.dtype();

    // WebGPU sort path does not support F64 directly; compute in F32 and cast back.
    if dtype == DType::F64 {
        let a_f32 = client.cast(a, DType::F32)?;
        let out_f32 = quantile_impl(client, &a_f32, q, dim, keepdim, interpolation)?;
        return client.cast(&out_f32, DType::F64);
    }

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            return Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device());
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
        return Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device());
    }

    // Sort along dimension using WebGPU sort
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);

    // Calculate quantile indices using shared logic
    let (floor_idx, ceil_idx, frac) =
        crate::runtime::common::statistics_common::compute_quantile_indices(q, dim_size);

    // Check for empty output
    let out_numel = out_shape.iter().product::<usize>();
    if out_numel == 0 {
        return Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device());
    }

    // Gather floor and ceil values using index_select
    // Create index tensors for the floor and ceil positions
    let floor_idx_tensor =
        Tensor::<WgpuRuntime>::from_slice(&[floor_idx as i32], &[1], client.device())?;
    let ceil_idx_tensor =
        Tensor::<WgpuRuntime>::from_slice(&[ceil_idx as i32], &[1], client.device())?;

    // Select floor and ceil values along dimension
    let floor_vals = client.index_select(&sorted, dim_idx, &floor_idx_tensor)?;
    let ceil_vals = client.index_select(&sorted, dim_idx, &ceil_idx_tensor)?;

    // Squeeze to remove the indexed dimension
    let mut floor_shape = Vec::with_capacity(shape.len() - 1);
    for (i, &s) in shape.iter().enumerate() {
        if i != dim_idx {
            floor_shape.push(s);
        }
    }
    let ceil_shape = floor_shape.clone();

    let floor_vals = floor_vals.reshape(&floor_shape)?;
    let ceil_vals = ceil_vals.reshape(&ceil_shape)?;

    // Convert to F32 for interpolation if needed
    let floor_f32 = if dtype != DType::F32 {
        client.cast(&floor_vals, DType::F32)?
    } else {
        floor_vals
    };

    let ceil_f32 = if dtype != DType::F32 {
        client.cast(&ceil_vals, DType::F32)?
    } else {
        ceil_vals
    };

    // Linear interpolation: result = floor + frac * (ceil - floor)
    let diff = client.sub(&ceil_f32, &floor_f32)?;
    let scaled_diff = client.mul_scalar(&diff, frac)?;
    let result_f32 = client.add(&floor_f32, &scaled_diff)?;

    // Cast back to original dtype if needed
    let result = if dtype != DType::F32 {
        client.cast(&result_f32, dtype)?
    } else {
        result_f32
    };

    // Add back the reduced dimension if keepdim
    if keepdim {
        let mut final_shape = result.shape().to_vec();
        final_shape.insert(dim_idx, 1);
        result.reshape(&final_shape)
    } else {
        Ok(result)
    }
}

/// Compute percentile (quantile * 100) along a dimension.
pub fn percentile_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    p: f64,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
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
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    quantile_impl(client, a, 0.5, dim, keepdim, "linear")
}
