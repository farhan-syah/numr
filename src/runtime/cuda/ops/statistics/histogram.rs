//! Histogram operations for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

use super::{create_bin_edges, read_scalar_f64};

/// Compute histogram of values entirely on GPU.
///
/// # Implementation Notes
///
/// Algorithm:
/// 1. Cast input to F32 if needed
/// 2. Compute min/max values (GPU reduce operations)
/// 3. Compute bin width = (max - min) / bins
/// 4. For each element, compute bin index: min(floor((x - min) / bin_width), bins - 1)
/// 5. Create one-hot encoding [n, bins]: indicator matrix
/// 6. Sum along axis 0 to get bin counts [bins]
/// 7. All operations run on GPU - no CPU transfers
pub fn histogram_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    if bins == 0 {
        return Err(Error::InvalidArgument {
            arg: "bins",
            reason: "Number of bins must be positive".to_string(),
        });
    }

    let dtype = a.dtype();
    let numel = a.numel();

    if numel == 0 {
        let (min_val, max_val) = range.unwrap_or((0.0, 1.0));
        let hist = Tensor::<CudaRuntime>::zeros(&[bins], DType::I64, &client.device)?;
        let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;
        return Ok((hist, edges));
    }

    // Flatten and cast to F32 for computation (stable for histogram)
    let flat = a.reshape(&[numel])?;
    let flat_f32 = if dtype == DType::F32 {
        flat.clone()
    } else {
        client.cast(&flat, DType::F32)?
    };

    // Determine range (compute on GPU, read scalars)
    let (min_val, max_val) = if let Some((min, max)) = range {
        if min >= max {
            return Err(Error::InvalidArgument {
                arg: "range",
                reason: format!("Range min ({}) must be less than max ({})", min, max),
            });
        }
        (min, max)
    } else {
        let min_tensor = client.min(&flat_f32, &[], false)?;
        let max_tensor = client.max(&flat_f32, &[], false)?;
        let min_val = read_scalar_f64(&min_tensor)?;
        let max_val = read_scalar_f64(&max_tensor)?;

        // Handle case where all values are the same
        if (min_val - max_val).abs() < f64::EPSILON {
            (min_val - 0.5, max_val + 0.5)
        } else {
            (min_val, max_val)
        }
    };

    // Compute on GPU:
    // 1. Shift: x - min_val
    let shifted = client.sub_scalar(&flat_f32, min_val)?;

    // 2. Compute bin width
    let bin_width = (max_val - min_val) / bins as f64;

    // 3. Divide by bin width: (x - min) / bin_width
    let bin_indices_f = client.div_scalar(&shifted, bin_width)?;

    // 4. Floor to get bin indices
    let bin_indices_floored = client.floor(&bin_indices_f)?;

    // 5. Clamp to [0, bins-1]: min(index, bins-1)
    let bin_max_scalar = (bins - 1) as f32;
    let bin_indices_clamped = if bins > 1 {
        // Use minimum to clamp to bins-1
        let ones = Tensor::<CudaRuntime>::ones(&[numel], DType::F32, &client.device)?;
        let bin_max_tensor = client.mul_scalar(&ones, bin_max_scalar as f64)?;
        client.minimum(&bin_indices_floored, &bin_max_tensor)?
    } else {
        bin_indices_floored
    };

    // 6. Convert to I64 for one_hot
    let bin_indices_i64 = client.cast(&bin_indices_clamped, DType::I64)?;

    // 7. Create one-hot encoding [numel, bins]
    let one_hot = client.one_hot(&bin_indices_i64, bins)?;

    // 8. Sum along axis 0 to get counts
    let hist = client.sum(&one_hot, &[0], false)?;

    // 9. Cast to I64 for output type
    let hist_i64 = client.cast(&hist, DType::I64)?;

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist_i64, edges))
}
