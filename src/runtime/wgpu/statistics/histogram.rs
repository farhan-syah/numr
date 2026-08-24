//! Histogram operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps};
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

use super::{create_bin_edges, tensor_to_f64};

/// Compute histogram of values entirely on GPU.
///
/// # Algorithm
///
/// 1. Determine min/max values from input data (GPU reductions)
/// 2. Compute bin width: `(max - min) / bins`
/// 3. For each element, compute bin index: `floor((x - min) / bin_width)`
/// 4. Clamp bin indices to valid range [0, bins-1]
/// 5. Cast indices to I64 for one_hot operation
/// 6. Use one_hot to create indicator matrix: shape [numel, bins]
/// 7. Sum along axis 0 to get histogram counts
///
/// # GPU-Only Operations
///
/// All computation stays on GPU - no CPU transfers except for scalar min/max values
/// needed to determine bin ranges (control-flow transfers).
pub fn histogram_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
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
        let hist = Tensor::<WgpuRuntime>::zeros(&[bins], DType::I64, client.device())?;
        let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;
        return Ok((hist, edges));
    }

    // Flatten input
    let flat = a.reshape(&[numel])?;

    // Determine range
    let (min_val, max_val) = if let Some((min, max)) = range {
        if min >= max {
            return Err(Error::InvalidArgument {
                arg: "range",
                reason: format!("Range min ({}) must be less than max ({})", min, max),
            });
        }
        (min, max)
    } else {
        let min_tensor = client.min(&flat, &[], false)?;
        let max_tensor = client.max(&flat, &[], false)?;
        let min_val = tensor_to_f64(client, &min_tensor)?;
        let max_val = tensor_to_f64(client, &max_tensor)?;

        // Handle case where all values are the same
        if (min_val - max_val).abs() < f64::EPSILON {
            (min_val - 0.5, max_val + 0.5)
        } else {
            (min_val, max_val)
        }
    };

    // Cast to F32 for computation if needed
    let flat_f32 = if dtype != DType::F32 {
        client.cast(&flat, DType::F32)?
    } else {
        flat.clone()
    };

    // Compute bin width
    let bin_width = (max_val - min_val) / bins as f64;

    // Compute bin indices: floor((x - min_val) / bin_width)
    let shifted = client.sub_scalar(&flat_f32, min_val)?;
    let normalized = client.div_scalar(&shifted, bin_width)?;
    let floored = client.floor(&normalized)?;

    // Clamp to [0, bins-1]
    let bin_indices = client.clamp(&floored, 0.0, (bins - 1) as f64)?;

    // Cast to I64 for one_hot indexing
    let bin_indices_i64 = client.cast(&bin_indices, DType::I64)?;

    // Create one-hot encoding: shape [numel, bins]
    let one_hot_matrix = client.one_hot(&bin_indices_i64, bins)?;

    // Sum along axis 0 to get histogram counts: shape [bins]
    let hist = client.sum(&one_hot_matrix, &[0], false)?;

    // Cast result back to I64
    let hist = client.cast(&hist, DType::I64)?;

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist, edges))
}
