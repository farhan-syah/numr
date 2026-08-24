//! Histogram operations for CPU runtime.

use super::super::helpers::{dispatch_dtype, ensure_contiguous};
use super::super::{CpuClient, CpuRuntime};
use super::{create_bin_edges, histogram_kernel, tensor_to_f64};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ReduceOps;
use crate::tensor::Tensor;

/// Compute histogram of values.
///
/// # Arguments
///
/// * `client` - The CPU runtime client
/// * `a` - Input tensor (will be flattened)
/// * `bins` - Number of histogram bins (must be > 0)
/// * `range` - Optional (min, max) range; defaults to (a.min(), a.max())
///
/// # Returns
///
/// Tuple of (histogram counts as I64 tensor, bin edges tensor).
///
/// # Errors
///
/// - `InvalidArgument` if bins is 0
/// - `InvalidArgument` if range min >= max
pub fn histogram_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    bins: usize,
    range: Option<(f64, f64)>,
) -> Result<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> {
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
        let hist = Tensor::<CpuRuntime>::zeros(&[bins], DType::I64, &client.device)?;
        let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;
        return Ok((hist, edges));
    }

    // Flatten input
    let flat = a.reshape(&[numel])?;
    let flat_contig = ensure_contiguous(&flat)?;
    let flat_ptr = flat_contig.ptr();

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
        let min_val = tensor_to_f64(&min_tensor)?;
        let max_val = tensor_to_f64(&max_tensor)?;

        // Handle case where all values are the same
        if (min_val - max_val).abs() < f64::EPSILON {
            (min_val - 0.5, max_val + 0.5)
        } else {
            (min_val, max_val)
        }
    };

    // Create histogram counts tensor
    let hist = Tensor::<CpuRuntime>::zeros(&[bins], DType::I64, &client.device)?;
    let hist_ptr = hist.ptr() as *mut i64;

    // Compute histogram using optimized kernel
    dispatch_dtype!(dtype, T => {
        unsafe {
            histogram_kernel::<T>(flat_ptr as *const T, hist_ptr, numel, bins, min_val, max_val);
        }
    }, "histogram");

    // Create bin edges
    let edges = create_bin_edges(client, min_val, max_val, bins, dtype)?;

    Ok((hist, edges))
}
