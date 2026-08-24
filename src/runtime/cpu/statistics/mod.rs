//! Statistics operations for CPU runtime
//!
//! Implements quantile, percentile, median, histogram, skewness, kurtosis, and mode.
//!
//! This module uses optimized CPU kernels with direct memory access for maximum
//! performance, while sharing common logic with other backends via the
//! `statistics_common` module.
//!
//! # Module Organization
//!
//! - `quantile` - Quantile, percentile, and median operations
//! - `histogram` - Histogram computation
//! - `moments` - Skewness and kurtosis (moment statistics)
//! - `mode` - Mode (most frequent value) computation

mod histogram;
mod mode;
mod moments;
mod quantile;

// Re-export all public API functions
pub use histogram::histogram_impl;
pub use mode::mode_impl;
pub use moments::{kurtosis_impl, skew_impl};
pub use quantile::{median_impl, percentile_impl, quantile_impl};

// ============================================================================
// Shared Imports
// ============================================================================

use super::helpers::dispatch_dtype;
use super::{CpuClient, CpuRuntime};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::common::statistics_common::{self, compute_bin_edges_f64};
use crate::tensor::Tensor;

// Re-export Interpolation for submodules
pub(crate) use crate::runtime::common::statistics_common::Interpolation;

// ============================================================================
// Optimized CPU Kernels
// ============================================================================

/// Quantile kernel - computes quantile from pre-sorted data.
///
/// # Safety
///
/// Caller must ensure:
/// - `sorted` points to valid memory of size `outer_size * reduce_size * inner_size`
/// - `out` points to valid memory of size `outer_size * inner_size`
/// - `reduce_size > 0`
/// - `q` is in [0.0, 1.0]
#[inline]
pub(crate) unsafe fn quantile_kernel<T: Element>(
    sorted: *const T,
    out: *mut T,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    q: f64,
    interp: Interpolation,
) {
    if reduce_size == 0 {
        return;
    }

    // Validate bounds in debug builds
    let total_input_size = outer_size * reduce_size * inner_size;
    let total_output_size = outer_size * inner_size;
    debug_assert!(
        total_input_size <= isize::MAX as usize,
        "Input array too large"
    );
    debug_assert!(
        total_output_size <= isize::MAX as usize,
        "Output array too large"
    );

    // Compute quantile indices using shared logic
    let (floor_idx, ceil_idx, frac) = statistics_common::compute_quantile_indices(q, reduce_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            let base_idx = outer * reduce_size * inner_size + inner;
            let out_idx = outer * inner_size + inner;

            // Debug bounds checks
            debug_assert!(
                base_idx + ceil_idx * inner_size < total_input_size,
                "Input index {} out of bounds (size {})",
                base_idx + ceil_idx * inner_size,
                total_input_size
            );
            debug_assert!(
                out_idx < total_output_size,
                "Output index {} out of bounds (size {})",
                out_idx,
                total_output_size
            );

            // SAFETY: bounds checked by debug_assert! above
            unsafe {
                let lower_val = (*sorted.add(base_idx + floor_idx * inner_size)).to_f64();
                let upper_val = (*sorted.add(base_idx + ceil_idx * inner_size)).to_f64();
                let value = interp.interpolate(lower_val, upper_val, frac);

                // SAFETY: out_idx bounds checked by debug_assert! above
                *out.add(out_idx) = T::from_f64(value);
            }
        }
    }
}

/// Histogram kernel - counts values into bins.
///
/// # Safety
///
/// Caller must ensure:
/// - `data` points to valid memory of size `numel`
/// - `counts` points to zero-initialized memory of size `bins`
/// - `bins > 0`
/// - `max_val > min_val`
#[inline]
pub(crate) unsafe fn histogram_kernel<T: Element>(
    data: *const T,
    counts: *mut i64,
    numel: usize,
    bins: usize,
    min_val: f64,
    max_val: f64,
) {
    debug_assert!(bins > 0, "bins must be positive");
    debug_assert!(max_val > min_val, "max_val must be greater than min_val");

    let bin_width = (max_val - min_val) / bins as f64;

    for i in 0..numel {
        // SAFETY: i < numel, and caller guarantees data has numel elements
        unsafe {
            let val = (*data.add(i)).to_f64();
            let bin_idx = statistics_common::compute_bin_index(val, min_val, bin_width, bins);

            debug_assert!(
                bin_idx < bins,
                "bin_idx {} out of bounds (bins {})",
                bin_idx,
                bins
            );

            // SAFETY: bin_idx < bins is guaranteed by compute_bin_index clamping
            *counts.add(bin_idx) += 1;
        }
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create bin edges tensor from computed f64 edges.
pub(crate) fn create_bin_edges(
    client: &CpuClient,
    min_val: f64,
    max_val: f64,
    bins: usize,
    dtype: DType,
) -> Result<Tensor<CpuRuntime>> {
    let edges_data = compute_bin_edges_f64(min_val, max_val, bins);

    // Create tensor and copy data based on dtype
    let edges = Tensor::<CpuRuntime>::try_empty(&[bins + 1], dtype, &client.device)?;
    let edges_ptr = edges.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            let out_slice = std::slice::from_raw_parts_mut(edges_ptr as *mut T, bins + 1);
            for (i, &val) in edges_data.iter().enumerate() {
                out_slice[i] = T::from_f64(val);
            }
        }
    }, "histogram_edges");

    Ok(edges)
}

/// Extract scalar f64 value from tensor.
pub(crate) fn tensor_to_f64(t: &Tensor<CpuRuntime>) -> Result<f64> {
    let dtype = t.dtype();
    let ptr = t.ptr();

    let val = dispatch_dtype!(dtype, T => {
        unsafe { (*(ptr as *const T)).to_f64() }
    }, "tensor_to_f64");

    Ok(val)
}
