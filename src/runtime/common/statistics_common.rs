//! Shared statistics utilities for all runtime backends
//!
//! This module contains common types and helper functions used by the statistics
//! implementations across CPU, CUDA, and WebGPU backends. Centralizing this code
//! ensures consistency and maintainability (DRY principle).

use crate::dtype::Element;
use crate::error::{Error, Result};

// ============================================================================
// Constants
// ============================================================================

/// Small epsilon value to prevent division by zero in statistics calculations.
///
/// This value is chosen to be small enough to not affect typical computations
/// while still preventing numerical instability when dividing by near-zero
/// standard deviations.
///
/// For F32 computations, this provides protection without significant precision loss.
/// For F64 computations, consider using `EPSILON_F64` for tighter tolerance.
pub const DIVISION_EPSILON: f64 = 1e-10;

// ============================================================================
// Interpolation Methods
// ============================================================================

/// Interpolation methods for quantile computation.
///
/// These methods determine how to compute quantile values when the desired
/// quantile falls between two data points.
///
/// # Methods
///
/// - `Linear`: Linear interpolation between the two nearest points (default)
/// - `Lower`: Use the lower of the two surrounding values
/// - `Higher`: Use the higher of the two surrounding values
/// - `Nearest`: Use the value closest to the computed index
/// - `Midpoint`: Use the average of the two surrounding values
///
/// # Example
///
/// For data `[1, 2, 3, 4]` and q=0.25 (index = 0.75):
/// - Linear: `1 * 0.25 + 2 * 0.75 = 1.75`
/// - Lower: `1` (floor index = 0)
/// - Higher: `2` (ceil index = 1)
/// - Nearest: `2` (0.75 rounds to 1)
/// - Midpoint: `(1 + 2) / 2 = 1.5`
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Interpolation {
    /// Linear interpolation between surrounding data points.
    /// Result = lower * (1 - frac) + upper * frac
    #[default]
    Linear,
    /// Use the lower of the two surrounding data points.
    Lower,
    /// Use the higher of the two surrounding data points.
    Higher,
    /// Use the data point nearest to the computed index.
    Nearest,
    /// Use the arithmetic mean of the two surrounding data points.
    Midpoint,
}

impl Interpolation {
    /// Parse an interpolation method from a string.
    ///
    /// # Arguments
    ///
    /// * `s` - The interpolation method name (case-insensitive)
    ///
    /// # Returns
    ///
    /// The corresponding `Interpolation` variant, or an error if the string
    /// is not a valid interpolation method.
    ///
    /// # Valid Options
    ///
    /// - "linear" (default, recommended)
    /// - "lower"
    /// - "higher"
    /// - "nearest"
    /// - "midpoint"
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if the string is not a valid method.
    pub fn parse(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "linear" => Ok(Interpolation::Linear),
            "lower" => Ok(Interpolation::Lower),
            "higher" => Ok(Interpolation::Higher),
            "nearest" => Ok(Interpolation::Nearest),
            "midpoint" => Ok(Interpolation::Midpoint),
            _ => Err(Error::InvalidArgument {
                arg: "interpolation",
                reason: format!(
                    "Invalid interpolation method '{}'. Valid options: linear, lower, higher, nearest, midpoint",
                    s
                ),
            }),
        }
    }

    /// Compute the interpolated value given sorted data indices.
    ///
    /// # Arguments
    ///
    /// * `lower_val` - The value at the floor index
    /// * `upper_val` - The value at the ceil index
    /// * `frac` - The fractional part of the virtual index (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The interpolated value according to this method.
    #[inline]
    pub fn interpolate(&self, lower_val: f64, upper_val: f64, frac: f64) -> f64 {
        match self {
            Interpolation::Linear => lower_val * (1.0 - frac) + upper_val * frac,
            Interpolation::Lower => lower_val,
            Interpolation::Higher => upper_val,
            Interpolation::Nearest => {
                if frac < 0.5 {
                    lower_val
                } else {
                    upper_val
                }
            }
            Interpolation::Midpoint => (lower_val + upper_val) / 2.0,
        }
    }
}

// ============================================================================
// Bin Edge Computation
// ============================================================================

/// Compute histogram bin edges as a vector of f64 values.
///
/// Creates `bins + 1` evenly spaced edge values from `min_val` to `max_val`.
///
/// # Arguments
///
/// * `min_val` - The minimum value (left edge of first bin)
/// * `max_val` - The maximum value (right edge of last bin)
/// * `bins` - The number of bins (must be > 0)
///
/// # Returns
///
/// A vector of `bins + 1` edge values.
///
#[inline]
pub fn compute_bin_edges_f64(min_val: f64, max_val: f64, bins: usize) -> Vec<f64> {
    let bin_width = (max_val - min_val) / bins as f64;
    (0..=bins).map(|i| min_val + i as f64 * bin_width).collect()
}

/// Compute the bin index for a value given histogram parameters.
///
/// # Arguments
///
/// * `value` - The value to bin
/// * `min_val` - The minimum value of the histogram range
/// * `bin_width` - The width of each bin
/// * `bins` - The total number of bins
///
/// # Returns
///
/// The bin index, clamped to `[0, bins - 1]`.
///
/// # Notes
///
/// - Values below `min_val` are placed in bin 0
/// - Values at or above `max_val` are placed in the last bin
#[inline]
pub fn compute_bin_index(value: f64, min_val: f64, bin_width: f64, bins: usize) -> usize {
    let idx = ((value - min_val) / bin_width).floor() as isize;
    if idx < 0 {
        0
    } else if idx >= bins as isize {
        bins - 1
    } else {
        idx as usize
    }
}

// ============================================================================
// Quantile Index Computation
// ============================================================================

/// Compute quantile indices and interpolation fraction.
///
/// Given a quantile value `q` and array size `n`, computes:
/// - `floor_idx`: The lower bounding index
/// - `ceil_idx`: The upper bounding index (clamped to n-1)
/// - `frac`: The fractional part for interpolation
///
/// # Arguments
///
/// * `q` - Quantile value in [0.0, 1.0]
/// * `n` - Size of the array (must be > 0)
///
/// # Returns
///
/// Tuple of `(floor_idx, ceil_idx, frac)`.
///
/// # Example
///
/// For n=5 (indices 0-4):
/// - q=0.0: (0, 0, 0.0) -> first element
/// - q=0.5: (2, 2, 0.0) -> middle element
/// - q=0.25: (1, 1, 0.0) -> second element
/// - q=0.75: (3, 3, 0.0) -> fourth element
/// - q=1.0: (4, 4, 0.0) -> last element
#[inline]
pub fn compute_quantile_indices(q: f64, n: usize) -> (usize, usize, f64) {
    debug_assert!(n > 0, "Array size must be positive");
    debug_assert!((0.0..=1.0).contains(&q), "Quantile must be in [0, 1]");

    let virtual_idx = q * (n - 1) as f64;
    let floor_idx = virtual_idx.floor() as usize;
    let ceil_idx = (virtual_idx.ceil() as usize).min(n - 1);
    let frac = virtual_idx - floor_idx as f64;

    (floor_idx, ceil_idx, frac)
}

// ============================================================================
// Generic GPU Moment Statistics (Skewness / Kurtosis)
// ============================================================================

/// Compute skewness (third standardized moment) using primitive tensor ops.
///
/// This is a generic composite implementation usable by any backend (CUDA, WebGPU, etc.)
/// that implements the required primitive operation traits.
///
/// Uses `mul` chains instead of `pow_scalar` to correctly handle negative values
/// (GPU `powf(negative, 3.0)` returns NaN).
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn skew_composite<R, C>(
    client: &C,
    a: &crate::tensor::Tensor<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: crate::runtime::Runtime<DType = crate::dtype::DType>,
    C: crate::ops::BinaryOps<R>
        + crate::ops::ReduceOps<R>
        + crate::ops::StatisticalOps<R>
        + crate::runtime::RuntimeClient<R>,
{
    let dtype = a.dtype();

    // skew = E[(X - mean)^3] / std^3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute third moment via multiplications (avoids NaN from pow on negatives)
    let centered_sq = client.mul(&centered, &centered)?;
    let centered_cubed = client.mul(&centered_sq, &centered)?;
    let m3 = client.mean(&centered_cubed, dims, keepdim)?;

    // Compute std^3
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_sq = client.mul(&std_val, &std_val)?;
    let std_cubed = client.mul(&std_sq, &std_val)?;

    // skew = m3 / std^3 (with epsilon to avoid division by zero)
    let epsilon = crate::tensor::Tensor::<R>::full_scalar(
        std_cubed.shape(),
        dtype,
        DIVISION_EPSILON,
        client.device(),
    )?;
    let std_cubed_safe = client.add(&std_cubed, &epsilon)?;

    client.div(&m3, &std_cubed_safe)
}

/// Compute excess kurtosis (fourth standardized moment - 3) using primitive tensor ops.
///
/// This is a generic composite implementation usable by any backend (CUDA, WebGPU, etc.)
/// that implements the required primitive operation traits.
///
/// Uses `mul` chains instead of `pow_scalar` to correctly handle negative values.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn kurtosis_composite<R, C>(
    client: &C,
    a: &crate::tensor::Tensor<R>,
    dims: &[usize],
    keepdim: bool,
    correction: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: crate::runtime::Runtime<DType = crate::dtype::DType>,
    C: crate::ops::BinaryOps<R>
        + crate::ops::ReduceOps<R>
        + crate::ops::StatisticalOps<R>
        + crate::runtime::RuntimeClient<R>,
{
    let dtype = a.dtype();

    // kurtosis = E[(X - mean)^4] / std^4 - 3
    let mean = client.mean(a, dims, true)?;
    let centered = client.sub(a, &mean)?;

    // Compute fourth moment via multiplications
    let centered_sq = client.mul(&centered, &centered)?;
    let centered_fourth = client.mul(&centered_sq, &centered_sq)?;
    let m4 = client.mean(&centered_fourth, dims, keepdim)?;

    // Compute std^4
    let std_val = client.std(a, dims, keepdim, correction)?;
    let std_sq = client.mul(&std_val, &std_val)?;
    let std_fourth = client.mul(&std_sq, &std_sq)?;

    // kurtosis = m4 / std^4 - 3 (with epsilon to avoid division by zero)
    let epsilon = crate::tensor::Tensor::<R>::full_scalar(
        std_fourth.shape(),
        dtype,
        DIVISION_EPSILON,
        client.device(),
    )?;
    let std_fourth_safe = client.add(&std_fourth, &epsilon)?;

    let ratio = client.div(&m4, &std_fourth_safe)?;
    let three =
        crate::tensor::Tensor::<R>::full_scalar(ratio.shape(), dtype, 3.0, client.device())?;
    client.sub(&ratio, &three)
}

// ============================================================================
// CPU Scalar Moment Statistics (Skewness / Kurtosis)
// ============================================================================

/// Compute the skewness (third standardized moment) of a data slice.
///
/// Skewness measures the asymmetry of a distribution:
/// - Positive skewness: tail extends toward positive values (right-skewed)
/// - Negative skewness: tail extends toward negative values (left-skewed)
/// - Zero skewness: symmetric distribution
///
/// # Formula
///
/// ```text
/// skewness = E[(X - μ)³] / σ³
///          = m₃ / m₂^(3/2)
/// ```
///
/// where m₂ is the second central moment (variance) and m₃ is the third
/// central moment.
///
/// # Arguments
///
/// * `data` - Slice of input values
/// * `_correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// The skewness value, or 0.0 if the data has fewer than 3 elements
/// or zero variance.
///
/// # Numerical Stability
///
/// Uses f64 accumulation internally for precision, regardless of input type.
pub fn compute_skewness<T: Element>(data: &[T], _correction: usize) -> f64 {
    let n = data.len();
    if n < 3 {
        return 0.0;
    }

    // Compute mean with f64 precision
    let sum: f64 = data.iter().map(|v| v.to_f64()).sum();
    let mean = sum / n as f64;

    // Compute second and third central moments
    let mut m2 = 0.0f64;
    let mut m3 = 0.0f64;
    for &val in data {
        let diff = val.to_f64() - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }

    m2 /= n as f64;
    m3 /= n as f64;

    // skewness = m3 / m2^(3/2)
    let std = m2.sqrt();
    if std < DIVISION_EPSILON {
        0.0
    } else {
        m3 / (std * std * std)
    }
}

/// Compute the excess kurtosis (fourth standardized moment - 3) of a data slice.
///
/// Kurtosis measures the "tailedness" of a distribution:
/// - Positive excess kurtosis (leptokurtic): heavier tails than normal
/// - Negative excess kurtosis (platykurtic): lighter tails than normal
/// - Zero excess kurtosis (mesokurtic): similar to normal distribution
///
/// # Formula
///
/// ```text
/// excess_kurtosis = E[(X - μ)⁴] / σ⁴ - 3
///                 = m₄ / m₂² - 3
/// ```
///
/// where m₂ is the second central moment and m₄ is the fourth central moment.
/// The subtraction of 3 gives "excess" kurtosis, making the normal distribution
/// have kurtosis of 0.
///
/// # Arguments
///
/// * `data` - Slice of input values
/// * `_correction` - Bias correction (reserved for future use)
///
/// # Returns
///
/// The excess kurtosis value, or 0.0 if the data has fewer than 4 elements
/// or zero variance.
///
/// # Numerical Stability
///
/// Uses f64 accumulation internally for precision, regardless of input type.
pub fn compute_kurtosis<T: Element>(data: &[T], _correction: usize) -> f64 {
    let n = data.len();
    if n < 4 {
        return 0.0;
    }

    // Compute mean with f64 precision
    let sum: f64 = data.iter().map(|v| v.to_f64()).sum();
    let mean = sum / n as f64;

    // Compute second and fourth central moments
    let mut m2 = 0.0f64;
    let mut m4 = 0.0f64;
    for &val in data {
        let diff = val.to_f64() - mean;
        let diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }

    m2 /= n as f64;
    m4 /= n as f64;

    // excess kurtosis = m4 / m2^2 - 3
    if m2 < DIVISION_EPSILON {
        0.0
    } else {
        m4 / (m2 * m2) - 3.0
    }
}

// ============================================================================
// Mode Computation
// ============================================================================

/// Result of mode computation for a single reduction.
#[derive(Debug, Clone)]
pub struct ModeResult<T> {
    /// The most frequent value
    pub value: T,
    /// The number of times it appears
    pub count: i64,
}

/// Compute the mode (most frequent value) from a sorted slice.
///
/// The input must be sorted in ascending order. If multiple values have the
/// same maximum frequency, the smallest value (which appears first in sorted
/// order) is returned.
///
/// # Arguments
///
/// * `sorted` - Slice of sorted input values (ascending order)
///
/// # Returns
///
/// `ModeResult` containing the mode value and its count.
///
/// # Algorithm
///
/// Uses run-length encoding on sorted data:
/// 1. Track current run (value and count)
/// 2. Track best run (highest count seen)
/// 3. Iterate through sorted data, updating runs
/// 4. Return the value with highest count
///
/// # Complexity
///
/// O(n) time, O(1) space.
///
/// # Panics
///
/// Panics if the input slice is empty.
pub fn compute_mode<T: Element>(sorted: &[T]) -> ModeResult<T> {
    debug_assert!(!sorted.is_empty(), "Cannot compute mode of empty slice");

    if sorted.len() == 1 {
        return ModeResult {
            value: sorted[0],
            count: 1,
        };
    }

    let mut best_value = sorted[0];
    let mut best_count: i64 = 1;

    let mut current_value = sorted[0];
    let mut current_count: i64 = 1;

    for &val in &sorted[1..] {
        // Compare using f64 representation for consistent comparison across types
        if val.to_f64() == current_value.to_f64() {
            current_count += 1;
        } else {
            // New run started
            if current_count > best_count {
                best_value = current_value;
                best_count = current_count;
            }
            current_value = val;
            current_count = 1;
        }
    }

    // Check final run
    if current_count > best_count {
        best_value = current_value;
        best_count = current_count;
    }

    ModeResult {
        value: best_value,
        count: best_count,
    }
}

/// Compute mode values from sorted data with strided access.
///
/// This function computes mode values from pre-sorted data, handling
/// the strided iteration pattern common to tensor reduction operations.
///
/// # Arguments
///
/// * `sorted` - Slice of sorted input data
/// * `outer_size` - Number of outer iterations (product of dimensions before reduce dim)
/// * `reduce_size` - Size of the dimension being reduced (sorted along)
/// * `inner_size` - Number of inner iterations (product of dimensions after reduce dim)
///
/// # Returns
///
/// Tuple of (mode_values, mode_counts) with length `outer_size * inner_size` each.
///
/// # Memory Layout
///
/// Input is assumed to be in row-major order with the reduce dimension
/// being the middle dimension:
/// ```text
/// [outer][reduce][inner]
/// ```
pub fn compute_mode_strided<T: Element>(
    sorted: &[T],
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) -> (Vec<T>, Vec<i64>) {
    let output_size = outer_size * inner_size;
    let mut values = Vec::with_capacity(output_size);
    let mut counts = Vec::with_capacity(output_size);

    for outer in 0..outer_size {
        for inner in 0..inner_size {
            // Extract the slice for this reduction
            let mut slice_data = Vec::with_capacity(reduce_size);
            for r in 0..reduce_size {
                let idx = outer * reduce_size * inner_size + r * inner_size + inner;
                slice_data.push(sorted[idx]);
            }

            // Data should already be sorted, but slice extraction breaks sorting
            // for non-contiguous access. Sort the extracted slice.
            slice_data.sort_by(|a, b| {
                a.to_f64()
                    .partial_cmp(&b.to_f64())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let result = compute_mode(&slice_data);
            values.push(result.value);
            counts.push(result.count);
        }
    }

    (values, counts)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_from_str() {
        assert_eq!(
            Interpolation::parse("linear").unwrap(),
            Interpolation::Linear
        );
        assert_eq!(
            Interpolation::parse("LINEAR").unwrap(),
            Interpolation::Linear
        );
        assert_eq!(Interpolation::parse("lower").unwrap(), Interpolation::Lower);
        assert_eq!(
            Interpolation::parse("higher").unwrap(),
            Interpolation::Higher
        );
        assert_eq!(
            Interpolation::parse("nearest").unwrap(),
            Interpolation::Nearest
        );
        assert_eq!(
            Interpolation::parse("midpoint").unwrap(),
            Interpolation::Midpoint
        );
        assert!(Interpolation::parse("invalid").is_err());
    }

    #[test]
    fn test_interpolation_values() {
        let lower = 1.0;
        let upper = 2.0;
        let frac = 0.75;

        assert!((Interpolation::Linear.interpolate(lower, upper, frac) - 1.75).abs() < 1e-10);
        assert!((Interpolation::Lower.interpolate(lower, upper, frac) - 1.0).abs() < 1e-10);
        assert!((Interpolation::Higher.interpolate(lower, upper, frac) - 2.0).abs() < 1e-10);
        assert!((Interpolation::Nearest.interpolate(lower, upper, frac) - 2.0).abs() < 1e-10);
        assert!((Interpolation::Midpoint.interpolate(lower, upper, frac) - 1.5).abs() < 1e-10);
    }

    #[test]
    fn test_compute_bin_edges() {
        let edges = compute_bin_edges_f64(0.0, 10.0, 5);
        assert_eq!(edges.len(), 6);
        assert!((edges[0] - 0.0).abs() < 1e-10);
        assert!((edges[2] - 4.0).abs() < 1e-10);
        assert!((edges[5] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_bin_index() {
        let min = 0.0;
        let width = 2.0;
        let bins = 5;

        assert_eq!(compute_bin_index(0.5, min, width, bins), 0);
        assert_eq!(compute_bin_index(2.5, min, width, bins), 1);
        assert_eq!(compute_bin_index(-1.0, min, width, bins), 0); // Clamped
        assert_eq!(compute_bin_index(100.0, min, width, bins), 4); // Clamped
    }

    #[test]
    fn test_compute_quantile_indices() {
        let (f, c, frac) = compute_quantile_indices(0.5, 5);
        assert_eq!(f, 2);
        assert_eq!(c, 2);
        assert!(frac.abs() < 1e-10);

        let (f, c, frac) = compute_quantile_indices(0.25, 5);
        assert_eq!(f, 1);
        assert_eq!(c, 1);
        assert!(frac.abs() < 1e-10);
    }

    #[test]
    fn test_skewness_symmetric() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let skew = compute_skewness(&data, 0);
        assert!(
            skew.abs() < 0.1,
            "Symmetric data should have near-zero skewness"
        );
    }

    #[test]
    fn test_kurtosis_uniform() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kurt = compute_kurtosis(&data, 0);
        // Uniform-like distribution should have negative excess kurtosis
        assert!(
            kurt < 0.0,
            "Uniform-like data should have negative kurtosis"
        );
    }

    #[test]
    fn test_compute_mode_simple() {
        // Simple case: 2 appears 3 times
        let data: Vec<f32> = vec![1.0, 2.0, 2.0, 2.0, 3.0];
        let result = compute_mode(&data);
        assert!((result.value - 2.0).abs() < 1e-10);
        assert_eq!(result.count, 3);
    }

    #[test]
    fn test_compute_mode_all_unique() {
        // All unique: return smallest (first) with count 1
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = compute_mode(&data);
        assert!((result.value - 1.0).abs() < 1e-10);
        assert_eq!(result.count, 1);
    }

    #[test]
    fn test_compute_mode_tie() {
        // Tie: 1 and 3 both appear twice, return smallest (1)
        let data: Vec<f32> = vec![1.0, 1.0, 2.0, 3.0, 3.0];
        let result = compute_mode(&data);
        assert!((result.value - 1.0).abs() < 1e-10);
        assert_eq!(result.count, 2);
    }

    #[test]
    fn test_compute_mode_single_element() {
        let data: Vec<f32> = vec![42.0];
        let result = compute_mode(&data);
        assert!((result.value - 42.0).abs() < 1e-10);
        assert_eq!(result.count, 1);
    }

    #[test]
    fn test_compute_mode_all_same() {
        let data: Vec<f32> = vec![7.0, 7.0, 7.0, 7.0];
        let result = compute_mode(&data);
        assert!((result.value - 7.0).abs() < 1e-10);
        assert_eq!(result.count, 4);
    }
}
