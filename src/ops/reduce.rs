//! Reduction operations helpers
//!
//! This module contains helper types and functions for reduction operations.
//! The actual operations are defined in the `TensorOps` trait.

use crate::dtype::DType;

/// Reduction operation kind
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum of elements
    Sum,
    /// Mean of elements
    Mean,
    /// Maximum element
    Max,
    /// Minimum element
    Min,
    /// Product of elements
    Prod,
    /// Logical AND (for bool tensors)
    All,
    /// Logical OR (for bool tensors)
    Any,
}

/// Accumulation precision for reduction operations.
///
/// Controls the intermediate precision used during reduction for reduced-precision types:
/// - F16/BF16: Can use Native, FP32, or FP64 (default: Native)
/// - FP8: Can use BF16, FP32, or FP64 (default: FP32) - no native FP8 arithmetic
/// - F32: Can use Native or FP64 (default: Native)
/// - F64/integers: Always use native precision
///
/// # Memory vs Precision Trade-off
///
/// | Precision | Memory per element | Use case |
/// |-----------|-------------------|----------|
/// | Native | dtype size | Default, least memory |
/// | BF16 | 2 bytes | FP8 with moderate precision |
/// | FP32 | 4 bytes | Good numerical stability |
/// | FP64 | 8 bytes | Maximum precision (math/science) |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum AccumulationPrecision {
    /// Use native dtype for accumulation.
    /// Least memory usage, may have reduced precision for large reductions.
    /// For FP8, this is equivalent to FP32 (no native FP8 arithmetic).
    #[default]
    Native,
    /// Use BF16 for accumulation (for FP8 types).
    /// Uses less shared memory than FP32 (2 bytes vs 4 bytes per element).
    /// For F16/BF16, this is equivalent to Native or FP32 respectively.
    BF16,
    /// Use FP32 for accumulation.
    /// Good numerical stability for large reductions.
    /// Uses 4 bytes per element.
    FP32,
    /// Use FP64 for accumulation.
    /// Maximum precision for math/science applications.
    /// Uses 8 bytes per element.
    FP64,
}

impl AccumulationPrecision {
    /// Resolve `Native` into the precision a reduction actually accumulates in
    /// for `dtype`.
    ///
    /// `Native` means "let the library choose". For a float narrower than F32
    /// the library always chooses FP32, because accumulating in F16, BF16, or
    /// FP8 saturates: once the accumulator's spacing exceeds twice the
    /// increment every further addition rounds away and the sum stalls on a
    /// constant. Summing 512 BF16 values of `11.76` stalls at exactly `4096`,
    /// so the mean comes back as exactly `8.0` whatever the inputs were.
    ///
    /// F32, F64, integers, and every explicitly requested precision are
    /// returned unchanged, so this never alters an F32 or F64 reduction.
    #[inline]
    pub fn resolve(self, dtype: DType) -> Self {
        if matches!(self, Self::Native) && dtype.is_narrow_float() {
            Self::FP32
        } else {
            self
        }
    }
}

/// Compute output shape for reduction
///
/// # Arguments
/// * `input_shape` - Shape of input tensor
/// * `dims` - Dimensions to reduce over
/// * `keepdim` - If true, keep reduced dimensions as size 1
pub fn reduce_output_shape(input_shape: &[usize], dims: &[usize], keepdim: bool) -> Vec<usize> {
    if keepdim {
        // Keep all dimensions, set reduced ones to 1
        input_shape
            .iter()
            .enumerate()
            .map(|(i, &s)| if dims.contains(&i) { 1 } else { s })
            .collect()
    } else {
        // Remove reduced dimensions
        input_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !dims.contains(i))
            .map(|(_, &s)| s)
            .collect()
    }
}

/// Compute the strides for a single-dimension reduction (used by argmax/argmin).
///
/// Returns `(outer_size, reduce_size, inner_size)` where:
/// - `outer_size`: product of dimensions before the reduced dimension
/// - `reduce_size`: size of the dimension being reduced
/// - `inner_size`: product of dimensions after the reduced dimension
///
/// This is the standard decomposition for implementing reduce operations that
/// iterate over outer × inner combinations, each reducing over reduce_size elements.
///
/// # Arguments
/// * `shape` - Shape of the input tensor
/// * `dim` - The dimension to reduce over
///
#[inline]
pub fn compute_reduce_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer_size: usize = shape[..dim].iter().product::<usize>().max(1);
    let reduce_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product::<usize>().max(1);
    (outer_size, reduce_size, inner_size)
}

/// Compute output shape for a single-dimension reduction (used by argmax/argmin).
///
/// This is a convenience wrapper around [`reduce_output_shape`] for the common
/// case of reducing over exactly one dimension.
///
/// # Arguments
/// * `shape` - Shape of the input tensor
/// * `dim` - The dimension to reduce over
/// * `keepdim` - If true, keep the reduced dimension as size 1
#[inline]
pub fn reduce_dim_output_shape(shape: &[usize], dim: usize, keepdim: bool) -> Vec<usize> {
    reduce_output_shape(shape, &[dim], keepdim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_output_shape() {
        // Reduce single dim without keepdim
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[1], false), vec![2, 4]);

        // Reduce single dim with keepdim
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[1], true), vec![2, 1, 4]);

        // Reduce multiple dims
        assert_eq!(reduce_output_shape(&[2, 3, 4], &[0, 2], false), vec![3]);
        assert_eq!(
            reduce_output_shape(&[2, 3, 4], &[0, 2], true),
            vec![1, 3, 1]
        );

        // Reduce all dims
        assert_eq!(
            reduce_output_shape(&[2, 3, 4], &[0, 1, 2], false),
            Vec::<usize>::new()
        );
        assert_eq!(
            reduce_output_shape(&[2, 3, 4], &[0, 1, 2], true),
            vec![1, 1, 1]
        );
    }

    #[test]
    fn test_compute_reduce_strides() {
        let (outer, reduce, inner) = compute_reduce_strides(&[2, 3, 4], 1);
        assert_eq!((outer, reduce, inner), (2, 3, 4));

        let (outer, reduce, inner) = compute_reduce_strides(&[2, 3, 4], 0);
        assert_eq!((outer, reduce, inner), (1, 2, 12));

        let (outer, reduce, inner) = compute_reduce_strides(&[2, 3, 4], 2);
        assert_eq!((outer, reduce, inner), (6, 4, 1));
    }
}
