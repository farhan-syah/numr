//! Reduction operations helpers
//!
//! This module contains helper types and functions for reduction operations.
//! The actual operations are defined in the `TensorOps` trait.

use crate::dtype::DType;
use crate::error::{Error, Result};

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
/// Never floor `outer_size` or `inner_size` at 1: an empty slice already products
/// to 1, so a clamp fires only on a genuinely zero dimension, and then reports an
/// extent the allocation does not have — a CPU loop past the end of the buffer, or
/// a GPU grid over elements that do not exist. Callers guard on a zero-element
/// input or output before looping or launching instead.
#[inline]
pub fn compute_reduce_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer_size: usize = shape[..dim].iter().product();
    let reduce_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    (outer_size, reduce_size, inner_size)
}

/// True when this dtype's reductions carry infinity as their `max`/`min`
/// identity.
///
/// F64, F32, F16 and BF16 do. Integers and bool have no infinity. FP8 is left
/// out on purpose: its CUDA accumulator traits seed from `-/+FP8_*_MAX`, so an
/// infinite identity here would answer something the GPU never produces.
#[inline]
fn reduces_with_infinite_identity(dtype: DType) -> bool {
    matches!(dtype, DType::F64 | DType::F32 | DType::F16 | DType::BF16)
}

/// Value a `max` reduction must produce when it folds over zero elements.
///
/// A reduction over an empty set is the operation's identity. For a float wide
/// enough to hold it that is negative infinity — the true identity, and what the
/// CUDA kernels already seed with. Every other dtype takes the identity of the
/// same monoid inside its own range: the dtype's minimum.
#[inline]
pub fn max_identity(dtype: DType) -> f64 {
    if reduces_with_infinite_identity(dtype) {
        f64::NEG_INFINITY
    } else {
        dtype.min_value()
    }
}

/// Value a `min` reduction must produce when it folds over zero elements.
///
/// The mirror of [`max_identity`]: positive infinity for F64/F32/F16/BF16, the
/// dtype's own maximum for FP8, integers and bool.
#[inline]
pub fn min_identity(dtype: DType) -> f64 {
    if reduces_with_infinite_identity(dtype) {
        f64::INFINITY
    } else {
        dtype.max_value()
    }
}

/// Reject `argmax`/`argmin` over a zero-length dimension.
///
/// Every other reduction folds an empty set to an identity, but an index
/// reduction has no index to name: there is no element to point at, and reading
/// one anyway walks off an empty allocation. Every backend calls this before it
/// loops or launches.
#[inline]
pub fn ensure_arg_reduce_dim_nonempty(
    reduce_size: usize,
    dim: usize,
    op: &'static str,
) -> Result<()> {
    if reduce_size == 0 {
        return Err(Error::InvalidArgument {
            arg: "dim",
            reason: format!("{op}: dimension {dim} has length 0, so no index is a valid answer"),
        });
    }
    Ok(())
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
