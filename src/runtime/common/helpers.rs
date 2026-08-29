//! Shared helper functions for runtime backends
//!
//! This module contains helper functions that are used across multiple backends
//! (CPU, CUDA, WebGPU) to reduce code duplication and maintain consistency.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::broadcast_shape;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// Dimension Normalization
// ============================================================================

/// Normalize a dimension index, handling negative indexing.
///
/// Converts a potentially negative dimension index to a positive index,
/// following NumPy/PyTorch conventions where -1 refers to the last dimension,
/// -2 to the second-to-last, etc.
///
/// # Arguments
///
/// * `dim` - The dimension index (can be negative)
/// * `ndim` - The number of dimensions in the tensor
///
/// # Returns
///
/// The normalized (positive) dimension index.
///
/// # Errors
///
/// Returns `Error::InvalidDimension` if the dimension is out of bounds.
#[inline]
pub fn normalize_dim(dim: isize, ndim: usize) -> Result<usize> {
    let dim_idx = if dim < 0 {
        let adjusted = ndim as isize + dim;
        if adjusted < 0 {
            return Err(Error::InvalidDimension { dim, ndim });
        }
        adjusted as usize
    } else {
        dim as usize
    };

    if dim_idx >= ndim {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    Ok(dim_idx)
}

// ============================================================================
// Utility Operation Validation
// ============================================================================

/// Validate arange parameters and compute the number of elements.
///
/// # Arguments
///
/// * `start` - Start value (inclusive)
/// * `stop` - Stop value (exclusive)
/// * `step` - Step between consecutive values
///
/// # Returns
///
/// The number of elements in the resulting tensor, or an error if parameters are invalid.
///
/// # Errors
///
/// * `InvalidArgument` if step is zero
/// * `InvalidArgument` if step direction doesn't match start→stop direction
#[inline]
pub fn validate_arange(start: f64, stop: f64, step: f64) -> Result<usize> {
    // Step cannot be zero
    if step == 0.0 {
        return Err(Error::InvalidArgument {
            arg: "step",
            reason: "step cannot be zero".to_string(),
        });
    }

    // Step direction must match start→stop direction
    if (stop > start && step < 0.0) || (stop < start && step > 0.0) {
        return Err(Error::InvalidArgument {
            arg: "step",
            reason: "step sign must match direction from start to stop".to_string(),
        });
    }

    // Calculate number of elements
    let numel = if start == stop {
        0
    } else {
        ((stop - start) / step).ceil() as usize
    };

    Ok(numel)
}

/// Validate linspace parameters.
///
/// # Arguments
///
/// * `steps` - Number of points to generate
///
/// # Returns
///
/// Validate eye (identity matrix) parameters.
///
/// # Arguments
///
/// * `n` - Number of rows
/// * `m` - Number of columns (if None, defaults to n)
///
/// # Returns
///
/// The (rows, cols) dimensions of the identity matrix.
#[inline]
pub fn validate_eye(n: usize, m: Option<usize>) -> (usize, usize) {
    let cols = m.unwrap_or(n);
    (n, cols)
}

/// Ensure a tensor is contiguous in memory.
///
/// If the tensor is already contiguous (elements laid out consecutively in memory),
/// returns a clone (zero-copy, just increments Arc refcount). Otherwise,
/// creates a new contiguous copy of the data by materializing the strided view.
///
/// This is typically required before passing tensors to backend kernels
/// that expect contiguous memory layout, or for operations that need
/// to work with the underlying data pointer directly.
///
/// # Parameters
///
/// * `tensor` - The input tensor which may or may not be contiguous
///
/// # Returns
///
/// A new tensor that is guaranteed to be contiguous. If the input was already
/// contiguous, this is zero-copy (just clones the Arc). Otherwise, data is copied.
#[inline]
pub fn ensure_contiguous<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
) -> crate::error::Result<Tensor<R>> {
    if tensor.is_contiguous() {
        Ok(tensor.clone())
    } else {
        tensor.contiguous()
    }
}

// ============================================================================
// Binary Operation Validation
// ============================================================================

/// Validate that two tensors have matching dtypes for binary operations.
///
/// This is a shared helper used by CPU, CUDA, and WebGPU backends to ensure
/// consistent dtype validation across all runtimes.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// The common dtype if both tensors have the same dtype, or an error if they differ.
///
/// # Errors
///
/// Returns `Error::DTypeMismatch` if the tensors have different dtypes.
#[inline]
pub fn validate_binary_dtypes<R: Runtime<DType = DType>>(
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<DType> {
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    Ok(a.dtype())
}

/// Output dtype of `pow_scalar` for a given input dtype and exponent.
///
/// An op's output dtype is a function of its input dtypes, never its input
/// values. `pow_scalar`'s exponent is a host-side parameter, not tensor data, so
/// it is known before launch and participates in that function.
///
/// An integer raised to a power is an integer only when the exponent is a
/// non-negative whole number. Every other exponent — negative, fractional,
/// infinite, or NaN — yields a non-integer real, so the result is F64. This is
/// why numpy promotes `int ** 0.5` while refusing an integer array exponent:
/// there the exponent is data, and dtype inference cannot read data.
///
/// Float, complex, and bool dtypes pass through unchanged.
#[inline]
pub fn pow_scalar_output_dtype(dtype: DType, scalar: f64) -> DType {
    // `fract()` is NaN for an infinite exponent and for NaN itself, so both fail
    // this test and promote, which is what their real-valued results require.
    let integral_result = scalar >= 0.0 && scalar.fract() == 0.0;
    if dtype.is_int() && !integral_result {
        DType::F64
    } else {
        dtype
    }
}

/// Cap an integer power's exponent above 1024 to preserve parity.
///
/// Above 1024, the result depends only on the base's magnitude (0, 1, or
/// saturated to the dtype bound) and the exponent's parity, since any base
/// outside `{-1, 0, 1}` has already saturated well before an exponent that
/// large. Capping to 1024 (even) or 1025 (odd) keeps the parity a negative
/// base needs while leaving the value small enough to cast losslessly into a
/// kernel's integer type. Caller must already know the exponent is a
/// non-negative whole number reaching an integer dtype — this does not
/// re-check that.
///
/// CUDA device code (`numr_ipow_scalar` in `runtime/cuda/kernels/ipow.cuh`)
/// cannot call this function, so it duplicates the cap by hand; that
/// duplicate must keep agreeing with this one.
#[inline]
pub fn cap_ipow_exponent(scalar: f64) -> f64 {
    if scalar > 1024.0 {
        1024.0 + f64::from(scalar % 2.0 != 0.0)
    } else {
        scalar
    }
}

/// Compute broadcast shape for binary operations.
///
/// Returns the output shape after broadcasting, or an error if shapes are incompatible.
/// Follows NumPy broadcasting rules.
///
/// This is a shared helper used by CPU, CUDA, and WebGPU backends to ensure
/// consistent broadcast shape computation across all runtimes.
///
/// # Arguments
///
/// * `a` - First tensor
/// * `b` - Second tensor
///
/// # Returns
///
/// The broadcast output shape, or an error if shapes are incompatible.
///
/// # Errors
///
/// Returns `Error::BroadcastError` if shapes cannot be broadcast together.
#[inline]
pub fn compute_broadcast_shape<R: Runtime<DType = DType>>(
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> Result<Vec<usize>> {
    broadcast_shape(a.shape(), b.shape()).ok_or_else(|| Error::BroadcastError {
        lhs: a.shape().to_vec(),
        rhs: b.shape().to_vec(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pow_scalar_output_dtype_promotes_only_non_integral_exponents() {
        // A non-negative whole exponent keeps the integer dtype.
        for scalar in [0.0, 1.0, 2.0, 63.0, 1e30] {
            assert_eq!(pow_scalar_output_dtype(DType::I64, scalar), DType::I64);
            assert_eq!(pow_scalar_output_dtype(DType::U32, scalar), DType::U32);
        }

        // Every other exponent yields a non-integer real.
        for scalar in [0.5, -1.0, -0.5, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            assert_eq!(pow_scalar_output_dtype(DType::I64, scalar), DType::F64);
            assert_eq!(pow_scalar_output_dtype(DType::I32, scalar), DType::F64);
        }

        // Float dtypes pass through for every exponent.
        for scalar in [0.5, -1.0, 2.0, f64::NAN] {
            assert_eq!(pow_scalar_output_dtype(DType::F32, scalar), DType::F32);
            assert_eq!(pow_scalar_output_dtype(DType::F64, scalar), DType::F64);
        }
    }

    #[test]
    fn test_normalize_dim_positive() {
        assert_eq!(normalize_dim(0, 3).unwrap(), 0);
        assert_eq!(normalize_dim(2, 3).unwrap(), 2);
    }

    #[test]
    fn test_normalize_dim_negative() {
        assert_eq!(normalize_dim(-1, 3).unwrap(), 2);
        assert_eq!(normalize_dim(-3, 3).unwrap(), 0);
    }

    #[test]
    fn test_normalize_dim_out_of_bounds() {
        assert!(normalize_dim(3, 3).is_err());
        assert!(normalize_dim(-4, 3).is_err());
    }

    #[test]
    fn test_ensure_contiguous() {
        use crate::runtime::cpu::{CpuDevice, CpuRuntime};
        use crate::tensor::Tensor;

        let device = CpuDevice::new();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        assert!(a.is_contiguous());
        let c = ensure_contiguous(&a).unwrap();
        assert!(c.is_contiguous());
    }
}
