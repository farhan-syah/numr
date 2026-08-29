//! The dtype rules shared by `matmul`, `matmul_bias`, and the GEMM epilogue.
//!
//! These live in one module because I8 is an exception every backend has to
//! apply identically: an I8 matmul is a quantized accumulation, so its result
//! is I32, not I8. Encoding that once here is what keeps CPU and CUDA from
//! drifting apart on it.

use crate::dtype::DType;
use crate::error::{Error, Result};

/// The dtype a matmul writes for this element dtype.
///
/// I8 is the one width that widens: `A @ B` on I8 operands is a quantized
/// accumulation whose products already leave I8's range, so both `matmul` and
/// `matmul_bias` allocate an I32 output (CPU runs `matmul_i8_to_i32_kernel`,
/// CUDA runs `matmul_i8_i32_tiled_64x64x8_4x4`). Every other width writes its
/// own dtype.
#[inline]
pub fn matmul_output_dtype(elem_dtype: DType) -> DType {
    if elem_dtype == DType::I8 {
        DType::I32
    } else {
        elem_dtype
    }
}

/// Validate the A, B, and bias dtypes for `matmul_bias`.
///
/// This is the **canonical** dtype validation for matmul_bias - use this
/// function in all backend implementations to ensure consistent error handling.
///
/// A and B must share an element dtype. The bias must match the *output* dtype
/// ([`matmul_output_dtype`]), which differs from the element dtype for I8 only:
/// the bias seeds the wide accumulator, so an I8 bias would cap at 127 the one
/// value the widened output exists to carry.
///
/// # Returns
/// - `Ok(elem_dtype)` - the shared A/B element dtype, which is what the kernels
///   dispatch on. The output dtype is [`matmul_output_dtype`] of it.
/// - `Err(DTypeMismatch)` if A and B differ, or if the bias differs from the
///   output dtype at a width that does not widen.
/// - `Err(InvalidArgument)` if the bias differs at a widening width, where the
///   expected dtype is not the element dtype and the caller needs it named.
pub fn validate_matmul_bias_dtypes(
    a_dtype: DType,
    b_dtype: DType,
    bias_dtype: DType,
) -> Result<DType> {
    if a_dtype != b_dtype {
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: b_dtype,
        });
    }

    let out_dtype = matmul_output_dtype(a_dtype);
    if bias_dtype != out_dtype {
        // At a widening width the expected bias dtype is not the operand dtype,
        // so `DTypeMismatch` (a bare pair) would not tell the caller which of
        // the two it got wrong.
        if out_dtype != a_dtype {
            return Err(Error::InvalidArgument {
                arg: "bias",
                reason: format!(
                    "matmul_bias on {a_dtype:?} operands accumulates into {out_dtype:?}, \
                     so the bias must be {out_dtype:?}, got {bias_dtype:?}"
                ),
            });
        }
        return Err(Error::DTypeMismatch {
            lhs: a_dtype,
            rhs: bias_dtype,
        });
    }

    Ok(a_dtype)
}

/// Validate the A, B, and bias dtypes for a GEMM epilogue op.
///
/// Same rule as [`validate_matmul_bias_dtypes`] except at the widening widths:
/// the epilogue kernels (`matmul_bias_activation`, `matmul_bias_residual`, and
/// the backward form) have no widening variant, so they would read the wider
/// bias buffer as the element type. Rejecting I8 here is what stops that from
/// being a silent misread.
///
/// # Returns
/// - `Ok(dtype)` - the dtype shared by all three operands and the output.
/// - `Err(UnsupportedDType)` if the element dtype widens, naming `op`.
pub fn validate_gemm_epilogue_dtypes(
    a_dtype: DType,
    b_dtype: DType,
    bias_dtype: DType,
    op: &'static str,
) -> Result<DType> {
    if matmul_output_dtype(a_dtype) != a_dtype {
        return Err(Error::UnsupportedDType { dtype: a_dtype, op });
    }
    validate_matmul_bias_dtypes(a_dtype, b_dtype, bias_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_output_dtype_widens_only_i8() {
        assert_eq!(matmul_output_dtype(DType::I8), DType::I32);
        assert_eq!(matmul_output_dtype(DType::I16), DType::I16);
        assert_eq!(matmul_output_dtype(DType::I32), DType::I32);
        assert_eq!(matmul_output_dtype(DType::U8), DType::U8);
        assert_eq!(matmul_output_dtype(DType::F32), DType::F32);
    }

    #[test]
    fn test_validate_matmul_bias_dtypes() {
        // All same dtype - should succeed
        assert!(validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).is_ok());
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::F32).unwrap(),
            DType::F32
        );
        assert_eq!(
            validate_matmul_bias_dtypes(DType::F64, DType::F64, DType::F64).unwrap(),
            DType::F64
        );

        // A and B mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F64, DType::F32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::F64);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }

        // A and bias mismatch
        let result = validate_matmul_bias_dtypes(DType::F32, DType::F32, DType::I32);
        assert!(result.is_err());
        match result {
            Err(Error::DTypeMismatch { lhs, rhs }) => {
                assert_eq!(lhs, DType::F32);
                assert_eq!(rhs, DType::I32);
            }
            _ => panic!("Expected DTypeMismatch error"),
        }
    }

    /// I8 operands take an I32 bias, and the returned dtype stays the element
    /// dtype the kernels dispatch on.
    #[test]
    fn test_validate_matmul_bias_dtypes_i8_takes_i32_bias() {
        assert_eq!(
            validate_matmul_bias_dtypes(DType::I8, DType::I8, DType::I32).unwrap(),
            DType::I8
        );
    }

    /// An I8 bias names the expected dtype rather than reporting a bare pair.
    #[test]
    fn test_validate_matmul_bias_dtypes_i8_bias_rejected() {
        let result = validate_matmul_bias_dtypes(DType::I8, DType::I8, DType::I8);
        match result {
            Err(Error::InvalidArgument { arg, reason }) => {
                assert_eq!(arg, "bias");
                assert!(reason.contains("matmul_bias"), "reason: {reason}");
                assert!(reason.contains("I32"), "reason: {reason}");
                assert!(reason.contains("I8"), "reason: {reason}");
            }
            other => panic!("Expected InvalidArgument, got {other:?}"),
        }
    }

    /// The epilogue ops have no widening kernel, so I8 is refused before a
    /// wider bias can be read as the element type.
    #[test]
    fn test_validate_gemm_epilogue_dtypes_rejects_i8() {
        let result =
            validate_gemm_epilogue_dtypes(DType::I8, DType::I8, DType::I32, "matmul_bias_residual");
        match result {
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(dtype, DType::I8);
                assert_eq!(op, "matmul_bias_residual");
            }
            other => panic!("Expected UnsupportedDType, got {other:?}"),
        }

        let ok = validate_gemm_epilogue_dtypes(
            DType::I16,
            DType::I16,
            DType::I16,
            "matmul_bias_residual",
        );
        assert_eq!(ok.unwrap(), DType::I16);
    }
}
