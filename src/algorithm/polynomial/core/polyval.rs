//! Polynomial evaluation using Horner's method

use super::{DTypeSupport, create_index_tensor};
use crate::algorithm::polynomial::helpers::{
    validate_polynomial_coeffs, validate_polynomial_dtype,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, IndexingOps, ScalarOps, ShapeOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Evaluate polynomial at given points using Horner's method
///
/// # Algorithm
///
/// Horner's method:
/// ```text
/// result = cₙ
/// for i in (n-1)..0:
///     result = result * x + cᵢ
/// ```
///
/// This is numerically stable and requires only n multiplications and n additions.
///
/// # Implementation
///
/// Uses only tensor operations (no GPU↔CPU transfers):
/// - `index_select` to access each coefficient as a tensor
/// - `add` with broadcasting instead of `add_scalar`
pub fn polyval_impl<R, C>(
    client: &C,
    coeffs: &Tensor<R>,
    x: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + BinaryOps<R> + ScalarOps<R> + IndexingOps<R> + ShapeOps<R>,
{
    validate_polynomial_dtype(coeffs.dtype())?;
    validate_polynomial_dtype(x.dtype())?;
    dtype_support.check(coeffs.dtype(), "polyval")?;

    if coeffs.dtype() != x.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: coeffs.dtype(),
            rhs: x.dtype(),
        });
    }

    let n = validate_polynomial_coeffs(coeffs.shape())?;
    let device = client.device();
    let index_dtype = dtype_support.index_dtype;

    // Degree 0 polynomial: constant
    if n == 1 {
        // Get the constant coefficient as a tensor and broadcast to x shape
        let idx = create_index_tensor::<R>(0, index_dtype, device)?;
        let c0 = client.index_select(coeffs, 0, &idx)?; // Shape [1]
        // Broadcast c0 to x's shape and make contiguous
        let result = c0.broadcast_to(x.shape())?;
        return result.contiguous();
    }

    // Horner's method using tensor operations
    // Start with leading coefficient (coeffs[n-1])
    let last_idx = create_index_tensor::<R>(n - 1, index_dtype, device)?;
    let mut result = client.index_select(coeffs, 0, &last_idx)?; // Shape [1]

    // Broadcast to x's shape for the first multiplication
    // Make contiguous since broadcast_to creates a non-contiguous view
    result = result.broadcast_to(x.shape())?.contiguous()?;

    // Iterate from second-highest to lowest coefficient
    for i in (0..n - 1).rev() {
        // result = result * x + coeffs[i]
        result = client.mul(&result, x)?;

        // Get coefficient i as tensor
        let idx = create_index_tensor::<R>(i, index_dtype, device)?;
        let coeff_i = client.index_select(coeffs, 0, &idx)?; // Shape [1]

        // Add with broadcasting (coeff_i broadcasts to result's shape)
        result = client.add(&result, &coeff_i)?;
    }

    Ok(result)
}
