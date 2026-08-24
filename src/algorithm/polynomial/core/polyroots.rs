//! Polynomial root finding via companion matrix eigendecomposition

use super::{DTypeSupport, create_arange_tensor, create_index_tensor};
use crate::algorithm::linalg::LinearAlgebraAlgorithms;
use crate::algorithm::polynomial::helpers::validate_polynomial_coeffs;
use crate::algorithm::polynomial::helpers::validate_polynomial_dtype;
use crate::algorithm::polynomial::types::PolynomialRoots;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    BinaryOps, CompareOps, IndexingOps, LinalgOps, ReduceOps, ScalarOps, ShapeOps, UtilityOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Find polynomial roots via companion matrix eigendecomposition
///
/// # Algorithm
///
/// For polynomial p(x) = c₀ + c₁x + ... + cₙxⁿ:
/// 1. Normalize to monic form: divide all coefficients by cₙ
/// 2. Build companion matrix C of the monic polynomial
/// 3. Find eigenvalues of C (which are the roots)
///
/// The companion matrix for monic polynomial xⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₀:
/// ```text
/// C = [ 0   0   ...  0  -a₀  ]
///     [ 1   0   ...  0  -a₁  ]
///     [ 0   1   ...  0  -a₂  ]
///     [ .   .   ...  .   .   ]
///     [ 0   0   ...  1  -aₙ₋₁]
/// ```
///
/// # Implementation
///
/// Uses only tensor operations (no GPU↔CPU transfers):
/// - `eye` for subdiagonal structure
/// - `index_select` for coefficient access
/// - `scatter` for matrix construction
/// - `div` for normalization
///
/// # Degenerate Polynomials
///
/// If the leading coefficient is zero, division produces Inf and eigendecomposition
/// returns NaN/Inf roots. Users should check for finite roots when appropriate.
pub fn polyroots_impl<R, C>(
    client: &C,
    coeffs: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<PolynomialRoots<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + LinearAlgebraAlgorithms<R>
        + BinaryOps<R>
        + ScalarOps<R>
        + LinalgOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>
        + CompareOps<R>,
{
    validate_polynomial_dtype(coeffs.dtype())?;
    dtype_support.check(coeffs.dtype(), "polyroots")?;

    let n = validate_polynomial_coeffs(coeffs.shape())?;
    let dtype = coeffs.dtype();
    let device = client.device();
    let index_dtype = dtype_support.index_dtype;

    // Degree 0 polynomial (constant) - no roots
    if n == 1 {
        return Ok(PolynomialRoots {
            roots_real: Tensor::try_zeros(&[0], dtype, device)?,
            roots_imag: Tensor::try_zeros(&[0], dtype, device)?,
        });
    }

    // For degree n-1 polynomial (n coefficients), companion matrix is (n-1)×(n-1)
    let degree = n - 1;

    // Get leading coefficient as tensor
    // Select last coefficient: coeffs[n-1]
    let last_idx = create_index_tensor::<R>(n - 1, index_dtype, device)?;
    let leading_tensor = client.index_select(coeffs, 0, &last_idx)?; // Shape [1]

    // Note: We skip validation of leading coefficient being non-zero.
    // If it's zero, the division will produce Inf, and eigendecomposition
    // will return NaN/Inf roots, which is mathematically correct behavior
    // for a degenerate polynomial. Users should check for finite roots.

    // Build companion matrix using tensor operations
    // Start with zeros matrix
    let mut companion = Tensor::try_zeros(&[degree, degree], dtype, device)?;

    // Build subdiagonal: positions (i+1, i) for i in 0..degree-1 should be 1
    // This can be done by shifting an identity matrix
    if degree > 1 {
        // Create (degree-1) x (degree-1) identity and embed it
        let sub_eye = client.eye(degree - 1, None, dtype)?;
        // Pad to create subdiagonal pattern
        // eye is at positions (0..d-1, 0..d-1), we need it at (1..d, 0..d-1)
        let zeros_row = Tensor::try_zeros(&[1, degree - 1], dtype, device)?;
        let zeros_col = Tensor::try_zeros(&[degree, 1], dtype, device)?;
        // Stack: [zeros_row; sub_eye] gives (degree x degree-1) with subdiagonal
        let sub_with_top = client.cat(&[&zeros_row, &sub_eye], 0)?; // [degree, degree-1]
        // Add zeros column on right to make it [degree, degree]
        companion = client.cat(&[&sub_with_top, &zeros_col], 1)?;
    }

    // Build last column: -coeffs[0:degree] / leading
    // Select coefficients 0 to degree-1
    let coeff_indices = create_arange_tensor::<R>(0, degree, index_dtype, device)?;
    let lower_coeffs = client.index_select(coeffs, 0, &coeff_indices)?; // Shape [degree]

    // Negate and divide by leading coefficient
    let neg_coeffs = client.neg(&lower_coeffs)?;
    // Broadcast leading_tensor [1] to [degree] for division
    // Make contiguous since broadcast_to creates a non-contiguous view
    let leading_broadcast = leading_tensor.broadcast_to(&[degree])?.contiguous()?;
    let last_col = client.div(&neg_coeffs, &leading_broadcast)?; // Shape [degree]

    // Now we need to set the last column of companion to last_col
    // Reshape last_col to [degree, 1] for column assignment
    let last_col_2d = last_col.reshape(&[degree, 1])?;

    // Create column indices (all pointing to last column = degree-1)
    let col_indices = Tensor::try_full_scalar(&[degree], index_dtype, (degree - 1) as f64, device)?;

    // Use scatter to set last column
    let col_indices_2d = col_indices.reshape(&[degree, 1])?;
    companion = client.scatter(&companion, 1, &col_indices_2d, &last_col_2d)?;

    // Compute eigenvalues using general eigendecomposition
    let eig = client.eig_decompose(&companion)?;

    Ok(PolynomialRoots {
        roots_real: eig.eigenvalues_real,
        roots_imag: eig.eigenvalues_imag,
    })
}
