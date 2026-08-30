//! Shared logic for quasi-random sequence operations.
//!
//! This module provides common parameter validation and Sobol direction vector
//! computation to ensure consistency across CPU, CUDA, and WebGPU backends
//! and eliminate code duplication.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cpu::kernels::sobol_data::ArchivedSobolPolynomial;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::runtime::cpu::kernels::sobol_data::get_polynomial;

/// Maximum number of dimensions supported for Sobol sequences with full direction numbers.
///
/// Uses Joe & Kuo (2008) direction numbers for high-quality quasi-random sequences.
/// All backends support the full 21,201 dimensions from the Joe & Kuo dataset.
pub const SOBOL_MAX_DIMENSIONS: usize = 21201;

/// Maximum number of dimensions supported for Halton sequences.
pub const HALTON_MAX_DIMENSIONS: usize = 100;

/// Number of bits for direction vectors (32-bit precision).
pub const SOBOL_BITS: usize = 32;

// ============================================================================
// Sobol Direction Vector Computation
// ============================================================================

/// Compute direction vectors from polynomial data using the recurrence relation.
///
/// The recurrence (from Joe & Kuo 2008):
/// For i >= s:
///   m_i = m_{i-s} XOR (m_{i-s} >> s) XOR sum_{j=1}^{s-1} [a_j * (m_{i-j} >> j)]
///
/// where a_j is the j-th bit of coefficient 'a' (from LSB).
#[inline]
pub fn compute_direction_vectors(poly: &ArchivedSobolPolynomial) -> [u32; SOBOL_BITS] {
    let s = poly.degree as usize;
    // Convert from archived little-endian to native u32
    let a: u32 = poly.coeff.into();

    let mut v = [0u32; SOBOL_BITS];

    // Initialize with provided m values, left-shifted to fill MSBs
    for (i, m) in poly.m_values.iter().enumerate() {
        let m_native: u32 = (*m).into();
        v[i] = m_native << (SOBOL_BITS - 1 - i);
    }

    // Recurrence for remaining positions
    for i in s..SOBOL_BITS {
        // Start with the required terms: m_{i-s} XOR (m_{i-s} >> s)
        let mut vi = v[i - s] ^ (v[i - s] >> s);

        // Add middle terms based on polynomial coefficients.
        // a encodes coefficients a_{s-1}, a_{s-2}, ..., a_1 from MSB to LSB.
        //
        // The middle terms are NOT shifted. Writing the recurrence over the
        // integers m_i it is `2^k a_k m_{i-k}`, but over the direction numbers
        // v_i = m_i / 2^i the powers of two cancel, leaving `a_k v_{i-k}`. Only
        // the `v[i - s] >> s` term above survives the change of variable. A
        // `>> j` here makes the leading direction-number matrix singular, which
        // collapses the dimension: every point is then visited twice over the
        // first 2^k draws instead of once.
        for j in 1..s {
            if (a >> (s - 1 - j)) & 1 != 0 {
                vi ^= v[i - j];
            }
        }

        v[i] = vi;
    }

    v
}

/// Direction vectors for dimension 0 (implicit: all powers of 2).
#[inline]
pub fn dimension_zero_vectors() -> [u32; SOBOL_BITS] {
    let mut v = [0u32; SOBOL_BITS];
    for i in 0..SOBOL_BITS {
        v[i] = 1u32 << (SOBOL_BITS - 1 - i);
    }
    v
}

/// Compute all direction vectors for the given number of dimensions.
///
/// Returns a flattened vector of length `dimension * SOBOL_BITS` containing
/// the direction vectors for each dimension concatenated.
#[inline]
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub fn compute_all_direction_vectors(dimension: usize) -> Vec<u32> {
    let mut all_vectors = Vec::with_capacity(dimension * SOBOL_BITS);

    for d in 0..dimension {
        let v = if d == 0 {
            dimension_zero_vectors()
        } else {
            // Dimension was validated by caller, so polynomial must exist.
            // If missing, the embedded sobol_data.bin is corrupted.
            let poly = get_polynomial(d + 1)
                .expect("INTERNAL: sobol_data.bin corrupted - missing polynomial");
            compute_direction_vectors(poly)
        };
        all_vectors.extend_from_slice(&v);
    }

    all_vectors
}

/// Validates common parameters for quasi-random sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate (must be > 0)
/// * `dimension` - Dimensionality of the sequence (must be > 0 and <= max_dim)
/// * `max_dim` - Maximum dimension supported for this algorithm
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns `InvalidArgument` if:
/// - `n_points` is 0
/// - `dimension` is 0
/// - `dimension` exceeds `max_dim`
#[inline]
pub fn validate_basic_params(
    n_points: usize,
    dimension: usize,
    max_dim: usize,
    op: &'static str,
) -> Result<()> {
    if n_points == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_points",
            reason: format!("{} requires at least 1 point", op),
        });
    }

    if dimension == 0 {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!("{} requires at least 1 dimension", op),
        });
    }

    if dimension > max_dim {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!(
                "{} supports up to {} dimensions, got {}",
                op, max_dim, dimension
            ),
        });
    }

    Ok(())
}

/// Validates data type support for quasi-random operations.
///
/// # Arguments
///
/// * `dtype` - Data type to validate
/// * `supported_dtypes` - Slice of supported DType values
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns `UnsupportedDType` if the dtype is not in the supported list.
#[inline]
pub fn validate_dtype(dtype: DType, supported_dtypes: &[DType], op: &'static str) -> Result<()> {
    if !supported_dtypes.contains(&dtype) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Validates parameters for Sobol sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `dimension` - Dimensionality of the sequence
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if basic validation or dtype validation fails.
#[inline]
pub fn validate_sobol_params(
    n_points: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    validate_basic_params(n_points, dimension, SOBOL_MAX_DIMENSIONS, op)?;
    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

/// Validates parameters for Halton sequence generation.
///
/// # Arguments
///
/// * `n_points` - Number of points to generate
/// * `dimension` - Dimensionality of the sequence
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if basic validation or dtype validation fails.
#[inline]
pub fn validate_halton_params(
    n_points: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    validate_basic_params(n_points, dimension, HALTON_MAX_DIMENSIONS, op)?;
    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

/// Validates parameters for Latin Hypercube Sampling.
///
/// # Arguments
///
/// * `n_samples` - Number of samples to generate
/// * `dimension` - Dimensionality of the samples
/// * `dtype` - Data type
/// * `supported_dtypes` - Slice of supported DType values for this backend
/// * `op` - Operation name for error messages
///
/// # Errors
///
/// Returns error if validation fails. Note: Latin Hypercube has no
/// dimension limit (other than memory constraints).
#[inline]
pub fn validate_latin_hypercube_params(
    n_samples: usize,
    dimension: usize,
    dtype: DType,
    supported_dtypes: &[DType],
    op: &'static str,
) -> Result<()> {
    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: format!("{} requires at least 1 sample", op),
        });
    }

    if dimension == 0 {
        return Err(Error::InvalidArgument {
            arg: "dimension",
            reason: format!("{} requires at least 1 dimension", op),
        });
    }

    validate_dtype(dtype, supported_dtypes, op)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_basic_params() {
        // Valid parameters
        assert!(validate_basic_params(10, 5, 100, "test").is_ok());

        // Zero points
        assert!(validate_basic_params(0, 5, 100, "test").is_err());

        // Zero dimension
        assert!(validate_basic_params(10, 0, 100, "test").is_err());

        // Dimension exceeds limit
        assert!(validate_basic_params(10, 101, 100, "test").is_err());
    }

    #[test]
    fn test_validate_dtype() {
        let supported = &[DType::F32, DType::F64];

        // Valid dtype
        assert!(validate_dtype(DType::F32, supported, "test").is_ok());
        assert!(validate_dtype(DType::F64, supported, "test").is_ok());

        // Unsupported dtype
        assert!(validate_dtype(DType::I32, supported, "test").is_err());
    }

    #[test]
    fn test_validate_sobol_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(validate_sobol_params(10, 5, DType::F32, supported, "sobol").is_ok());
        assert!(validate_sobol_params(10, 1000, DType::F32, supported, "sobol").is_ok());
        assert!(validate_sobol_params(10, 21201, DType::F32, supported, "sobol").is_ok());

        // Invalid dimension (beyond 21,201)
        assert!(validate_sobol_params(10, 21202, DType::F32, supported, "sobol").is_err());

        // Unsupported dtype
        assert!(validate_sobol_params(10, 5, DType::I32, supported, "sobol").is_err());
    }

    #[test]
    fn test_validate_halton_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(validate_halton_params(10, 5, DType::F32, supported, "halton").is_ok());

        // Invalid dimension (Halton max is 100)
        assert!(validate_halton_params(10, 101, DType::F32, supported, "halton").is_err());
    }

    #[test]
    fn test_validate_latin_hypercube_params() {
        let supported = &[DType::F32, DType::F64];

        // Valid parameters
        assert!(
            validate_latin_hypercube_params(10, 5, DType::F32, supported, "latin_hypercube")
                .is_ok()
        );

        // Zero samples
        assert!(
            validate_latin_hypercube_params(0, 5, DType::F32, supported, "latin_hypercube")
                .is_err()
        );

        // Zero dimension
        assert!(
            validate_latin_hypercube_params(10, 0, DType::F32, supported, "latin_hypercube")
                .is_err()
        );
    }

    #[test]
    fn test_dimension_zero_vectors() {
        let v = dimension_zero_vectors();

        // Dimension 0 should be powers of 2: 2^31, 2^30, ..., 2^0
        assert_eq!(v.len(), SOBOL_BITS);
        for i in 0..SOBOL_BITS {
            assert_eq!(v[i], 1u32 << (SOBOL_BITS - 1 - i));
        }
    }

    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    fn test_compute_all_direction_vectors_length() {
        // Test that we get the right number of vectors
        let dim = 5;
        let vectors = compute_all_direction_vectors(dim);
        assert_eq!(vectors.len(), dim * SOBOL_BITS);
    }

    #[test]
    #[cfg(any(feature = "cuda", feature = "wgpu"))]
    fn test_compute_all_direction_vectors_dimension_0() {
        // First dimension should match dimension_zero_vectors()
        let vectors = compute_all_direction_vectors(3);
        let d0 = dimension_zero_vectors();

        for i in 0..SOBOL_BITS {
            assert_eq!(vectors[i], d0[i]);
        }
    }
}
