//! Polynomial construction from roots via iterative convolution

use super::{DTypeSupport, convolve_impl, create_index_tensor};
use crate::algorithm::fft::FftAlgorithms;
use crate::algorithm::polynomial::helpers::{validate_polynomial_dtype, validate_polynomial_roots};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ComplexOps, IndexingOps, ReduceOps, ShapeOps, UnaryOps, UtilityOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Construct polynomial from roots via iterative convolution
///
/// # Algorithm
///
/// Uses complex polynomial multiplication to handle both real and complex roots.
/// For polynomial p(x) and root r = a + bi, we compute p(x) * (x - r) using
/// complex arithmetic on separate real/imaginary tensor pairs.
///
/// Complex polynomial multiplication:
/// - (p_real + i*p_imag) * (q_real + i*q_imag)
/// - result_real = conv(p_real, q_real) - conv(p_imag, q_imag)
/// - result_imag = conv(p_real, q_imag) + conv(p_imag, q_real)
///
/// If roots come in proper conjugate pairs, the final imaginary part will be ~0.
///
/// # Implementation
///
/// Uses only tensor operations (no GPU↔CPU transfers):
/// - `index_select` to access individual roots
/// - `cat` to build linear factors
/// - Tensor-based convolution via outer product + scatter_reduce
pub fn polyfromroots_impl<R, C>(
    client: &C,
    roots_real: &Tensor<R>,
    roots_imag: &Tensor<R>,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + IndexingOps<R>
        + ShapeOps<R>
        + UtilityOps<R>
        + ReduceOps<R>
        + FftAlgorithms<R>
        + ComplexOps<R>,
{
    validate_polynomial_dtype(roots_real.dtype())?;
    validate_polynomial_dtype(roots_imag.dtype())?;
    dtype_support.check(roots_real.dtype(), "polyfromroots")?;

    if roots_real.dtype() != roots_imag.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: roots_real.dtype(),
            rhs: roots_imag.dtype(),
        });
    }

    let n_roots = validate_polynomial_roots(roots_real.shape())?;
    let n_imag = validate_polynomial_roots(roots_imag.shape())?;

    if n_roots != n_imag {
        return Err(Error::ShapeMismatch {
            expected: vec![n_roots],
            got: vec![n_imag],
        });
    }

    let dtype = roots_real.dtype();
    let device = client.device();
    let index_dtype = dtype_support.index_dtype;

    // Empty roots → constant polynomial [1.0]
    if n_roots == 0 {
        return Tensor::full_scalar(&[1], dtype, 1.0, device);
    }

    // Start with polynomial p(x) = 1 (represented as complex: real=[1], imag=[0])
    let mut p_real = Tensor::full_scalar(&[1], dtype, 1.0, device)?;
    let mut p_imag = Tensor::full_scalar(&[1], dtype, 0.0, device)?;

    // For each root r, multiply polynomial by (x - r)
    // Factor (x - r) where r = r_real + i*r_imag:
    //   factor_real = [-r_real, 1]
    //   factor_imag = [-r_imag, 0]
    for i in 0..n_roots {
        let idx = create_index_tensor::<R>(i, index_dtype, device)?;

        // Get root i as [1]-shaped tensors
        let r_real = client.index_select(roots_real, 0, &idx)?;
        let r_imag = client.index_select(roots_imag, 0, &idx)?;

        // Build factor (x - r):
        // factor_real = [-r_real, 1]
        // factor_imag = [-r_imag, 0]
        let neg_r_real = client.neg(&r_real)?;
        let neg_r_imag = client.neg(&r_imag)?;
        let one = Tensor::full_scalar(&[1], dtype, 1.0, device)?;
        let zero = Tensor::full_scalar(&[1], dtype, 0.0, device)?;

        let factor_real = client.cat(&[&neg_r_real, &one], 0)?;
        let factor_imag = client.cat(&[&neg_r_imag, &zero], 0)?;

        // Complex polynomial multiplication:
        // new_real = conv(p_real, factor_real) - conv(p_imag, factor_imag)
        // new_imag = conv(p_real, factor_imag) + conv(p_imag, factor_real)
        let conv_rr = convolve_impl(client, &p_real, &factor_real, dtype_support)?;
        let conv_ii = convolve_impl(client, &p_imag, &factor_imag, dtype_support)?;
        let conv_ri = convolve_impl(client, &p_real, &factor_imag, dtype_support)?;
        let conv_ir = convolve_impl(client, &p_imag, &factor_real, dtype_support)?;

        p_real = client.sub(&conv_rr, &conv_ii)?;
        p_imag = client.add(&conv_ri, &conv_ir)?;
    }

    // If roots were proper conjugate pairs, p_imag should be ~0
    // Return only the real part
    Ok(p_real)
}
