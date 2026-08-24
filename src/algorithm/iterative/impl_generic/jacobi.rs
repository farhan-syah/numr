//! Generic weighted Jacobi iteration
//!
//! x_{k+1} = x_k + omega * D^{-1} * (b - A*x_k)
//!
//! Simple stationary iterative method, best for diagonally dominant systems
//! or as a smoother inside multigrid.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{extract_diagonal_inv, vector_norm};
use super::super::types::{JacobiOptions, JacobiResult};

/// Generic weighted Jacobi implementation
///
/// Algorithm:
/// ```text
/// D_inv = 1 / diag(A)           (precomputed once)
/// x = x0 (or zeros)
/// for iter = 1, 2, ...:
///     r = b - A*x
///     x = x + omega * D_inv * r  (element-wise multiply)
///     if ||r|| < tol: return
/// ```
pub fn jacobi_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: JacobiOptions,
) -> Result<JacobiResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseOps<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let n = super::super::traits::validate_iterative_inputs(a.shape, b, x0)?;
    let device = b.device();
    let dtype = b.dtype();

    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "jacobi",
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::try_zeros(&[n], dtype, device)?,
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(JacobiResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    // Precompute D_inv = 1/diag(A) — one-time setup
    let d_inv = extract_diagonal_inv(client, a)?;

    for iter in 0..options.max_iter {
        // r = b - A*x
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;

        let res_norm = vector_norm(client, &r)?;
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(JacobiResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // x = x + omega * D_inv * r
        let d_inv_r = client.mul(&d_inv, &r)?;
        let update = client.mul_scalar(&d_inv_r, options.omega)?;
        x = client.add(&x, &update)?;
    }

    // Final residual
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r)?;

    Ok(JacobiResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
