//! Generic CGS (Conjugate Gradient Squared) implementation
//!
//! Sonneveld's CGS for non-symmetric systems. Faster convergence than BiCGSTAB
//! when it works, but can be less stable (residual norms may oscillate wildly).

use crate::algorithm::sparse_linalg::{IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{BREAKDOWN_TOL, apply_ilu0_preconditioner, vector_dot, vector_norm};
use super::super::types::{CgsOptions, CgsResult, PreconditionerType};

/// Generic preconditioned CGS implementation
///
/// Algorithm (Sonneveld):
/// ```text
/// x = x0, r = b - A*x, r_hat = r
/// rho = <r_hat, r>, u = r, p = r
///
/// for iter = 1, 2, ...:
///     p_hat = M^-1 * p
///     v = A * p_hat
///     sigma = <r_hat, v>
///     alpha = rho / sigma
///
///     q = u - alpha * v
///     u_plus_q = u + q
///     uq_hat = M^-1 * u_plus_q
///     x = x + alpha * uq_hat
///     r = r - alpha * A * uq_hat
///
///     if ||r|| < tol: return
///
///     rho_new = <r_hat, r>
///     beta = rho_new / rho
///     u = r + beta * q
///     p = u + beta * (q + beta * p)
///     rho = rho_new
/// ```
pub fn cgs_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: CgsOptions,
) -> Result<CgsResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseLinAlgAlgorithms<R>
        + SparseOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>,
{
    let n = super::super::traits::validate_iterative_inputs(a.shape, b, x0)?;
    let device = b.device();
    let dtype = b.dtype();

    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op: "cgs" });
    }

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => Some(client.ilu0(a, IluOptions::default())?),
        PreconditionerType::Amg => {
            return Err(Error::Internal(
                "AMG preconditioner not supported for CGS — use amg_preconditioned_cg".to_string(),
            ));
        }
        PreconditionerType::Ic0 => {
            return Err(Error::Internal(
                "IC0 preconditioner not supported for CGS — use ILU0".to_string(),
            ));
        }
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(CgsResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    // r = b - A*x
    let ax = a.spmv(&x)?;
    let mut r = client.sub(b, &ax)?;

    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();

    let mut rho = vector_dot(client, &r_hat, &r)?;

    // u = r, p = r
    let mut u = r.clone();
    let mut p = r.clone();

    for iter in 0..options.max_iter {
        if rho.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &r)?;
            return Ok(CgsResult {
                solution: x,
                iterations: iter,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        // p_hat = M^-1 * p
        let p_hat = apply_ilu0_preconditioner(client, &precond, &p)?;

        // v = A * p_hat
        let v = a.spmv(&p_hat)?;

        // sigma = <r_hat, v>
        let sigma = vector_dot(client, &r_hat, &v)?;
        if sigma.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &r)?;
            return Ok(CgsResult {
                solution: x,
                iterations: iter,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }
        let alpha = rho / sigma;

        // q = u - alpha * v
        let v_scaled = client.mul_scalar(&v, alpha)?;
        let q = client.sub(&u, &v_scaled)?;

        // u_plus_q = u + q
        let u_plus_q = client.add(&u, &q)?;

        // uq_hat = M^-1 * (u + q)
        let uq_hat = apply_ilu0_preconditioner(client, &precond, &u_plus_q)?;

        // x = x + alpha * uq_hat
        let uq_scaled = client.mul_scalar(&uq_hat, alpha)?;
        x = client.add(&x, &uq_scaled)?;

        // r = r - alpha * A * uq_hat
        let a_uq = a.spmv(&uq_hat)?;
        let a_uq_scaled = client.mul_scalar(&a_uq, alpha)?;
        r = client.sub(&r, &a_uq_scaled)?;

        let res_norm = vector_norm(client, &r)?;
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(CgsResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // rho_new = <r_hat, r>
        let rho_new = vector_dot(client, &r_hat, &r)?;

        let beta = rho_new / rho;

        // u = r + beta * q
        let q_scaled = client.mul_scalar(&q, beta)?;
        u = client.add(&r, &q_scaled)?;

        // p = u + beta * (q + beta * p)
        let p_scaled = client.mul_scalar(&p, beta)?;
        let q_plus_bp = client.add(&q, &p_scaled)?;
        let qbp_scaled = client.mul_scalar(&q_plus_bp, beta)?;
        p = client.add(&u, &qbp_scaled)?;

        rho = rho_new;
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(CgsResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
