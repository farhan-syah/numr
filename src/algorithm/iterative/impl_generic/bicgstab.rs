//! Generic BiCGSTAB implementation
//!
//! Bi-Conjugate Gradient Stabilized method for non-symmetric sparse systems.
//! Alternative to GMRES with fixed memory footprint.

use crate::algorithm::sparse_linalg::{IluDecomposition, IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{apply_ilu0_preconditioner, vector_dot, vector_norm};
use super::super::types::{BiCgStabOptions, BiCgStabResult, PreconditionerType};

/// Generic BiCGSTAB implementation
///
/// Implements right-preconditioned BiCGSTAB.
/// Uses less memory than GMRES(m) but convergence can be less predictable.
pub fn bicgstab_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: BiCgStabOptions,
) -> Result<BiCgStabResult<R>>
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

    // Validate dtype is floating point
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "bicgstab",
        });
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::try_zeros(&[n], dtype, device)?,
    };

    // Compute preconditioner if requested
    let precond: Option<IluDecomposition<R>> = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => {
            let ilu = client.ilu0(a, IluOptions::default())?;
            Some(ilu)
        }
        PreconditionerType::Amg => {
            return Err(Error::Internal(
                "AMG preconditioner not supported for BiCGSTAB - use amg_preconditioned_cg"
                    .to_string(),
            ));
        }
        PreconditionerType::Ic0 => {
            return Err(Error::Internal(
                "IC0 preconditioner not yet supported for BiCGSTAB - use ILU0".to_string(),
            ));
        }
    };

    // Compute ||b|| for relative tolerance
    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(BiCgStabResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    // r = b - A @ x
    let ax = a.spmv(&x)?;
    let mut r = client.sub(b, &ax)?;

    // r_hat = r (shadow residual, kept constant)
    let r_hat = r.clone();

    // Initialize vectors
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;

    let mut v = Tensor::<R>::try_zeros(&[n], dtype, device)?;
    let mut p = Tensor::<R>::try_zeros(&[n], dtype, device)?;

    for iter in 0..options.max_iter {
        // rho_new = <r_hat, r>
        let rho_new = vector_dot(client, &r_hat, &r)?;

        // Check for breakdown
        if rho_new.abs() < 1e-40 {
            // BiCGSTAB breakdown - shadow residual orthogonal to residual
            let res_norm = vector_norm(client, &r)?;
            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        // beta = (rho_new / rho) * (alpha / omega)
        let beta = (rho_new / rho) * (alpha / omega);

        // p = r + beta * (p - omega * v)
        // p = r + beta * p - beta * omega * v
        let p_scaled = client.mul_scalar(&p, beta)?;
        let v_scaled = client.mul_scalar(&v, beta * omega)?;
        let temp = client.sub(&p_scaled, &v_scaled)?;
        p = client.add(&r, &temp)?;

        // Apply preconditioner: p_hat = M^-1 @ p
        let p_hat = apply_ilu0_preconditioner(client, &precond, &p)?;

        // v = A @ p_hat
        v = a.spmv(&p_hat)?;

        // alpha = rho_new / <r_hat, v>
        let r_hat_v = vector_dot(client, &r_hat, &v)?;
        if r_hat_v.abs() < 1e-40 {
            // Another breakdown case
            let res_norm = vector_norm(client, &r)?;
            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }
        alpha = rho_new / r_hat_v;

        // s = r - alpha * v
        let v_scaled = client.mul_scalar(&v, alpha)?;
        let s = client.sub(&r, &v_scaled)?;

        // Check convergence on s
        let s_norm = vector_norm(client, &s)?;
        if s_norm < options.atol || s_norm / b_norm < options.rtol {
            // x = x + alpha * p_hat
            let p_hat_scaled = client.mul_scalar(&p_hat, alpha)?;
            x = client.add(&x, &p_hat_scaled)?;

            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: s_norm,
                converged: true,
            });
        }

        // Apply preconditioner: s_hat = M^-1 @ s
        let s_hat = apply_ilu0_preconditioner(client, &precond, &s)?;

        // t = A @ s_hat
        let t = a.spmv(&s_hat)?;

        // omega = <t, s> / <t, t>
        let t_s = vector_dot(client, &t, &s)?;
        let t_t = vector_dot(client, &t, &t)?;
        if t_t.abs() < 1e-40 {
            // Breakdown
            let res_norm = vector_norm(client, &s)?;
            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: false,
            });
        }
        omega = t_s / t_t;

        // x = x + alpha * p_hat + omega * s_hat
        let p_hat_scaled = client.mul_scalar(&p_hat, alpha)?;
        let s_hat_scaled = client.mul_scalar(&s_hat, omega)?;
        x = client.add(&x, &p_hat_scaled)?;
        x = client.add(&x, &s_hat_scaled)?;

        // r = s - omega * t
        let t_scaled = client.mul_scalar(&t, omega)?;
        r = client.sub(&s, &t_scaled)?;

        // Update rho
        rho = rho_new;

        // Check convergence
        let res_norm = vector_norm(client, &r)?;
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Check for stagnation
        if omega.abs() < 1e-40 {
            return Ok(BiCgStabResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: false,
            });
        }
    }

    // Max iterations reached
    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(BiCgStabResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
