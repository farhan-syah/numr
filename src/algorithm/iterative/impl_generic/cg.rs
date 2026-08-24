//! Generic Conjugate Gradient implementation
//!
//! Preconditioned CG (Hestenes-Stiefel) for symmetric positive definite systems.

use crate::algorithm::sparse_linalg::{IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{BREAKDOWN_TOL, apply_ilu0_preconditioner, vector_dot, vector_norm};
use super::super::types::{CgOptions, CgResult, PreconditionerType};

/// Generic preconditioned CG implementation
///
/// Algorithm (Hestenes-Stiefel):
/// ```text
/// x = x0, r = b - A*x, z = M^-1*r, p = z, rz = <r,z>
/// for iter = 1, 2, ...:
///     Ap = A*p
///     alpha = rz / <p, Ap>
///     x = x + alpha*p
///     r = r - alpha*Ap
///     if ||r|| < tol: return
///     z = M^-1*r
///     rz_new = <r, z>
///     beta = rz_new / rz
///     p = z + beta*p
///     rz = rz_new
/// ```
pub fn cg_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: CgOptions,
) -> Result<CgResult<R>>
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
        return Err(Error::UnsupportedDType { dtype, op: "cg" });
    }

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::try_zeros(&[n], dtype, device)?,
    };

    // Compute preconditioner
    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => Some(client.ilu0(a, IluOptions::default())?),
        PreconditionerType::Amg => {
            return Err(Error::Internal(
                "AMG preconditioner not supported here — use amg_preconditioned_cg".to_string(),
            ));
        }
        PreconditionerType::Ic0 => {
            return Err(Error::Internal(
                "IC0 preconditioner not supported for CG — use ILU0".to_string(),
            ));
        }
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(CgResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    // r = b - A*x
    let ax = a.spmv(&x)?;
    let mut r = client.sub(b, &ax)?;

    // z = M^-1 * r
    let mut z = apply_ilu0_preconditioner(client, &precond, &r)?;

    // p = z
    let mut p = z.clone();

    // rz = <r, z>
    let mut rz = vector_dot(client, &r, &z)?;

    for iter in 0..options.max_iter {
        // Ap = A * p
        let ap = a.spmv(&p)?;

        // alpha = rz / <p, Ap>
        let p_ap = vector_dot(client, &p, &ap)?;
        if p_ap.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &r)?;
            return Ok(CgResult {
                solution: x,
                iterations: iter,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }
        let alpha = rz / p_ap;

        // x = x + alpha * p
        let p_scaled = client.mul_scalar(&p, alpha)?;
        x = client.add(&x, &p_scaled)?;

        // r = r - alpha * Ap
        let ap_scaled = client.mul_scalar(&ap, alpha)?;
        r = client.sub(&r, &ap_scaled)?;

        let res_norm = vector_norm(client, &r)?;
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(CgResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // z = M^-1 * r
        z = apply_ilu0_preconditioner(client, &precond, &r)?;

        let rz_new = vector_dot(client, &r, &z)?;

        if rz.abs() < BREAKDOWN_TOL {
            return Ok(CgResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: false,
            });
        }
        let beta = rz_new / rz;

        // p = z + beta * p
        let p_scaled = client.mul_scalar(&p, beta)?;
        p = client.add(&z, &p_scaled)?;

        rz = rz_new;
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(CgResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
