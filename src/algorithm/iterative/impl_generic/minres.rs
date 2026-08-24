//! Generic MINRES implementation
//!
//! Paige-Saunders MINRES for symmetric (possibly indefinite) systems.
//! Uses Lanczos tridiagonalization with QR factorization via Givens rotations.

use crate::algorithm::sparse_linalg::{IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{BREAKDOWN_TOL, apply_ilu0_preconditioner, vector_dot, vector_norm};
use super::super::types::{MinresOptions, MinresResult, PreconditionerType};

/// Generic MINRES implementation following Saad, "Iterative Methods for
/// Sparse Linear Systems", 2nd ed., Algorithm 6.12.
///
/// At iteration k, the Lanczos process produces tridiagonal T_k.
/// Column k of T_{k+1,k} has entries [beta_{k-1}, alpha_k, beta_k] at rows [k-2, k-1, k].
/// (The entry at k-2 is really beta_{k-1} only for k≥2; for k=1 there's no row k-2.)
///
/// Previous Givens rotations G_{k-2} and G_{k-1} are applied, then G_k is
/// chosen to eliminate the subdiagonal of R.
pub fn minres_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: MinresOptions,
) -> Result<MinresResult<R>>
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
        return Err(Error::UnsupportedDType {
            dtype,
            op: "minres",
        });
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
                "AMG preconditioner not supported for MINRES — use amg_preconditioned_cg"
                    .to_string(),
            ));
        }
        PreconditionerType::Ic0 => {
            return Err(Error::Internal(
                "IC0 preconditioner not supported for MINRES — use ILU0".to_string(),
            ));
        }
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(MinresResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    let ax = a.spmv(&x)?;
    let r0 = client.sub(b, &ax)?;
    let beta1 = vector_norm(client, &r0)?;

    if beta1 < options.atol || beta1 / b_norm < options.rtol {
        return Ok(MinresResult {
            solution: x,
            iterations: 0,
            residual_norm: beta1,
            converged: true,
        });
    }

    // Lanczos vectors
    let mut v_old = Tensor::<R>::zeros(&[n], dtype, device)?;
    let mut v = client.mul_scalar(&r0, 1.0 / beta1)?;

    let mut beta = beta1;

    // Givens rotation parameters from two previous steps
    let mut c1 = 1.0_f64; // c_{k-1}
    let mut s1 = 0.0_f64; // s_{k-1}
    let mut c2 = 1.0_f64; // c_{k-2}
    let mut s2 = 0.0_f64; // s_{k-2}

    // Direction vectors for solution update (d_{k-1}, d_{k-2})
    let mut d1 = Tensor::<R>::zeros(&[n], dtype, device)?;
    let mut d2 = Tensor::<R>::zeros(&[n], dtype, device)?;

    // Right-hand side of the least-squares problem: Q * (beta1 * e1)
    // We track the last two entries: tau_{k-1} and tau_k
    let mut phibar = beta1; // = |tau_k| before rotation G_k

    for iter in 0..options.max_iter {
        let z = apply_ilu0_preconditioner(client, &precond, &v)?;

        // Lanczos: w = A*z - beta * v_old
        let az = a.spmv(&z)?;
        let vold_s = client.mul_scalar(&v_old, beta)?;
        let mut w = client.sub(&az, &vold_s)?;

        let alpha = vector_dot(client, &v, &w)?;

        let vs = client.mul_scalar(&v, alpha)?;
        w = client.sub(&w, &vs)?;

        let beta_new = vector_norm(client, &w)?;

        // QR update for column j (= iter) of T_{j+2, j+1}.
        //
        // Column j of the tridiagonal has entries:
        //   row j-1: beta    (current Lanczos subdiag, only if j >= 1)
        //   row j:   alpha
        //   row j+1: beta_new
        //
        // Apply previous rotations then compute new one:
        //   G_{j-2} on rows (j-2, j-1): [0, beta] -> [s2*beta, c2*beta]
        //   G_{j-1} on rows (j-1, j): [c2*beta, alpha] -> [delta, gamma_bar]
        //   G_j on rows (j, j+1): [gamma_bar, beta_new] -> [gamma, 0]

        // Step 1: G_{j-2} on rows (j-2, j-1) of [0, beta]:
        let eps = s2 * beta;
        let beta_hat = c2 * beta;

        // Step 2: G_{j-1} on rows (j-1, j) of [beta_hat, alpha]:
        let delta = c1 * beta_hat + s1 * alpha;
        let gamma_bar = -s1 * beta_hat + c1 * alpha;

        // Step 3: G_j on rows (j, j+1) of [gamma_bar, beta_new]:

        let (c_new, s_new, gamma) = super::super::helpers::givens_rotation(gamma_bar, beta_new);

        // Residual update: Q * (beta1 * e1) right-hand side
        let phi = c_new * phibar;
        phibar *= -s_new; // Note: sign matters for correctness

        // Direction vector: d_new = (z - eps * d2 - delta * d1) / gamma
        if gamma.abs() < BREAKDOWN_TOL {
            return Ok(MinresResult {
                solution: x,
                iterations: iter,
                residual_norm: phibar.abs(),
                converged: phibar.abs() < options.atol || phibar.abs() / b_norm < options.rtol,
            });
        }

        let d2_s = client.mul_scalar(&d2, eps)?;
        let d1_s = client.mul_scalar(&d1, delta)?;
        let d_new_num = client.sub(&z, &d2_s)?;
        let d_new_num = client.sub(&d_new_num, &d1_s)?;
        let d_new = client.mul_scalar(&d_new_num, 1.0 / gamma)?;

        // x = x + phi * d_new
        let d_step = client.mul_scalar(&d_new, phi)?;
        x = client.add(&x, &d_step)?;

        let res_norm = phibar.abs();
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(MinresResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        if beta_new.abs() < BREAKDOWN_TOL {
            return Ok(MinresResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        // Shift for next iteration
        v_old = v;
        v = client.mul_scalar(&w, 1.0 / beta_new)?;

        d2 = d1;
        d1 = d_new;

        beta = beta_new;

        c2 = c1;
        s2 = s1;
        c1 = c_new;
        s1 = s_new;
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(MinresResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
