//! Generic QMR (Quasi-Minimal Residual) implementation
//!
//! Based on the QMR algorithm from Barrett et al., "Templates for the
//! Solution of Linear Systems: Building Blocks for Iterative Methods".
//!
//! QMR maintains a quasi-minimal residual property, providing smoother
//! convergence than BiCGSTAB for many non-symmetric systems.

use crate::algorithm::sparse_linalg::{IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{BREAKDOWN_TOL, apply_ilu0_preconditioner, vector_dot, vector_norm};
use super::super::types::{PreconditionerType, QmrOptions, QmrResult};

/// Generic QMR implementation
///
/// Uses coupled two-term Lanczos biorthogonalization with quasi-minimal
/// residual smoothing. Follows the Templates book algorithm.
pub fn qmr_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: QmrOptions,
) -> Result<QmrResult<R>>
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
        return Err(Error::UnsupportedDType { dtype, op: "qmr" });
    }

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => Some(client.ilu0(a, IluOptions::default())?),
        PreconditionerType::Ic0 | PreconditionerType::Amg => {
            return Err(Error::Internal(
                "Only None and Ilu0 preconditioners supported for QMR".to_string(),
            ));
        }
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(QmrResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    // Build A^T for transpose SpMV
    let at = a.transpose().to_csr()?;

    // r = b - A*x
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;

    let r_norm = vector_norm(client, &r)?;
    if r_norm < options.atol || r_norm / b_norm < options.rtol {
        return Ok(QmrResult {
            solution: x,
            iterations: 0,
            residual_norm: r_norm,
            converged: true,
        });
    }

    // v_tilde = r, w_tilde = r
    let mut v_tilde = r.clone();
    let mut w_tilde = r.clone();

    let mut rho = vector_norm(client, &v_tilde)?;
    let mut xi = vector_norm(client, &w_tilde)?;

    let mut gamma_prev = 1.0_f64;
    let mut eta = -1.0_f64;
    let mut theta_prev = 0.0_f64;

    let mut v = client.mul_scalar(&v_tilde, 1.0 / rho)?;
    let mut w = client.mul_scalar(&w_tilde, 1.0 / xi)?;

    let mut d = Tensor::<R>::zeros(&[n], dtype, device)?;
    let mut s = Tensor::<R>::zeros(&[n], dtype, device)?;

    let mut p;
    let mut q;
    let mut epsilon_prev = 0.0_f64;

    // Track p_prev and q_prev for the recurrence
    let mut p_prev = Tensor::<R>::zeros(&[n], dtype, device)?;
    let mut q_prev = Tensor::<R>::zeros(&[n], dtype, device)?;

    let mut residual = r;

    for iter in 0..options.max_iter {
        let delta = vector_dot(client, &w, &v)?;
        if delta.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &residual)?;
            return Ok(QmrResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        // Apply preconditioner
        let y = apply_ilu0_preconditioner(client, &precond, &v)?;
        let z = apply_ilu0_preconditioner(client, &precond, &w)?;

        // p, q direction vectors
        if iter == 0 {
            p = y.clone();
            q = z.clone();
        } else {
            let coeff_p = (xi * delta) / epsilon_prev;
            let coeff_q = (rho * delta) / epsilon_prev;
            let pp = client.mul_scalar(&p_prev, coeff_p)?;
            let qq = client.mul_scalar(&q_prev, coeff_q)?;
            p = client.sub(&y, &pp)?;
            q = client.sub(&z, &qq)?;
        }

        // epsilon = <q, A*p>
        let ap = a.spmv(&p)?;
        let epsilon = vector_dot(client, &q, &ap)?;
        if epsilon.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &residual)?;
            return Ok(QmrResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        let beta = epsilon / delta;

        // v_tilde = A*p - beta*v
        let bv = client.mul_scalar(&v, beta)?;
        v_tilde = client.sub(&ap, &bv)?;

        // w_tilde = A^T*q - conj(beta)*w
        let atq = at.spmv(&q)?;
        let bw = client.mul_scalar(&w, beta)?;
        w_tilde = client.sub(&atq, &bw)?;

        let rho_new = vector_norm(client, &v_tilde)?;
        let xi_new = vector_norm(client, &w_tilde)?;

        // QMR quasi-minimization
        let theta = rho_new / (gamma_prev * beta.abs());
        let gamma = 1.0 / (1.0 + theta * theta).sqrt();

        if gamma.abs() < BREAKDOWN_TOL {
            let res_norm = vector_norm(client, &residual)?;
            return Ok(QmrResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        let eta_new = -eta * rho * gamma * gamma / (beta * gamma_prev * gamma_prev);

        // d = eta*p + (theta_prev*gamma)^2 * d_prev
        let tg2 = (theta_prev * gamma) * (theta_prev * gamma);
        let ep = client.mul_scalar(&p, eta_new)?;
        let td = client.mul_scalar(&d, tg2)?;
        d = client.add(&ep, &td)?;

        // s = eta*A*p + (theta_prev*gamma)^2 * s_prev
        let eap = client.mul_scalar(&ap, eta_new)?;
        let ts = client.mul_scalar(&s, tg2)?;
        s = client.add(&eap, &ts)?;

        // x = x + d
        x = client.add(&x, &d)?;

        // r = r - s
        residual = client.sub(&residual, &s)?;

        let res_norm = vector_norm(client, &residual)?;
        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(QmrResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }

        // Check true residual periodically to guard against drift
        if (iter + 1) % 50 == 0 {
            let ax_check = a.spmv(&x)?;
            residual = client.sub(b, &ax_check)?;
            let true_norm = vector_norm(client, &residual)?;
            if true_norm < options.atol || true_norm / b_norm < options.rtol {
                return Ok(QmrResult {
                    solution: x,
                    iterations: iter + 1,
                    residual_norm: true_norm,
                    converged: true,
                });
            }
        }

        // Update for next iteration
        if rho_new < BREAKDOWN_TOL || xi_new < BREAKDOWN_TOL {
            return Ok(QmrResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: res_norm < options.atol || res_norm / b_norm < options.rtol,
            });
        }

        v = client.mul_scalar(&v_tilde, 1.0 / rho_new)?;
        w = client.mul_scalar(&w_tilde, 1.0 / xi_new)?;

        p_prev = p;
        q_prev = q;
        rho = rho_new;
        xi = xi_new;
        gamma_prev = gamma;
        theta_prev = theta;
        eta = eta_new;
        epsilon_prev = epsilon;
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(QmrResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}
