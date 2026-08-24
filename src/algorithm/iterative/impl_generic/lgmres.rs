//! Generic LGMRES (Loose GMRES) implementation
//!
//! LGMRES augments the Krylov subspace with error approximation vectors
//! from previous restart cycles (Baker, Jessup, Manteuffel 2005).
//!
//! Key difference from standard GMRES(m): after each restart, the Krylov
//! subspace is augmented with k_aug approximate error vectors computed as
//! the difference between consecutive iterates. This accelerates
//! convergence across restarts.

use crate::algorithm::sparse_linalg::{IluOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{
    apply_ilu0_preconditioner, givens_rotation, solve_upper_triangular, update_solution,
    vector_dot, vector_norm,
};
use super::super::types::{LgmresOptions, LgmresResult, PreconditionerType};

/// Generic LGMRES implementation
///
/// Algorithm: GMRES(m) augmented with k_aug error approximation vectors.
///
/// Each restart cycle:
/// 1. Compute residual r = b - Ax
/// 2. Build augmented Krylov basis: first k_aug vectors from previous
///    error approximations, then m standard Arnoldi vectors
/// 3. Solve least-squares problem over augmented basis
/// 4. Update solution and store error approximation for next restart
pub fn lgmres_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: LgmresOptions,
) -> Result<LgmresResult<R>>
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
            op: "lgmres",
        });
    }

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::try_zeros(&[n], dtype, device)?,
    };

    // Compute preconditioner if requested
    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => Some(client.ilu0(a, IluOptions::default())?),
        PreconditionerType::Ic0 | PreconditionerType::Amg => {
            return Err(Error::Internal(
                "Only None and Ilu0 preconditioners supported for LGMRES".to_string(),
            ));
        }
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < options.atol {
        return Ok(LgmresResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    let m = options.restart;
    let k_aug = options.k_aug;
    let mut total_iterations = 0;

    // Augmentation vectors from previous restarts (error approximations)
    // These are the delta_x vectors from previous restart cycles
    let mut aug_vectors: Vec<Tensor<R>> = Vec::new();

    // Outer restart loop
    for _restart_cycle in 0..(options.max_iter / m.max(1) + 1) {
        // Save x at start of cycle for error approximation
        let x_start = x.clone();

        // r = b - A*x
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;
        let beta = vector_norm(client, &r)?;

        // Check convergence
        if beta < options.atol || beta / b_norm < options.rtol {
            return Ok(LgmresResult {
                solution: x,
                iterations: total_iterations,
                residual_norm: beta,
                converged: true,
            });
        }

        // v[0] = r / beta
        let v0 = client.mul_scalar(&r, 1.0 / beta)?;
        let mut v_basis: Vec<Tensor<R>> = vec![v0];
        let mut z_basis: Vec<Tensor<R>> = Vec::new();
        let mut h_matrix: Vec<Vec<f64>> = Vec::new();
        let mut cs: Vec<f64> = Vec::new();
        let mut sn: Vec<f64> = Vec::new();
        let mut g: Vec<f64> = vec![beta];

        // Total inner dimension = augmentation vectors + Arnoldi vectors
        let n_aug = aug_vectors.len().min(k_aug);
        let inner_dim = m + n_aug;

        let mut j = 0;
        while j < inner_dim && total_iterations < options.max_iter {
            total_iterations += 1;

            let vj = &v_basis[j];

            // Determine what to multiply:
            // - First n_aug steps: use augmentation vectors as search directions
            // - Remaining steps: standard Arnoldi with preconditioner
            let z = if j < n_aug {
                // Use augmentation vector as preconditioned direction
                aug_vectors[aug_vectors.len() - n_aug + j].clone()
            } else {
                apply_ilu0_preconditioner(client, &precond, vj)?
            };

            let w = a.spmv(&z)?;
            z_basis.push(z);

            // Arnoldi orthogonalization (modified Gram-Schmidt)
            let mut h_col: Vec<f64> = Vec::with_capacity(j + 2);
            let mut w_current = w;

            for i in 0..=j {
                let h_ij = vector_dot(client, &w_current, &v_basis[i])?;
                h_col.push(h_ij);
                let scaled_vi = client.mul_scalar(&v_basis[i], h_ij)?;
                w_current = client.sub(&w_current, &scaled_vi)?;
            }

            let h_jp1_j = vector_norm(client, &w_current)?;
            h_col.push(h_jp1_j);

            // Apply previous Givens rotations
            for i in 0..j {
                let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
                h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1];
                h_col[i] = temp;
            }

            // Compute new Givens rotation
            let (c, s, r_val) = givens_rotation(h_col[j], h_col[j + 1]);
            cs.push(c);
            sn.push(s);
            h_col[j] = r_val;
            h_col[j + 1] = 0.0;

            let g_old_j = g[j];
            g.push(-s * g_old_j);
            g[j] = c * g_old_j;

            h_matrix.push(h_col);

            // Check convergence
            let res_norm = g[j + 1].abs();
            if res_norm < options.atol || res_norm / b_norm < options.rtol {
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(LgmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: res_norm,
                    converged: true,
                });
            }

            // Lucky breakdown
            if h_jp1_j < 1e-14 {
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(LgmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: g[j + 1].abs(),
                    converged: true,
                });
            }

            let v_jp1 = client.mul_scalar(&w_current, 1.0 / h_jp1_j)?;
            v_basis.push(v_jp1);

            j += 1;
        }

        // End of restart cycle — update solution
        if !h_matrix.is_empty() {
            let y = solve_upper_triangular(&h_matrix, &g[..j]);
            x = update_solution(client, &x, &z_basis, &y)?;
        }

        // Compute error approximation: delta_x = x - x_start
        let delta_x = client.sub(&x, &x_start)?;
        let delta_norm = vector_norm(client, &delta_x)?;

        // Store normalized error approximation for next restart
        if delta_norm > 1e-14 {
            let aug_vec = client.mul_scalar(&delta_x, 1.0 / delta_norm)?;

            // Keep only the most recent k_aug vectors
            if aug_vectors.len() >= k_aug {
                aug_vectors.remove(0);
            }
            aug_vectors.push(aug_vec);
        }
    }

    // Final residual
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r)?;

    Ok(LgmresResult {
        solution: x,
        iterations: total_iterations,
        residual_norm: final_residual,
        converged: false,
    })
}
