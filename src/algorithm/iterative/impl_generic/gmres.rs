//! Generic GMRES implementation
//!
//! Right-preconditioned GMRES with restarts using Arnoldi iteration
//! and Givens rotations for numerical stability.
//!
//! All operations use tensor primitives - no GPU↔CPU transfers.

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
use super::super::types::{
    ConvergenceReason, GmresDiagnostics, GmresOptions, GmresResult, PreconditionerType,
};

/// Generic GMRES implementation
///
/// Implements right-preconditioned GMRES with Arnoldi iteration and Givens rotations.
/// All operations are performed via tensor primitives to ensure no GPU↔CPU transfers.
///
/// # Type Parameters
///
/// * `R` - Runtime (CPU, CUDA, WebGPU)
/// * `C` - Client type implementing required operations
pub fn gmres_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: GmresOptions,
) -> Result<GmresResult<R>>
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
        return Err(Error::UnsupportedDType { dtype, op: "gmres" });
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    // Compute preconditioner if requested
    let precond = match options.preconditioner {
        PreconditionerType::None => None,
        PreconditionerType::Ilu0 => {
            let ilu = client.ilu0(a, IluOptions::default())?;
            Some(ilu)
        }
        PreconditionerType::Amg => {
            return Err(Error::Internal(
                "AMG preconditioner not supported for GMRES — use amg_preconditioned_cg"
                    .to_string(),
            ));
        }
        PreconditionerType::Ic0 => {
            return Err(Error::Internal(
                "IC0 preconditioner not yet supported for GMRES - use ILU0".to_string(),
            ));
        }
    };

    // Compute initial residual norm ||b||
    let b_norm = vector_norm(client, b)?;

    // Track residual history if requested
    let mut residual_history = if options.track_residual_history {
        Vec::with_capacity(options.max_iter)
    } else {
        Vec::new()
    };

    // Helper to build diagnostics
    let build_diagnostics = |initial_residual_norm: f64, history: Vec<f64>| GmresDiagnostics {
        rtol: options.rtol,
        atol: options.atol,
        max_iter: options.max_iter,
        restart: options.restart,
        initial_residual_norm,
        rhs_norm: b_norm,
        residual_history: history,
    };

    if b_norm < options.atol {
        // b is essentially zero, x = 0 is the solution
        let reason = ConvergenceReason::ZeroRhs;
        return Ok(GmresResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
            reason,
            diagnostics: build_diagnostics(b_norm, residual_history),
        });
    }

    let m = options.restart;
    let mut total_iterations = 0;
    let mut initial_residual_norm = 0.0;

    // Outer restart loop
    for restart_cycle in 0..(options.max_iter / m + 1) {
        // r = b - A @ x
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;

        // beta = ||r||
        let beta = vector_norm(client, &r)?;

        // Track initial residual for diagnostics
        if restart_cycle == 0 {
            initial_residual_norm = beta;
        }

        // Check convergence
        let atol_met = beta < options.atol;
        let rtol_met = beta / b_norm < options.rtol;
        if atol_met || rtol_met {
            let reason = if atol_met && rtol_met {
                ConvergenceReason::BothTolerances
            } else if atol_met {
                ConvergenceReason::AbsoluteTolerance
            } else {
                ConvergenceReason::RelativeTolerance
            };
            return Ok(GmresResult {
                solution: x,
                iterations: total_iterations,
                residual_norm: beta,
                converged: true,
                reason,
                diagnostics: build_diagnostics(initial_residual_norm, residual_history),
            });
        }

        // v[0] = r / beta
        let v0 = client.mul_scalar(&r, 1.0 / beta)?;

        // Krylov basis vectors V = [v0, v1, ..., vm]
        let mut v_basis: Vec<Tensor<R>> = vec![v0];

        // Preconditioned basis vectors Z = [z0, z1, ..., zm] where z_j = M^-1 @ v_j
        // We store these to avoid recomputing during solution update
        let mut z_basis: Vec<Tensor<R>> = Vec::with_capacity(m);

        // Hessenberg matrix H (m+1 x m) stored as columns
        // H[i][j] = h_{i,j} where i is row, j is column
        // We store the upper Hessenberg entries
        let mut h_matrix: Vec<Vec<f64>> = Vec::with_capacity(m);

        // Givens rotation coefficients
        let mut cs: Vec<f64> = Vec::with_capacity(m);
        let mut sn: Vec<f64> = Vec::with_capacity(m);

        // Right-hand side of least squares: g = beta * e_1
        let mut g: Vec<f64> = vec![beta];

        let mut j = 0;
        while j < m && total_iterations < options.max_iter {
            total_iterations += 1;

            // z_j = M^-1 @ v[j], w = A @ z_j
            let vj = &v_basis[j];
            let z = apply_ilu0_preconditioner(client, &precond, vj)?;
            let w = a.spmv(&z)?;
            z_basis.push(z); // Store for later use in solution update

            // Arnoldi orthogonalization (modified Gram-Schmidt)
            let mut h_col: Vec<f64> = Vec::with_capacity(j + 2);
            let mut w_current = w;

            for i in 0..=j {
                // h_{i,j} = <w, v_i>
                let h_ij = vector_dot(client, &w_current, &v_basis[i])?;
                h_col.push(h_ij);

                // w = w - h_{i,j} * v_i
                let scaled_vi = client.mul_scalar(&v_basis[i], h_ij)?;
                w_current = client.sub(&w_current, &scaled_vi)?;
            }

            // h_{j+1,j} = ||w||
            let h_jp1_j = vector_norm(client, &w_current)?;
            h_col.push(h_jp1_j);

            // Apply previous Givens rotations to the new column
            for i in 0..j {
                let temp = cs[i] * h_col[i] + sn[i] * h_col[i + 1];
                h_col[i + 1] = -sn[i] * h_col[i] + cs[i] * h_col[i + 1];
                h_col[i] = temp;
            }

            // Compute new Givens rotation
            let (c, s, r) = givens_rotation(h_col[j], h_col[j + 1]);
            cs.push(c);
            sn.push(s);

            // Apply to H column
            h_col[j] = r;
            h_col[j + 1] = 0.0;

            // Apply to g
            let g_old_j = g[j];
            g.push(-s * g_old_j);
            g[j] = c * g_old_j;

            h_matrix.push(h_col);

            // Check convergence (|g[j+1]| is the residual norm)
            let res_norm = g[j + 1].abs();

            // Track residual if requested
            if options.track_residual_history {
                residual_history.push(res_norm);
            }

            let atol_met = res_norm < options.atol;
            let rtol_met = res_norm / b_norm < options.rtol;
            if atol_met || rtol_met {
                // Solve upper triangular system H @ y = g
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);

                // x = x + Z @ y (using stored preconditioned basis)
                x = update_solution(client, &x, &z_basis, &y)?;

                let reason = if atol_met && rtol_met {
                    ConvergenceReason::BothTolerances
                } else if atol_met {
                    ConvergenceReason::AbsoluteTolerance
                } else {
                    ConvergenceReason::RelativeTolerance
                };

                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: res_norm,
                    converged: true,
                    reason,
                    diagnostics: build_diagnostics(initial_residual_norm, residual_history),
                });
            }

            // Check for lucky breakdown
            if h_jp1_j < 1e-14 {
                // Solve and update solution
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(GmresResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: g[j + 1].abs(),
                    converged: true,
                    reason: ConvergenceReason::LuckyBreakdown,
                    diagnostics: build_diagnostics(initial_residual_norm, residual_history),
                });
            }

            // v[j+1] = w / h_{j+1,j}
            let v_jp1 = client.mul_scalar(&w_current, 1.0 / h_jp1_j)?;
            v_basis.push(v_jp1);

            j += 1;
        }

        // End of restart cycle - update solution
        if !h_matrix.is_empty() {
            let y = solve_upper_triangular(&h_matrix, &g[..j]);
            x = update_solution(client, &x, &z_basis, &y)?;
        }
    }

    // Compute final residual
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r)?;

    Ok(GmresResult {
        solution: x,
        iterations: total_iterations,
        residual_norm: final_residual,
        converged: false,
        reason: ConvergenceReason::MaxIterationsReached,
        diagnostics: build_diagnostics(initial_residual_norm, residual_history),
    })
}
