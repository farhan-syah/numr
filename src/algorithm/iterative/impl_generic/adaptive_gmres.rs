//! Adaptive GMRES implementation
//!
//! GMRES with automatic preconditioner upgrading when convergence stagnates.

use crate::algorithm::sparse_linalg::{IlukOptions, SparseLinAlgAlgorithms};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{
    apply_iluk_preconditioner, detect_stagnation, givens_rotation, solve_upper_triangular,
    update_solution, vector_dot, vector_norm,
};
use super::super::types::{
    AdaptiveGmresResult, AdaptivePreconditionerOptions, ConvergenceReason, GmresOptions,
};

/// Adaptive GMRES implementation
///
/// Automatically upgrades preconditioner when stagnation is detected.
///
/// # Algorithm
///
/// 1. Start with ILU(initial_level) preconditioner
/// 2. Run GMRES, tracking residual history
/// 3. If stagnation detected (residual not decreasing by `reduction_factor` over `window_size` iterations):
///    a. Upgrade preconditioner to next ILU level
///    b. Optionally restart GMRES from current solution
/// 4. Repeat until converged or max_upgrades exceeded
pub fn adaptive_gmres_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    gmres_opts: GmresOptions,
    adaptive_opts: AdaptivePreconditionerOptions,
) -> Result<AdaptiveGmresResult<R>>
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
            op: "adaptive_gmres",
        });
    }

    // Initialize solution
    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    // Compute ||b|| for relative tolerance check
    let b_norm = vector_norm(client, b)?;
    if b_norm < gmres_opts.atol {
        // b is essentially zero
        return Ok(AdaptiveGmresResult {
            solution: x,
            total_iterations: 0,
            residual_norm: b_norm,
            converged: true,
            final_level: adaptive_opts.initial_level,
            upgrades: 0,
            ilu_metrics: Vec::new(),
            reason: ConvergenceReason::ZeroRhs,
        });
    }

    let mut current_level = adaptive_opts.initial_level;
    let mut upgrades = 0;
    let mut ilu_metrics = Vec::new();
    let mut total_iterations = 0;
    let mut residual_history: Vec<f64> = Vec::new();

    loop {
        // Compute ILU(k) preconditioner at current level
        let iluk_opts = IlukOptions {
            fill_level: current_level,
            drop_tolerance: 0.0,
            diagonal_shift: 1e-10, // Small shift for stability
            pivot_threshold: 0.1,
        };
        let ilu = client.iluk(a, iluk_opts)?;
        ilu_metrics.push(ilu.metrics.clone());

        // Run GMRES with this preconditioner
        let result = gmres_with_iluk(
            client,
            a,
            b,
            &x,
            &ilu,
            &gmres_opts,
            &adaptive_opts.stagnation,
            b_norm,
            &mut residual_history,
        )?;

        total_iterations += result.iterations;
        x = result.solution;

        // Check if converged
        if result.converged {
            return Ok(AdaptiveGmresResult {
                solution: x,
                total_iterations,
                residual_norm: result.residual_norm,
                converged: true,
                final_level: current_level,
                upgrades,
                ilu_metrics,
                reason: result.reason,
            });
        }

        // Check if stagnation detected and can upgrade
        let should_upgrade = matches!(result.reason, ConvergenceReason::Stagnation)
            && upgrades < adaptive_opts.max_upgrades
            && current_level.upgrade().is_some();

        if should_upgrade {
            current_level = current_level
                .upgrade()
                .expect("upgrade checked via is_some() above");
            upgrades += 1;

            // Optionally reset residual history on upgrade
            if adaptive_opts.restart_on_upgrade {
                residual_history.clear();
            }
        } else {
            // Can't upgrade or max upgrades reached
            return Ok(AdaptiveGmresResult {
                solution: x,
                total_iterations,
                residual_norm: result.residual_norm,
                converged: false,
                final_level: current_level,
                upgrades,
                ilu_metrics,
                reason: result.reason,
            });
        }
    }
}

/// Internal GMRES result with stagnation detection
struct GmresInternalResult<R: Runtime> {
    solution: Tensor<R>,
    iterations: usize,
    residual_norm: f64,
    converged: bool,
    reason: ConvergenceReason,
}

/// Run GMRES with ILU(k) preconditioner and stagnation detection
fn gmres_with_iluk<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: &Tensor<R>,
    ilu: &crate::algorithm::sparse_linalg::IlukDecomposition<R>,
    opts: &GmresOptions,
    stagnation: &super::super::types::StagnationParams,
    b_norm: f64,
    residual_history: &mut Vec<f64>,
) -> Result<GmresInternalResult<R>>
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
    let mut x = x0.clone();
    let m = opts.restart;
    let mut total_iterations = 0;

    // Outer restart loop
    for _restart_cycle in 0..(opts.max_iter / m + 1) {
        // r = b - A @ x
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;

        // beta = ||r||
        let beta = vector_norm(client, &r)?;

        // Check convergence
        if beta < opts.atol || beta / b_norm < opts.rtol {
            let atol_met = beta < opts.atol;
            let rtol_met = beta / b_norm < opts.rtol;
            let reason = if atol_met && rtol_met {
                ConvergenceReason::BothTolerances
            } else if atol_met {
                ConvergenceReason::AbsoluteTolerance
            } else {
                ConvergenceReason::RelativeTolerance
            };
            return Ok(GmresInternalResult {
                solution: x,
                iterations: total_iterations,
                residual_norm: beta,
                converged: true,
                reason,
            });
        }

        // v[0] = r / beta
        let v0 = client.mul_scalar(&r, 1.0 / beta)?;

        // Krylov basis
        let mut v_basis: Vec<Tensor<R>> = vec![v0];
        let mut z_basis: Vec<Tensor<R>> = Vec::with_capacity(m);
        let mut h_matrix: Vec<Vec<f64>> = Vec::with_capacity(m);
        let mut cs: Vec<f64> = Vec::with_capacity(m);
        let mut sn: Vec<f64> = Vec::with_capacity(m);
        let mut g: Vec<f64> = vec![beta];

        let mut j = 0;
        while j < m && total_iterations < opts.max_iter {
            total_iterations += 1;

            // Apply preconditioner: z = M^-1 @ v[j]
            let vj = &v_basis[j];
            let z = apply_iluk_preconditioner(client, ilu, vj)?;
            let w = a.spmv(&z)?;
            z_basis.push(z);

            // Arnoldi orthogonalization
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
            let (c, s, r) = givens_rotation(h_col[j], h_col[j + 1]);
            cs.push(c);
            sn.push(s);

            h_col[j] = r;
            h_col[j + 1] = 0.0;

            let g_old_j = g[j];
            g.push(-s * g_old_j);
            g[j] = c * g_old_j;

            h_matrix.push(h_col);

            let res_norm = g[j + 1].abs();
            residual_history.push(res_norm);

            // Check convergence
            if res_norm < opts.atol || res_norm / b_norm < opts.rtol {
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                let atol_met = res_norm < opts.atol;
                let rtol_met = res_norm / b_norm < opts.rtol;
                let reason = if atol_met && rtol_met {
                    ConvergenceReason::BothTolerances
                } else if atol_met {
                    ConvergenceReason::AbsoluteTolerance
                } else {
                    ConvergenceReason::RelativeTolerance
                };

                return Ok(GmresInternalResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: res_norm,
                    converged: true,
                    reason,
                });
            }

            // Check for stagnation
            if detect_stagnation(residual_history, stagnation) {
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(GmresInternalResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: res_norm,
                    converged: false,
                    reason: ConvergenceReason::Stagnation,
                });
            }

            // Check for lucky breakdown
            if h_jp1_j < 1e-14 {
                let y = solve_upper_triangular(&h_matrix, &g[..j + 1]);
                x = update_solution(client, &x, &z_basis, &y)?;

                return Ok(GmresInternalResult {
                    solution: x,
                    iterations: total_iterations,
                    residual_norm: g[j + 1].abs(),
                    converged: true,
                    reason: ConvergenceReason::LuckyBreakdown,
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

    // Max iterations reached
    let ax = a.spmv(&x)?;
    let r = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r)?;

    Ok(GmresInternalResult {
        solution: x,
        iterations: total_iterations,
        residual_norm: final_residual,
        converged: false,
        reason: ConvergenceReason::MaxIterationsReached,
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::helpers::detect_stagnation;
    use super::super::super::types::StagnationParams;

    #[test]
    fn test_stagnation_detection() {
        let params = StagnationParams {
            reduction_factor: 0.5,
            window_size: 3,
            min_iterations: 2,
        };

        // Not enough iterations
        let history = vec![1.0, 0.9];
        assert!(!detect_stagnation(&history, &params));

        // Enough iterations, but still improving
        let history = vec![1.0, 0.8, 0.6, 0.4, 0.2];
        assert!(!detect_stagnation(&history, &params));

        // Stagnation: not improving enough
        let history = vec![1.0, 0.9, 0.85, 0.8, 0.75, 0.72];
        assert!(detect_stagnation(&history, &params));
    }
}
