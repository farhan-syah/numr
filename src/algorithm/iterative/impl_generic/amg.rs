//! Algebraic Multigrid (AMG) preconditioner
//!
//! Classical Ruge-Stüben AMG with:
//! - Strength-of-connection based coarsening
//! - PMIS independent set selection
//! - Classical interpolation
//! - Galerkin coarse grid operators
//! - Weighted Jacobi smoothing
//! - V-cycle iteration
//!
//! Used as a preconditioner for CG/GMRES on SPD systems,
//! especially those arising from discretized PDEs.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{extract_diagonal_inv, vector_dot, vector_norm};
use super::super::types::{AmgHierarchy, AmgOptions};
use super::amg_coarsen::{
    build_interpolation, build_restriction, galerkin_coarse_operator, pmis_coarsening,
    strength_of_connection,
};

/// Build the AMG multigrid hierarchy (setup phase)
///
/// This is CPU-side structural analysis that builds:
/// - Coarse grid operators at each level (A_0, A_1, ..., A_L)
/// - Interpolation (prolongation) operators (P_0, P_1, ..., P_{L-1})
/// - Restriction operators (R_0, R_1, ..., R_{L-1})
/// - Diagonal inverses for Jacobi smoothing at each level
///
/// The setup is done once and the hierarchy is reused for many V-cycles.
pub fn amg_setup<R, C>(client: &C, a: &CsrData<R>, options: AmgOptions) -> Result<AmgHierarchy<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseOps<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let dtype = a.values().dtype();
    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "amg_setup",
        });
    }

    let device = a.values().device();
    let mut operators: Vec<CsrData<R>> = vec![a.clone()];
    let mut prolongations: Vec<CsrData<R>> = Vec::new();
    let mut restrictions: Vec<CsrData<R>> = Vec::new();
    let mut diag_inv: Vec<Tensor<R>> = Vec::new();

    // Compute D_inv for finest level
    diag_inv.push(extract_diagonal_inv(client, a)?);

    let mut current_n = a.shape[0];

    for _level in 0..options.max_levels - 1 {
        if current_n <= options.coarse_size {
            break;
        }

        let current_a = operators
            .last()
            .expect("operators always contains at least the original matrix");

        // Extract CSR structure for CPU-side graph analysis. PMIS coarsening
        // is inherently sequential (greedy independent-set with data dependencies
        // between iterations) and Galerkin triple-product uses irregular HashMap
        // accumulation — both are *slower* on GPU than CPU. This runs once during
        // setup; the V-cycle iteration loop is fully on-device with zero transfers.
        let rp: Vec<i64> = current_a.row_ptrs().to_vec();
        let ci: Vec<i64> = current_a.col_indices().to_vec();
        let vv: Vec<f64> = match dtype {
            DType::F32 => current_a
                .values()
                .to_vec::<f32>()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::F64 => current_a.values().to_vec::<f64>(),
            _ => unreachable!("dtype validated as F32 or F64 above"),
        };

        // Strength of connection
        let strong = strength_of_connection(&rp, &ci, &vv, current_n, options.strength_threshold);

        // Coarsening
        let splitting = pmis_coarsening(&strong, current_n);

        if splitting.n_coarse == 0 || splitting.n_coarse >= current_n {
            break; // Can't coarsen further
        }

        // Build interpolation P
        let p = build_interpolation::<R>(&rp, &ci, &vv, current_n, &splitting, &strong, device)?;

        // Build restriction R = P^T
        let r = build_restriction::<R>(&p)?;

        // Extract P structure for Galerkin product
        let p_rp: Vec<i64> = p.row_ptrs().to_vec();
        let p_ci: Vec<i64> = p.col_indices().to_vec();
        let p_vv: Vec<f64> = match dtype {
            DType::F32 => p
                .values()
                .to_vec::<f32>()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::F64 => p.values().to_vec::<f64>(),
            _ => unreachable!("dtype validated as F32 or F64 above"),
        };

        // Build coarse operator A_c = R * A * P
        let a_coarse = galerkin_coarse_operator::<R>(
            &rp,
            &ci,
            &vv,
            current_n,
            &p_rp,
            &p_ci,
            &p_vv,
            splitting.n_coarse,
            device,
        )?;

        // Compute D_inv for coarse level
        let d_inv_coarse = extract_diagonal_inv(client, &a_coarse)?;

        prolongations.push(p);
        restrictions.push(r);
        diag_inv.push(d_inv_coarse);
        current_n = splitting.n_coarse;
        operators.push(a_coarse);
    }

    let num_levels = operators.len();

    Ok(AmgHierarchy {
        operators,
        prolongations,
        restrictions,
        diag_inv,
        options,
        num_levels,
    })
}

/// Apply one AMG V-cycle: smooth → restrict → coarse-solve → prolongate → smooth
///
/// This is the preconditioner application: given residual r, compute z ≈ A^{-1} r
pub fn amg_vcycle<R, C>(
    client: &C,
    hierarchy: &AmgHierarchy<R>,
    rhs: &Tensor<R>,
    level: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseOps<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let a = &hierarchy.operators[level];
    let n = a.shape[0];
    let dtype = rhs.dtype();
    let device = rhs.device();

    // Base case: direct solve at coarsest level (small system)
    if level == hierarchy.num_levels - 1 || n <= hierarchy.options.coarse_size {
        // Use many Jacobi iterations as a "direct" solve for the small system
        let d_inv = &hierarchy.diag_inv[level];
        let omega = hierarchy.options.smoother_omega;
        let mut x = Tensor::<R>::zeros(&[n], dtype, device)?;

        for _ in 0..50 {
            let ax = a.spmv(&x)?;
            let r = client.sub(rhs, &ax)?;
            let d_inv_r = client.mul(d_inv, &r)?;
            let update = client.mul_scalar(&d_inv_r, omega)?;
            x = client.add(&x, &update)?;
        }
        return Ok(x);
    }

    let d_inv = &hierarchy.diag_inv[level];
    let omega = hierarchy.options.smoother_omega;

    // Pre-smoothing: weighted Jacobi iterations
    let mut x = Tensor::<R>::zeros(&[n], dtype, device)?;
    for _ in 0..hierarchy.options.smoother_sweeps {
        let ax = a.spmv(&x)?;
        let r = client.sub(rhs, &ax)?;
        let d_inv_r = client.mul(d_inv, &r)?;
        let update = client.mul_scalar(&d_inv_r, omega)?;
        x = client.add(&x, &update)?;
    }

    // Compute residual
    let ax = a.spmv(&x)?;
    let residual = client.sub(rhs, &ax)?;

    // Restrict residual to coarse level: r_c = R * r
    let r_coarse = hierarchy.restrictions[level].spmv(&residual)?;

    // Recursive V-cycle on coarse level
    let e_coarse = amg_vcycle(client, hierarchy, &r_coarse, level + 1)?;

    // Prolongate correction: e = P * e_c
    let e_fine = hierarchy.prolongations[level].spmv(&e_coarse)?;

    // Apply correction
    x = client.add(&x, &e_fine)?;

    // Post-smoothing: weighted Jacobi iterations
    for _ in 0..hierarchy.options.smoother_sweeps {
        let ax = a.spmv(&x)?;
        let r = client.sub(rhs, &ax)?;
        let d_inv_r = client.mul(d_inv, &r)?;
        let update = client.mul_scalar(&d_inv_r, omega)?;
        x = client.add(&x, &update)?;
    }

    Ok(x)
}

/// Apply AMG as a preconditioner within CG
///
/// Solves Ax = b using CG with AMG V-cycle as preconditioner M^{-1}
pub fn amg_preconditioned_cg<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    hierarchy: &AmgHierarchy<R>,
    max_iter: usize,
    rtol: f64,
    atol: f64,
) -> Result<(Tensor<R>, usize, f64, bool)>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseOps<R> + BinaryOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let n = a.shape[0];
    let dtype = b.dtype();
    let device = b.device();

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    let b_norm = vector_norm(client, b)?;
    if b_norm < atol {
        return Ok((x, 0, b_norm, true));
    }

    // r = b - A*x
    let ax = a.spmv(&x)?;
    let mut r = client.sub(b, &ax)?;

    // z = M^{-1} r (AMG V-cycle)
    let mut z = amg_vcycle(client, hierarchy, &r, 0)?;

    let mut p = z.clone();
    let mut rz = vector_dot(client, &r, &z)?;

    for iter in 0..max_iter {
        let ap = a.spmv(&p)?;

        // alpha = rz / <p, Ap>
        let p_ap = vector_dot(client, &p, &ap)?;

        if p_ap.abs() < 1e-40 {
            let res_norm = vector_norm(client, &r)?;
            return Ok((
                x,
                iter,
                res_norm,
                res_norm < atol || res_norm / b_norm < rtol,
            ));
        }

        let alpha = rz / p_ap;

        let ps = client.mul_scalar(&p, alpha)?;
        x = client.add(&x, &ps)?;

        let aps = client.mul_scalar(&ap, alpha)?;
        r = client.sub(&r, &aps)?;

        let res_norm = vector_norm(client, &r)?;
        if res_norm < atol || res_norm / b_norm < rtol {
            return Ok((x, iter + 1, res_norm, true));
        }

        // z = M^{-1} r
        z = amg_vcycle(client, hierarchy, &r, 0)?;

        let rz_new = vector_dot(client, &r, &z)?;

        if rz.abs() < 1e-40 {
            return Ok((x, iter + 1, res_norm, false));
        }

        let beta_val = rz_new / rz;
        let pbs = client.mul_scalar(&p, beta_val)?;
        p = client.add(&z, &pbs)?;
        rz = rz_new;
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_res = vector_norm(client, &r_final)?;

    Ok((x, max_iter, final_res, false))
}
