//! Generic SOR (Successive Over-Relaxation) implementation
//!
//! Forward-sweep SOR with relaxation parameter omega, implemented via
//! the matrix splitting formulation:
//!
//!   x_{k+1} = x_k + (D + omega*L)^{-1} * omega * (b - A*x_k)
//!
//! where D = diag(A), L = strict lower triangle of A.
//!
//! The lower triangular matrix (D + omega*L) is built once at setup from
//! the CSR sparsity structure. All iteration work uses on-device operations
//! (SpMV and sparse triangular solve).

use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::vector_norm;
use super::super::types::{SorOptions, SorResult};

/// Generic SOR implementation via sparse triangular solve
///
/// Each iteration:
/// 1. r = b - A*x  (SpMV on device)
/// 2. rhs = omega * r  (scalar mul on device)
/// 3. delta = (D + omega*L)^{-1} * rhs  (sparse forward substitution on device)
/// 4. x = x + delta  (vector add on device)
pub fn sor_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    b: &Tensor<R>,
    x0: Option<&Tensor<R>>,
    options: SorOptions,
) -> Result<SorResult<R>>
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
    let dtype = b.dtype();
    let device = b.device();

    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op: "sor" });
    }

    let b_norm = vector_norm(client, b)?;

    let mut x = match x0 {
        Some(x0) => x0.clone(),
        None => Tensor::<R>::zeros(&[n], dtype, device)?,
    };

    if b_norm < options.atol {
        return Ok(SorResult {
            solution: x,
            iterations: 0,
            residual_norm: b_norm,
            converged: true,
        });
    }

    let omega = options.omega;

    // Build lower triangular iteration matrix (D + omega*L) once at setup.
    // This extracts A's CSR sparsity structure (one-time structural analysis)
    // and constructs the triangular operator on device for forward solves.
    let lower_tri = build_sor_lower_triangular::<R>(a, omega, device)?;

    for iter in 0..options.max_iter {
        // r = b - A*x (on device)
        let ax = a.spmv(&x)?;
        let r = client.sub(b, &ax)?;

        // rhs = omega * r (on device)
        let rhs = client.mul_scalar(&r, omega)?;

        // delta = (D + omega*L)^{-1} * rhs (forward substitution on device)
        let delta = client.sparse_solve_triangular(&lower_tri, &rhs, true, false)?;

        // x = x + delta (on device)
        x = client.add(&x, &delta)?;

        // Check convergence with true residual
        let ax = a.spmv(&x)?;
        let r_check = client.sub(b, &ax)?;
        let res_norm = vector_norm(client, &r_check)?;

        if res_norm < options.atol || res_norm / b_norm < options.rtol {
            return Ok(SorResult {
                solution: x,
                iterations: iter + 1,
                residual_norm: res_norm,
                converged: true,
            });
        }
    }

    let ax = a.spmv(&x)?;
    let r_final = client.sub(b, &ax)?;
    let final_residual = vector_norm(client, &r_final)?;

    Ok(SorResult {
        solution: x,
        iterations: options.max_iter,
        residual_norm: final_residual,
        converged: false,
    })
}

/// Build the lower triangular matrix (D + omega*L) from A's CSR structure.
///
/// One-time structural setup: extracts A's sparsity pattern to construct
/// the SOR iteration matrix. Entry (i,j):
/// - j < i:  omega * a_ij  (scaled strict lower triangle)
/// - j == i: a_ii           (diagonal, unscaled)
/// - j > i:  excluded       (upper triangle)
fn build_sor_lower_triangular<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    omega: f64,
    device: &R::Device,
) -> Result<CsrData<R>> {
    let n = a.shape[0];
    let dtype = a.values().dtype();

    // Extract CSR structure for one-time triangular matrix construction.
    // This is structural setup analogous to AMG coarsening — graph analysis
    // on the sparsity pattern, not iteration-loop data processing.
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();
    let values: Vec<f64> = match dtype {
        DType::F32 => a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&v| v as f64)
            .collect(),
        DType::F64 => a.values().to_vec::<f64>(),
        _ => unreachable!(),
    };

    let mut lt_rp = Vec::with_capacity(n + 1);
    let mut lt_ci = Vec::new();
    let mut lt_vv = Vec::new();

    lt_rp.push(0i64);

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        let mut row_entries: Vec<(i64, f64)> = Vec::new();

        for idx in start..end {
            let j = col_indices[idx] as usize;
            let v = values[idx];

            if j < i {
                // Strict lower triangle: scale by omega
                row_entries.push((j as i64, omega * v));
            } else if j == i {
                // Diagonal: unscaled
                row_entries.push((j as i64, v));
            }
        }

        row_entries.sort_by_key(|&(c, _)| c);

        for (c, v) in row_entries {
            lt_ci.push(c);
            lt_vv.push(v);
        }

        lt_rp.push(lt_ci.len() as i64);
    }

    let rp_t = Tensor::<R>::from_slice(&lt_rp, &[lt_rp.len()], device)?;
    let ci_t = Tensor::<R>::from_slice(&lt_ci, &[lt_ci.len()], device)?;
    let vv_t = Tensor::<R>::from_slice(&lt_vv, &[lt_vv.len()], device)?;

    CsrData::new(rp_t, ci_t, vv_t, [n, n])
}
