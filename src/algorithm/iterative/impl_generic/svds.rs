//! Generic sparse SVD via Lanczos bidiagonalization
//!
//! Computes the k largest (or smallest) singular values and vectors of a
//! large sparse matrix using the Golub-Kahan-Lanczos bidiagonalization.
//!
//! This is more numerically stable than eigendecomposing A^T*A.

use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, LinalgOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{
    INVARIANT_SUBSPACE_TOL, REORTH_TOL, accumulate_basis_combination, vector_dot, vector_norm,
};
use super::super::types::{SparseSvdResult, SvdsOptions, WhichSingularValues};
use super::dense_eig::tridiagonal_eig;

/// Generic sparse SVD implementation via Lanczos bidiagonalization
///
/// Algorithm (Golub-Kahan):
/// 1. Build bidiagonal B from A using Lanczos bidiagonalization:
///    - p_1 = random, q_1 = A^T*p_1 / ||A^T*p_1||
///    - for j = 1, ..., ncv:
///      p = A*q_j - alpha_j*p_j;  beta_j = ||p||; p_{j+1} = p/beta_j
///      q = A^T*p_{j+1} - beta_j*q_j;  alpha_{j+1} = ||q||; q_{j+1} = q/alpha_{j+1}
/// 2. B is bidiagonal with diagonal alphas and super-diagonal betas
/// 3. SVD of B gives singular values of A (small dense problem)
/// 4. Ritz vectors give approximate singular vectors of A
pub fn svds_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    k: usize,
    options: SvdsOptions,
) -> Result<SparseSvdResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseLinAlgAlgorithms<R>
        + SparseOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + LinalgOps<R>
        + ShapeOps<R>,
{
    let [m, n_cols] = a.shape;
    let min_dim = m.min(n_cols);

    if k == 0 || k > min_dim {
        return Err(Error::Internal(format!(
            "k must be in [1, min(m,n)], got k={k}, min(m,n)={min_dim}"
        )));
    }

    let device = a.values().device();
    let dtype = a.values().dtype();

    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType { dtype, op: "svds" });
    }

    // Build A^T for transpose spmv
    let at = a.transpose().to_csr()?;

    // Krylov dimension
    let ncv = options.ncv.unwrap_or_else(|| (2 * k + 1).min(min_dim));
    let ncv = ncv.max(k + 1).min(min_dim);

    // Starting vector: q_1 = [1, 1, ..., 1] / sqrt(n)
    let ones_n = Tensor::<R>::try_ones(&[n_cols], dtype, device)?;
    let scale = 1.0 / (n_cols as f64).sqrt();
    let mut q_curr = client.mul_scalar(&ones_n, scale)?;

    // Lanczos bidiagonalization
    let mut p_basis: Vec<Tensor<R>> = Vec::with_capacity(ncv); // left vectors (m-dim)
    let mut q_basis: Vec<Tensor<R>> = Vec::with_capacity(ncv); // right vectors (n-dim)
    let mut alphas: Vec<f64> = Vec::with_capacity(ncv);
    let mut betas: Vec<f64> = Vec::with_capacity(ncv);

    let mut invariant_found = false;

    for restart_iter in 0..options.max_iter {
        if invariant_found {
            break;
        }
        let start_j = q_basis.len();

        for j in start_j..ncv {
            q_basis.push(q_curr.clone());

            // p = A * q_j
            let mut p = a.spmv(&q_curr)?;

            // Subtract beta_{j-1} * p_{j-1}
            if j > 0 {
                let beta_prev = betas[j - 1];
                let bps = client.mul_scalar(&p_basis[j - 1], beta_prev)?;
                p = client.sub(&p, &bps)?;
            }

            // Reorthogonalize against all previous p vectors
            for pi in p_basis.iter() {
                let h = vector_dot(client, pi, &p)?;
                if h.abs() > REORTH_TOL {
                    let pis = client.mul_scalar(pi, h)?;
                    p = client.sub(&p, &pis)?;
                }
            }

            let alpha_j = vector_norm(client, &p)?;
            alphas.push(alpha_j);

            if alpha_j < INVARIANT_SUBSPACE_TOL {
                invariant_found = true;
                break;
            }

            let p_next = client.mul_scalar(&p, 1.0 / alpha_j)?;
            p_basis.push(p_next.clone());

            // q = A^T * p_{j+1} - alpha_j * q_j
            let atq = at.spmv(&p_next)?;
            let aqs = client.mul_scalar(&q_curr, alpha_j)?;
            let mut q = client.sub(&atq, &aqs)?;

            // Reorthogonalize against all previous q vectors
            for qi in q_basis.iter() {
                let h = vector_dot(client, qi, &q)?;
                if h.abs() > REORTH_TOL {
                    let qis = client.mul_scalar(qi, h)?;
                    q = client.sub(&q, &qis)?;
                }
            }

            let beta_j = vector_norm(client, &q)?;
            betas.push(beta_j);

            if beta_j < INVARIANT_SUBSPACE_TOL {
                invariant_found = true;
                break;
            }

            q_curr = client.mul_scalar(&q, 1.0 / beta_j)?;
        }

        let kk = alphas.len();
        if kk == 0 {
            return Err(Error::Internal(
                "Lanczos bidiagonalization produced no vectors".to_string(),
            ));
        }

        // Compute SVD of bidiagonal matrix B
        // B^T * B is tridiagonal: use tridiagonal eigensolver
        // The eigenvalues of B^T*B are the squared singular values of B.
        //
        // Diagonal of B^T*B: alpha_i^2 + beta_{i-1}^2
        // Off-diagonal: alpha_i * beta_i
        let mut btb_diag = Vec::with_capacity(kk);
        let mut btb_offdiag = Vec::with_capacity(kk);

        for i in 0..kk {
            let d = alphas[i] * alphas[i]
                + if i > 0 {
                    betas[i - 1] * betas[i - 1]
                } else {
                    0.0
                };
            btb_diag.push(d);
            if i + 1 < kk && i < betas.len() {
                btb_offdiag.push(alphas[i] * betas[i]);
            }
        }

        let (eig_vals, eig_vecs) = tridiagonal_eig(&btb_diag, &btb_offdiag);

        // Singular values = sqrt(eigenvalues of B^T*B)
        let mut sv_indices: Vec<usize> = (0..eig_vals.len()).collect();

        match options.which {
            WhichSingularValues::Largest => {
                sv_indices.sort_by(|&a_idx, &b_idx| {
                    eig_vals[b_idx]
                        .partial_cmp(&eig_vals[a_idx])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            WhichSingularValues::Smallest => {
                sv_indices.sort_by(|&a_idx, &b_idx| {
                    eig_vals[a_idx]
                        .partial_cmp(&eig_vals[b_idx])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        sv_indices.truncate(k.min(kk));

        // Check convergence
        let last_beta = if betas.len() >= kk && kk > 0 {
            *betas.last().unwrap_or(&0.0)
        } else {
            0.0
        };

        let mut nconv = 0;
        for &idx in &sv_indices {
            let error = (last_beta * eig_vecs[idx][kk - 1]).abs();
            if error < options.tol {
                nconv += 1;
            }
        }

        if nconv >= k || kk >= min_dim || invariant_found {
            let k_actual = k.min(sv_indices.len());
            let mut singular_values_data = Vec::with_capacity(k_actual);
            let mut right_vectors: Vec<Tensor<R>> = Vec::with_capacity(k_actual);
            let mut left_vectors: Vec<Tensor<R>> = Vec::with_capacity(k_actual);

            for &idx in sv_indices.iter().take(k_actual) {
                let sv = eig_vals[idx].max(0.0).sqrt();
                singular_values_data.push(sv);

                // Right singular vector: V_k = Q * z (Ritz vector)
                let right = accumulate_basis_combination(
                    client,
                    &q_basis[..kk],
                    &eig_vecs[idx],
                    n_cols,
                    dtype,
                    device,
                )?;
                right_vectors.push(right.clone());

                // Left singular vector: U_k = A * V_k / sigma_k
                if sv > INVARIANT_SUBSPACE_TOL {
                    let av = a.spmv(&right)?;
                    let left = client.mul_scalar(&av, 1.0 / sv)?;
                    left_vectors.push(left);
                } else {
                    left_vectors.push(Tensor::<R>::try_zeros(&[m], dtype, device)?);
                }
            }

            let singular_values =
                Tensor::<R>::try_from_slice(&singular_values_data, &[k_actual], device)?;

            // Assemble U = [n, k] and Vt = [k, n] on-device via stack
            let u = assemble_column_matrix(client, &left_vectors)?;
            let vt = assemble_row_matrix(client, &right_vectors)?;

            return Ok(SparseSvdResult {
                u,
                singular_values,
                vt,
                iterations: restart_iter + 1,
                converged: nconv >= k,
                nconv,
            });
        }

        // Thick restart: keep k wanted Ritz vectors
        let k_keep = k.min(sv_indices.len());
        let mut new_q_basis: Vec<Tensor<R>> = Vec::with_capacity(k_keep);
        let mut new_alphas: Vec<f64> = Vec::with_capacity(k_keep);

        for &idx in sv_indices.iter().take(k_keep) {
            let ritz = accumulate_basis_combination(
                client,
                &q_basis[..kk],
                &eig_vecs[idx],
                n_cols,
                dtype,
                device,
            )?;
            new_alphas.push(eig_vals[idx].max(0.0).sqrt());
            new_q_basis.push(ritz);
        }

        q_basis = new_q_basis;
        p_basis.clear();
        alphas = new_alphas;
        betas.clear();
    }

    Err(Error::Internal(format!(
        "Sparse SVD did not converge after {} restart cycles",
        options.max_iter
    )))
}

/// Assemble k column vectors of length n into [n, k] tensor (on-device via stack + transpose)
fn assemble_column_matrix<R, C>(client: &C, columns: &[Tensor<R>]) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ShapeOps<R>,
{
    let refs: Vec<&Tensor<R>> = columns.iter().collect();
    // Stack [v1, v2, ..., vk] → [k, n], then transpose → [n, k]
    let stacked = client.stack(&refs, 0)?;
    Ok(stacked.transpose(0, 1)?)
}

/// Assemble k row vectors of length n into [k, n] tensor (on-device via stack)
fn assemble_row_matrix<R, C>(client: &C, rows: &[Tensor<R>]) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ShapeOps<R>,
{
    let refs: Vec<&Tensor<R>> = rows.iter().collect();
    client.stack(&refs, 0)
}
