//! Generic Lanczos eigensolver for large sparse symmetric matrices
//!
//! Implicitly Restarted Lanczos Method (IRLM) with thick restart.
//! Computes a few eigenvalues/eigenvectors of A where A is symmetric.

use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, LinalgOps, ReduceOps, ScalarOps, UnaryOps};
use crate::runtime::Runtime;
use crate::sparse::{CsrData, SparseOps};
use crate::tensor::Tensor;

use super::super::helpers::{
    INVARIANT_SUBSPACE_TOL, REORTH_TOL, accumulate_basis_combination, vector_dot, vector_norm,
};
use super::super::types::{SparseEigOptions, SparseEigResult};
use super::dense_eig::{select_eigenvalues, tridiagonal_eig};

/// Generic Lanczos eigensolver implementation
///
/// Uses thick-restart Lanczos to find k eigenvalues of a symmetric matrix.
///
/// Algorithm:
/// 1. Build Lanczos basis V and tridiagonal T via 3-term recurrence
/// 2. Compute eigenvalues of T (small dense problem)
/// 3. Check convergence of Ritz values
/// 4. If not converged, do implicit restart (thick restart) and repeat
pub fn lanczos_eig_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    k: usize,
    options: SparseEigOptions,
) -> Result<SparseEigResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: SparseOps<R>,
    C: SparseLinAlgAlgorithms<R>
        + SparseOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + LinalgOps<R>,
{
    let [nrows, ncols] = a.shape;
    if nrows != ncols {
        return Err(Error::ShapeMismatch {
            expected: vec![nrows, nrows],
            got: vec![nrows, ncols],
        });
    }
    let n = nrows;

    if k == 0 || k > n {
        return Err(Error::Internal(format!(
            "k must be in [1, n], got k={k}, n={n}"
        )));
    }

    let device = a.values().device();
    let dtype = a.values().dtype();

    if !matches!(dtype, DType::F32 | DType::F64) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "lanczos_eig",
        });
    }

    // Krylov dimension
    let ncv = options.ncv.unwrap_or_else(|| (2 * k + 1).min(n));
    let ncv = ncv.max(k + 1).min(n);

    // Deterministic starting vector: v = [1, 1, ..., 1] / sqrt(n)
    let ones = Tensor::<R>::ones(&[n], dtype, device);
    let scale = 1.0 / (n as f64).sqrt();
    let mut v_curr = client.mul_scalar(&ones, scale)?;

    let mut v_basis: Vec<Tensor<R>> = Vec::with_capacity(ncv);
    let mut alphas: Vec<f64> = Vec::with_capacity(ncv);
    let mut betas: Vec<f64> = Vec::with_capacity(ncv);

    let mut v_prev = Tensor::<R>::zeros(&[n], dtype, device);
    let mut beta_prev = 0.0_f64;

    for restart_iter in 0..options.max_iter {
        let start_j = v_basis.len();

        for j in start_j..ncv {
            v_basis.push(v_curr.clone());

            let av = a.spmv(&v_curr)?;
            let alpha_j = vector_dot(client, &v_curr, &av)?;
            alphas.push(alpha_j);

            // w = av - alpha_j * v_curr - beta_{j-1} * v_prev
            let av_s = client.mul_scalar(&v_curr, alpha_j)?;
            let mut w = client.sub(&av, &av_s)?;
            if j > 0 || beta_prev > 0.0 {
                let vp_s = client.mul_scalar(&v_prev, beta_prev)?;
                w = client.sub(&w, &vp_s)?;
            }

            // Full reorthogonalization (CGS against all previous vectors)
            for vi in v_basis.iter() {
                let h = vector_dot(client, vi, &w)?;
                if h.abs() > REORTH_TOL {
                    let vi_s = client.mul_scalar(vi, h)?;
                    w = client.sub(&w, &vi_s)?;
                }
            }

            let beta_j = vector_norm(client, &w)?;

            if beta_j < INVARIANT_SUBSPACE_TOL {
                betas.push(0.0);
                break;
            }

            betas.push(beta_j);
            v_prev = v_curr;
            v_curr = client.mul_scalar(&w, 1.0 / beta_j)?;
            beta_prev = beta_j;
        }

        let m = alphas.len();
        if m == 0 {
            return Err(Error::Internal("Lanczos produced no vectors".to_string()));
        }

        // Solve tridiagonal eigenvalue problem (small dense, CPU-side)
        let (eig_vals, eig_vecs) = tridiagonal_eig(&alphas, &betas);
        let indices = select_eigenvalues(&eig_vals, k.min(m), &options.which);

        // Check convergence: |beta_m * s_{m,i}| < tol
        let last_beta = if betas.len() >= m && m > 0 {
            betas[m - 1]
        } else {
            0.0
        };

        let mut nconv = 0;
        for &idx in &indices {
            let error = (last_beta * eig_vecs[idx][m - 1]).abs();
            if error < options.tol {
                nconv += 1;
            }
        }

        if nconv >= k || m >= n {
            // Converged — compute Ritz vectors entirely on device
            let k_actual = k.min(indices.len());
            let mut eigenvalue_data = Vec::with_capacity(k_actual);
            let mut ritz_vectors: Vec<Tensor<R>> = Vec::with_capacity(k_actual);

            for &idx in indices.iter().take(k_actual) {
                eigenvalue_data.push(eig_vals[idx]);
                let ritz = accumulate_basis_combination(
                    client,
                    &v_basis[..m],
                    &eig_vecs[idx],
                    n,
                    dtype,
                    device,
                )?;
                ritz_vectors.push(ritz);
            }

            let eigenvalues = Tensor::<R>::from_slice(&eigenvalue_data, &[k_actual], device);
            let eigenvectors = assemble_column_matrix::<R>(&ritz_vectors, n, k_actual, device)?;

            return Ok(SparseEigResult {
                eigenvalues,
                eigenvectors,
                iterations: restart_iter + 1,
                converged: nconv >= k,
                nconv,
            });
        }

        // Thick restart: keep k wanted Ritz vectors as new basis
        let k_keep = k.min(indices.len());
        let mut new_basis: Vec<Tensor<R>> = Vec::with_capacity(k_keep);
        let mut new_alphas: Vec<f64> = Vec::with_capacity(k_keep);

        for &idx in indices.iter().take(k_keep) {
            let ritz = accumulate_basis_combination(
                client,
                &v_basis[..m],
                &eig_vecs[idx],
                n,
                dtype,
                device,
            )?;
            new_alphas.push(eig_vals[idx]);
            new_basis.push(ritz);
        }

        v_basis = new_basis;
        alphas = new_alphas;
        betas.clear();
        beta_prev = last_beta;
    }

    Err(Error::Internal(format!(
        "Lanczos eigensolver did not converge after {} restart cycles",
        options.max_iter
    )))
}

/// Assemble k column vectors of length n into a [n, k] tensor.
///
/// Each column vector is transferred once from device to host, then the
/// complete matrix is transferred back. This is O(k) transfers for final
/// output assembly — not used in any iterative loop.
fn assemble_column_matrix<R: Runtime<DType = DType>>(
    columns: &[Tensor<R>],
    n: usize,
    k: usize,
    device: &R::Device,
) -> Result<Tensor<R>> {
    let mut flat = vec![0.0f64; n * k];
    for (col, tensor) in columns.iter().enumerate() {
        let col_data: Vec<f64> = tensor.to_vec();
        flat[col * n..(col + 1) * n].copy_from_slice(&col_data);
    }
    Tensor::<R>::try_from_slice(&flat, &[n, k], device)
}
