//! Generic Arnoldi eigensolver (IRAM) for large sparse non-symmetric matrices
//!
//! Implicitly Restarted Arnoldi Method — the algorithm behind ARPACK's dnaupd.
//! Computes a few eigenvalues/eigenvectors of a general (non-symmetric) sparse matrix.

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
use super::super::types::{SparseEigComplexResult, SparseEigOptions};
use super::dense_eig::{hessenberg_eig, select_eigenvalues_complex};

/// Generic Arnoldi eigensolver implementation
///
/// Uses Implicitly Restarted Arnoldi (IRA) with thick restart to find k
/// eigenvalues of a general (non-symmetric) sparse matrix.
///
/// Algorithm:
/// 1. Build Arnoldi factorization: A*V_m = V_m*H_m + h_{m+1,m}*v_{m+1}*e_m^T
/// 2. Compute eigenvalues of H_m (small dense Hessenberg eigenvalue problem)
/// 3. Check convergence of Ritz values
/// 4. If not converged, thick restart with wanted Ritz vectors and extend again
pub fn arnoldi_eig_impl<R, C>(
    client: &C,
    a: &CsrData<R>,
    k: usize,
    options: SparseEigOptions,
) -> Result<SparseEigComplexResult<R>>
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
            op: "arnoldi_eig",
        });
    }

    let ncv = options.ncv.unwrap_or_else(|| (2 * k + 1).min(n));
    let ncv = ncv.max(k + 1).min(n);

    // Deterministic starting vector
    let ones = Tensor::<R>::try_ones(&[n], dtype, device)?;
    let scale = 1.0 / (n as f64).sqrt();
    let v0 = client.mul_scalar(&ones, scale)?;

    let mut v_basis: Vec<Tensor<R>> = Vec::with_capacity(ncv + 1);
    let mut h_columns: Vec<Vec<f64>> = Vec::with_capacity(ncv);

    v_basis.push(v0);

    for restart_iter in 0..options.max_iter {
        let start_j = h_columns.len();

        for j in start_j..ncv {
            let vj = &v_basis[j];
            let mut w = a.spmv(vj)?;

            // Modified Gram-Schmidt
            let mut h_col = vec![0.0f64; j + 2];
            for i in 0..=j {
                let h_ij = vector_dot(client, &v_basis[i], &w)?;
                h_col[i] = h_ij;
                if h_ij.abs() > REORTH_TOL {
                    let scaled = client.mul_scalar(&v_basis[i], h_ij)?;
                    w = client.sub(&w, &scaled)?;
                }
            }

            // Re-orthogonalization pass
            for i in 0..=j {
                let h_corr = vector_dot(client, &v_basis[i], &w)?;
                if h_corr.abs() > REORTH_TOL {
                    h_col[i] += h_corr;
                    let scaled = client.mul_scalar(&v_basis[i], h_corr)?;
                    w = client.sub(&w, &scaled)?;
                }
            }

            let h_next = vector_norm(client, &w)?;
            h_col[j + 1] = h_next;
            h_columns.push(h_col);

            if h_next < INVARIANT_SUBSPACE_TOL {
                break;
            }

            let v_next = client.mul_scalar(&w, 1.0 / h_next)?;
            if v_basis.len() <= j + 1 {
                v_basis.push(v_next);
            } else {
                v_basis[j + 1] = v_next;
            }
        }

        let m = h_columns.len();
        if m == 0 {
            return Err(Error::Internal("Arnoldi produced no vectors".to_string()));
        }

        // Build dense Hessenberg matrix
        let mut h_dense = vec![vec![0.0f64; m]; m];
        for j in 0..m {
            for i in 0..h_columns[j].len().min(m) {
                h_dense[i][j] = h_columns[j][i];
            }
        }

        let (eig_real, eig_imag, schur_vecs) = hessenberg_eig(&h_dense, m);
        let indices = select_eigenvalues_complex(&eig_real, &eig_imag, k.min(m), &options.which);

        // Check convergence: error ≈ |h_{m+1,m} * y_m|
        let last_h = if m > 0 && h_columns[m - 1].len() > m {
            h_columns[m - 1][m]
        } else {
            0.0
        };

        let mut nconv = 0;
        for &idx in &indices {
            if idx < schur_vecs.len() && !schur_vecs[idx].is_empty() {
                let last_comp = schur_vecs[idx].last().copied().unwrap_or(0.0);
                let error = (last_h * last_comp).abs();
                if error < options.tol {
                    nconv += 1;
                }
            }
        }

        if nconv >= k || m >= n {
            return build_result(
                client,
                &v_basis,
                &eig_real,
                &eig_imag,
                &schur_vecs,
                &indices,
                k,
                m,
                n,
                dtype,
                device,
                restart_iter + 1,
                nconv >= k,
                nconv,
            );
        }

        // Thick restart: keep k wanted Ritz vectors
        thick_restart(
            client,
            &mut v_basis,
            &mut h_columns,
            a,
            &schur_vecs,
            &indices,
            k,
            m,
            n,
            dtype,
            device,
        )?;
    }

    Err(Error::Internal(format!(
        "Arnoldi eigensolver did not converge after {} restart cycles",
        options.max_iter
    )))
}

/// Build the final eigensolver result from converged Ritz pairs.
///
/// Ritz vectors are accumulated on-device using [`accumulate_basis_combination`],
/// then assembled into the output matrix with O(k) final transfers.
#[allow(clippy::too_many_arguments)]
fn build_result<R, C>(
    client: &C,
    v_basis: &[Tensor<R>],
    eig_real: &[f64],
    eig_imag: &[f64],
    schur_vecs: &[Vec<f64>],
    indices: &[usize],
    k: usize,
    m: usize,
    n: usize,
    dtype: DType,
    device: &R::Device,
    iterations: usize,
    converged: bool,
    nconv: usize,
) -> Result<SparseEigComplexResult<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ScalarOps<R>,
{
    let k_actual = k.min(indices.len());
    let mut eig_real_out = Vec::with_capacity(k_actual);
    let mut eig_imag_out = Vec::with_capacity(k_actual);
    let mut ritz_vectors: Vec<Tensor<R>> = Vec::with_capacity(k_actual);

    for &idx in indices.iter().take(k_actual) {
        eig_real_out.push(eig_real[idx]);
        eig_imag_out.push(eig_imag[idx]);

        if idx < schur_vecs.len() {
            let y = &schur_vecs[idx];
            let ritz = accumulate_basis_combination(
                client,
                &v_basis[..y.len().min(m)],
                y,
                n,
                dtype,
                device,
            )?;
            ritz_vectors.push(ritz);
        } else {
            ritz_vectors.push(Tensor::<R>::try_zeros(&[n], dtype, device)?);
        }
    }

    let eigenvalues_real = Tensor::<R>::try_from_slice(&eig_real_out, &[k_actual], device)?;
    let eigenvalues_imag = Tensor::<R>::try_from_slice(&eig_imag_out, &[k_actual], device)?;

    // Assemble eigenvector columns into [n, k] matrix
    let mut flat = vec![0.0f64; n * k_actual];
    for (col, ritz) in ritz_vectors.iter().enumerate() {
        let col_data: Vec<f64> = ritz.to_vec();
        flat[col * n..(col + 1) * n].copy_from_slice(&col_data);
    }
    let eigenvectors = Tensor::<R>::try_from_slice(&flat, &[n, k_actual], device)?;

    Ok(SparseEigComplexResult {
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors,
        iterations,
        converged,
        nconv,
    })
}

/// Perform thick restart: keep k wanted Ritz vectors, re-orthogonalize,
/// rebuild Hessenberg entries, and generate continuation vector.
#[allow(clippy::too_many_arguments)]
fn thick_restart<R, C>(
    client: &C,
    v_basis: &mut Vec<Tensor<R>>,
    h_columns: &mut Vec<Vec<f64>>,
    a: &CsrData<R>,
    schur_vecs: &[Vec<f64>],
    indices: &[usize],
    k: usize,
    m: usize,
    n: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<()>
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
    let k_keep = k.min(indices.len());
    let mut new_basis: Vec<Tensor<R>> = Vec::with_capacity(k_keep);

    for &idx in indices.iter().take(k_keep) {
        if idx < schur_vecs.len() {
            let y = &schur_vecs[idx];
            let mut ritz = accumulate_basis_combination(
                client,
                &v_basis[..y.len().min(m)],
                y,
                n,
                dtype,
                device,
            )?;
            let ritz_norm = vector_norm(client, &ritz)?;
            if ritz_norm > REORTH_TOL {
                ritz = client.mul_scalar(&ritz, 1.0 / ritz_norm)?;
            }
            new_basis.push(ritz);
        }
    }

    // Re-orthogonalize the kept basis
    for i in 1..new_basis.len() {
        let mut vi = new_basis[i].clone();
        for j in 0..i {
            let h = vector_dot(client, &new_basis[j], &vi)?;
            if h.abs() > REORTH_TOL {
                let scaled = client.mul_scalar(&new_basis[j], h)?;
                vi = client.sub(&vi, &scaled)?;
            }
        }
        let vi_norm = vector_norm(client, &vi)?;
        if vi_norm > REORTH_TOL {
            vi = client.mul_scalar(&vi, 1.0 / vi_norm)?;
        }
        new_basis[i] = vi;
    }

    // Rebuild H and generate continuation vector
    *v_basis = new_basis;
    h_columns.clear();

    let k_len = v_basis.len();
    for j in 0..k_len {
        let vj = &v_basis[j];
        let av = a.spmv(vj)?;
        let mut h_col = vec![0.0f64; j + 2];
        for i in 0..=j {
            h_col[i] = vector_dot(client, &v_basis[i], &av)?;
        }
        if j + 1 < k_len {
            h_col[j + 1] = vector_dot(client, &v_basis[j + 1], &av)?;
        } else {
            // Last kept vector: compute residual to get continuation vector v_{k+1}
            let mut w = av;
            for i in 0..=j {
                if h_col[i].abs() > REORTH_TOL {
                    let scaled = client.mul_scalar(&v_basis[i], h_col[i])?;
                    w = client.sub(&w, &scaled)?;
                }
            }
            let h_next = vector_norm(client, &w)?;
            h_col[j + 1] = h_next;
            if h_next > INVARIANT_SUBSPACE_TOL {
                let v_next = client.mul_scalar(&w, 1.0 / h_next)?;
                v_basis.push(v_next);
            }
        }
        h_columns.push(h_col);
    }

    Ok(())
}
