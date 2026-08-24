//! CPU implementation of sparse Householder QR factorization
//!
//! Column-wise left-looking Householder QR with partial pivoting (rank detection).

use crate::algorithm::sparse_linalg::qr::symbolic::sparse_qr_symbolic;
use crate::algorithm::sparse_linalg::qr::types::{QrFactors, QrMetrics, QrOptions, QrSymbolic};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CscData;
use crate::tensor::Tensor;

use super::algorithm::{apply_qt, back_substitute, householder_qr};
use super::helpers::{
    create_r_tensor, create_vector_tensor, extract_values_f64, extract_values_f64_tensor,
};

/// Sparse QR factorization with precomputed symbolic information (CPU)
pub fn sparse_qr_cpu<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &QrSymbolic,
    options: &QrOptions,
) -> Result<QrFactors<R>> {
    let (factors, _metrics) = sparse_qr_cpu_with_metrics(a, symbolic, options)?;
    Ok(factors)
}

/// Sparse QR factorization with metrics (CPU)
pub fn sparse_qr_cpu_with_metrics<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    symbolic: &QrSymbolic,
    options: &QrOptions,
) -> Result<(QrFactors<R>, QrMetrics)> {
    let [m, n] = a.shape;

    if m != symbolic.m || n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.m, symbolic.n],
            got: vec![m, n],
        });
    }

    if m < n {
        return Err(Error::Internal(
            "sparse_qr: requires m >= n (more rows than columns)".to_string(),
        ));
    }

    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();
    let values = extract_values_f64(a)?;

    let result = householder_qr(
        m,
        n,
        &col_ptrs,
        &row_indices,
        &values,
        &symbolic.col_perm,
        options,
    )?;

    let device = a.values().device();
    let dtype = a.values().dtype();

    let r = create_r_tensor::<R>(
        m,
        n,
        &result.r_col_ptrs,
        &result.r_row_indices,
        &result.r_values,
        dtype,
        device,
    )?;

    let original_nnz = values.len();
    let r_nnz = result.r_values.len();

    let factors = QrFactors {
        householder_vectors: result.householder_vectors,
        tau: result.tau,
        r,
        col_perm: symbolic.col_perm.clone(),
        rank: result.rank,
        gpu_householder_values: None,
        gpu_tau: None,
    };

    let metrics = QrMetrics {
        original_nnz,
        r_nnz,
        fill_ratio: if original_nnz > 0 {
            r_nnz as f64 / original_nnz as f64
        } else {
            0.0
        },
        numerical_rank: result.rank,
    };

    Ok((factors, metrics))
}

/// Sparse QR factorization without precomputed symbolic information (CPU)
pub fn sparse_qr_simple_cpu<R: Runtime<DType = DType>>(
    a: &CscData<R>,
    options: &QrOptions,
) -> Result<QrFactors<R>> {
    let [m, n] = a.shape;
    let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
    let row_indices: Vec<i64> = a.row_indices().to_vec();

    let symbolic = sparse_qr_symbolic(&col_ptrs, &row_indices, m, n, options)?;
    sparse_qr_cpu(a, &symbolic, options)
}

/// Solve A*x = b using precomputed QR factors (square full-rank systems)
///
/// Computes x = P * R^{-1} * Q^T * b
pub fn sparse_qr_solve_cpu<R: Runtime<DType = DType>>(
    factors: &QrFactors<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>> {
    let [m, n] = factors.r.shape;
    let b_shape = b.shape();

    if b_shape.is_empty() || b_shape[0] != m {
        return Err(Error::ShapeMismatch {
            expected: vec![m],
            got: b_shape.to_vec(),
        });
    }

    if factors.rank < n {
        return Err(Error::Internal(format!(
            "sparse_qr_solve: matrix is rank-deficient (rank {} < n {})",
            factors.rank, n
        )));
    }

    let b_vals = extract_values_f64_tensor(b)?;

    // Step 1: Compute Q^T * b by applying Householder reflectors
    let mut qtb = b_vals;
    apply_qt(&factors.householder_vectors, &factors.tau, &mut qtb);

    // Step 2: Back-substitute R * x = (Q^T * b)[0:n]
    let r_col_ptrs: Vec<i64> = factors.r.col_ptrs().to_vec();
    let r_row_indices: Vec<i64> = factors.r.row_indices().to_vec();
    let r_values = extract_values_f64(&factors.r)?;

    let mut x = vec![0.0f64; n];
    back_substitute(n, &r_col_ptrs, &r_row_indices, &r_values, &qtb[..n], &mut x)?;

    // Step 3: Apply column permutation: x_orig[col_perm[k]] = x[k]
    let mut x_perm = vec![0.0f64; n];
    for (k, &orig_col) in factors.col_perm.iter().enumerate() {
        x_perm[orig_col] = x[k];
    }

    create_vector_tensor::<R>(&x_perm, b.dtype(), b.device())
}

/// Solve least-squares min ||A*x - b||_2 using QR factors (overdetermined systems)
///
/// For m > n: x = P * R[0:n, 0:n]^{-1} * (Q^T * b)[0:n]
pub fn sparse_qr_least_squares_cpu<R: Runtime<DType = DType>>(
    factors: &QrFactors<R>,
    b: &Tensor<R>,
) -> Result<Tensor<R>> {
    let [m, n] = factors.r.shape;
    let b_shape = b.shape();

    if b_shape.is_empty() || b_shape[0] != m {
        return Err(Error::ShapeMismatch {
            expected: vec![m],
            got: b_shape.to_vec(),
        });
    }

    if factors.rank == 0 {
        return Err(Error::Internal(
            "sparse_qr_least_squares: matrix has zero rank".to_string(),
        ));
    }

    let b_vals = extract_values_f64_tensor(b)?;

    // Step 1: Compute Q^T * b
    let mut qtb = b_vals;
    apply_qt(&factors.householder_vectors, &factors.tau, &mut qtb);

    // Step 2: Back-substitute R[0:rank, 0:rank] * x = (Q^T * b)[0:rank]
    let r_col_ptrs: Vec<i64> = factors.r.col_ptrs().to_vec();
    let r_row_indices: Vec<i64> = factors.r.row_indices().to_vec();
    let r_values = extract_values_f64(&factors.r)?;

    let rank = factors.rank;
    let mut x = vec![0.0f64; n];
    back_substitute(
        rank,
        &r_col_ptrs,
        &r_row_indices,
        &r_values,
        &qtb[..rank],
        &mut x,
    )?;
    // Columns rank..n remain zero (minimum-norm solution)

    // Step 3: Apply column permutation
    let mut x_perm = vec![0.0f64; n];
    for (k, &orig_col) in factors.col_perm.iter().enumerate() {
        if k < n {
            x_perm[orig_col] = x[k];
        }
    }

    create_vector_tensor::<R>(&x_perm, b.dtype(), b.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuRuntime;

    fn cpu_device() -> <CpuRuntime as Runtime>::Device {
        <CpuRuntime as Runtime>::Device::default()
    }

    /// Create a 4x4 tridiagonal SPD matrix in CSC format
    fn create_tridiagonal_4x4() -> CscData<CpuRuntime> {
        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        CscData::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &cpu_device()).unwrap()
    }

    /// Create a 5x3 overdetermined matrix in CSC format
    fn create_overdetermined_5x3() -> CscData<CpuRuntime> {
        let col_ptrs = vec![0i64, 3, 6, 8];
        let row_indices = vec![0i64, 2, 4, 1, 3, 4, 0, 3];
        let values = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        CscData::from_slices(&col_ptrs, &row_indices, &values, [5, 3], &cpu_device()).unwrap()
    }

    fn verify_ax_eq_b(a_dense: &[&[f64]], x: &[f64], b: &[f64]) {
        let m = a_dense.len();
        let n = x.len();
        for i in 0..m {
            let mut ax_i = 0.0;
            for j in 0..n {
                ax_i += a_dense[i][j] * x[j];
            }
            assert!(
                (ax_i - b[i]).abs() < 1e-10,
                "A*x[{}] = {}, expected {}",
                i,
                ax_i,
                b[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_simple_square() {
        let a = create_tridiagonal_4x4();
        let options = QrOptions::no_ordering();
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        assert_eq!(factors.rank, 4);
        assert_eq!(factors.householder_vectors.len(), 4);
        assert_eq!(factors.tau.len(), 4);
    }

    #[test]
    fn test_sparse_qr_solve_square() {
        let a = create_tridiagonal_4x4();
        let options = QrOptions::no_ordering();
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &cpu_device())
            .unwrap();
        let x = sparse_qr_solve_cpu(&factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        let a_dense: &[&[f64]] = &[
            &[4.0, 1.0, 0.0, 0.0],
            &[1.0, 4.0, 1.0, 0.0],
            &[0.0, 1.0, 4.0, 1.0],
            &[0.0, 0.0, 1.0, 4.0],
        ];
        verify_ax_eq_b(a_dense, &x_vals, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_sparse_qr_overdetermined_least_squares() {
        let a = create_overdetermined_5x3();
        let options = QrOptions::no_ordering();
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        assert_eq!(factors.rank, 3);

        let b = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f64, 2.0, 3.0, 4.0, 5.0],
            &[5],
            &cpu_device(),
        )
        .unwrap();
        let x = sparse_qr_least_squares_cpu(&factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        // Verify optimality: A^T * (A*x - b) ≈ 0
        let a_dense = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ];
        let b_vals = [1.0, 2.0, 3.0, 4.0, 5.0];

        let mut residual = vec![0.0f64; 5];
        for i in 0..5 {
            for j in 0..3 {
                residual[i] += a_dense[i][j] * x_vals[j];
            }
            residual[i] -= b_vals[i];
        }

        for j in 0..3 {
            let mut at_r = 0.0;
            for i in 0..5 {
                at_r += a_dense[i][j] * residual[i];
            }
            assert!(
                at_r.abs() < 1e-10,
                "A^T * residual[{}] = {}, expected ~0",
                j,
                at_r
            );
        }
    }

    #[test]
    fn test_sparse_qr_rank_deficient() {
        // Rank-2 matrix (3x3) where col 2 = col 0 + col 1
        let col_ptrs = vec![0i64, 2, 4, 7];
        let row_indices = vec![0i64, 2, 1, 2, 0, 1, 2];
        let values = vec![1.0f64, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0];
        let a = CscData::<CpuRuntime>::from_slices(
            &col_ptrs,
            &row_indices,
            &values,
            [3, 3],
            &cpu_device(),
        )
        .unwrap();

        let options = QrOptions::no_ordering();
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        assert!(
            factors.rank < 3,
            "Expected rank < 3, got rank = {}",
            factors.rank
        );
    }

    #[test]
    fn test_sparse_qr_with_colamd() {
        let a = create_tridiagonal_4x4();
        let options = QrOptions::default(); // Uses Colamd
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        assert_eq!(factors.rank, 4);

        let b = Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 0.0, 0.0, 0.0], &[4], &cpu_device())
            .unwrap();
        let x = sparse_qr_solve_cpu(&factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        let a_dense: &[&[f64]] = &[
            &[4.0, 1.0, 0.0, 0.0],
            &[1.0, 4.0, 1.0, 0.0],
            &[0.0, 1.0, 4.0, 1.0],
            &[0.0, 0.0, 1.0, 4.0],
        ];
        verify_ax_eq_b(a_dense, &x_vals, &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparse_qr_known_diagonal() {
        // 2x2 identity matrix: QR should give R = I
        let col_ptrs = vec![0i64, 1, 2];
        let row_indices = vec![0i64, 1];
        let values = vec![1.0f64, 1.0];
        let a = CscData::<CpuRuntime>::from_slices(
            &col_ptrs,
            &row_indices,
            &values,
            [2, 2],
            &cpu_device(),
        )
        .unwrap();

        let options = QrOptions::no_ordering();
        let factors = sparse_qr_simple_cpu(&a, &options).unwrap();

        assert_eq!(factors.rank, 2);

        // R diagonal should be ±1
        let r_values: Vec<f64> = factors.r.values().to_vec();
        let r_row_indices: Vec<i64> = factors.r.row_indices().to_vec();
        let r_col_ptrs: Vec<i64> = factors.r.col_ptrs().to_vec();

        for col in 0..2 {
            let start = r_col_ptrs[col] as usize;
            let end = r_col_ptrs[col + 1] as usize;
            for idx in start..end {
                if r_row_indices[idx] as usize == col {
                    assert!(
                        (r_values[idx].abs() - 1.0).abs() < 1e-10,
                        "R[{},{}] = {}, expected ±1",
                        r_row_indices[idx],
                        col,
                        r_values[idx]
                    );
                }
            }
        }
    }

    #[test]
    fn test_sparse_qr_metrics() {
        let a = create_tridiagonal_4x4();
        let options = QrOptions::no_ordering();

        let col_ptrs: Vec<i64> = a.col_ptrs().to_vec();
        let row_indices: Vec<i64> = a.row_indices().to_vec();
        let symbolic = sparse_qr_symbolic(&col_ptrs, &row_indices, 4, 4, &options).unwrap();

        let (factors, metrics) = sparse_qr_cpu_with_metrics(&a, &symbolic, &options).unwrap();

        assert_eq!(metrics.original_nnz, 10);
        assert_eq!(metrics.numerical_rank, 4);
        assert!(metrics.r_nnz > 0);
        assert!(metrics.fill_ratio > 0.0);
        assert_eq!(factors.rank, 4);
    }
}
