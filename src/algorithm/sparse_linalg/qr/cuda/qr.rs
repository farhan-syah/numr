//! CUDA sparse QR public API: factorize and simple
//!
//! Delegates GPU factorization to `factorize.rs`, solve to `solve.rs`.

use crate::algorithm::sparse_linalg::qr::symbolic::sparse_qr_symbolic;
use crate::algorithm::sparse_linalg::qr::types::{QrFactors, QrOptions, QrSymbolic};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::sparse::CscData;

use super::factorize::run_factorization;

/// Sparse QR factorization with precomputed symbolic information (CUDA)
///
/// Uses GPU kernels with zero intermediate transfers. Householder vectors and tau
/// stay GPU-resident. Only R structural data (diag, off-diag) transferred to CPU
/// for CSC construction.
pub fn sparse_qr_cuda(
    client: &CudaClient,
    a: &CscData<CudaRuntime>,
    symbolic: &QrSymbolic,
    options: &QrOptions,
) -> Result<QrFactors<CudaRuntime>> {
    let [m, n] = a.shape;
    let dtype = a.values().dtype();

    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr_cuda",
        });
    }

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

    match dtype {
        DType::F32 => run_factorization::<f32>(client, a, symbolic, options, m, n),
        DType::F64 => run_factorization::<f64>(client, a, symbolic, options, m, n),
        _ => unreachable!(),
    }
}

/// Sparse QR factorization without precomputed symbolic information (CUDA)
///
/// `col_ptrs_host` and `row_indices_host` must be the CPU-resident structural
/// data for `a` (the same values used to construct `a` via `CscData::from_slices`).
/// These are kept CPU-side to avoid GPU→CPU transfers during symbolic analysis,
/// which requires irregular graph traversal and runs on the CPU.
pub fn sparse_qr_simple_cuda(
    client: &CudaClient,
    a: &CscData<CudaRuntime>,
    col_ptrs_host: &[i64],
    row_indices_host: &[i64],
    options: &QrOptions,
) -> Result<QrFactors<CudaRuntime>> {
    let [m, n] = a.shape;
    let symbolic = sparse_qr_symbolic(col_ptrs_host, row_indices_host, m, n, options)?;
    sparse_qr_cuda(client, a, &symbolic, options)
}

#[cfg(test)]
mod tests {
    use super::super::sparse_qr_solve_cuda;
    use super::*;
    use crate::runtime::cuda::CudaDevice;
    use crate::tensor::Tensor;

    fn cuda_setup() -> Option<(<CudaRuntime as crate::runtime::Runtime>::Device, CudaClient)> {
        if !crate::runtime::cuda::is_cuda_available() {
            return None;
        }
        let device = <CudaRuntime as crate::runtime::Runtime>::Device::new(0);
        let client = CudaClient::new(CudaDevice::new(0)).expect("CUDA device required");
        Some((device, client))
    }

    #[test]
    fn test_sparse_qr_cuda_simple_square() {
        let Some((device, client)) = cuda_setup() else {
            return;
        };

        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let a =
            CscData::<CudaRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        let options = QrOptions::no_ordering();
        let factors =
            sparse_qr_simple_cuda(&client, &a, &col_ptrs, &row_indices, &options).unwrap();

        assert_eq!(factors.rank, 4);
        // GPU factorization keeps Householder data GPU-resident only
        assert!(factors.gpu_householder_values.is_some());
        assert!(factors.gpu_tau.is_some());
    }

    #[test]
    fn test_sparse_qr_cuda_solve() {
        let Some((device, client)) = cuda_setup() else {
            return;
        };

        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f64, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let a =
            CscData::<CudaRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        let options = QrOptions::no_ordering();
        let factors =
            sparse_qr_simple_cuda(&client, &a, &col_ptrs, &row_indices, &options).unwrap();

        let b =
            Tensor::<CudaRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let x = sparse_qr_solve_cuda(&client, &factors, &b).unwrap();
        let x_vals: Vec<f64> = x.to_vec();

        // Verify A*x ≈ b
        let a_dense: &[&[f64]] = &[
            &[4.0, 1.0, 0.0, 0.0],
            &[1.0, 4.0, 1.0, 0.0],
            &[0.0, 1.0, 4.0, 1.0],
            &[0.0, 0.0, 1.0, 4.0],
        ];
        let b_vals = [1.0, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let mut ax_i = 0.0;
            for j in 0..4 {
                ax_i += a_dense[i][j] * x_vals[j];
            }
            assert!(
                (ax_i - b_vals[i]).abs() < 1e-8,
                "A*x[{}] = {}, expected {}",
                i,
                ax_i,
                b_vals[i]
            );
        }
    }

    #[test]
    fn test_sparse_qr_cuda_f32() {
        let Some((device, client)) = cuda_setup() else {
            return;
        };

        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f32, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let a =
            CscData::<CudaRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        let options = QrOptions::no_ordering();
        let factors =
            sparse_qr_simple_cuda(&client, &a, &col_ptrs, &row_indices, &options).unwrap();

        assert_eq!(factors.rank, 4);
    }
}
