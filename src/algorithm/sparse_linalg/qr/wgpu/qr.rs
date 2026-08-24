//! WebGPU sparse QR public API: factorize and simple
//!
//! F32 only. Delegates GPU factorization to `factorize.rs`, solve to `solve.rs`.

#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::qr::symbolic::sparse_qr_symbolic;
#[cfg(feature = "wgpu")]
use crate::algorithm::sparse_linalg::qr::types::{QrFactors, QrOptions};
#[cfg(feature = "wgpu")]
use crate::dtype::DType;
#[cfg(feature = "wgpu")]
use crate::error::{Error, Result};
#[cfg(feature = "wgpu")]
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
#[cfg(feature = "wgpu")]
use crate::sparse::CscData;

/// Sparse QR factorization with precomputed symbolic information (WebGPU)
///
/// F32 only. Uses GPU kernels with zero intermediate transfers.
#[cfg(feature = "wgpu")]
pub fn sparse_qr_wgpu(
    client: &WgpuClient,
    a: &CscData<WgpuRuntime>,
    symbolic: &crate::algorithm::sparse_linalg::qr::types::QrSymbolic,
    options: &QrOptions,
) -> Result<QrFactors<WgpuRuntime>> {
    let [m, n] = a.shape;
    let dtype = a.values().dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "sparse_qr_wgpu",
        });
    }

    if m != symbolic.m || n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.m, symbolic.n],
            got: vec![m, n],
        });
    }

    if m < n {
        return Err(Error::Internal("sparse_qr: requires m >= n".to_string()));
    }

    super::factorize::run_factorization_wgpu(client, a, symbolic, options)
}

/// Sparse QR factorization without precomputed symbolic information (WebGPU)
///
/// `col_ptrs_host` and `row_indices_host` must be the CPU-resident structural
/// data for `a` (the same values used to construct `a` via `CscData::from_slices`).
/// These are kept CPU-side to avoid GPU→CPU transfers during symbolic analysis,
/// which requires irregular graph traversal and runs on the CPU.
#[cfg(feature = "wgpu")]
pub fn sparse_qr_simple_wgpu(
    client: &WgpuClient,
    a: &CscData<WgpuRuntime>,
    col_ptrs_host: &[i64],
    row_indices_host: &[i64],
    options: &QrOptions,
) -> Result<QrFactors<WgpuRuntime>> {
    let [m, n] = a.shape;
    let symbolic = sparse_qr_symbolic(col_ptrs_host, row_indices_host, m, n, options)?;
    sparse_qr_wgpu(client, a, &symbolic, options)
}

#[cfg(test)]
#[cfg(feature = "wgpu")]
mod tests {
    use super::super::sparse_qr_solve_wgpu;
    use super::*;
    use crate::tensor::Tensor;

    fn wgpu_device() -> <WgpuRuntime as crate::runtime::Runtime>::Device {
        <WgpuRuntime as crate::runtime::Runtime>::Device::default()
    }

    fn get_wgpu_client() -> WgpuClient {
        WgpuClient::new(wgpu_device()).expect("WGPU device required")
    }

    #[test]
    fn test_sparse_qr_wgpu_simple_square() {
        let device = wgpu_device();
        let client = get_wgpu_client();

        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f32, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let a =
            CscData::<WgpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        let options = QrOptions::no_ordering();
        let factors =
            sparse_qr_simple_wgpu(&client, &a, &col_ptrs, &row_indices, &options).unwrap();

        assert_eq!(factors.rank, 4);
        // GPU factorization keeps Householder data GPU-resident only
        assert!(factors.gpu_householder_values.is_some());
        assert!(factors.gpu_tau.is_some());
    }

    #[test]
    fn test_sparse_qr_wgpu_solve() {
        let device = wgpu_device();
        let client = get_wgpu_client();

        let col_ptrs = vec![0i64, 2, 5, 8, 10];
        let row_indices = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
        let values = vec![4.0f32, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0];
        let a =
            CscData::<WgpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [4, 4], &device)
                .unwrap();

        let options = QrOptions::no_ordering();
        let factors =
            sparse_qr_simple_wgpu(&client, &a, &col_ptrs, &row_indices, &options).unwrap();

        let b = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let x = sparse_qr_solve_wgpu(&client, &factors, &b).unwrap();
        let x_vals: Vec<f32> = x.to_vec();

        let a_dense: &[&[f32]] = &[
            &[4.0, 1.0, 0.0, 0.0],
            &[1.0, 4.0, 1.0, 0.0],
            &[0.0, 1.0, 4.0, 1.0],
            &[0.0, 0.0, 1.0, 4.0],
        ];
        let b_vals = [1.0f32, 2.0, 3.0, 4.0];
        for i in 0..4 {
            let mut ax_i: f32 = 0.0;
            for j in 0..4 {
                ax_i += a_dense[i][j] * x_vals[j];
            }
            assert!(
                (ax_i - b_vals[i]).abs() < 1e-4,
                "A*x[{}] = {}, expected {}",
                i,
                ax_i,
                b_vals[i]
            );
        }
    }
}
