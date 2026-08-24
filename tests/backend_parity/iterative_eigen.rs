//! Integration tests for sparse eigensolvers and SVD

use numr::algorithm::iterative::{
    IterativeSolvers, SparseEigOptions, SvdsOptions, WhichEigenvalues, WhichSingularValues,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::sparse::CsrData;
use numr::tensor::Tensor;

fn get_client() -> CpuClient {
    let device = CpuRuntime::default_device();
    CpuRuntime::default_client(&device)
}

fn create_1d_laplacian(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64);
        }
        col_indices.push(i as i64);
        values.push(2.0f64);
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.0f64);
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor =
        Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
    let col_indices_tensor =
        Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device).unwrap();
    let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device).unwrap();

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

fn create_nonsymmetric(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64);
        }
        col_indices.push(i as i64);
        values.push(3.0f64);
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.5f64);
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor =
        Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
    let col_indices_tensor =
        Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device).unwrap();
    let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device).unwrap();

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

// ============================================================================
// Lanczos Eigensolver Tests
// ============================================================================

#[test]
fn test_lanczos_laplacian_largest() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 5;
    let a = create_1d_laplacian(n, device);

    let result = client
        .sparse_eig_symmetric(
            &a,
            2,
            SparseEigOptions {
                which: WhichEigenvalues::LargestMagnitude,
                tol: 1e-8,
                ..Default::default()
            },
        )
        .expect("Lanczos should succeed");

    assert!(
        result.converged,
        "Lanczos should converge for 5x5 Laplacian"
    );
    assert_eq!(result.nconv, 2, "Should find 2 eigenvalues");

    let eig_data: Vec<f64> = result.eigenvalues.to_vec();
    let max_eig = eig_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (max_eig - 3.732).abs() < 0.1,
        "Largest eigenvalue ≈ 3.732, got {}",
        max_eig
    );
}

#[test]
fn test_lanczos_identity_eigenvalues() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 4;
    let row_ptrs = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4], &[n + 1], device).unwrap();
    let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3], &[n], device).unwrap();
    let values = Tensor::<CpuRuntime>::from_slice(&[1.0f64; 4], &[n], device).unwrap();
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let result = client
        .sparse_eig_symmetric(
            &a,
            2,
            SparseEigOptions {
                tol: 1e-8,
                ..Default::default()
            },
        )
        .expect("Lanczos on identity should succeed");

    let eig_data: Vec<f64> = result.eigenvalues.to_vec();
    for &ev in &eig_data {
        assert!(
            (ev - 1.0).abs() < 1e-6,
            "Identity eigenvalue should be 1.0, got {}",
            ev
        );
    }
}

// ============================================================================
// Arnoldi Eigensolver Tests
// ============================================================================

#[test]
fn test_arnoldi_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 6;
    let a = create_nonsymmetric(n, device);

    let result = client
        .sparse_eig(
            &a,
            2,
            SparseEigOptions {
                which: WhichEigenvalues::LargestMagnitude,
                tol: 1e-6,
                max_iter: 100,
                ..Default::default()
            },
        )
        .expect("Arnoldi should succeed");

    assert!(result.nconv >= 1, "Should find at least 1 eigenvalue");

    let eig_real: Vec<f64> = result.eigenvalues_real.to_vec();
    for &er in &eig_real {
        assert!(
            er > 0.0,
            "Eigenvalue real part should be positive, got {}",
            er
        );
    }
}

#[test]
fn test_arnoldi_symmetric_matches_lanczos() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 5;
    let a = create_1d_laplacian(n, device);

    let lanczos_result = client
        .sparse_eig_symmetric(
            &a,
            2,
            SparseEigOptions {
                which: WhichEigenvalues::LargestMagnitude,
                tol: 1e-8,
                ..Default::default()
            },
        )
        .expect("Lanczos should succeed");

    let arnoldi_result = client
        .sparse_eig(
            &a,
            2,
            SparseEigOptions {
                which: WhichEigenvalues::LargestMagnitude,
                tol: 1e-6,
                max_iter: 100,
                ..Default::default()
            },
        )
        .expect("Arnoldi should succeed");

    let lanczos_eigs: Vec<f64> = lanczos_result.eigenvalues.to_vec();
    let arnoldi_eigs: Vec<f64> = arnoldi_result.eigenvalues_real.to_vec();

    let lanczos_max = lanczos_eigs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let arnoldi_max = arnoldi_eigs
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (lanczos_max - arnoldi_max).abs() < 0.5,
        "Lanczos ({}) vs Arnoldi ({}) largest eigenvalue",
        lanczos_max,
        arnoldi_max
    );
}

// ============================================================================
// Sparse SVD Tests
// ============================================================================

#[test]
fn test_svds_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 8;
    let a = create_1d_laplacian(n, device);

    let result = client
        .svds(
            &a,
            2,
            SvdsOptions {
                ncv: Some(10),
                max_iter: 100,
                tol: 1e-6,
                which: WhichSingularValues::Largest,
            },
        )
        .expect("SVDs should succeed");

    assert!(result.nconv >= 1, "Should find at least 1 singular value");

    let sigma: Vec<f64> = result.singular_values.to_vec();
    let max_sigma = sigma.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert!(
        max_sigma > 3.0,
        "Largest singular value should be > 3, got {}",
        max_sigma
    );
}

#[test]
fn test_svds_identity() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 5;
    let row_ptrs =
        Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4, 5], &[n + 1], device).unwrap();
    let col_indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 3, 4], &[n], device).unwrap();
    let values = Tensor::<CpuRuntime>::from_slice(&[2.0f64; 5], &[n], device).unwrap();
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let result = client
        .svds(
            &a,
            2,
            SvdsOptions {
                ncv: Some(8),
                max_iter: 100,
                tol: 1e-8,
                which: WhichSingularValues::Largest,
            },
        )
        .expect("SVDs on scaled identity should succeed");

    let sigma: Vec<f64> = result.singular_values.to_vec();
    for &s in &sigma {
        assert!(
            (s - 2.0).abs() < 1e-4,
            "Singular values of 2*I should be 2.0, got {}",
            s
        );
    }
}
