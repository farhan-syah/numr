//! Integration tests for relaxation and advanced iterative solvers (Jacobi, SOR, LGMRES, QMR, AMG)

use numr::algorithm::iterative::{
    IterativeSolvers, JacobiOptions, LgmresOptions, PreconditionerType, QmrOptions, SorOptions,
};
use numr::ops::BinaryOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::sparse::CsrData;
use numr::tensor::Tensor;

use super::iterative_solvers::{create_1d_laplacian, create_nonsymmetric, get_client};

/// Create diagonally dominant matrix: 4I + Laplacian
fn create_diag_dominant(n: usize, device: &<CpuRuntime as Runtime>::Device) -> CsrData<CpuRuntime> {
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
        values.push(6.0f64);
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

// ============================================================================
// Jacobi Tests
// ============================================================================

#[test]
fn test_jacobi_diag_dominant() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_diag_dominant(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .jacobi(
            &a,
            &b,
            None,
            JacobiOptions {
                max_iter: 500,
                omega: 2.0 / 3.0,
                rtol: 1e-10,
                atol: 1e-14,
            },
        )
        .expect("Jacobi should succeed");

    assert!(result.converged, "Jacobi should converge on diag dominant");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

// ============================================================================
// SOR Tests
// ============================================================================

#[test]
fn test_sor_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .sor(
            &a,
            &b,
            None,
            SorOptions {
                max_iter: 500,
                omega: 1.5,
                rtol: 1e-10,
                atol: 1e-14,
            },
        )
        .expect("SOR should succeed");

    assert!(result.converged, "SOR should converge on Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_sor_vs_jacobi() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_diag_dominant(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let jacobi_result = client
        .jacobi(
            &a,
            &b,
            None,
            JacobiOptions {
                max_iter: 500,
                omega: 2.0 / 3.0,
                ..Default::default()
            },
        )
        .expect("Jacobi should succeed");

    let sor_result = client
        .sor(
            &a,
            &b,
            None,
            SorOptions {
                max_iter: 500,
                omega: 1.2,
                ..Default::default()
            },
        )
        .expect("SOR should succeed");

    assert!(jacobi_result.converged);
    assert!(sor_result.converged);
    assert!(
        sor_result.iterations <= jacobi_result.iterations,
        "SOR ({}) should converge no slower than Jacobi ({})",
        sor_result.iterations,
        jacobi_result.iterations
    );
}

// ============================================================================
// LGMRES Tests
// ============================================================================

#[test]
fn test_lgmres_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .lgmres(
            &a,
            &b,
            None,
            LgmresOptions {
                max_iter: 100,
                restart: 10,
                k_aug: 3,
                rtol: 1e-10,
                atol: 1e-14,
                preconditioner: PreconditionerType::None,
            },
        )
        .expect("LGMRES should succeed");

    assert!(result.converged, "LGMRES should converge");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_lgmres_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .lgmres(&a, &b, None, LgmresOptions::default())
        .expect("LGMRES should succeed");

    assert!(result.converged, "LGMRES should converge on Laplacian");
}

// ============================================================================
// QMR Tests
// ============================================================================

#[test]
fn test_qmr_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .qmr(
            &a,
            &b,
            None,
            QmrOptions {
                max_iter: 200,
                rtol: 1e-10,
                atol: 1e-14,
                preconditioner: PreconditionerType::None,
            },
        )
        .expect("QMR should succeed");

    assert!(result.converged, "QMR should converge");
    assert!(
        result.residual_norm < 1e-6,
        "Residual: {}",
        result.residual_norm
    );

    let ax = a.spmv(&result.solution).expect("spmv");
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-6, "Verification residual: {}", res_norm);
}

#[test]
fn test_qmr_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .qmr(&a, &b, None, QmrOptions::default())
        .expect("QMR should succeed");

    assert!(result.converged, "QMR should converge on SPD Laplacian");
}

// ============================================================================
// AMG Tests
// ============================================================================

#[test]
fn test_amg_preconditioned_cg() {
    use numr::algorithm::iterative::{AmgOptions, amg_preconditioned_cg, amg_setup};

    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 20;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[n], device).unwrap();

    let hierarchy =
        amg_setup(&client, &a, AmgOptions::default()).expect("AMG setup should succeed");

    let (solution, iterations, residual_norm, converged) =
        amg_preconditioned_cg(&client, &a, &b, None, &hierarchy, 100, 1e-10, 1e-14)
            .expect("AMG-CG should succeed");

    assert!(converged, "AMG-preconditioned CG should converge");
    assert!(residual_norm < 1e-8, "Residual: {}", residual_norm);
    assert!(iterations > 0, "Should take at least 1 iteration");

    let ax = a.spmv(&solution).expect("spmv");
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}
