//! Integration tests for iterative solvers and sparse eigensolvers

use numr::algorithm::iterative::{
    CgOptions, CgsOptions, IterativeSolvers, MinresOptions, PreconditionerType,
};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
use numr::sparse::CsrData;
use numr::tensor::Tensor;

pub fn get_client() -> CpuClient {
    let device = CpuRuntime::default_device();
    CpuRuntime::default_client(&device)
}

/// Create 1D Laplacian (SPD tridiagonal): diag=2, off-diag=-1
pub fn create_1d_laplacian(
    n: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> CsrData<CpuRuntime> {
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
        Tensor::<CpuRuntime>::try_from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
    let col_indices_tensor =
        Tensor::<CpuRuntime>::try_from_slice(&col_indices, &[col_indices.len()], device).unwrap();
    let values_tensor =
        Tensor::<CpuRuntime>::try_from_slice(&values, &[values.len()], device).unwrap();

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

/// Create a non-symmetric sparse matrix for CGS/Arnoldi testing
/// Convection-diffusion: tridiagonal with asymmetric off-diags
pub fn create_nonsymmetric(
    n: usize,
    device: &<CpuRuntime as Runtime>::Device,
) -> CsrData<CpuRuntime> {
    let mut row_ptrs = Vec::with_capacity(n + 1);
    let mut col_indices = Vec::new();
    let mut values = Vec::new();

    row_ptrs.push(0i64);
    for i in 0..n {
        if i > 0 {
            col_indices.push((i - 1) as i64);
            values.push(-1.0f64); // diffusion
        }
        col_indices.push(i as i64);
        values.push(3.0f64); // diagonal dominance
        if i < n - 1 {
            col_indices.push((i + 1) as i64);
            values.push(-1.5f64); // convection (asymmetric)
        }
        row_ptrs.push(col_indices.len() as i64);
    }

    let row_ptrs_tensor =
        Tensor::<CpuRuntime>::try_from_slice(&row_ptrs, &[row_ptrs.len()], device).unwrap();
    let col_indices_tensor =
        Tensor::<CpuRuntime>::try_from_slice(&col_indices, &[col_indices.len()], device).unwrap();
    let values_tensor =
        Tensor::<CpuRuntime>::try_from_slice(&values, &[values.len()], device).unwrap();

    CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
        .expect("CSR creation should succeed")
}

// ============================================================================
// CG Tests
// ============================================================================

#[test]
fn test_cg_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    assert!(result.converged, "CG should converge on SPD Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual too large: {}",
        result.residual_norm
    );

    // Verify solution
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_cg_with_preconditioner() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 20;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
    let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[n], device).unwrap();

    let result_plain = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    let result_ilu = client
        .cg(
            &a,
            &b,
            None,
            CgOptions {
                preconditioner: PreconditionerType::Ilu0,
                ..Default::default()
            },
        )
        .expect("CG+ILU should succeed");

    assert!(result_plain.converged);
    assert!(result_ilu.converged);
    assert!(
        result_ilu.iterations <= result_plain.iterations,
        "ILU should help: {} vs {}",
        result_ilu.iterations,
        result_plain.iterations
    );
}

#[test]
fn test_cg_identity() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 5;
    let row_ptrs =
        Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 2, 3, 4, 5], &[n + 1], device).unwrap();
    let col_indices =
        Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 2, 3, 4], &[n], device).unwrap();
    let values = Tensor::<CpuRuntime>::try_from_slice(&[1.0f64; 5], &[n], device).unwrap();
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[n], device).unwrap();

    let result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    assert!(result.converged);
    assert_eq!(result.iterations, 1, "Identity should converge in 1 iter");

    let x_data: Vec<f64> = result.solution.to_vec();
    for (i, &val) in x_data.iter().enumerate() {
        assert!((val - (i as f64 + 1.0)).abs() < 1e-10);
    }
}

// ============================================================================
// CGS Tests
// ============================================================================

#[test]
fn test_cgs_nonsymmetric() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_nonsymmetric(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .cgs(&a, &b, None, CgsOptions::default())
        .expect("CGS should succeed");

    assert!(result.converged, "CGS should converge");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_cgs_laplacian() {
    // CGS also works on SPD systems
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        &[n],
        device,
    )
    .unwrap();

    let result = client
        .cgs(&a, &b, None, CgsOptions::default())
        .expect("CGS should succeed");

    assert!(result.converged, "CGS should converge on Laplacian");
}

// ============================================================================
// MINRES Tests
// ============================================================================

#[test]
fn test_minres_laplacian() {
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[n], device).unwrap();

    let result = client
        .minres(&a, &b, None, MinresOptions::default())
        .expect("MINRES should succeed");

    assert!(result.converged, "MINRES should converge on SPD Laplacian");
    assert!(
        result.residual_norm < 1e-8,
        "Residual: {}",
        result.residual_norm
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-8, "Verification residual: {}", res_norm);
}

#[test]
fn test_minres_symmetric_indefinite() {
    // Create symmetric indefinite matrix: [-2, 1; 1, -2] shifted to have mixed eigenvalues
    // A = [[1, 2], [2, -1]] — symmetric, eigenvalues ≈ 2.236 and -2.236
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 2;
    let row_ptrs = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 2, 4], &[3], device).unwrap();
    let col_indices = Tensor::<CpuRuntime>::try_from_slice(&[0i64, 1, 0, 1], &[4], device).unwrap();
    let values =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 2.0, -1.0], &[4], device).unwrap();
    let a = CsrData::new(row_ptrs, col_indices, values, [n, n]).unwrap();

    let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f64, 1.0], &[n], device).unwrap();

    let result = client
        .minres(&a, &b, None, MinresOptions::default())
        .expect("MINRES should succeed on indefinite system");

    assert!(
        result.converged,
        "MINRES should converge on symmetric indefinite"
    );

    // Verify
    let ax = a.spmv(&result.solution).expect("spmv");
    use numr::ops::BinaryOps;
    let residual = client.sub(&b, &ax).expect("sub");
    let res_data: Vec<f64> = residual.to_vec();
    let res_norm: f64 = res_data.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(res_norm < 1e-6, "Verification residual: {}", res_norm);
}

// ============================================================================
// Solver comparison test
// ============================================================================

#[test]
fn test_cg_vs_gmres_on_spd() {
    // CG and GMRES should produce the same solution on SPD systems
    let client = get_client();
    let device = &CpuRuntime::default_device();

    let n = 10;
    let a = create_1d_laplacian(n, device);
    let b_data: Vec<f64> = (0..n).map(|i| (i as f64 + 1.0).sin()).collect();
    let b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[n], device).unwrap();

    let cg_result = client
        .cg(&a, &b, None, CgOptions::default())
        .expect("CG should succeed");

    let gmres_result = client
        .gmres(
            &a,
            &b,
            None,
            numr::algorithm::iterative::GmresOptions::default(),
        )
        .expect("GMRES should succeed");

    assert!(cg_result.converged);
    assert!(gmres_result.converged);

    let cg_x: Vec<f64> = cg_result.solution.to_vec();
    let gmres_x: Vec<f64> = gmres_result.solution.to_vec();
    for i in 0..n {
        assert!(
            (cg_x[i] - gmres_x[i]).abs() < 1e-6,
            "CG and GMRES solutions differ at [{}]: {} vs {}",
            i,
            cg_x[i],
            gmres_x[i]
        );
    }
}
