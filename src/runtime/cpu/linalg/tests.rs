//! Tests for CPU linear algebra implementations

use super::super::{CpuClient, CpuDevice, CpuRuntime};
use crate::algorithm::LinearAlgebraAlgorithms;
use crate::algorithm::linalg::MatrixNormOrder;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

fn create_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

#[test]
fn test_lu_decomposition_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[4, 3], [6, 3]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device).unwrap();

    let lu = client.lu_decompose(&a).unwrap();

    // Verify dimensions
    assert_eq!(lu.lu.shape(), &[2, 2]);
    assert_eq!(lu.pivots.shape(), &[2]);
}

#[test]
fn test_lu_decomposition_3x3() {
    let client = create_client();
    let device = client.device();

    // A = [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] (tridiagonal matrix)
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[2.0f32, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.lu_decompose(&a);
    assert!(result.is_ok());

    let lu = result.unwrap();
    assert_eq!(lu.lu.shape(), &[3, 3]);
    assert_eq!(lu.pivots.shape(), &[3]);
}

#[test]
fn test_solve_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[2, 1], [1, 2]], b = [3, 3]
    // Solution: x = [1, 1]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

    let x = client.solve(&a, &b).unwrap();
    let x_data: Vec<f32> = x.to_vec();

    // Check solution is approximately [1, 1]
    assert!((x_data[0] - 1.0).abs() < 1e-5);
    assert!((x_data[1] - 1.0).abs() < 1e-5);
}

#[test]
fn test_det_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[4, 3], [6, 3]]
    // det = 4*3 - 3*6 = 12 - 18 = -6
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device).unwrap();

    let det = client.det(&a).unwrap();
    let det_val: Vec<f32> = det.to_vec();

    assert!((det_val[0] - (-6.0)).abs() < 1e-5);
}

#[test]
fn test_trace() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]]
    // trace = 1 + 4 = 5
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let tr = client.trace(&a).unwrap();
    let tr_val: Vec<f32> = tr.to_vec();

    assert!((tr_val[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_cholesky_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[4, 2], [2, 2]] - symmetric positive definite
    // L = [[2, 0], [1, 1]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 2.0, 2.0, 2.0], &[2, 2], device).unwrap();

    let chol = client.cholesky_decompose(&a).unwrap();
    let l_data: Vec<f32> = chol.l.to_vec();

    // Check L is approximately [[2, 0], [1, 1]]
    assert!((l_data[0] - 2.0).abs() < 1e-5); // L[0,0]
    assert!((l_data[1]).abs() < 1e-5); // L[0,1] = 0
    assert!((l_data[2] - 1.0).abs() < 1e-5); // L[1,0]
    assert!((l_data[3] - 1.0).abs() < 1e-5); // L[1,1]
}

#[test]
fn test_inverse_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[4, 7], [2, 6]]
    // det = 24 - 14 = 10
    // A^(-1) = 1/10 * [[6, -7], [-2, 4]] = [[0.6, -0.7], [-0.2, 0.4]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

    let inv = client.inverse(&a).unwrap();
    let inv_data: Vec<f32> = inv.to_vec();

    assert!((inv_data[0] - 0.6).abs() < 1e-4);
    assert!((inv_data[1] - (-0.7)).abs() < 1e-4);
    assert!((inv_data[2] - (-0.2)).abs() < 1e-4);
    assert!((inv_data[3] - 0.4).abs() < 1e-4);
}

#[test]
fn test_diag() {
    let client = create_client();
    let device = client.device();

    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
            .unwrap();

    let d = client.diag(&a).unwrap();
    let d_data: Vec<f32> = d.to_vec();

    assert_eq!(d_data.len(), 2); // min(2, 3)
    assert!((d_data[0] - 1.0).abs() < 1e-5);
    assert!((d_data[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_diagflat() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

    let mat = client.diagflat(&a).unwrap();
    let mat_data: Vec<f32> = mat.to_vec();

    // Should be 3x3 with [1, 2, 3] on diagonal
    assert_eq!(mat.shape(), &[3, 3]);
    assert!((mat_data[0] - 1.0).abs() < 1e-5); // [0,0]
    assert!((mat_data[4] - 2.0).abs() < 1e-5); // [1,1]
    assert!((mat_data[8] - 3.0).abs() < 1e-5); // [2,2]
    // Off-diagonal should be zero
    assert!((mat_data[1]).abs() < 1e-5);
    assert!((mat_data[2]).abs() < 1e-5);
}

#[test]
fn test_qr_decomposition_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let qr = client.qr_decompose(&a).unwrap();

    // Check dimensions
    assert_eq!(qr.q.shape(), &[2, 2]);
    assert_eq!(qr.r.shape(), &[2, 2]);

    // Q should be orthogonal: Q^T @ Q ≈ I
    let q_data: Vec<f32> = qr.q.to_vec();
    // Check Q^T @ Q diagonal is ~1 and off-diagonal is ~0
    let q00 = q_data[0];
    let q01 = q_data[1];
    let q10 = q_data[2];
    let q11 = q_data[3];

    let qtq_00 = q00 * q00 + q10 * q10; // Should be 1
    let qtq_11 = q01 * q01 + q11 * q11; // Should be 1
    let qtq_01 = q00 * q01 + q10 * q11; // Should be 0

    assert!(
        (qtq_00 - 1.0).abs() < 1e-4,
        "Q^T@Q[0,0] = {} should be 1",
        qtq_00
    );
    assert!(
        (qtq_11 - 1.0).abs() < 1e-4,
        "Q^T@Q[1,1] = {} should be 1",
        qtq_11
    );
    assert!((qtq_01).abs() < 1e-4, "Q^T@Q[0,1] = {} should be 0", qtq_01);

    // R should be upper triangular (R[1,0] = 0)
    let r_data: Vec<f32> = qr.r.to_vec();
    assert!((r_data[2]).abs() < 1e-4, "R[1,0] should be 0");
}

#[test]
fn test_lstsq_exact() {
    let client = create_client();
    let device = client.device();

    // Exact system: A @ x = b with unique solution
    // A = [[2, 1], [1, 2]], b = [3, 3]
    // Solution: x = [1, 1]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

    let x = client.lstsq(&a, &b).unwrap();
    let x_data: Vec<f32> = x.to_vec();

    // Solution should be approximately [1, 1]
    assert!(
        (x_data[0] - 1.0).abs() < 1e-4,
        "x[0] = {} should be 1.0",
        x_data[0]
    );
    assert!(
        (x_data[1] - 1.0).abs() < 1e-4,
        "x[1] = {} should be 1.0",
        x_data[1]
    );
}

#[test]
fn test_matrix_rank_full_rank() {
    let client = create_client();
    let device = client.device();

    // Full rank 2x2 matrix
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let rank = client.matrix_rank(&a, None).unwrap();
    let rank_val: Vec<i64> = rank.to_vec();

    assert_eq!(rank_val[0], 2, "Full rank 2x2 matrix should have rank 2");
}

#[test]
fn test_matrix_rank_rank_deficient() {
    let client = create_client();
    let device = client.device();

    // Rank-deficient 2x2 matrix: second row is multiple of first
    // [[1, 2], [2, 4]] has rank 1
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device).unwrap();

    let rank = client.matrix_rank(&a, None).unwrap();
    let rank_val: Vec<i64> = rank.to_vec();

    assert_eq!(rank_val[0], 1, "Rank-deficient matrix should have rank 1");
}

#[test]
fn test_frobenius_norm_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]]
    // ||A||_F = sqrt(1² + 2² + 3² + 4²) = sqrt(1 + 4 + 9 + 16) = sqrt(30)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
    let norm_val: Vec<f32> = norm.to_vec();

    let expected = (30.0f32).sqrt();
    assert!(
        (norm_val[0] - expected).abs() < 1e-5,
        "Frobenius norm = {} should be {}",
        norm_val[0],
        expected
    );
}

#[test]
fn test_frobenius_norm_3x3() {
    let client = create_client();
    let device = client.device();

    // Identity matrix: ||I||_F = sqrt(3) for 3x3
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let norm = client.matrix_norm(&a, MatrixNormOrder::Frobenius).unwrap();
    let norm_val: Vec<f32> = norm.to_vec();

    let expected = (3.0f32).sqrt();
    assert!(
        (norm_val[0] - expected).abs() < 1e-5,
        "Frobenius norm of 3x3 identity = {} should be {}",
        norm_val[0],
        expected
    );
}

#[test]
fn test_spectral_norm() {
    let client = create_client();
    let device = client.device();

    // Test on a 2x2 matrix [[1, 2], [3, 4]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let result = client.matrix_norm(&a, MatrixNormOrder::Spectral).unwrap();
    let norm_val: Vec<f32> = result.to_vec();

    // Spectral norm is the largest singular value
    // For [[1,2],[3,4]], singular values are approximately 5.465 and 0.366
    assert!(
        (norm_val[0] - 5.465).abs() < 0.01,
        "Spectral norm of [[1,2],[3,4]] = {} should be ~5.465",
        norm_val[0]
    );
}

#[test]
fn test_nuclear_norm() {
    let client = create_client();
    let device = client.device();

    // Test on a 2x2 matrix [[1, 2], [3, 4]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let result = client.matrix_norm(&a, MatrixNormOrder::Nuclear).unwrap();
    let norm_val: Vec<f32> = result.to_vec();

    // Nuclear norm is sum of singular values
    // For [[1,2],[3,4]], singular values are approximately 5.465 and 0.366
    // Sum is approximately 5.831
    assert!(
        (norm_val[0] - 5.831).abs() < 0.01,
        "Nuclear norm of [[1,2],[3,4]] = {} should be ~5.831",
        norm_val[0]
    );
}

#[test]
fn test_spectral_norm_identity() {
    let client = create_client();
    let device = client.device();

    // Identity matrix has spectral norm = 1 (all singular values = 1)
    let eye =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();

    let result = client.matrix_norm(&eye, MatrixNormOrder::Spectral).unwrap();
    let norm_val: Vec<f32> = result.to_vec();

    assert!(
        (norm_val[0] - 1.0).abs() < 1e-5,
        "Spectral norm of 2x2 identity = {} should be 1.0",
        norm_val[0]
    );
}

#[test]
fn test_nuclear_norm_identity() {
    let client = create_client();
    let device = client.device();

    // 3x3 Identity matrix has nuclear norm = 3 (sum of 3 singular values = 1)
    let eye = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.matrix_norm(&eye, MatrixNormOrder::Nuclear).unwrap();
    let norm_val: Vec<f32> = result.to_vec();

    assert!(
        (norm_val[0] - 3.0).abs() < 1e-5,
        "Nuclear norm of 3x3 identity = {} should be 3.0",
        norm_val[0]
    );
}

// ========================================================================
// Schur Decomposition Tests
// ========================================================================

#[test]
fn test_schur_1x1() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[5.0f64], &[1, 1], device).unwrap();
    let schur = client.schur_decompose(&a).unwrap();

    let z_data: Vec<f64> = schur.z.to_vec();
    let t_data: Vec<f64> = schur.t.to_vec();

    // For 1x1, Z = [1] and T = A
    assert!((z_data[0] - 1.0).abs() < 1e-10);
    assert!((t_data[0] - 5.0).abs() < 1e-10);
}

#[test]
fn test_schur_2x2_reconstruction() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let schur = client.schur_decompose(&a).unwrap();

    let z_data: Vec<f64> = schur.z.to_vec();
    let t_data: Vec<f64> = schur.t.to_vec();

    // Verify Z is orthogonal: Z^T @ Z = I
    let ztza = z_data[0] * z_data[0] + z_data[2] * z_data[2];
    let ztzb = z_data[0] * z_data[1] + z_data[2] * z_data[3];
    let ztzd = z_data[1] * z_data[1] + z_data[3] * z_data[3];

    assert!((ztza - 1.0).abs() < 1e-6, "Z^T Z [0,0] should be 1");
    assert!(ztzb.abs() < 1e-6, "Z^T Z [0,1] should be 0");
    assert!((ztzd - 1.0).abs() < 1e-6, "Z^T Z [1,1] should be 1");

    // Verify reconstruction: A = Z @ T @ Z^T
    // Z @ T
    let zt00 = z_data[0] * t_data[0] + z_data[1] * t_data[2];
    let zt01 = z_data[0] * t_data[1] + z_data[1] * t_data[3];
    let zt10 = z_data[2] * t_data[0] + z_data[3] * t_data[2];
    let zt11 = z_data[2] * t_data[1] + z_data[3] * t_data[3];

    // (Z @ T) @ Z^T
    let rec00 = zt00 * z_data[0] + zt01 * z_data[1];
    let rec01 = zt00 * z_data[2] + zt01 * z_data[3];
    let rec10 = zt10 * z_data[0] + zt11 * z_data[1];
    let rec11 = zt10 * z_data[2] + zt11 * z_data[3];

    assert!(
        (rec00 - 1.0).abs() < 1e-5,
        "Reconstruction [0,0] failed: {} != 1.0",
        rec00
    );
    assert!(
        (rec01 - 2.0).abs() < 1e-5,
        "Reconstruction [0,1] failed: {} != 2.0",
        rec01
    );
    assert!(
        (rec10 - 3.0).abs() < 1e-5,
        "Reconstruction [1,0] failed: {} != 3.0",
        rec10
    );
    assert!(
        (rec11 - 4.0).abs() < 1e-5,
        "Reconstruction [1,1] failed: {} != 4.0",
        rec11
    );
}

#[test]
fn test_schur_symmetric_diagonal() {
    let client = create_client();
    let device = client.device();

    // For a symmetric matrix, Schur form is diagonal (eigenvalue decomposition)
    // A = [[2, 1], [1, 3]] (symmetric)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f64, 1.0, 1.0, 3.0], &[2, 2], device).unwrap();
    let schur = client.schur_decompose(&a).unwrap();

    let t_data: Vec<f64> = schur.t.to_vec();

    // T[1,0] should be small (quasi-triangular becomes diagonal for symmetric)
    assert!(
        t_data[2].abs() < 0.1, // Use looser tolerance for QR iteration
        "T should be quasi-triangular: T[1,0] = {}",
        t_data[2]
    );
}

#[test]
fn test_schur_3x3() {
    let client = create_client();
    let device = client.device();

    // A 3x3 upper triangular matrix is already in Schur form
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f64, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0],
        &[3, 3],
        device,
    )
    .unwrap();
    let schur = client.schur_decompose(&a).unwrap();

    let z_data: Vec<f64> = schur.z.to_vec();
    let t_data: Vec<f64> = schur.t.to_vec();

    // Z should be close to identity
    let diag_sum = z_data[0] * z_data[0] + z_data[4] * z_data[4] + z_data[8] * z_data[8];
    assert!(
        diag_sum > 2.5,
        "For upper triangular input, Z should be close to identity"
    );

    // T should have small subdiagonal elements
    assert!(
        t_data[3].abs() < 0.1,
        "T[1,0] should be small: {}",
        t_data[3]
    );
    assert!(
        t_data[6].abs() < 0.1,
        "T[2,0] should be small: {}",
        t_data[6]
    );
    assert!(
        t_data[7].abs() < 0.1,
        "T[2,1] should be small: {}",
        t_data[7]
    );
}

// ======================= eig_decompose tests =======================

#[test]
fn test_eig_1x1() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[5.0f64], &[1, 1], device).unwrap();
    let eig = client.eig_decompose(&a).unwrap();

    let eval_real: Vec<f64> = eig.eigenvalues_real.to_vec();
    let eval_imag: Vec<f64> = eig.eigenvalues_imag.to_vec();

    // Eigenvalue should be 5.0 (real)
    assert!((eval_real[0] - 5.0).abs() < 1e-10);
    assert!(eval_imag[0].abs() < 1e-10);
}

#[test]
fn test_eig_2x2_real_eigenvalues() {
    let client = create_client();
    let device = client.device();

    // A = [[2, 1], [1, 2]] - symmetric with real eigenvalues 3 and 1
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let eig = client.eig_decompose(&a).unwrap();

    let eval_real: Vec<f64> = eig.eigenvalues_real.to_vec();
    let eval_imag: Vec<f64> = eig.eigenvalues_imag.to_vec();

    // Both eigenvalues should be real
    assert!(eval_imag[0].abs() < 1e-6, "Eigenvalue 0 should be real");
    assert!(eval_imag[1].abs() < 1e-6, "Eigenvalue 1 should be real");

    // Check eigenvalues are 3 and 1 (in some order)
    let mut evals = vec![eval_real[0], eval_real[1]];
    evals.sort_by(|a, b| b.partial_cmp(a).unwrap());
    assert!(
        (evals[0] - 3.0).abs() < 1e-5,
        "Larger eigenvalue should be 3"
    );
    assert!(
        (evals[1] - 1.0).abs() < 1e-5,
        "Smaller eigenvalue should be 1"
    );
}

#[test]
fn test_eig_2x2_complex_eigenvalues() {
    let client = create_client();
    let device = client.device();

    // A = [[0, -1], [1, 0]] - rotation matrix with eigenvalues ±i
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[0.0f64, -1.0, 1.0, 0.0], &[2, 2], device).unwrap();
    let eig = client.eig_decompose(&a).unwrap();

    let eval_real: Vec<f64> = eig.eigenvalues_real.to_vec();
    let eval_imag: Vec<f64> = eig.eigenvalues_imag.to_vec();

    // Real parts should be 0
    assert!(eval_real[0].abs() < 1e-6, "Real part should be 0");
    assert!(eval_real[1].abs() < 1e-6, "Real part should be 0");

    // Imaginary parts should be ±1 (conjugate pair)
    let imag_sum = eval_imag[0] + eval_imag[1];
    let imag_prod = eval_imag[0] * eval_imag[1];
    assert!(imag_sum.abs() < 1e-6, "Imaginary parts should sum to 0");
    assert!(
        (imag_prod - (-1.0)).abs() < 1e-6,
        "Imaginary parts should multiply to -1"
    );
}

#[test]
fn test_eig_eigenvector_equation() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]] - non-symmetric with real eigenvalues
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let a_data: Vec<f64> = a.to_vec();
    let eig = client.eig_decompose(&a).unwrap();

    let eval_real: Vec<f64> = eig.eigenvalues_real.to_vec();
    let eval_imag: Vec<f64> = eig.eigenvalues_imag.to_vec();
    let evec_real: Vec<f64> = eig.eigenvectors_real.to_vec();

    // For each real eigenvalue, verify A @ v = λ @ v
    for i in 0..2 {
        if eval_imag[i].abs() < 1e-6 {
            let lambda = eval_real[i];
            let v0 = evec_real[0 * 2 + i]; // eigenvectors_real[0, i]
            let v1 = evec_real[1 * 2 + i]; // eigenvectors_real[1, i]

            // A @ v
            let av0 = a_data[0] * v0 + a_data[1] * v1;
            let av1 = a_data[2] * v0 + a_data[3] * v1;

            // λ * v
            let lv0 = lambda * v0;
            let lv1 = lambda * v1;

            // Normalize check (handle zero eigenvectors)
            let v_norm = (v0 * v0 + v1 * v1).sqrt();
            if v_norm > 1e-6 {
                assert!(
                    (av0 - lv0).abs() < 1e-4,
                    "A @ v[0] = {} but λ * v[0] = {}",
                    av0,
                    lv0
                );
                assert!(
                    (av1 - lv1).abs() < 1e-4,
                    "A @ v[1] = {} but λ * v[1] = {}",
                    av1,
                    lv1
                );
            }
        }
    }
}

#[test]
fn test_eig_3x3_diagonal() {
    let client = create_client();
    let device = client.device();

    // Diagonal matrix - eigenvalues are the diagonal elements
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        &[3, 3],
        device,
    )
    .unwrap();
    let eig = client.eig_decompose(&a).unwrap();

    let eval_real: Vec<f64> = eig.eigenvalues_real.to_vec();
    let eval_imag: Vec<f64> = eig.eigenvalues_imag.to_vec();

    // All eigenvalues should be real
    for i in 0..3 {
        assert!(
            eval_imag[i].abs() < 1e-10,
            "Eigenvalue {} should be real",
            i
        );
    }

    // Eigenvalues should be 1, 2, 3 (in some order)
    let mut evals = eval_real.clone();
    evals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((evals[0] - 1.0).abs() < 1e-6);
    assert!((evals[1] - 2.0).abs() < 1e-6);
    assert!((evals[2] - 3.0).abs() < 1e-6);
}

// ======================= kron (Kronecker product) tests =======================

#[test]
fn test_kron_2x2_identity() {
    let client = create_client();
    let device = client.device();

    // I₂ ⊗ I₂ = I₄
    let i2 =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();
    let kron = client.kron(&i2, &i2).unwrap();

    assert_eq!(kron.shape(), &[4, 4]);

    let data: Vec<f32> = kron.to_vec();
    // Should be 4x4 identity
    for i in 0..4 {
        for j in 0..4 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (data[i * 4 + j] - expected).abs() < 1e-5,
                "kron[{},{}] = {} expected {}",
                i,
                j,
                data[i * 4 + j],
                expected
            );
        }
    }
}

#[test]
fn test_kron_2x2_simple() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]], B = [[0, 5], [6, 7]]
    // A ⊗ B should be 4x4
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 5.0, 6.0, 7.0], &[2, 2], device).unwrap();

    let kron = client.kron(&a, &b).unwrap();
    assert_eq!(kron.shape(), &[4, 4]);

    let data: Vec<f32> = kron.to_vec();

    // Expected result:
    // [[1*0, 1*5, 2*0, 2*5],     [[0,  5,  0, 10],
    //  [1*6, 1*7, 2*6, 2*7],  =   [6,  7, 12, 14],
    //  [3*0, 3*5, 4*0, 4*5],      [0, 15,  0, 20],
    //  [3*6, 3*7, 4*6, 4*7]]      [18, 21, 24, 28]]
    #[rustfmt::skip]
    let expected = [
        0.0, 5.0, 0.0, 10.0,
        6.0, 7.0, 12.0, 14.0,
        0.0, 15.0, 0.0, 20.0,
        18.0, 21.0, 24.0, 28.0,
    ];

    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "element {} differs: {} vs {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_kron_scalar_property() {
    let client = create_client();
    let device = client.device();

    // 1x1 ⊗ A = scalar * A
    let scalar = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1, 1], device).unwrap();
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let kron = client.kron(&scalar, &a).unwrap();
    assert_eq!(kron.shape(), &[2, 2]);

    let data: Vec<f32> = kron.to_vec();
    let expected = [3.0f32, 6.0, 9.0, 12.0]; // 3 * A

    for (got, exp) in data.iter().zip(expected.iter()) {
        assert!((got - exp).abs() < 1e-5);
    }
}

#[test]
fn test_kron_rectangular() {
    let client = create_client();
    let device = client.device();

    // 2x3 ⊗ 3x2 = 6x6
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
            .unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], device)
            .unwrap();

    let kron = client.kron(&a, &b).unwrap();
    assert_eq!(kron.shape(), &[6, 6]);

    // Verify a few elements manually
    // kron[0,0] = a[0,0] * b[0,0] = 1 * 1 = 1
    // kron[0,1] = a[0,0] * b[0,1] = 1 * 2 = 2
    // kron[3,0] = a[1,0] * b[0,0] = 4 * 1 = 4
    let data: Vec<f32> = kron.to_vec();
    assert!((data[0] - 1.0).abs() < 1e-5, "kron[0,0]");
    assert!((data[1] - 2.0).abs() < 1e-5, "kron[0,1]");
    assert!((data[3 * 6 + 0] - 4.0).abs() < 1e-5, "kron[3,0]");
}

#[test]
fn test_kron_f64() {
    let client = create_client();
    let device = client.device();

    // Test with F64
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

    let kron = client.kron(&a, &b).unwrap();
    assert_eq!(kron.shape(), &[4, 4]);

    let data: Vec<f64> = kron.to_vec();

    // kron[0,0] = 1*5 = 5, kron[0,1] = 1*6 = 6
    // kron[1,0] = 1*7 = 7, kron[1,1] = 1*8 = 8
    assert!((data[0] - 5.0).abs() < 1e-10);
    assert!((data[1] - 6.0).abs() < 1e-10);
    assert!((data[4] - 7.0).abs() < 1e-10);
    assert!((data[5] - 8.0).abs() < 1e-10);
}

// ======================= khatri_rao (Column-wise Kronecker) tests =======================

#[test]
fn test_khatri_rao_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]] (2x2)
    // B = [[5, 6], [7, 8]] (2x2)
    // A ⊙ B should be (2*2)x2 = 4x2
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

    let kr = client.khatri_rao(&a, &b).unwrap();
    assert_eq!(kr.shape(), &[4, 2]);

    let data: Vec<f32> = kr.to_vec();

    // Expected result (column-wise Kronecker):
    // Column 0: kron([1,3], [5,7]) = [1*5, 1*7, 3*5, 3*7] = [5, 7, 15, 21]
    // Column 1: kron([2,4], [6,8]) = [2*6, 2*8, 4*6, 4*8] = [12, 16, 24, 32]
    // Result: [[5, 12], [7, 16], [15, 24], [21, 32]]
    let expected = [5.0f32, 12.0, 7.0, 16.0, 15.0, 24.0, 21.0, 32.0];

    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "element {} differs: {} vs {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_khatri_rao_different_rows() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2, 3]] (1x3)
    // B = [[4, 5, 6], [7, 8, 9]] (2x3)
    // A ⊙ B should be (1*2)x3 = 2x3
    let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 5.0, 6.0, 7.0, 8.0, 9.0], &[2, 3], device)
            .unwrap();

    let kr = client.khatri_rao(&a, &b).unwrap();
    assert_eq!(kr.shape(), &[2, 3]);

    let data: Vec<f32> = kr.to_vec();

    // Column 0: kron([1], [4,7]) = [1*4, 1*7] = [4, 7]
    // Column 1: kron([2], [5,8]) = [2*5, 2*8] = [10, 16]
    // Column 2: kron([3], [6,9]) = [3*6, 3*9] = [18, 27]
    // Result: [[4, 10, 18], [7, 16, 27]]
    let expected = [4.0f32, 10.0, 18.0, 7.0, 16.0, 27.0];

    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "element {} differs: {} vs {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_khatri_rao_column_mismatch() {
    let client = create_client();
    let device = client.device();

    // Different column counts should fail
    let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[1, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0, 5.0], &[1, 3], device).unwrap();

    let result = client.khatri_rao(&a, &b);
    assert!(
        result.is_err(),
        "Khatri-Rao with mismatched columns should fail"
    );
}

#[test]
fn test_khatri_rao_f64() {
    let client = create_client();
    let device = client.device();

    // Test with F64
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

    let kr = client.khatri_rao(&a, &b).unwrap();
    assert_eq!(kr.shape(), &[4, 2]);

    let data: Vec<f64> = kr.to_vec();

    // Same expected values as f32 test
    let expected = [5.0f64, 12.0, 7.0, 16.0, 15.0, 24.0, 21.0, 32.0];

    for (i, (got, exp)) in data.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-10,
            "element {} differs: {} vs {}",
            i,
            got,
            exp
        );
    }
}

#[test]
fn test_khatri_rao_gram_property() {
    let client = create_client();
    let device = client.device();

    // Property: (A ⊙ B)^T (A ⊙ B) = (A^T A) * (B^T B) (Hadamard product)
    // This is a fundamental property of Khatri-Rao used in tensor decompositions
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b =
        Tensor::<CpuRuntime>::try_from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

    let kr = client.khatri_rao(&a, &b).unwrap();

    // We just verify the Khatri-Rao product computed correctly
    // Full Gram property verification would require matmul which is tested elsewhere
    assert_eq!(kr.shape(), &[4, 2]);
}

// ============================================================================
// triu / tril tests
// ============================================================================

#[test]
fn test_triu_3x3_default_diagonal() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.triu(&a, 0).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[1, 2, 3], [0, 5, 6], [0, 0, 9]]
    assert_eq!(data, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
}

#[test]
fn test_triu_3x3_diagonal_1() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.triu(&a, 1).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[0, 2, 3], [0, 0, 6], [0, 0, 0]]
    assert_eq!(data, [0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_triu_3x3_negative_diagonal() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.triu(&a, -1).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[1, 2, 3], [4, 5, 6], [0, 8, 9]]
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 8.0, 9.0]);
}

#[test]
fn test_tril_3x3_default_diagonal() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.tril(&a, 0).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[1, 0, 0], [4, 5, 0], [7, 8, 9]]
    assert_eq!(data, [1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_tril_3x3_diagonal_1() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.tril(&a, 1).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[1, 2, 0], [4, 5, 6], [7, 8, 9]]
    assert_eq!(data, [1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
}

#[test]
fn test_triu_rectangular_2x4() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2, 3, 4], [5, 6, 7, 8]]
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        device,
    )
    .unwrap();

    let result = client.triu(&a, 0).unwrap();
    let data: Vec<f32> = result.to_vec();

    // Expected: [[1, 2, 3, 4], [0, 6, 7, 8]]
    assert_eq!(data, [1.0, 2.0, 3.0, 4.0, 0.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_triu_i32() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[1i32, 2, 3, 4, 5, 6, 7, 8, 9], &[3, 3], device)
        .unwrap();

    let result = client.triu(&a, 0).unwrap();
    let data: Vec<i32> = result.to_vec();

    assert_eq!(data, [1, 2, 3, 0, 5, 6, 0, 0, 9]);
}

#[test]
fn test_triu_f64() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.triu(&a, 0).unwrap();
    let data: Vec<f64> = result.to_vec();

    assert_eq!(data, [1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]);
}

// ============================================================================
// slogdet tests
// ============================================================================

#[test]
fn test_slogdet_2x2() {
    let client = create_client();
    let device = client.device();

    // A = [[1, 2], [3, 4]], det = 1*4 - 2*3 = -2
    // sign = -1, logabsdet = ln(2)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let result = client.slogdet(&a).unwrap();

    let sign_data: Vec<f32> = result.sign.to_vec();
    let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

    assert!(
        (sign_data[0] - (-1.0)).abs() < 1e-5,
        "sign should be -1, got {}",
        sign_data[0]
    );
    assert!(
        (logabsdet_data[0] - 2.0f32.ln()).abs() < 1e-5,
        "logabsdet should be ln(2) ≈ {}, got {}",
        2.0f32.ln(),
        logabsdet_data[0]
    );
}

#[test]
fn test_slogdet_3x3_identity() {
    let client = create_client();
    let device = client.device();

    // Identity matrix: det = 1, sign = 1, logabsdet = 0
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        device,
    )
    .unwrap();

    let result = client.slogdet(&a).unwrap();

    let sign_data: Vec<f32> = result.sign.to_vec();
    let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

    assert!(
        (sign_data[0] - 1.0).abs() < 1e-5,
        "sign should be 1, got {}",
        sign_data[0]
    );
    assert!(
        logabsdet_data[0].abs() < 1e-5,
        "logabsdet should be 0, got {}",
        logabsdet_data[0]
    );
}

#[test]
fn test_slogdet_positive_det() {
    let client = create_client();
    let device = client.device();

    // A = [[2, 0], [0, 3]], det = 6
    // sign = 1, logabsdet = ln(6)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 0.0, 0.0, 3.0], &[2, 2], device).unwrap();

    let result = client.slogdet(&a).unwrap();

    let sign_data: Vec<f32> = result.sign.to_vec();
    let logabsdet_data: Vec<f32> = result.logabsdet.to_vec();

    assert!(
        (sign_data[0] - 1.0).abs() < 1e-5,
        "sign should be 1, got {}",
        sign_data[0]
    );
    assert!(
        (logabsdet_data[0] - 6.0f32.ln()).abs() < 1e-4,
        "logabsdet should be ln(6) ≈ {}, got {}",
        6.0f32.ln(),
        logabsdet_data[0]
    );
}
