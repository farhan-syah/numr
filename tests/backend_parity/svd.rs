//! Integration tests for Singular Value Decomposition (SVD)
//!
//! Tests verify:
//! - Reconstruction: ||A - U @ diag(S) @ V^T|| < tolerance
//! - Orthogonality: U^T @ U ≈ I, V @ V^T ≈ I
//! - Singular values in descending order
//! - Edge cases: identity, single row/column, zero matrix
//! - Backend parity: CPU results match expected values

use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::ops::{LinalgOps, MatmulOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Helper Functions
// ============================================================================

fn create_client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

/// Assert all values are close within tolerance
fn assert_allclose(a: &[f32], b: &[f32], rtol: f32, atol: f32, msg: &str) {
    assert_eq!(a.len(), b.len(), "{}: length mismatch", msg);
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();
        assert!(
            diff <= tol,
            "{}: element {} differs: {} vs {} (diff={}, tol={})",
            msg,
            i,
            x,
            y,
            diff,
            tol
        );
    }
}

/// Check if matrix is close to identity
fn assert_near_identity(data: &[f32], n: usize, tol: f32, msg: &str) {
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = data[i * n + j];
            let diff = (actual - expected).abs();
            assert!(
                diff <= tol,
                "{}: element [{},{}] differs: {} vs {} (diff={})",
                msg,
                i,
                j,
                actual,
                expected,
                diff
            );
        }
    }
}

/// Check if singular values are in descending order
fn assert_descending(s: &[f32], msg: &str) {
    for i in 1..s.len() {
        assert!(
            s[i - 1] >= s[i] - 1e-6,
            "{}: s[{}]={} should be >= s[{}]={}",
            msg,
            i - 1,
            s[i - 1],
            i,
            s[i]
        );
    }
}

/// Compute Frobenius norm of a vector (flattened matrix)
fn frobenius_norm(data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// Basic SVD Tests
// ============================================================================

#[test]
fn test_svd_2x2() {
    let (client, device) = create_client();

    // Simple 2x2 matrix
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 1.0, 1.0, 3.0], &[2, 2], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // Check shapes
    assert_eq!(svd.u.shape(), &[2, 2], "U shape");
    assert_eq!(svd.s.shape(), &[2], "S shape");
    assert_eq!(svd.vt.shape(), &[2, 2], "V^T shape");

    // Get data
    let _u: Vec<f32> = svd.u.to_vec();
    let s: Vec<f32> = svd.s.to_vec();
    let _vt: Vec<f32> = svd.vt.to_vec();

    // Check singular values are descending
    assert_descending(&s, "singular values");

    // Known singular values for [[3,1],[1,3]]: 4 and 2
    assert!((s[0] - 4.0).abs() < 1e-5, "Expected s[0]=4, got {}", s[0]);
    assert!((s[1] - 2.0).abs() < 1e-5, "Expected s[1]=2, got {}", s[1]);

    // Verify orthogonality: U^T @ U ≈ I
    let ut = svd.u.transpose(0, 1).unwrap();
    let ut_u = client.matmul(&ut, &svd.u).unwrap();
    let ut_u_data: Vec<f32> = ut_u.to_vec();
    assert_near_identity(&ut_u_data, 2, 1e-5, "U^T @ U");

    // Verify orthogonality: V @ V^T ≈ I (note: vt is already V^T)
    let v = svd.vt.transpose(0, 1).unwrap();
    let v_vt = client.matmul(&v, &svd.vt).unwrap();
    let v_vt_data: Vec<f32> = v_vt.to_vec();
    assert_near_identity(&v_vt_data, 2, 1e-5, "V @ V^T");

    // Verify reconstruction: A ≈ U @ diag(S) @ V^T
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();
    let a_data: Vec<f32> = a.to_vec();
    assert_allclose(&reconstructed_data, &a_data, 1e-5, 1e-5, "reconstruction");
}

#[test]
fn test_svd_3x2_tall() {
    let (client, device) = create_client();

    // 3x2 matrix (m > n)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
            .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // Check shapes: thin SVD gives U [m, k], S [k], V^T [k, n] where k = min(m,n)
    assert_eq!(svd.u.shape(), &[3, 2], "U shape");
    assert_eq!(svd.s.shape(), &[2], "S shape");
    assert_eq!(svd.vt.shape(), &[2, 2], "V^T shape");

    // Check singular values are descending
    let s: Vec<f32> = svd.s.to_vec();
    assert_descending(&s, "singular values");
    assert!(s[0] > 0.0, "s[0] should be positive");
    assert!(s[1] >= 0.0, "s[1] should be non-negative");

    // Verify reconstruction
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();
    let a_data: Vec<f32> = a.to_vec();

    // Check reconstruction error
    let diff: Vec<f32> = reconstructed_data
        .iter()
        .zip(a_data.iter())
        .map(|(r, a)| r - a)
        .collect();
    let error = frobenius_norm(&diff);
    assert!(error < 1e-4, "Reconstruction error too large: {}", error);
}

#[test]
fn test_svd_2x3_wide() {
    let (client, device) = create_client();

    // 2x3 matrix (m < n)
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
            .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // Check shapes: thin SVD gives U [m, k], S [k], V^T [k, n] where k = min(m,n)
    assert_eq!(svd.u.shape(), &[2, 2], "U shape");
    assert_eq!(svd.s.shape(), &[2], "S shape");
    assert_eq!(svd.vt.shape(), &[2, 3], "V^T shape");

    // Check singular values are descending
    let s: Vec<f32> = svd.s.to_vec();
    assert_descending(&s, "singular values");

    // Verify reconstruction
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();
    let a_data: Vec<f32> = a.to_vec();

    let diff: Vec<f32> = reconstructed_data
        .iter()
        .zip(a_data.iter())
        .map(|(r, a)| r - a)
        .collect();
    let error = frobenius_norm(&diff);
    assert!(error < 1e-4, "Reconstruction error too large: {}", error);
}

#[test]
fn test_svd_identity() {
    let (client, device) = create_client();

    // 3x3 identity matrix
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    )
    .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // All singular values should be 1
    let s: Vec<f32> = svd.s.to_vec();
    for (i, &val) in s.iter().enumerate() {
        assert!(
            (val - 1.0).abs() < 1e-5,
            "s[{}] should be 1, got {}",
            i,
            val
        );
    }

    // Verify reconstruction
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();
    let a_data: Vec<f32> = a.to_vec();
    assert_allclose(
        &reconstructed_data,
        &a_data,
        1e-5,
        1e-5,
        "identity reconstruction",
    );
}

#[test]
fn test_svd_diagonal() {
    let (client, device) = create_client();

    // Diagonal matrix with known singular values
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[5.0f32, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    )
    .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    let s: Vec<f32> = svd.s.to_vec();
    assert_descending(&s, "singular values");

    // Singular values should be 5, 3, 1 (sorted descending)
    assert!((s[0] - 5.0).abs() < 1e-5, "Expected s[0]=5, got {}", s[0]);
    assert!((s[1] - 3.0).abs() < 1e-5, "Expected s[1]=3, got {}", s[1]);
    assert!((s[2] - 1.0).abs() < 1e-5, "Expected s[2]=1, got {}", s[2]);
}

#[test]
fn test_svd_rank_deficient() {
    let (client, device) = create_client();

    // Rank-1 matrix: outer product of [1, 2, 3] and [1, 2]
    // A = [[1, 2], [2, 4], [3, 6]]
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], &device)
            .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    let s: Vec<f32> = svd.s.to_vec();
    assert_descending(&s, "singular values");

    // Second singular value should be ~0 (rank-1 matrix)
    assert!(
        s[1] < 1e-4,
        "Second singular value should be ~0, got {}",
        s[1]
    );

    // First singular value should be non-zero
    assert!(
        s[0] > 1.0,
        "First singular value should be > 1, got {}",
        s[0]
    );
}

// ============================================================================
// F64 Tests (CPU only supports both F32 and F64)
// ============================================================================

#[test]
fn test_svd_f64() {
    let (client, device) = create_client();

    // 2x2 matrix with F64
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[3.0f64, 1.0, 1.0, 3.0], &[2, 2], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    assert_eq!(svd.u.shape(), &[2, 2]);
    assert_eq!(svd.s.shape(), &[2]);
    assert_eq!(svd.vt.shape(), &[2, 2]);

    let s: Vec<f64> = svd.s.to_vec();

    // Known singular values for [[3,1],[1,3]]: 4 and 2
    assert!((s[0] - 4.0).abs() < 1e-10, "Expected s[0]=4, got {}", s[0]);
    assert!((s[1] - 2.0).abs() < 1e-10, "Expected s[1]=2, got {}", s[1]);

    // Verify reconstruction with F64 precision
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f64> = reconstructed.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    for (i, (r, a)) in reconstructed_data.iter().zip(a_data.iter()).enumerate() {
        let diff = (r - a).abs();
        assert!(
            diff < 1e-10,
            "F64 reconstruction element {} differs: {} vs {} (diff={})",
            i,
            r,
            a,
            diff
        );
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_svd_single_element() {
    let (client, device) = create_client();

    // 1x1 matrix
    let a = Tensor::<CpuRuntime>::try_from_slice(&[5.0f32], &[1, 1], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    assert_eq!(svd.u.shape(), &[1, 1]);
    assert_eq!(svd.s.shape(), &[1]);
    assert_eq!(svd.vt.shape(), &[1, 1]);

    let s: Vec<f32> = svd.s.to_vec();
    assert!((s[0] - 5.0).abs() < 1e-5, "Expected s[0]=5, got {}", s[0]);
}

#[test]
fn test_svd_single_row() {
    let (client, device) = create_client();

    // 1x3 matrix (single row)
    let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // k = min(1, 3) = 1
    assert_eq!(svd.u.shape(), &[1, 1]);
    assert_eq!(svd.s.shape(), &[1]);
    assert_eq!(svd.vt.shape(), &[1, 3]);

    let s: Vec<f32> = svd.s.to_vec();
    // ||[1, 2, 3]|| = sqrt(14) ≈ 3.742
    let expected_norm = (14.0f32).sqrt();
    assert!(
        (s[0] - expected_norm).abs() < 1e-4,
        "Expected s[0]={}, got {}",
        expected_norm,
        s[0]
    );
}

#[test]
fn test_svd_single_column() {
    let (client, device) = create_client();

    // 3x1 matrix (single column)
    let a = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // k = min(3, 1) = 1
    assert_eq!(svd.u.shape(), &[3, 1]);
    assert_eq!(svd.s.shape(), &[1]);
    assert_eq!(svd.vt.shape(), &[1, 1]);

    let s: Vec<f32> = svd.s.to_vec();
    // ||[1, 2, 3]|| = sqrt(14) ≈ 3.742
    let expected_norm = (14.0f32).sqrt();
    assert!(
        (s[0] - expected_norm).abs() < 1e-4,
        "Expected s[0]={}, got {}",
        expected_norm,
        s[0]
    );
}

#[test]
fn test_svd_larger_matrix() {
    let (client, device) = create_client();

    // 5x4 matrix with random-ish values
    let data: Vec<f32> = (0..20).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let a = Tensor::<CpuRuntime>::try_from_slice(&data, &[5, 4], &device).unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // k = min(5, 4) = 4
    assert_eq!(svd.u.shape(), &[5, 4]);
    assert_eq!(svd.s.shape(), &[4]);
    assert_eq!(svd.vt.shape(), &[4, 4]);

    let s: Vec<f32> = svd.s.to_vec();
    assert_descending(&s, "singular values");

    // Verify reconstruction
    let s_diag = LinalgOps::diagflat(&client, &svd.s).unwrap();
    let u_s = client.matmul(&svd.u, &s_diag).unwrap();
    let reconstructed = client.matmul(&u_s, &svd.vt).unwrap();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();

    let diff: Vec<f32> = reconstructed_data
        .iter()
        .zip(data.iter())
        .map(|(r, a)| r - a)
        .collect();
    let error = frobenius_norm(&diff);
    assert!(error < 1e-4, "Reconstruction error too large: {}", error);
}

// ============================================================================
// Orthogonality Tests
// ============================================================================

#[test]
fn test_svd_u_orthogonality() {
    let (client, device) = create_client();

    // 4x3 matrix
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[4, 3],
        &device,
    )
    .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // U^T @ U should be identity (k x k)
    let ut = svd.u.transpose(0, 1).unwrap();
    let ut_u = client.matmul(&ut, &svd.u).unwrap();
    let ut_u_data: Vec<f32> = ut_u.to_vec();

    let k = 3; // min(4, 3)
    assert_near_identity(&ut_u_data, k, 1e-5, "U^T @ U orthogonality");
}

#[test]
fn test_svd_v_orthogonality() {
    let (client, device) = create_client();

    // 3x4 matrix
    let a = Tensor::<CpuRuntime>::try_from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[3, 4],
        &device,
    )
    .unwrap();

    let svd = client.svd_decompose(&a).unwrap();

    // V @ V^T should be identity (V^T @ V = I implies V @ V^T = I for thin V)
    let v = svd.vt.transpose(0, 1).unwrap();
    let vt_v = client.matmul(&svd.vt, &v).unwrap();
    let vt_v_data: Vec<f32> = vt_v.to_vec();

    let k = 3; // min(3, 4)
    assert_near_identity(&vt_v_data, k, 1e-5, "V^T @ V orthogonality");
}
