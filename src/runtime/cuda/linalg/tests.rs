//! Unit tests for CUDA linear algebra

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::algorithm::linalg::LinearAlgebraAlgorithms;
use crate::ops::MatmulOps;
use crate::runtime::cuda::{CudaDevice, is_cuda_available};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

fn create_client() -> Option<CudaClient> {
    if !is_cuda_available() {
        return None;
    }
    let device = CudaDevice::new(0);
    Some(CudaRuntime::default_client(&device))
}

#[test]
fn test_trace() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // 2x2 matrix: [[1, 2], [3, 4]]
    // trace = 1 + 4 = 5
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let t = LinearAlgebraAlgorithms::trace(&client, &a).unwrap();
    let result: Vec<f32> = t.to_vec();

    assert!((result[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_diag() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // 2x3 matrix
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], device)
        .unwrap();

    let d = LinearAlgebraAlgorithms::diag(&client, &a).unwrap();
    let result: Vec<f32> = d.to_vec();

    assert_eq!(result.len(), 2);
    assert!((result[0] - 1.0).abs() < 1e-5);
    assert!((result[1] - 5.0).abs() < 1e-5);
}

#[test]
fn test_diagflat() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

    let m = LinearAlgebraAlgorithms::diagflat(&client, &a).unwrap();
    let result: Vec<f32> = m.to_vec();

    assert_eq!(m.shape(), &[3, 3]);
    // Expected: [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
    assert!((result[0] - 1.0).abs() < 1e-5); // [0,0]
    assert!((result[1]).abs() < 1e-5); // [0,1]
    assert!((result[4] - 2.0).abs() < 1e-5); // [1,1]
    assert!((result[8] - 3.0).abs() < 1e-5); // [2,2]
}

#[test]
fn test_lu_decomposition() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // 2x2 matrix: [[4, 3], [6, 3]]
    let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 3.0, 6.0, 3.0], &[2, 2], device).unwrap();

    let lu = client.lu_decompose(&a).unwrap();

    assert_eq!(lu.lu.shape(), &[2, 2]);
    assert_eq!(lu.pivots.shape(), &[2]);
}

#[test]
fn test_cholesky() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Symmetric positive definite: [[4, 2], [2, 5]]
    let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 2.0, 2.0, 5.0], &[2, 2], device).unwrap();

    let chol = client.cholesky_decompose(&a).unwrap();

    assert_eq!(chol.l.shape(), &[2, 2]);

    // L should be lower triangular
    let l_data: Vec<f32> = chol.l.to_vec();
    assert!((l_data[1]).abs() < 1e-5); // Upper triangle should be 0
}

#[test]
fn test_det() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // 2x2 matrix: [[1, 2], [3, 4]]
    // det = 1*4 - 2*3 = -2
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let d = LinearAlgebraAlgorithms::det(&client, &a).unwrap();
    let result: Vec<f32> = d.to_vec();

    assert!((result[0] - (-2.0)).abs() < 1e-4);
}

#[test]
fn test_solve() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Solve [[2, 1], [1, 2]] @ x = [3, 3]
    // Solution: x = [1, 1]
    let a = Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 3.0], &[2], device).unwrap();

    let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
    let result: Vec<f32> = x.to_vec();

    assert!((result[0] - 1.0).abs() < 1e-4);
    assert!((result[1] - 1.0).abs() < 1e-4);
}

#[test]
fn test_inverse() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Test 2x2 matrix: [[4, 7], [2, 6]]
    // Inverse: [[0.6, -0.7], [-0.2, 0.4]]
    let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

    let inv = LinearAlgebraAlgorithms::inverse(&client, &a).unwrap();
    let result: Vec<f32> = inv.to_vec();

    // Check inverse values (det = 4*6 - 7*2 = 10)
    // inv = (1/10) * [[6, -7], [-2, 4]]
    assert!((result[0] - 0.6).abs() < 1e-4); // [0,0]
    assert!((result[1] - (-0.7)).abs() < 1e-4); // [0,1]
    assert!((result[2] - (-0.2)).abs() < 1e-4); // [1,0]
    assert!((result[3] - 0.4).abs() < 1e-4); // [1,1]
}

#[test]
fn test_inverse_identity() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // A @ A^-1 should equal I
    let a = Tensor::<CudaRuntime>::from_slice(&[4.0f32, 7.0, 2.0, 6.0], &[2, 2], device).unwrap();

    let inv = LinearAlgebraAlgorithms::inverse(&client, &a).unwrap();
    let product = client.matmul(&a, &inv).unwrap();
    let result: Vec<f32> = product.to_vec();

    // Should be identity matrix
    assert!((result[0] - 1.0).abs() < 1e-4); // [0,0]
    assert!((result[1]).abs() < 1e-4); // [0,1]
    assert!((result[2]).abs() < 1e-4); // [1,0]
    assert!((result[3] - 1.0).abs() < 1e-4); // [1,1]
}

#[test]
fn test_matrix_rank_full() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Full rank 2x2 matrix
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
    let result: Vec<i64> = rank.to_vec();

    assert_eq!(result[0], 2);
}

#[test]
fn test_matrix_rank_deficient() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Rank-deficient 2x2 matrix (rows are linearly dependent)
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0], &[2, 2], device).unwrap();

    let rank = LinearAlgebraAlgorithms::matrix_rank(&client, &a, None).unwrap();
    let result: Vec<i64> = rank.to_vec();

    assert_eq!(result[0], 1);
}

#[test]
fn test_qr_decomposition() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Test QR: A = Q @ R
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let qr = client.qr_decompose(&a).unwrap();

    // Verify Q @ R == A
    let reconstructed = client.matmul(&qr.q, &qr.r).unwrap();
    let a_data: Vec<f32> = a.to_vec();
    let reconstructed_data: Vec<f32> = reconstructed.to_vec();

    for i in 0..4 {
        assert!(
            (a_data[i] - reconstructed_data[i]).abs() < 1e-4,
            "Mismatch at {}: {} vs {}",
            i,
            a_data[i],
            reconstructed_data[i]
        );
    }
}

#[test]
fn test_solve_multi_rhs() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Solve A @ X = B where B has multiple columns
    // A = [[2, 1], [1, 2]], B = [[3, 4], [3, 5]]
    // Solutions: X[:, 0] = [1, 1], X[:, 1] = [1, 2]
    let a = Tensor::<CudaRuntime>::from_slice(&[2.0f32, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&[3.0f32, 4.0, 3.0, 5.0], &[2, 2], device).unwrap();

    let x = LinearAlgebraAlgorithms::solve(&client, &a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    let result: Vec<f32> = x.to_vec();

    // X[:, 0] = [1, 1] -> result[0], result[2]
    // X[:, 1] = [1, 2] -> result[1], result[3]
    assert!(
        (result[0] - 1.0).abs() < 1e-4,
        "X[0,0] = {} expected 1",
        result[0]
    );
    assert!(
        (result[1] - 1.0).abs() < 1e-4,
        "X[0,1] = {} expected 1",
        result[1]
    );
    assert!(
        (result[2] - 1.0).abs() < 1e-4,
        "X[1,0] = {} expected 1",
        result[2]
    );
    assert!(
        (result[3] - 2.0).abs() < 1e-4,
        "X[1,1] = {} expected 2",
        result[3]
    );
}

#[test]
fn test_lstsq_overdetermined() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Overdetermined system: A is 3x2, b is 3x1
    // A = [[1, 1], [1, 2], [1, 3]], b = [1, 2, 3]
    // Least squares solution minimizes ||Ax - b||^2
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device)
        .unwrap();
    let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], device).unwrap();

    let x = LinearAlgebraAlgorithms::lstsq(&client, &a, &b).unwrap();
    assert_eq!(x.shape(), &[2]);
    let result: Vec<f32> = x.to_vec();

    // For this system, the solution is approximately x = [0, 1]
    assert!((result[0]).abs() < 0.1, "x[0] = {} expected ~0", result[0]);
    assert!(
        (result[1] - 1.0).abs() < 0.1,
        "x[1] = {} expected ~1",
        result[1]
    );
}

#[test]
fn test_lstsq_multi_rhs() {
    let Some(client) = create_client() else {
        return;
    };
    let device = client.device();

    // Overdetermined system with multiple RHS
    // A is 3x2, B is 3x2
    let a = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2], device)
        .unwrap();
    // B = [[1, 2], [2, 4], [3, 6]] (second column is 2x first)
    let b = Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 2.0, 4.0, 3.0, 6.0], &[3, 2], device)
        .unwrap();

    let x = LinearAlgebraAlgorithms::lstsq(&client, &a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    let result: Vec<f32> = x.to_vec();

    // Second solution should be 2x the first
    // X[:, 0] ≈ [0, 1], X[:, 1] ≈ [0, 2]
    assert!(
        (result[0]).abs() < 0.1,
        "X[0,0] = {} expected ~0",
        result[0]
    );
    assert!(
        (result[1]).abs() < 0.1,
        "X[0,1] = {} expected ~0",
        result[1]
    );
    assert!(
        (result[2] - 1.0).abs() < 0.1,
        "X[1,0] = {} expected ~1",
        result[2]
    );
    assert!(
        (result[3] - 2.0).abs() < 0.1,
        "X[1,1] = {} expected ~2",
        result[3]
    );
}
