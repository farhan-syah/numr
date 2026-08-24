//! Unit tests for advanced decompositions

use super::super::super::{CpuClient, CpuDevice, CpuRuntime};
use crate::algorithm::linalg::{LinearAlgebraAlgorithms, SchurDecomposition};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

fn create_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

fn assert_close(a: f64, b: f64, tol: f64) {
    assert!(
        (a - b).abs() < tol,
        "Expected {} to be close to {}, diff = {}",
        a,
        b,
        (a - b).abs()
    );
}

fn matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut c = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                c[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], n: usize) -> Vec<f64> {
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[j * n + i] = a[i * n + j];
        }
    }
    t
}

fn is_orthogonal(q: &[f64], n: usize, tol: f64) -> bool {
    let qt = transpose(q, n);
    let qtq = matrix_multiply(&qt, q, n);
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            if (qtq[i * n + j] - expected).abs() > tol {
                return false;
            }
        }
    }
    true
}

// =========================================================================
// rsf2csf tests
// =========================================================================

#[test]
fn test_rsf2csf_1x1_real_eigenvalue() {
    let client = create_client();
    let device = client.device();

    let t = Tensor::<CpuRuntime>::from_slice(&[3.0f64], &[1, 1], device).unwrap();
    let z = Tensor::<CpuRuntime>::from_slice(&[1.0f64], &[1, 1], device).unwrap();
    let schur = SchurDecomposition { z, t };

    let result = client.rsf2csf(&schur).unwrap();

    let t_real: Vec<f64> = result.t_real.to_vec();
    let t_imag: Vec<f64> = result.t_imag.to_vec();

    assert_close(t_real[0], 3.0, 1e-10);
    assert_close(t_imag[0], 0.0, 1e-10);
}

#[test]
fn test_rsf2csf_2x2_complex_block() {
    let client = create_client();
    let device = client.device();

    // 2x2 block with complex eigenvalues: eigenvalues are 2 ± i
    let t = Tensor::<CpuRuntime>::from_slice(&[2.0f64, -1.0, 1.0, 2.0], &[2, 2], device).unwrap();
    let z = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();
    let schur = SchurDecomposition { z, t };

    let result = client.rsf2csf(&schur).unwrap();

    let t_real: Vec<f64> = result.t_real.to_vec();
    let t_imag: Vec<f64> = result.t_imag.to_vec();

    // Diagonal should have eigenvalues 2 + i and 2 - i
    assert_close(t_real[0], 2.0, 1e-6);
    assert_close(t_imag[0], 1.0, 1e-6);
    assert_close(t_real[3], 2.0, 1e-6);
    assert_close(t_imag[3], -1.0, 1e-6);

    // Subdiagonal should be zero
    assert_close(t_real[2], 0.0, 1e-10);
    assert_close(t_imag[2], 0.0, 1e-10);
}

#[test]
fn test_rsf2csf_empty_matrix() {
    let client = create_client();
    let device = client.device();

    let t = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();
    let z = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();
    let schur = SchurDecomposition { z, t };

    let result = client.rsf2csf(&schur).unwrap();
    assert_eq!(result.t_real.shape(), &[0, 0]);
}

// =========================================================================
// QZ decomposition tests
// =========================================================================

#[test]
fn test_qz_identity_matrices() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();

    let eig_real: Vec<f64> = result.eigenvalues_real.to_vec();
    let eig_imag: Vec<f64> = result.eigenvalues_imag.to_vec();

    for i in 0..2 {
        assert_close(eig_real[i], 1.0, 1e-6);
        assert_close(eig_imag[i], 0.0, 1e-10);
    }
}

#[test]
fn test_qz_diagonal_matrices() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 0.0, 0.0, 6.0], &[2, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 2.0], &[2, 2], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();

    let eig_real: Vec<f64> = result.eigenvalues_real.to_vec();

    let mut eigs: Vec<f64> = eig_real.clone();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());

    assert_close(eigs[0], 2.0, 1e-6);
    assert_close(eigs[1], 3.0, 1e-6);
}

#[test]
fn test_qz_orthogonality() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f64, 6.0, 7.0, 8.0], &[2, 2], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();

    let q: Vec<f64> = result.q.to_vec();
    let z: Vec<f64> = result.z.to_vec();

    assert!(is_orthogonal(&q, 2, 1e-6), "Q is not orthogonal");
    assert!(is_orthogonal(&z, 2, 1e-6), "Z is not orthogonal");
}

#[test]
fn test_qz_empty_matrix() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();
    assert_eq!(result.s.shape(), &[0, 0]);
    assert_eq!(result.eigenvalues_real.shape(), &[0]);
}

fn is_upper_triangular(mat: &[f64], n: usize, tol: f64) -> bool {
    for i in 0..n {
        for j in 0..i {
            if mat[i * n + j].abs() > tol {
                return false;
            }
        }
    }
    true
}

#[test]
fn test_qz_t_upper_triangular() {
    let client = create_client();
    let device = client.device();

    // 3×3 non-trivial matrices
    #[rustfmt::skip]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f64, 2.0, 3.0,
          4.0, 5.0, 6.0,
          7.0, 8.0, 10.0],
        &[3, 3], device).unwrap();
    #[rustfmt::skip]
    let b = Tensor::<CpuRuntime>::from_slice(
        &[2.0f64, 1.0, 0.0,
          1.0, 3.0, 1.0,
          0.0, 1.0, 2.0],
        &[3, 3], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();
    let t_data: Vec<f64> = result.t.to_vec();
    assert!(
        is_upper_triangular(&t_data, 3, 1e-10),
        "T is not upper triangular: {:?}",
        t_data
    );
}

#[test]
fn test_qz_factorization_identity() {
    let client = create_client();
    let device = client.device();

    #[rustfmt::skip]
    let a_data = [1.0f64, 2.0, 3.0,
                  4.0, 5.0, 6.0,
                  7.0, 8.0, 10.0];
    #[rustfmt::skip]
    let b_data = [2.0f64, 1.0, 0.0,
                  1.0, 3.0, 1.0,
                  0.0, 1.0, 2.0];

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[3, 3], device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[3, 3], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();
    let q_vec: Vec<f64> = result.q.to_vec();
    let z_vec: Vec<f64> = result.z.to_vec();
    let s_vec: Vec<f64> = result.s.to_vec();
    let t_vec: Vec<f64> = result.t.to_vec();

    let n = 3;
    // Verify A = Q * S * Z^T
    let zt = transpose(&z_vec, n);
    let qs = matrix_multiply(&q_vec, &s_vec, n);
    let qszt = matrix_multiply(&qs, &zt, n);
    for i in 0..n * n {
        assert!(
            (qszt[i] - a_data[i]).abs() < 1e-8,
            "A = Q*S*Z^T failed at {}: {} vs {}",
            i,
            qszt[i],
            a_data[i]
        );
    }

    // Verify B = Q * T * Z^T
    let qt_b = matrix_multiply(&q_vec, &t_vec, n);
    let qtzt = matrix_multiply(&qt_b, &zt, n);
    for i in 0..n * n {
        assert!(
            (qtzt[i] - b_data[i]).abs() < 1e-8,
            "B = Q*T*Z^T failed at {}: {} vs {}",
            i,
            qtzt[i],
            b_data[i]
        );
    }

    // Verify Q and Z are orthogonal
    assert!(is_orthogonal(&q_vec, n, 1e-10), "Q not orthogonal");
    assert!(is_orthogonal(&z_vec, n, 1e-10), "Z not orthogonal");

    // Verify T is upper triangular
    assert!(
        is_upper_triangular(&t_vec, n, 1e-10),
        "T not upper triangular"
    );
}

#[test]
fn test_qz_5x5_upper_triangular_t() {
    let client = create_client();
    let device = client.device();

    // 5×5 test with a mix of real and potentially complex eigenvalues
    #[rustfmt::skip]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[4.0f64, 1.0, 2.0, 0.0, 1.0,
          1.0, 3.0, 1.0, 1.0, 0.0,
          2.0, 1.0, 5.0, 2.0, 1.0,
          0.0, 1.0, 2.0, 6.0, 3.0,
          1.0, 0.0, 1.0, 3.0, 7.0],
        &[5, 5], device).unwrap();
    #[rustfmt::skip]
    let b = Tensor::<CpuRuntime>::from_slice(
        &[2.0f64, 0.5, 0.0, 0.0, 0.0,
          0.5, 3.0, 0.5, 0.0, 0.0,
          0.0, 0.5, 2.0, 0.5, 0.0,
          0.0, 0.0, 0.5, 3.0, 0.5,
          0.0, 0.0, 0.0, 0.5, 2.0],
        &[5, 5], device).unwrap();

    let result = client.qz_decompose(&a, &b).unwrap();
    let t_data: Vec<f64> = result.t.to_vec();
    let q_data: Vec<f64> = result.q.to_vec();
    let z_data: Vec<f64> = result.z.to_vec();
    let s_data: Vec<f64> = result.s.to_vec();

    let n = 5;
    assert!(
        is_upper_triangular(&t_data, n, 1e-8),
        "T not upper triangular for 5×5"
    );
    assert!(is_orthogonal(&q_data, n, 1e-8), "Q not orthogonal for 5×5");
    assert!(is_orthogonal(&z_data, n, 1e-8), "Z not orthogonal for 5×5");

    // Verify A = Q * S * Z^T
    let a_orig: Vec<f64> = a.to_vec();
    let zt = transpose(&z_data, n);
    let qs = matrix_multiply(&q_data, &s_data, n);
    let qszt = matrix_multiply(&qs, &zt, n);
    for i in 0..n * n {
        assert!(
            (qszt[i] - a_orig[i]).abs() < 1e-6,
            "A = Q*S*Z^T failed at {}: {} vs {}",
            i,
            qszt[i],
            a_orig[i]
        );
    }

    // Verify B = Q * T * Z^T
    let b_orig: Vec<f64> = b.to_vec();
    let qt_b = matrix_multiply(&q_data, &t_data, n);
    let qtzt = matrix_multiply(&qt_b, &zt, n);
    for i in 0..n * n {
        assert!(
            (qtzt[i] - b_orig[i]).abs() < 1e-6,
            "B = Q*T*Z^T failed at {}: {} vs {}",
            i,
            qtzt[i],
            b_orig[i]
        );
    }
}

// =========================================================================
// Polar decomposition tests
// =========================================================================

#[test]
fn test_polar_orthogonal_input() {
    let client = create_client();
    let device = client.device();

    let angle = std::f64::consts::PI / 4.0;
    let c = angle.cos();
    let s = angle.sin();
    let a = Tensor::<CpuRuntime>::from_slice(&[c, -s, s, c], &[2, 2], device).unwrap();

    let result = client.polar_decompose(&a).unwrap();

    let u: Vec<f64> = result.u.to_vec();
    let p: Vec<f64> = result.p.to_vec();

    assert!(is_orthogonal(&u, 2, 1e-6), "U is not orthogonal");

    assert_close(p[0], 1.0, 1e-6);
    assert_close(p[1], 0.0, 1e-6);
    assert_close(p[2], 0.0, 1e-6);
    assert_close(p[3], 1.0, 1e-6);
}

#[test]
fn test_polar_symmetric_input() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f64, 1.0, 1.0, 2.0], &[2, 2], device).unwrap();

    let result = client.polar_decompose(&a).unwrap();

    let u: Vec<f64> = result.u.to_vec();
    let p: Vec<f64> = result.p.to_vec();

    assert!(is_orthogonal(&u, 2, 1e-6), "U is not orthogonal");
    assert_close(p[1], p[2], 1e-6);
}

#[test]
fn test_polar_reconstruction() {
    let client = create_client();
    let device = client.device();

    let a_data = [1.0f64, 2.0, 3.0, 4.0];
    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], device).unwrap();

    let result = client.polar_decompose(&a).unwrap();

    let u: Vec<f64> = result.u.to_vec();
    let p: Vec<f64> = result.p.to_vec();

    let reconstructed = matrix_multiply(&u, &p, 2);

    for i in 0..4 {
        assert_close(reconstructed[i], a_data[i], 1e-6);
    }
}

#[test]
fn test_polar_p_symmetric() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[2, 2], device).unwrap();

    let result = client.polar_decompose(&a).unwrap();
    let p: Vec<f64> = result.p.to_vec();

    assert_close(p[1], p[2], 1e-6);
}

#[test]
fn test_polar_empty_matrix() {
    let client = create_client();
    let device = client.device();

    let a = Tensor::<CpuRuntime>::from_slice(&[] as &[f64], &[0, 0], device).unwrap();

    let result = client.polar_decompose(&a).unwrap();
    assert_eq!(result.u.shape(), &[0, 0]);
    assert_eq!(result.p.shape(), &[0, 0]);
}
