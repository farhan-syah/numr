//! Integration tests for signm, fractional matrix power, and funm

// migrated: use shared common utilities from parent integration test crate

use crate::common::{assert_allclose_f64, create_cpu_client};
use numr::algorithm::linalg::MatrixFunctionsAlgorithms;
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

// ============================================================================
// Matrix Sign Function (signm) Tests
// ============================================================================

#[test]
fn test_signm_positive_definite() {
    // sign(I) = I
    let (client, device) = create_cpu_client();

    let identity = Tensor::<CpuRuntime>::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.signm(&identity).expect("signm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-6, 1e-6, "signm(I) = I");
}

#[test]
fn test_signm_negative_definite() {
    // sign(-I) = -I
    let (client, device) = create_cpu_client();

    let neg_identity =
        Tensor::<CpuRuntime>::from_slice(&[-1.0, 0.0, 0.0, -1.0], &[2, 2], &device).unwrap();
    let result = client.signm(&neg_identity).expect("signm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![-1.0, 0.0, 0.0, -1.0];

    assert_allclose_f64(&result_data, &expected, 1e-6, 1e-6, "signm(-I) = -I");
}

#[test]
fn test_signm_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[5.0], &[1, 1], &device).unwrap();
    let result = client.signm(&a).expect("signm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    assert_eq!(result_data[0], 1.0, "signm(5) = 1");

    let b = Tensor::<CpuRuntime>::from_slice(&[-3.0], &[1, 1], &device).unwrap();
    let result_b = client.signm(&b).expect("signm should succeed");

    let result_b_data: Vec<f64> = result_b.to_vec();
    assert_eq!(result_b_data[0], -1.0, "signm(-3) = -1");
}

#[test]
fn test_signm_involutory() {
    // sign(A)^2 = I
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 1.0, 0.5, 3.0], &[2, 2], &device).unwrap();
    let sign_a = client.signm(&a).expect("signm should succeed");

    let sign_squared = client
        .matmul(&sign_a, &sign_a)
        .expect("matmul should succeed");

    let result_data: Vec<f64> = sign_squared.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-5, 1e-5, "signm(A)^2 = I");
}

#[test]
fn test_signm_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client.signm(&empty).expect("signm of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}

// ============================================================================
// Fractional Matrix Power Tests
// ============================================================================

#[test]
fn test_fractional_power_zero() {
    // A^0 = I
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 1.0, 0.5, 3.0], &[2, 2], &device).unwrap();
    let result = client
        .fractional_matrix_power(&a, 0.0)
        .expect("A^0 should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "A^0 = I");
}

#[test]
fn test_fractional_power_one() {
    // A^1 = A
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 1.0, 0.5, 3.0], &[2, 2], &device).unwrap();
    let result = client
        .fractional_matrix_power(&a, 1.0)
        .expect("A^1 should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    assert_allclose_f64(&result_data, &a_data, 1e-10, 1e-10, "A^1 = A");
}

#[test]
fn test_fractional_power_half() {
    // A^0.5 = sqrtm(A)
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[4.0, 0.0, 0.0, 9.0], &[2, 2], &device).unwrap();
    let result = client
        .fractional_matrix_power(&a, 0.5)
        .expect("A^0.5 should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![2.0, 0.0, 0.0, 3.0];

    assert_allclose_f64(&result_data, &expected, 1e-6, 1e-6, "A^0.5 = sqrtm(A)");
}

#[test]
fn test_fractional_power_integer() {
    // A^2 via integer power
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 0.0, 3.0], &[2, 2], &device).unwrap();

    let result = client
        .fractional_matrix_power(&a, 2.0)
        .expect("A^2 should succeed");

    // A^2 = A @ A
    let expected = client.matmul(&a, &a).expect("matmul should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected_data: Vec<f64> = expected.to_vec();

    assert_allclose_f64(&result_data, &expected_data, 1e-10, 1e-10, "A^2");
}

#[test]
fn test_fractional_power_negative() {
    // A^{-1} = inverse(A)
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 1.0, 1.0, 3.0], &[2, 2], &device).unwrap();

    let result = client
        .fractional_matrix_power(&a, -1.0)
        .expect("A^{-1} should succeed");

    // Verify: A @ A^{-1} = I
    let product = client.matmul(&a, &result).expect("matmul should succeed");

    let product_data: Vec<f64> = product.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&product_data, &expected, 1e-8, 1e-8, "A @ A^{-1} = I");
}

#[test]
fn test_fractional_power_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[8.0], &[1, 1], &device).unwrap();
    let result = client
        .fractional_matrix_power(&a, 1.0 / 3.0)
        .expect("8^(1/3) should succeed");

    let result_data: Vec<f64> = result.to_vec();
    assert!((result_data[0] - 2.0).abs() < 1e-8, "8^(1/3) = 2");
}

#[test]
fn test_fractional_power_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client
        .fractional_matrix_power(&empty, 2.5)
        .expect("fractional power of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}

// ============================================================================
// General Matrix Function (funm) Tests
// ============================================================================

#[test]
fn test_funm_exp() {
    // funm with exp should match expm
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[0.5, 0.1, 0.1, 0.3], &[2, 2], &device).unwrap();

    let expm_result = client.expm(&a).expect("expm should succeed");
    let funm_result = client
        .funm(&a, |x| x.exp())
        .expect("funm(exp) should succeed");

    let expm_data: Vec<f64> = expm_result.to_vec();
    let funm_data: Vec<f64> = funm_result.to_vec();

    assert_allclose_f64(&funm_data, &expm_data, 1e-6, 1e-6, "funm(exp) = expm");
}

#[test]
fn test_funm_identity() {
    // funm with identity function returns the matrix
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

    let result = client.funm(&a, |x| x).expect("funm(id) should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    assert_allclose_f64(&result_data, &a_data, 1e-8, 1e-8, "funm(id) = A");
}

#[test]
fn test_funm_diagonal() {
    // funm on diagonal: f(diag(d)) = diag(f(d))
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 0.0, 0.0, 3.0], &[2, 2], &device).unwrap();

    let result = client
        .funm(&a, |x| x * x + 1.0)
        .expect("funm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    // f(2) = 5, f(3) = 10
    let expected = vec![5.0, 0.0, 0.0, 10.0];

    assert_allclose_f64(
        &result_data,
        &expected,
        1e-8,
        1e-8,
        "funm(x^2+1) on diagonal",
    );
}

#[test]
fn test_funm_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[4.0], &[1, 1], &device).unwrap();
    let result = client
        .funm(&a, |x| x.sqrt())
        .expect("funm(sqrt) should succeed");

    let result_data: Vec<f64> = result.to_vec();
    assert!((result_data[0] - 2.0).abs() < 1e-10, "funm(sqrt)(4) = 2");
}

#[test]
fn test_funm_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client
        .funm(&empty, |x| x.sin())
        .expect("funm of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}
