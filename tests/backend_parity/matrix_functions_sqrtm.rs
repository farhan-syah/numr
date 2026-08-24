//! Integration tests for matrix square root (sqrtm)

// migrated: use shared common utilities from parent integration test crate

use crate::common::{assert_allclose_f64, create_cpu_client};
use numr::algorithm::linalg::MatrixFunctionsAlgorithms;
use numr::dtype::DType;
use numr::ops::MatmulOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_sqrtm_identity() {
    // sqrt(I) = I
    let (client, device) = create_cpu_client();

    let identity = Tensor::<CpuRuntime>::from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.sqrtm(&identity).expect("sqrtm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-8, 1e-8, "sqrtm(I) = I");
}

#[test]
fn test_sqrtm_diagonal() {
    // sqrt(diag([a, b, c])) = diag([sqrt(a), sqrt(b), sqrt(c)])
    let (client, device) = create_cpu_client();

    let diag_matrix = Tensor::<CpuRuntime>::from_slice(
        &[4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.sqrtm(&diag_matrix).expect("sqrtm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];

    assert_allclose_f64(&result_data, &expected, 1e-8, 1e-8, "sqrtm(diag)");
}

#[test]
fn test_sqrtm_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[9.0], &[1, 1], &device).unwrap();
    let result = client.sqrtm(&a).expect("sqrtm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![3.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "sqrtm(1x1)");
}

#[test]
fn test_sqrtm_verify_squared() {
    // sqrt(A)^2 = A
    let (client, device) = create_cpu_client();

    // Positive definite matrix
    let a = Tensor::<CpuRuntime>::from_slice(&[4.0, 2.0, 2.0, 5.0], &[2, 2], &device).unwrap();

    let sqrt_a = client.sqrtm(&a).expect("sqrtm should succeed");

    // Compute sqrt_a @ sqrt_a
    let squared = client
        .matmul(&sqrt_a, &sqrt_a)
        .expect("matmul should succeed");

    let squared_data: Vec<f64> = squared.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    assert_allclose_f64(&squared_data, &a_data, 1e-6, 1e-6, "sqrtm(A)^2 = A");
}

#[test]
fn test_sqrtm_negative_eigenvalue_error() {
    let (client, device) = create_cpu_client();

    // Matrix with negative eigenvalue
    let a = Tensor::<CpuRuntime>::from_slice(&[-4.0], &[1, 1], &device).unwrap();

    let result = client.sqrtm(&a);
    assert!(result.is_err(), "sqrtm of negative should fail");
}

#[test]
fn test_sqrtm_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client.sqrtm(&empty).expect("sqrtm of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}

#[test]
fn test_sqrtm_f32() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::from_slice(&[4.0_f32, 0.0, 0.0, 9.0], &[2, 2], &device).unwrap();
    let result = client.sqrtm(&a).expect("sqrtm f32 should succeed");

    let result_data: Vec<f32> = result.to_vec();
    let expected = [2.0_f32, 0.0, 0.0, 3.0];

    for (i, (x, y)) in result_data.iter().zip(expected.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < 1e-4,
            "sqrtm f32: element {} differs: {} vs {}",
            i,
            x,
            y
        );
    }
}
