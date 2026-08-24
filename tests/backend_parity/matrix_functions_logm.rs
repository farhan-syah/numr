//! Integration tests for matrix logarithm (logm)

// migrated: use shared common utilities from parent integration test crate

use crate::common::{assert_allclose_f64, create_cpu_client};
use numr::algorithm::linalg::MatrixFunctionsAlgorithms;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_logm_identity() {
    // log(I) = 0
    let (client, device) = create_cpu_client();

    let identity = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.logm(&identity).expect("logm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "logm(I) = 0");
}

#[test]
fn test_logm_diagonal() {
    // log(diag([a, b, c])) = diag([log(a), log(b), log(c)])
    let (client, device) = create_cpu_client();

    let e = std::f64::consts::E;
    let diag_matrix = Tensor::<CpuRuntime>::try_from_slice(
        &[e, 0.0, 0.0, 0.0, e * e, 0.0, 0.0, 0.0, e * e * e],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.logm(&diag_matrix).expect("logm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "logm(diag)");
}

#[test]
fn test_logm_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[std::f64::consts::E], &[1, 1], &device).unwrap();
    let result = client.logm(&a).expect("logm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "logm(1x1)");
}

#[test]
fn test_logm_nonpositive_error() {
    let (client, device) = create_cpu_client();

    // Matrix with non-positive eigenvalue
    let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0], &[1, 1], &device).unwrap();

    let result = client.logm(&a);
    assert!(result.is_err(), "logm of zero should fail");
}

#[test]
fn test_logm_expm_inverse() {
    // log(exp(A)) = A for suitable A
    let (client, device) = create_cpu_client();

    // Small matrix to avoid numerical issues
    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[0.1, 0.05, 0.05, 0.2], &[2, 2], &device).unwrap();

    let exp_a = client.expm(&a).expect("expm should succeed");
    let log_exp_a = client.logm(&exp_a).expect("logm should succeed");

    let result_data: Vec<f64> = log_exp_a.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    assert_allclose_f64(&result_data, &a_data, 1e-6, 1e-6, "logm(expm(A)) = A");
}

#[test]
fn test_logm_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::try_zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client.logm(&empty).expect("logm of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}

#[test]
fn test_expm_logm_roundtrip() {
    // exp(log(A)) = A for positive definite A
    let (client, device) = create_cpu_client();

    // Positive definite matrix
    let a = Tensor::<CpuRuntime>::try_from_slice(&[2.0, 0.5, 0.5, 3.0], &[2, 2], &device).unwrap();

    let log_a = client.logm(&a).expect("logm should succeed");
    let exp_log_a = client.expm(&log_a).expect("expm should succeed");

    let result_data: Vec<f64> = exp_log_a.to_vec();
    let a_data: Vec<f64> = a.to_vec();

    assert_allclose_f64(&result_data, &a_data, 1e-6, 1e-6, "expm(logm(A)) = A");
}

#[test]
fn test_logm_f32() {
    let (client, device) = create_cpu_client();

    let e = std::f32::consts::E;
    let a = Tensor::<CpuRuntime>::try_from_slice(&[e, 0.0, 0.0, e * e], &[2, 2], &device).unwrap();
    let result = client.logm(&a).expect("logm f32 should succeed");

    let result_data: Vec<f32> = result.to_vec();
    let expected = [1.0_f32, 0.0, 0.0, 2.0];

    for (i, (x, y)) in result_data.iter().zip(expected.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < 1e-5,
            "logm f32: element {} differs: {} vs {}",
            i,
            x,
            y
        );
    }
}
