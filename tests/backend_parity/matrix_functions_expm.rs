//! Integration tests for matrix exponential (expm)

// migrated: use shared common utilities from parent integration test crate

use crate::common::{assert_allclose_f64, create_cpu_client};
use numr::algorithm::linalg::MatrixFunctionsAlgorithms;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[test]
fn test_expm_identity() {
    // exp(0) = I
    let (client, device) = create_cpu_client();

    let zeros = Tensor::<CpuRuntime>::try_zeros(&[3, 3], DType::F64, &device).unwrap();
    let result = client.expm(&zeros).expect("expm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "expm(0) = I");
}

#[test]
fn test_expm_diagonal() {
    // exp(diag([a, b, c])) = diag([exp(a), exp(b), exp(c)])
    let (client, device) = create_cpu_client();

    let diag_matrix = Tensor::<CpuRuntime>::try_from_slice(
        &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0],
        &[3, 3],
        &device,
    )
    .unwrap();
    let result = client.expm(&diag_matrix).expect("expm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![
        1.0_f64.exp(),
        0.0,
        0.0,
        0.0,
        2.0_f64.exp(),
        0.0,
        0.0,
        0.0,
        3.0_f64.exp(),
    ];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "expm(diag)");
}

#[test]
fn test_expm_1x1() {
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[2.5], &[1, 1], &device).unwrap();
    let result = client.expm(&a).expect("expm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![2.5_f64.exp()];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "expm(1x1)");
}

#[test]
fn test_expm_2x2_nilpotent() {
    // For strictly upper triangular matrix [[0, 1], [0, 0]]:
    // exp(A) = I + A = [[1, 1], [0, 1]]
    let (client, device) = create_cpu_client();

    let a = Tensor::<CpuRuntime>::try_from_slice(&[0.0, 1.0, 0.0, 0.0], &[2, 2], &device).unwrap();
    let result = client.expm(&a).expect("expm should succeed");

    let result_data: Vec<f64> = result.to_vec();
    let expected = vec![1.0, 1.0, 0.0, 1.0];

    assert_allclose_f64(&result_data, &expected, 1e-10, 1e-10, "expm(nilpotent)");
}

#[test]
fn test_expm_empty() {
    let (client, device) = create_cpu_client();

    let empty = Tensor::<CpuRuntime>::try_zeros(&[0, 0], DType::F64, &device).unwrap();
    let result = client.expm(&empty).expect("expm of empty should succeed");

    assert_eq!(result.shape(), &[0, 0]);
}

#[test]
fn test_expm_f32() {
    let (client, device) = create_cpu_client();

    let a =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0_f32, 0.0, 0.0, 2.0], &[2, 2], &device).unwrap();
    let result = client.expm(&a).expect("expm f32 should succeed");

    let result_data: Vec<f32> = result.to_vec();
    let expected = [1.0_f32.exp(), 0.0, 0.0, 2.0_f32.exp()];

    for (i, (x, y)) in result_data.iter().zip(expected.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < 1e-5,
            "expm f32: element {} differs: {} vs {}",
            i,
            x,
            y
        );
    }
}
