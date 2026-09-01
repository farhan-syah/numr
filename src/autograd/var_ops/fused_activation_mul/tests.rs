//! Tests for the fused activation-multiply autograd op.

use super::*;
use crate::autograd::backward;
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

#[test]
fn test_silu_mul_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, -1.0], &[3], &device).unwrap(),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap(),
        false,
    );

    let output = var_silu_mul(&a, &b, &client).unwrap();
    let data: Vec<f32> = output.tensor().to_vec();

    // silu(0)*1 = 0, silu(1)*2, silu(-1)*3
    assert!(data[0].abs() < 1e-6);
    let silu_1 = 1.0 / (1.0 + (-1.0f32).exp());
    assert!((data[1] - silu_1 * 2.0).abs() < 1e-4);
    let silu_neg1 = -1.0 / (1.0 + 1.0f32.exp());
    assert!((data[2] - silu_neg1 * 3.0).abs() < 1e-4);
}

#[test]
fn test_silu_mul_matches_separate_ops() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data = vec![0.5f32, -0.3, 1.2, -2.0, 0.0];
    let b_data = vec![1.0f32, 2.0, 0.5, -1.0, 3.0];

    // Fused
    let fused = client
        .silu_mul(
            &Tensor::<CpuRuntime>::from_slice(&a_data, &[5], &device).unwrap(),
            &Tensor::<CpuRuntime>::from_slice(&b_data, &[5], &device).unwrap(),
        )
        .unwrap();

    // Separate
    let silu_a = client
        .silu(&Tensor::<CpuRuntime>::from_slice(&a_data, &[5], &device).unwrap())
        .unwrap();
    let separate = client
        .mul(
            &silu_a,
            &Tensor::<CpuRuntime>::from_slice(&b_data, &[5], &device).unwrap(),
        )
        .unwrap();

    let fused_v: Vec<f32> = fused.to_vec();
    let separate_v: Vec<f32> = separate.to_vec();
    for i in 0..5 {
        assert!(
            (fused_v[i] - separate_v[i]).abs() < 1e-5,
            "mismatch at {i}: {} vs {}",
            fused_v[i],
            separate_v[i]
        );
    }
}

#[test]
fn test_silu_mul_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0], &[2], &device).unwrap(),
        true,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
        true,
    );

    let output = var_silu_mul(&a, &b, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
    let d_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();

    // Verify d_b = silu(a)
    for (i, &g) in [1.0f32, -1.0].iter().enumerate() {
        let expected = g / (1.0 + (-g).exp());
        assert!(
            (d_b[i] - expected).abs() < 1e-4,
            "d_b[{i}]: got {}, expected {expected}",
            d_b[i]
        );
    }

    // Verify d_a = b * silu'(a)
    for (i, (&g, &u)) in [1.0f32, -1.0].iter().zip([2.0f32, 3.0].iter()).enumerate() {
        let sig = 1.0 / (1.0 + (-g).exp());
        let silu_g = g * sig;
        let silu_deriv = sig * (1.0 + g - silu_g);
        let expected = u * silu_deriv;
        assert!(
            (d_a[i] - expected).abs() < 1e-4,
            "d_a[{i}]: got {}, expected {expected}",
            d_a[i]
        );
    }
}

#[test]
fn test_relu_mul_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 2.0], &[3], &device).unwrap(),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0, 5.0], &[3], &device).unwrap(),
        false,
    );

    let output = var_relu_mul(&a, &b, &client).unwrap();
    let data: Vec<f32> = output.tensor().to_vec();
    assert!((data[0] - 0.0).abs() < 1e-6);
    assert!((data[1] - 0.0).abs() < 1e-6);
    assert!((data[2] - 10.0).abs() < 1e-6);
}

#[test]
fn test_sigmoid_mul_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap(),
        true,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
        true,
    );

    let output = var_sigmoid_mul(&a, &b, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_a: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
    let d_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();

    // d_b = sigmoid(0) = 0.5
    assert!((d_b[0] - 0.5).abs() < 1e-4);

    // d_a = b * sigmoid'(0) = 2 * sigmoid(0)*(1-sigmoid(0)) = 2 * 0.25 = 0.5
    assert!((d_a[0] - 0.5).abs() < 1e-4);
}

#[test]
fn test_no_grad() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap(),
        false,
    );
    let b = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
        false,
    );

    let output = var_gelu_mul(&a, &b, &client).unwrap();
    assert!(!output.requires_grad());
}
