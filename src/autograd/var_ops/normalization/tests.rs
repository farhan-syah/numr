//! Tests for rms_norm, layer_norm, group_norm, and their fused-add variants.

use super::*;
use crate::autograd::backward;
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

#[test]
fn test_var_rms_norm_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        true,
    );

    let result = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();
    let data: Vec<f32> = result.tensor().to_vec();

    // rms = sqrt(mean([1, 4, 9, 16]) + 1e-5) = sqrt(7.5 + 1e-5) ~ 2.7386
    // output = [1/rms, 2/rms, 3/rms, 4/rms] * [1,1,1,1]
    let rms = (7.5f32 + 1e-5).sqrt();
    for i in 0..4 {
        let expected = (i as f32 + 1.0) / rms;
        assert!(
            (data[i] - expected).abs() < 1e-5,
            "data[{}] = {}, expected {}",
            i,
            data[i],
            expected,
        );
    }
}

#[test]
fn test_var_rms_norm_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap(),
        true,
    );

    let output = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();

    // Sum the output to get a scalar for backward
    // Sum over all dims to get a scalar for backward
    let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let grad_input = grads.get(input.id()).unwrap();
    let grad_weight = grads.get(weight.id()).unwrap();

    let gi: Vec<f32> = grad_input.to_vec();
    let gw: Vec<f32> = grad_weight.to_vec();

    // Verify gradients are finite and have correct shapes
    assert_eq!(gi.len(), 3);
    assert_eq!(gw.len(), 3);
    for val in &gi {
        assert!(val.is_finite(), "input gradient should be finite");
    }
    for val in &gw {
        assert!(val.is_finite(), "weight gradient should be finite");
    }
}

#[test]
fn test_var_layer_norm_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap(),
        true,
    );

    let result = var_layer_norm(&input, &weight, &bias, 1e-5, &client).unwrap();
    let data: Vec<f32> = result.tensor().to_vec();

    // mean = 2.5, var = mean([(-1.5)^2, (-0.5)^2, (0.5)^2, (1.5)^2]) = 1.25
    // rstd = 1/sqrt(1.25 + 1e-5)
    // output should have mean ~0 and unit variance
    let sum: f32 = data.iter().sum();
    assert!(sum.abs() < 1e-4, "layer norm output should have ~0 mean");
}

#[test]
fn test_var_layer_norm_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device).unwrap(),
        true,
    );

    let output = var_layer_norm(&input, &weight, &bias, 1e-5, &client).unwrap();

    // Sum over all dims to get a scalar for backward
    let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let grad_input = grads.get(input.id()).unwrap();
    let grad_weight = grads.get(weight.id()).unwrap();
    let grad_bias = grads.get(bias.id()).unwrap();

    let gi: Vec<f32> = grad_input.to_vec();
    let gw: Vec<f32> = grad_weight.to_vec();
    let gb: Vec<f32> = grad_bias.to_vec();

    // Verify shapes
    assert_eq!(gi.len(), 3);
    assert_eq!(gw.len(), 3);
    assert_eq!(gb.len(), 3);

    // For layer norm with sum loss:
    // d_bias = sum(grad_output) = [1, 1, 1] (each element contributes 1)
    for val in &gb {
        assert!(
            (*val - 1.0).abs() < 1e-5,
            "bias gradient should be 1.0, got {}",
            val,
        );
    }

    // d_input should sum to ~0 (layer norm property)
    let sum: f32 = gi.iter().sum();
    assert!(
        sum.abs() < 1e-5,
        "sum of input gradients should be ~0, got {}",
        sum,
    );

    // All gradients should be finite
    for val in &gi {
        assert!(val.is_finite());
    }
    for val in &gw {
        assert!(val.is_finite());
    }
}

#[test]
fn test_var_rms_norm_no_grad() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // When no inputs require grad, output should not track gradients
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
        false,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap(),
        false,
    );

    let result = var_rms_norm(&input, &weight, 1e-5, &client).unwrap();
    assert!(!result.requires_grad());
    assert!(result.grad_fn().is_none());
}

#[test]
fn test_var_group_norm_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [batch=1, channels=4, spatial=3], 2 groups
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[1, 4, 3],
            &device,
        )
        .unwrap(),
        false,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        false,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap(),
        false,
    );

    let result = var_group_norm(&input, &weight, &bias, 2, 1e-5, &client).unwrap();
    assert_eq!(result.tensor().shape(), &[1, 4, 3]);

    // Each group should have approximately zero mean
    let data: Vec<f32> = result.tensor().to_vec();
    // Group 0: channels 0,1 → indices 0..6
    let group0_sum: f32 = data[0..6].iter().sum();
    assert!(
        group0_sum.abs() < 1e-4,
        "group 0 mean should be ~0, sum={group0_sum}"
    );
    // Group 1: channels 2,3 → indices 6..12
    let group1_sum: f32 = data[6..12].iter().sum();
    assert!(
        group1_sum.abs() < 1e-4,
        "group 1 mean should be ~0, sum={group1_sum}"
    );
}

#[test]
fn test_var_group_norm_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [batch=1, channels=4, spatial=2], 2 groups
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[1, 4, 2],
            &device,
        )
        .unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap(),
        true,
    );

    let output = var_group_norm(&input, &weight, &bias, 2, 1e-5, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    let d_bias: Vec<f32> = grads.get(bias.id()).unwrap().to_vec();

    assert_eq!(d_input.len(), 8);
    assert_eq!(d_weight.len(), 4);
    assert_eq!(d_bias.len(), 4);

    // d_bias should be sum of grad_output over batch and spatial = [2, 2, 2, 2]
    for &b in &d_bias {
        assert!((b - 2.0).abs() < 1e-5, "d_bias should be 2.0, got {b}");
    }

    // All gradients should be finite
    for v in d_input.iter().chain(d_weight.iter()) {
        assert!(v.is_finite());
    }
}

#[test]
fn test_var_fused_add_rms_norm_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap(),
        true,
    );
    let residual = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[1, 4], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        true,
    );

    let result = var_fused_add_rms_norm(&x, &residual, &weight, 1e-5, &client).unwrap();
    let data: Vec<f32> = result.tensor().to_vec();

    assert_eq!(data.len(), 4);
    for val in &data {
        assert!(val.is_finite());
    }
}

#[test]
fn test_var_fused_add_rms_norm_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
        true,
    );
    let residual = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap(),
        true,
    );

    let output = var_fused_add_rms_norm(&x, &residual, &weight, 1e-5, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
    let gr: Vec<f32> = grads.get(residual.id()).unwrap().to_vec();
    let gw: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();

    assert_eq!(gx.len(), 3);
    assert_eq!(gr.len(), 3);
    assert_eq!(gw.len(), 3);

    // x and residual should get the same gradient
    for (a, b) in gx.iter().zip(gr.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "x and residual grads must match: {a} vs {b}"
        );
    }
    for val in gx.iter().chain(gw.iter()) {
        assert!(val.is_finite());
    }
}

#[test]
fn test_var_fused_add_rms_norm_no_grad() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap(),
        false,
    );
    let residual = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[1, 2], &device).unwrap(),
        false,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device).unwrap(),
        false,
    );

    let result = var_fused_add_rms_norm(&x, &residual, &weight, 1e-5, &client).unwrap();
    assert!(!result.requires_grad());
}

#[test]
fn test_var_fused_add_layer_norm_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device).unwrap(),
        true,
    );
    let residual = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[1, 4], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap(),
        true,
    );

    let result = var_fused_add_layer_norm(&x, &residual, &weight, &bias, 1e-5, &client).unwrap();
    let data: Vec<f32> = result.tensor().to_vec();

    // Layer norm output should have ~0 mean
    let sum: f32 = data.iter().sum();
    assert!(
        sum.abs() < 1e-4,
        "output should have ~0 mean, got sum={sum}"
    );
}

#[test]
fn test_var_fused_add_layer_norm_backward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let x = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap(),
        true,
    );
    let residual = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[1, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device).unwrap(),
        true,
    );

    let output = var_fused_add_layer_norm(&x, &residual, &weight, &bias, 1e-5, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
    let gr: Vec<f32> = grads.get(residual.id()).unwrap().to_vec();
    let gw: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    let gb: Vec<f32> = grads.get(bias.id()).unwrap().to_vec();

    // x and residual should get the same gradient
    for (a, b) in gx.iter().zip(gr.iter()) {
        assert!((a - b).abs() < 1e-5, "x and residual grads must match");
    }

    // d_bias should be [1, 1, 1] for sum loss
    for val in &gb {
        assert!(
            (*val - 1.0).abs() < 1e-5,
            "bias gradient should be 1.0, got {val}"
        );
    }

    for val in gx.iter().chain(gw.iter()) {
        assert!(val.is_finite());
    }
}
