//! Tests for conv1d forward/backward, including value-correctness checks
//! against an independently computed reference gradient.

use super::*;
use crate::autograd::backward;
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

#[test]
fn test_var_conv1d_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // weight: [out=1, in=1, kernel=1] → identity-like
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device).unwrap(),
        false,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device).unwrap(),
        false,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
    let data: Vec<f32> = output.tensor().to_vec();
    assert_eq!(data, vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_var_conv1d_backward_input() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device).unwrap(),
        true,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    // d_input should be weight broadcast: [2, 2, 2]
    assert_eq!(d_input, vec![2.0, 2.0, 2.0]);

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    // d_weight = sum of input = 1+2+3 = 6
    assert!((d_weight[0] - 6.0).abs() < 1e-5);
}

#[test]
fn test_var_conv1d_backward_with_bias() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 2], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device).unwrap(),
        true,
    );

    let output = var_conv1d(
        &input,
        &weight,
        Some(&bias),
        1,
        PaddingMode::Valid,
        1,
        1,
        &client,
    )
    .unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_bias: Vec<f32> = grads.get(bias.id()).unwrap().to_vec();
    // d_bias = sum of grad_output (all ones) over batch and length = 2
    assert!((d_bias[0] - 2.0).abs() < 1e-5);
}

#[test]
fn test_var_conv1d_kernel3() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // kernel_size=3, input_length=5 → output_length=3
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device)
            .unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device).unwrap(),
        true,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
    let data: Vec<f32> = output.tensor().to_vec();
    // [1+2+3, 2+3+4, 3+4+5] = [6, 9, 12]
    assert_eq!(data, vec![6.0, 9.0, 12.0]);

    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    // Each input position contributes to 1-3 output positions
    // pos 0: contributes to output 0 → weight[0] = 1
    // pos 1: contributes to outputs 0,1 → weight[1]+weight[0] = 2
    // pos 2: contributes to outputs 0,1,2 → weight[2]+weight[1]+weight[0] = 3
    // pos 3: contributes to outputs 1,2 → weight[2]+weight[1] = 2
    // pos 4: contributes to output 2 → weight[2] = 1
    assert_eq!(d_input, vec![1.0, 2.0, 3.0, 2.0, 1.0]);
}

// VALUE checks against a reference computed independently in-test, not
// against another backend. grad_output is all ones (loss = sum(output)):
//   d_input[n][ci][i]   = sum over (co,k) with i==o+k, valid o, of weight[co][ci][k]
//   d_weight[co][ci][k] = sum over (n,o) of input[n][ci][o+k]
//   d_bias[co]          = batch * output_length
// Pre-fix, conv1d_input_backward computed matmul(grad_g, weight_g.T).
// grad_g is [batch, c_out_per_group], weight_g.T is [c_in_per_group,
// c_out_per_group] — matmul needs grad_g's inner dim to equal
// weight_g.T's outer dim, only true when c_in_per_group == c_out_per_group.
// Every case below has c_in_per_group != c_out_per_group, so the pre-fix
// code errors on a matmul shape mismatch rather than just being wrong.

#[test]
fn conv1d_input_gradient_multichannel_matches_reference() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // input [batch=1, c_in=2, length=4], weight [c_out=3, c_in=2, k=2]
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 2, 4], &device).unwrap(),
        true,
    );
    let weight_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&weight_data, &[3, 2, 2], &device).unwrap(),
        true,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    assert_eq!(
        d_input,
        vec![15.0, 33.0, 33.0, 18.0, 21.0, 45.0, 45.0, 24.0]
    );

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    assert_eq!(
        d_weight,
        vec![
            6.0, 9.0, 18.0, 21.0, 6.0, 9.0, 18.0, 21.0, 6.0, 9.0, 18.0, 21.0
        ]
    );
}

#[test]
fn conv1d_input_gradient_asymmetric_channels_matches_reference() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // input [batch=2, c_in=3, length=3], weight [c_out=2, c_in=3, k=2]: c_out < c_in
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    ];
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input_data, &[2, 3, 3], &device).unwrap(),
        true,
    );
    let weight_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&weight_data, &[2, 3, 2], &device).unwrap(),
        true,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    assert_eq!(
        d_input,
        vec![
            8.0, 18.0, 10.0, 12.0, 26.0, 14.0, 16.0, 34.0, 18.0, 8.0, 18.0, 10.0, 12.0, 26.0, 14.0,
            16.0, 34.0, 18.0,
        ]
    );

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    assert_eq!(
        d_weight,
        vec![
            24.0, 28.0, 36.0, 40.0, 48.0, 52.0, 24.0, 28.0, 36.0, 40.0, 48.0, 52.0
        ]
    );

    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap(),
        true,
    );
    let output_b = var_conv1d(
        &input,
        &weight,
        Some(&bias),
        1,
        PaddingMode::Valid,
        1,
        1,
        &client,
    )
    .unwrap();
    let loss_b = crate::autograd::var_sum(&output_b, &[], false, &client).unwrap();
    let grads_b = backward(&loss_b, &client).unwrap();
    let d_bias: Vec<f32> = grads_b.get(bias.id()).unwrap().to_vec();
    // d_bias = batch * output_length = 2 * 2 = 4
    assert_eq!(d_bias, vec![4.0, 4.0]);
}

#[test]
fn conv1d_input_gradient_grouped_matches_reference() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // input [batch=1, c_in=4, length=3], groups=2: c_in_per_group=2 != c_out_per_group=3
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 4, 3], &device).unwrap(),
        true,
    );
    // weight [c_out=6, c_in_per_group=2, k=2]
    let weight_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&weight_data, &[6, 2, 2], &device).unwrap(),
        true,
    );

    let output = var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 2, &client).unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    assert_eq!(
        d_input,
        vec![
            15.0, 33.0, 18.0, 21.0, 45.0, 24.0, 51.0, 105.0, 54.0, 57.0, 117.0, 60.0
        ]
    );

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    assert_eq!(
        d_weight,
        vec![
            3.0, 5.0, 9.0, 11.0, 3.0, 5.0, 9.0, 11.0, 3.0, 5.0, 9.0, 11.0, 15.0, 17.0, 21.0, 23.0,
            15.0, 17.0, 21.0, 23.0, 15.0, 17.0, 21.0, 23.0,
        ]
    );
}
