//! Tests for conv2d forward/backward, including value-correctness checks
//! against an independently computed reference gradient.

use super::*;
use crate::autograd::backward;
use crate::runtime::cpu::{CpuDevice, CpuRuntime};
use crate::tensor::Tensor;

#[test]
fn test_var_conv2d_forward() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: [batch=1, c_in=1, h=2, w=2], weight: [c_out=1, c_in=1, kH=1, kW=1] = 2.0
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device).unwrap(),
        false,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1, 1], &device).unwrap(),
        false,
    );

    let output = var_conv2d(
        &input,
        &weight,
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let data: Vec<f32> = output.tensor().to_vec();
    assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_var_conv2d_backward_input() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: [1, 1, 2, 2], weight: [1, 1, 1, 1] = 2.0
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1, 1], &device).unwrap(),
        true,
    );

    let output = var_conv2d(
        &input,
        &weight,
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    // With 1x1 kernel of weight=2, d_input should be [2, 2, 2, 2]
    assert_eq!(d_input, vec![2.0, 2.0, 2.0, 2.0]);

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    // d_weight = sum of input = 1+2+3+4 = 10
    assert!((d_weight[0] - 10.0).abs() < 1e-5);
}

#[test]
fn test_var_conv2d_backward_with_bias() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: [1, 1, 2, 2], weight: [1, 1, 1, 1] = 1.0, bias: [1] = 10.0
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1, 1], &device).unwrap(),
        true,
    );
    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device).unwrap(),
        true,
    );

    let output = var_conv2d(
        &input,
        &weight,
        Some(&bias),
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_bias: Vec<f32> = grads.get(bias.id()).unwrap().to_vec();
    // d_bias = sum of grad_output (all ones) over batch, h, w = 2*2 = 4
    assert!((d_bias[0] - 4.0).abs() < 1e-5);
}

#[test]
fn test_var_conv2d_kernel2x2() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: [1, 1, 3, 3], weight: [1, 1, 2, 2] all ones
    // Output: [1, 1, 2, 2]
    #[rustfmt::skip]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
    ];
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device).unwrap(),
        true,
    );
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device).unwrap(),
        true,
    );

    let output = var_conv2d(
        &input,
        &weight,
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let data: Vec<f32> = output.tensor().to_vec();
    // [1+2+4+5, 2+3+5+6, 4+5+7+8, 5+6+8+9] = [12, 16, 24, 28]
    assert_eq!(data, vec![12.0, 16.0, 24.0, 28.0]);

    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    // Each input position contributes to 1-4 output positions (2x2 kernel, all 1s)
    // pos(0,0): out(0,0) → 1
    // pos(0,1): out(0,0)+out(0,1) → 2
    // pos(0,2): out(0,1) → 1
    // pos(1,0): out(0,0)+out(1,0) → 2
    // pos(1,1): out(0,0)+out(0,1)+out(1,0)+out(1,1) → 4
    // pos(1,2): out(0,1)+out(1,1) → 2
    // pos(2,0): out(1,0) → 1
    // pos(2,1): out(1,0)+out(1,1) → 2
    // pos(2,2): out(1,1) → 1
    assert_eq!(d_input, vec![1.0, 2.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0, 1.0]);
}

// VALUE checks against a reference computed independently in-test, not
// against another backend. grad_output is all ones (loss = sum(output)):
//   d_input[n][ci][ih][iw]  = sum over (co,kh,kw), ih==oh+kh, iw==ow+kw
//                             valid (oh,ow), of weight[co][ci][kh][kw]
//   d_weight[co][ci][kh][kw] = sum over (n,oh,ow) of input[n][ci][oh+kh][ow+kw]
//   d_bias[co] = batch * output_h * output_w
// Pre-fix, conv2d_input_backward computed matmul(grad_g, weight_g.T).
// grad_g is [batch, c_out_per_group], weight_g.T is [c_in_per_group,
// c_out_per_group]. c_in=2 != c_out=3 here, so the pre-fix matmul
// hits a shape mismatch rather than just being wrong.
#[test]
fn conv2d_input_gradient_multichannel_matches_reference() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // input [batch=1, c_in=2, h=3, w=3], weight [c_out=3, c_in=2, kh=2, kw=2]
    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0,
    ];
    let input = Var::new(
        Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 2, 3, 3], &device).unwrap(),
        true,
    );
    let weight_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    ];
    let weight = Var::new(
        Tensor::<CpuRuntime>::from_slice(&weight_data, &[3, 2, 2, 2], &device).unwrap(),
        true,
    );

    let output = var_conv2d(
        &input,
        &weight,
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
    let grads = backward(&loss, &client).unwrap();

    let d_input: Vec<f32> = grads.get(input.id()).unwrap().to_vec();
    assert_eq!(
        d_input,
        vec![
            27.0, 57.0, 30.0, 60.0, 126.0, 66.0, 33.0, 69.0, 36.0, 39.0, 81.0, 42.0, 84.0, 174.0,
            90.0, 45.0, 93.0, 48.0,
        ]
    );

    let d_weight: Vec<f32> = grads.get(weight.id()).unwrap().to_vec();
    let per_co = [12.0f32, 16.0, 24.0, 28.0, 48.0, 52.0, 60.0, 64.0];
    let expected_weight: Vec<f32> = per_co.iter().cycle().take(24).copied().collect();
    assert_eq!(d_weight, expected_weight);

    let bias = Var::new(
        Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0], &[3], &device).unwrap(),
        true,
    );
    let output_b = var_conv2d(
        &input,
        &weight,
        Some(&bias),
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        &client,
    )
    .unwrap();
    let loss_b = crate::autograd::var_sum(&output_b, &[], false, &client).unwrap();
    let grads_b = backward(&loss_b, &client).unwrap();
    let d_bias: Vec<f32> = grads_b.get(bias.id()).unwrap().to_vec();
    // d_bias = batch * output_h * output_w = 1 * 2 * 2 = 4
    assert_eq!(d_bias, vec![4.0, 4.0, 4.0]);
}
