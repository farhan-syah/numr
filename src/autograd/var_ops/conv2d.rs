//! Conv2d autograd operation
//!
//! Wraps `ConvOps::conv2d` with gradient tracking.
//!
//! Backward computes:
//! - d_input  = transposed convolution of grad_output with weight
//! - d_weight = cross-correlation of input with grad_output
//! - d_bias   = sum(grad_output) over batch and spatial dims

use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ConvOps, PaddingMode, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

use super::conv_common::compute_padding_2d;

/// Differentiable 2D convolution.
///
/// Wraps the forward `conv2d` and builds autograd graph for backward.
///
/// # Arguments
/// * `input` - Input Var of shape `[batch, in_channels, height, width]`
/// * `weight` - Weight Var of shape `[out_channels, in_channels/groups, kH, kW]`
/// * `bias` - Optional bias Var of shape `[out_channels]`
/// * `stride` - Stride as `(stride_h, stride_w)`
/// * `padding` - Padding mode
/// * `dilation` - Dilation as `(dilation_h, dilation_w)`
/// * `groups` - Groups
/// * `client` - Runtime client
pub fn var_conv2d<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let output = client.conv2d(
        input.tensor(),
        weight.tensor(),
        bias.map(|b| b.tensor()),
        stride,
        padding,
        dilation,
        groups,
    )?;

    let needs_grad =
        input.requires_grad() || weight.requires_grad() || bias.is_some_and(|b| b.requires_grad());

    if needs_grad {
        let grad_fn = Conv2dBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.map(|b| b.id()),
            input.tensor().clone(),
            weight.tensor().clone(),
            input.tensor().shape().to_vec(),
            stride,
            padding,
            dilation,
            groups,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.and_then(|b| b.grad_fn().cloned()),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for conv2d.
///
/// Computes gradients for input, weight, and bias using:
/// - d_input: transposed convolution (conv with flipped kernel, adjusted padding)
/// - d_weight: cross-correlation of input with grad_output
/// - d_bias: sum of grad_output over batch and spatial dims
pub struct Conv2dBackward<R: Runtime> {
    input_ids: Vec<crate::tensor::TensorId>,
    saved_input: crate::tensor::Tensor<R>,
    saved_weight: crate::tensor::Tensor<R>,
    input_shape: Vec<usize>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
    input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> Conv2dBackward<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_id: crate::tensor::TensorId,
        weight_id: crate::tensor::TensorId,
        bias_id: Option<crate::tensor::TensorId>,
        input: crate::tensor::Tensor<R>,
        weight: crate::tensor::Tensor<R>,
        input_shape: Vec<usize>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
        input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        let mut ids = vec![input_id, weight_id];
        if let Some(bid) = bias_id {
            ids.push(bid);
        }
        Self {
            input_ids: ids,
            saved_input: input,
            saved_weight: weight,
            input_shape,
            stride,
            padding,
            dilation,
            groups,
            input_grad_fn,
            weight_grad_fn,
            bias_grad_fn,
        }
    }
}

/// Compute conv2d backward for input using tensor operations.
///
/// d_input[n, c_in, h, w] = sum over c_out, kh, kw of:
///   weight[c_out, c_in, kh, kw] * grad_output[n, c_out, h*sh - pad_top + kh*dh, w*sw - pad_left + kw*dw]
fn conv2d_input_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    weight: &crate::tensor::Tensor<R>,
    input_shape: &[usize],
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let batch = input_shape[0];
    let _c_in = input_shape[1];
    let input_h = input_shape[2];
    let input_w = input_shape[3];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let kernel_h = weight.shape()[2];
    let kernel_w = weight.shape()[3];
    let output_h = grad_output.shape()[2];
    let output_w = grad_output.shape()[3];
    let c_out_per_group = c_out / groups;

    let (pad_top, _pad_bottom, pad_left, _pad_right) =
        compute_padding_2d(padding, kernel_h, kernel_w, dilation.0, dilation.1);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_input = crate::tensor::Tensor::<R>::try_zeros(input_shape, dtype, device)?;

    for kh in 0..kernel_h {
        for kw in 0..kernel_w {
            // Extract weight slice at [kh, kw]: weight[:, :, kh, kw] → [c_out, c_in_per_group]
            let weight_kh = weight.narrow(2, kh, 1)?;
            let weight_khkw = weight_kh.narrow(3, kw, 1)?;
            let weight_2d = weight_khkw.squeeze(Some(3)).squeeze(Some(2));

            for oh in 0..output_h {
                let ih_pos = oh * stride.0 + kh * dilation.0;
                if ih_pos < pad_top || ih_pos >= pad_top + input_h {
                    continue;
                }
                let ih = ih_pos - pad_top;

                for ow in 0..output_w {
                    let iw_pos = ow * stride.1 + kw * dilation.1;
                    if iw_pos < pad_left || iw_pos >= pad_left + input_w {
                        continue;
                    }
                    let iw = iw_pos - pad_left;

                    // grad_output[:, :, oh, ow] → [batch, c_out]
                    let grad_o = grad_output.narrow(2, oh, 1)?.narrow(3, ow, 1)?;
                    let grad_o_2d = grad_o.squeeze(Some(3)).squeeze(Some(2));

                    for g in 0..groups {
                        let c_in_start = g * c_in_per_group;
                        let c_out_start = g * c_out_per_group;

                        let grad_g = grad_o_2d.narrow(1, c_out_start, c_out_per_group)?;
                        let weight_g = weight_2d.narrow(0, c_out_start, c_out_per_group)?;

                        // [batch, c_out_per_group] @ [c_out_per_group, c_in_per_group]
                        let contrib_g = client.matmul(&grad_g, &weight_g.transpose(0, 1)?)?;

                        // Reshape to [batch, c_in_per_group, 1, 1]
                        let contrib_4d = contrib_g.reshape(&[batch, c_in_per_group, 1, 1])?;

                        // Get the slice at position (ih, iw) in the full d_input
                        let mut d_input_at = d_input.narrow(2, ih, 1)?.narrow(3, iw, 1)?;

                        // Get the group slice
                        let d_input_group = d_input_at.narrow(1, c_in_start, c_in_per_group)?;

                        // Add contribution
                        let updated_group = client.add(&d_input_group, &contrib_4d)?;

                        // Put back along dim 1
                        d_input_at =
                            client.slice_assign(&d_input_at, &updated_group, 1, c_in_start)?;

                        // Put back into d_input: first along dim 3 (width), then dim 2 (height)
                        let mut d_input_h = d_input.narrow(2, ih, 1)?;
                        d_input_h = client.slice_assign(&d_input_h, &d_input_at, 3, iw)?;
                        d_input = client.slice_assign(&d_input, &d_input_h, 2, ih)?;
                    }
                }
            }
        }
    }

    Ok(d_input)
}

/// Compute conv2d backward for weight using tensor operations.
///
/// d_weight[c_out, c_in, kh, kw] = sum over n, oh, ow of:
///   input[n, c_in, oh*sh - pad_top + kh*dh, ow*sw - pad_left + kw*dw] * grad_output[n, c_out, oh, ow]
fn conv2d_weight_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    input: &crate::tensor::Tensor<R>,
    weight_shape: &[usize],
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let _batch = input.shape()[0];
    let _c_in = input.shape()[1];
    let input_h = input.shape()[2];
    let input_w = input.shape()[3];
    let c_out = weight_shape[0];
    let c_in_per_group = weight_shape[1];
    let kernel_h = weight_shape[2];
    let kernel_w = weight_shape[3];
    let output_h = grad_output.shape()[2];
    let output_w = grad_output.shape()[3];
    let c_out_per_group = c_out / groups;

    let (pad_top, _pad_bottom, pad_left, _pad_right) =
        compute_padding_2d(padding, kernel_h, kernel_w, dilation.0, dilation.1);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_weight = crate::tensor::Tensor::<R>::try_zeros(weight_shape, dtype, device)?;

    for oh in 0..output_h {
        for ow in 0..output_w {
            // grad_output[:, :, oh, ow] → [batch, c_out]
            let grad_o = grad_output.narrow(2, oh, 1)?.narrow(3, ow, 1)?;
            let grad_o_2d = grad_o.squeeze(Some(3)).squeeze(Some(2));

            for kh in 0..kernel_h {
                let ih_pos = oh * stride.0 + kh * dilation.0;
                if ih_pos < pad_top || ih_pos >= pad_top + input_h {
                    continue;
                }
                let ih = ih_pos - pad_top;

                for kw in 0..kernel_w {
                    let iw_pos = ow * stride.1 + kw * dilation.1;
                    if iw_pos < pad_left || iw_pos >= pad_left + input_w {
                        continue;
                    }
                    let iw = iw_pos - pad_left;

                    // input[:, :, ih, iw] → [batch, c_in]
                    let input_hw = input.narrow(2, ih, 1)?.narrow(3, iw, 1)?;
                    let input_2d = input_hw.squeeze(Some(3)).squeeze(Some(2));

                    for g in 0..groups {
                        let c_in_start = g * c_in_per_group;
                        let c_out_start = g * c_out_per_group;

                        let input_g = input_2d.narrow(1, c_in_start, c_in_per_group)?;
                        let grad_g = grad_o_2d.narrow(1, c_out_start, c_out_per_group)?;

                        // [c_out_per_group, batch] @ [batch, c_in_per_group]
                        // = [c_out_per_group, c_in_per_group]
                        let contrib_2d = client.matmul(&grad_g.transpose(0, 1)?, &input_g)?;

                        // Reshape to [c_out_per_group, c_in_per_group, 1, 1]
                        let contrib_4d =
                            contrib_2d.reshape(&[c_out_per_group, c_in_per_group, 1, 1])?;

                        // Get the weight slice at kernel position (kh, kw)
                        let mut d_weight_at = d_weight.narrow(2, kh, 1)?.narrow(3, kw, 1)?;

                        // Get the group slice
                        let d_weight_group = d_weight_at.narrow(0, c_out_start, c_out_per_group)?;

                        // Add contribution
                        let updated_group = client.add(&d_weight_group, &contrib_4d)?;

                        // Put back along dim 0
                        d_weight_at =
                            client.slice_assign(&d_weight_at, &updated_group, 0, c_out_start)?;

                        // Put back into d_weight: first along dim 3, then dim 2
                        let mut d_weight_kh = d_weight.narrow(2, kh, 1)?;
                        d_weight_kh = client.slice_assign(&d_weight_kh, &d_weight_at, 3, kw)?;
                        d_weight = client.slice_assign(&d_weight, &d_weight_kh, 2, kh)?;
                    }
                }
            }
        }
    }

    Ok(d_weight)
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for Conv2dBackward<R>
where
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // The input gradient and the weight gradient are separate convolutions
        // sharing nothing but `grad_output`, so each is guarded. A frozen conv
        // layer skips the whole cross-correlation that builds d_weight.

        // d_input via transposed convolution
        let d_input = if needed[0] {
            Some(conv2d_input_backward::<R, _>(
                &client,
                grad_output,
                &self.saved_weight,
                &self.input_shape,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        // d_weight via cross-correlation
        let d_weight = if needed[1] {
            Some(conv2d_weight_backward::<R, _>(
                &client,
                grad_output,
                &self.saved_input,
                self.saved_weight.shape(),
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        // d_bias = sum over batch, height, and width dims
        let d_bias = if self.input_ids.len() > 2 && needed[2] {
            // grad_output shape: [batch, c_out, out_h, out_w]
            // sum over dim 0 (batch), dim 2 (height), dim 3 (width) → [c_out]
            let summed = client.sum(grad_output, &[0, 2, 3], false)?;
            Some(summed)
        } else {
            None
        };

        Ok(vec![d_input, d_weight, d_bias])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + ConvOps<R>
            + TensorOps<R>
            + ReduceOps<R>
            + BinaryOps<R>
            + ScalarOps<R>,
    {
        // First-order only for conv — second-order conv is rarely needed
        // Second-order traversal keeps every node, so ask for every gradient.
        let grads = self.backward_all(grad_output.tensor())?;
        Ok(grads
            .into_iter()
            .map(|g| g.map(|t| Var::new(t, true)))
            .collect())
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        let mut fns = vec![self.input_grad_fn.clone(), self.weight_grad_fn.clone()];
        if self.input_ids.len() > 2 {
            fns.push(self.bias_grad_fn.clone());
        }
        fns
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "Conv2dBackward"
    }
}

#[cfg(test)]
mod tests {
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
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            false,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1, 1, 1, 1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1, 1, 1, 1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 1, 2, 2], &device)
                .unwrap(),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1, 1, 1, 1], &device).unwrap(),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[10.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::try_from_slice(&input_data, &[1, 1, 3, 3], &device).unwrap(),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device)
                .unwrap(),
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
}
