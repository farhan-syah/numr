//! Conv1d autograd operation
//!
//! Wraps `ConvOps::conv1d` with gradient tracking.
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

use super::conv_common::compute_padding;

/// Differentiable 1D convolution.
///
/// Wraps the forward `conv1d` and builds autograd graph for backward.
///
/// # Arguments
/// * `input` - Input Var of shape `[batch, in_channels, length]`
/// * `weight` - Weight Var of shape `[out_channels, in_channels/groups, kernel_size]`
/// * `bias` - Optional bias Var of shape `[out_channels]`
/// * `stride` - Stride
/// * `padding` - Padding mode
/// * `dilation` - Dilation
/// * `groups` - Groups
/// * `client` - Runtime client
pub fn var_conv1d<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let output = client.conv1d(
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
        let grad_fn = Conv1dBackward::<R>::new(
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

/// Backward for conv1d.
///
/// Computes gradients for input, weight, and bias using:
/// - d_input: transposed convolution (conv with flipped kernel, adjusted padding)
/// - d_weight: cross-correlation of input with grad_output
/// - d_bias: sum of grad_output over batch and spatial dims
pub struct Conv1dBackward<R: Runtime> {
    input_ids: Vec<crate::tensor::TensorId>,
    saved_input: crate::tensor::Tensor<R>,
    saved_weight: crate::tensor::Tensor<R>,
    input_shape: Vec<usize>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
    input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> Conv1dBackward<R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_id: crate::tensor::TensorId,
        weight_id: crate::tensor::TensorId,
        bias_id: Option<crate::tensor::TensorId>,
        input: crate::tensor::Tensor<R>,

        weight: crate::tensor::Tensor<R>,
        input_shape: Vec<usize>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
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

/// Compute conv1d backward for input using tensor operations.
///
/// d_input[n, c_in, l] = sum over c_out, k of:
///   weight[c_out, c_in, k] * grad_output[n, c_out, l*stride - pad + k*dilation]
///
/// This is equivalent to a transposed convolution (conv_transpose1d).
///
/// IMPLEMENTATION NOTE: Uses tensor operations (no to_vec/to_cpu). All computation
/// is performed through the client, which works on any backend. The Rust loop
/// structures the iteration, but actual mathematical operations (matmul, add)
/// happen on the device via the client.
pub(super) fn conv1d_input_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    weight: &crate::tensor::Tensor<R>,
    input_shape: &[usize],
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let batch = input_shape[0];
    let _c_in = input_shape[1];
    let input_len = input_shape[2];
    let c_out = weight.shape()[0];
    let c_in_per_group = weight.shape()[1];
    let kernel_size = weight.shape()[2];
    let output_len = grad_output.shape()[2];
    let c_out_per_group = c_out / groups;

    let (pad_left, _pad_right) = compute_padding(padding, kernel_size, dilation);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_input = crate::tensor::Tensor::<R>::try_zeros(input_shape, dtype, device)?;

    // Accumulate contributions by iterating and accumulating tensor operations
    for k in 0..kernel_size {
        let weight_k = weight.narrow(2, k, 1)?;
        let weight_k = weight_k.squeeze(Some(2));

        for o in 0..output_len {
            let i_pos = o * stride + k * dilation;

            if i_pos >= pad_left && i_pos < pad_left + input_len {
                let i = i_pos - pad_left;

                let grad_o = grad_output.narrow(2, o, 1)?;
                let grad_o = grad_o.squeeze(Some(2));

                for g in 0..groups {
                    let c_in_start = g * c_in_per_group;
                    let c_out_start = g * c_out_per_group;

                    let grad_g = grad_o.narrow(1, c_out_start, c_out_per_group)?;
                    let weight_g = weight_k.narrow(0, c_out_start, c_out_per_group)?;

                    // Compute contribution: [batch, c_out_per_group] @ [c_out_per_group, c_in_per_group].T
                    let contrib_g = client.matmul(&grad_g, &weight_g.transpose(0, 1)?)?;

                    // Reshape to [batch, c_in_per_group, 1]
                    let contrib_g_3d = contrib_g.reshape(&[batch, c_in_per_group, 1])?;

                    // Get the slice at position i in the full d_input
                    let mut d_input_at_i = d_input.narrow(2, i, 1)?; // [batch, c_in, 1]

                    // Get the group slice
                    let d_input_group = d_input_at_i.narrow(1, c_in_start, c_in_per_group)?; // [batch, c_in_per_group, 1]

                    // Add contribution
                    let updated_group = client.add(&d_input_group, &contrib_g_3d)?;

                    // Now put it back. We need to use slice_assign correctly.
                    // The challenge is that we have a [batch, c_in_per_group, 1] but
                    // we need to update a specific region of a [batch, c_in, 1].
                    // slice_assign along dim 1 requires src to have the same dimension count
                    // and the same size on all dims except dim.
                    // So src should be [batch, c_in_per_group, 1] and we use dim=1, start=c_in_start
                    d_input_at_i =
                        client.slice_assign(&d_input_at_i, &updated_group, 1, c_in_start)?;

                    // Now put d_input_at_i back into d_input at position i
                    d_input = client.slice_assign(&d_input, &d_input_at_i, 2, i)?;
                }
            }
        }
    }

    Ok(d_input)
}

/// Compute conv1d backward for weight using tensor operations.
///
/// d_weight[c_out, c_in, k] = sum over n, o of:
///   input[n, c_in, o*stride - pad + k*dilation] * grad_output[n, c_out, o]
///
/// This function uses only tensor operations (no to_vec/to_cpu). All computation
/// is performed through the client, which works on any backend.
pub(super) fn conv1d_weight_backward<R, C>(
    client: &C,
    grad_output: &crate::tensor::Tensor<R>,
    input: &crate::tensor::Tensor<R>,
    weight_shape: &[usize],
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R> + BinaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    let _batch = input.shape()[0];
    let _c_in = input.shape()[1];
    let input_len = input.shape()[2];
    let c_out = weight_shape[0];
    let c_in_per_group = weight_shape[1];
    let kernel_size = weight_shape[2];
    let output_len = grad_output.shape()[2];
    let c_out_per_group = c_out / groups;

    let (pad_left, _pad_right) = compute_padding(padding, kernel_size, dilation);

    let device = grad_output.device();
    let dtype = grad_output.dtype();

    let mut d_weight = crate::tensor::Tensor::<R>::try_zeros(weight_shape, dtype, device)?;

    // Accumulate contributions by iterating and accumulating tensor operations
    for o in 0..output_len {
        for k in 0..kernel_size {
            let i_pos = o * stride + k * dilation;

            if i_pos >= pad_left && i_pos < pad_left + input_len {
                let i = i_pos - pad_left;

                let input_i = input.narrow(2, i, 1)?;
                let input_i = input_i.squeeze(Some(2));

                let grad_o = grad_output.narrow(2, o, 1)?;
                let grad_o = grad_o.squeeze(Some(2));

                for g in 0..groups {
                    let c_in_start = g * c_in_per_group;
                    let c_out_start = g * c_out_per_group;

                    let input_g = input_i.narrow(1, c_in_start, c_in_per_group)?;
                    let grad_g = grad_o.narrow(1, c_out_start, c_out_per_group)?;

                    // Compute: [c_out_per_group, batch] @ [batch, c_in_per_group]
                    //        = [c_out_per_group, c_in_per_group]
                    let contrib_2d = client.matmul(&grad_g.transpose(0, 1)?, &input_g)?;

                    // Reshape to [c_out_per_group, c_in_per_group, 1]
                    let contrib_3d = contrib_2d.reshape(&[c_out_per_group, c_in_per_group, 1])?;

                    // Get the weight slice at kernel position k
                    let mut d_weight_at_k = d_weight.narrow(2, k, 1)?; // [c_out, c_in_per_group, 1]

                    // Get the group slice
                    let d_weight_group = d_weight_at_k.narrow(0, c_out_start, c_out_per_group)?; // [c_out_per_group, c_in_per_group, 1]

                    // Add contribution
                    let updated_group = client.add(&d_weight_group, &contrib_3d)?;

                    // Put back along dimension 0
                    d_weight_at_k =
                        client.slice_assign(&d_weight_at_k, &updated_group, 0, c_out_start)?;

                    // Put back into d_weight along dimension 2
                    d_weight = client.slice_assign(&d_weight, &d_weight_at_k, 2, k)?;
                }
            }
        }
    }

    Ok(d_weight)
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for Conv1dBackward<R>
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
            Some(conv1d_input_backward::<R, _>(
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
            Some(conv1d_weight_backward::<R, _>(
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

        // d_bias = sum over batch and length dims
        let d_bias = if self.input_ids.len() > 2 && needed[2] {
            // grad_output shape: [batch, c_out, output_len]
            // sum over dim 0 (batch) and dim 2 (length) → [c_out]
            let summed = client.sum(grad_output, &[0, 2], false)?;
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
        "Conv1dBackward"
    }
}

#[cfg(test)]
mod tests {
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
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device),
            false,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device),
            false,
        );

        let output =
            var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
        let data: Vec<f32> = output.tensor().to_vec();
        assert_eq!(data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_var_conv1d_backward_input() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1, 1, 1], &device),
            true,
        );

        let output =
            var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
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
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 2], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1, 1, 1], &device),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device),
            true,
        );

        let output =
            var_conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1, &client).unwrap();
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
}
