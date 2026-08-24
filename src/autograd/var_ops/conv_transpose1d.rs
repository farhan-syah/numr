//! Autograd wrapper for transposed 1D convolution.
//!
//! Transposed convolution is the adjoint of convolution, which makes its
//! backward pass unusually clean: the gradient w.r.t. the input is an ordinary
//! `conv1d`, and the gradient w.r.t. the weight is exactly conv1d's own
//! weight-gradient with the roles of `input` and `grad_output` swapped. Both
//! are reused here rather than reimplemented.

use std::sync::Arc;

use super::conv1d::conv1d_weight_backward;
use crate::autograd::Var;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::PaddingMode;
use crate::ops::conv_transpose_common::validate_conv_transpose1d;
use crate::ops::traits::{BinaryOps, ConvOps, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};

/// Transposed 1D convolution with autograd.
///
/// * `input`  — `[batch, c_in, length]`
/// * `weight` — `[c_in, c_out / groups, kernel]` (input channels lead, unlike
///   `conv1d`)
/// * `bias`   — `[c_out]`
///
/// `output_padding` must be smaller than `max(stride, dilation)`; it resolves
/// the ambiguity where several input lengths map to the same output length.
#[allow(clippy::too_many_arguments)]
pub fn var_conv_transpose1d<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: Option<&Var<R>>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
    groups: usize,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
    R::Client: ConvOps<R> + TensorOps<R> + ReduceOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let output = client.conv_transpose1d(
        input.tensor(),
        weight.tensor(),
        bias.map(|b| b.tensor()),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
    )?;

    let needs_grad =
        input.requires_grad() || weight.requires_grad() || bias.is_some_and(|b| b.requires_grad());

    if !needs_grad {
        return Ok(Var::new(output, false));
    }

    // Resolve the padding actually used, so backward reproduces the exact
    // forward geometry instead of re-deriving `Same` from a different formula.
    let params = validate_conv_transpose1d(
        input.tensor().shape(),
        weight.tensor().shape(),
        bias.map(|b| b.tensor().shape()),
        stride,
        padding,
        output_padding,
        dilation,
        groups,
        input.tensor().dtype(),
        weight.tensor().dtype(),
        bias.map(|b| b.tensor().dtype()),
    )?;

    let grad_fn = ConvTranspose1dBackward::<R> {
        input_ids: {
            let mut ids = vec![input.id(), weight.id()];
            if let Some(b) = bias {
                ids.push(b.id());
            }
            ids
        },
        saved_input: input.tensor().clone(),
        saved_weight: weight.tensor().clone(),
        stride,
        resolved_padding: PaddingMode::Custom(params.pad_left, params.pad_right, 0, 0),
        dilation,
        groups,
        input_grad_fn: input.grad_fn().cloned(),
        weight_grad_fn: weight.grad_fn().cloned(),
        bias_grad_fn: bias.and_then(|b| b.grad_fn().cloned()),
    };
    Ok(Var::from_op(output, Arc::new(grad_fn)))
}

/// Backward for `conv_transpose1d`.
pub struct ConvTranspose1dBackward<R: Runtime> {
    input_ids: Vec<crate::tensor::TensorId>,
    saved_input: crate::tensor::Tensor<R>,
    saved_weight: crate::tensor::Tensor<R>,
    stride: usize,
    /// Always `Custom`, holding the padding the forward pass actually applied.
    resolved_padding: PaddingMode,
    dilation: usize,
    groups: usize,
    input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    weight_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    bias_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for ConvTranspose1dBackward<R>
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
        // sharing nothing but `grad_output`, so each is guarded. A frozen layer
        // skips the whole cross-correlation that builds d_weight.

        // d_input: transposed convolution is the adjoint of convolution, so the
        // input gradient is a plain conv1d of grad_output with the SAME weight —
        // whose `[c_in, c_out/groups, k]` layout is already conv1d's expected
        // `[c_out', c_in'/groups, k]` for this direction.
        let d_input = if needed[0] {
            Some(client.conv1d(
                grad_output,
                &self.saved_weight,
                None,
                self.stride,
                self.resolved_padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        // d_weight: conv1d's weight gradient with input/grad_output swapped.
        //   dW[ic, oc, k] = sum_{n,l} x[n, ic, l] * gy[n, oc, l*stride + k*dil - pad]
        // which is exactly conv1d_weight_backward(grad_output = x, input = gy).
        let d_weight = if needed[1] {
            Some(conv1d_weight_backward::<R, _>(
                &client,
                &self.saved_input,
                grad_output,
                self.saved_weight.shape(),
                self.stride,
                self.resolved_padding,
                self.dilation,
                self.groups,
            )?)
        } else {
            None
        };

        let d_bias = if self.input_ids.len() > 2 && needed[2] {
            Some(client.sum(grad_output, &[0, 2], false)?)
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
        // First-order only, matching conv1d/conv2d.
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
        "ConvTranspose1dBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn setup() -> (CpuDevice, <CpuRuntime as Runtime>::Client) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    /// Stride-2 upsampling with a length-2 kernel spreads each input sample
    /// across two output positions — the shape and values are hand-checkable.
    #[test]
    fn forward_upsamples_by_stride() {
        let (device, client) = setup();
        let input = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[1, 1, 2], &device).unwrap(),
            false,
        );
        // weight [c_in=1, c_out=1, k=2]
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 10.0], &[1, 1, 2], &device).unwrap(),
            false,
        );
        let out = var_conv_transpose1d(
            &input,
            &weight,
            None,
            2,
            PaddingMode::Valid,
            0,
            1,
            1,
            &client,
        )
        .unwrap();
        // out_len = (2-1)*2 + 2 = 4; contributions at 0,1 from x0 and 2,3 from x1
        assert_eq!(out.tensor().shape(), &[1, 1, 4]);
        let got: Vec<f32> = out.tensor().to_vec();
        assert_eq!(got, vec![1.0, 10.0, 2.0, 20.0]);
    }

    #[test]
    fn bias_is_added_per_output_channel() {
        let (device, client) = setup();
        let input = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1, 1, 1], &device).unwrap(),
            false,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 3.0], &[1, 2, 1], &device).unwrap(),
            false,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, -1.0], &[2], &device).unwrap(),
            false,
        );
        let out = var_conv_transpose1d(
            &input,
            &weight,
            Some(&bias),
            1,
            PaddingMode::Valid,
            0,
            1,
            1,
            &client,
        )
        .unwrap();
        let got: Vec<f32> = out.tensor().to_vec();
        assert_eq!(got, vec![2.5, 2.0]);
    }

    /// Gradients must reach input, weight AND bias — a severed path here would
    /// silently freeze any layer built on this op.
    #[test]
    fn backward_reaches_input_weight_and_bias() {
        let (device, client) = setup();
        let input = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, -2.0, 0.5], &[1, 1, 3], &device)
                .unwrap(),
            true,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, -1.5], &[1, 1, 2], &device).unwrap(),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.25f32], &[1], &device).unwrap(),
            true,
        );

        let out = var_conv_transpose1d(
            &input,
            &weight,
            Some(&bias),
            2,
            PaddingMode::Valid,
            0,
            1,
            1,
            &client,
        )
        .unwrap();
        let loss = crate::autograd::var_sum(&out, &[0, 1, 2], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gi: Vec<f32> = grads.get(input.id()).expect("input grad").to_vec();
        let gw: Vec<f32> = grads.get(weight.id()).expect("weight grad").to_vec();
        let gb: Vec<f32> = grads.get(bias.id()).expect("bias grad").to_vec();

        // d(sum)/dx[j] = sum_k w[k] = 0.5 - 1.5 = -1 for every interior position.
        for v in &gi {
            assert!((v - (-1.0)).abs() < 1e-5, "expected -1, got {v}");
        }
        // d(sum)/dw[k] = sum_j x[j] = -0.5
        for v in &gw {
            assert!((v - (-0.5)).abs() < 1e-5, "expected -0.5, got {v}");
        }
        // d(sum)/db = number of output positions = (3-1)*2 + 2 = 6
        assert!((gb[0] - 6.0).abs() < 1e-5, "expected 6, got {}", gb[0]);
    }

    /// `Same` padding is defined as "output length == input length * stride",
    /// which is the upsampling contract these layers rely on.
    #[test]
    fn same_padding_gives_exact_stride_upsampling() {
        let (device, client) = setup();
        let input = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&vec![0.5f32; 7], &[1, 1, 7], &device).unwrap(),
            false,
        );
        let weight = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&vec![0.1f32; 12], &[1, 1, 12], &device).unwrap(),
            false,
        );
        let out = var_conv_transpose1d(
            &input,
            &weight,
            None,
            2,
            PaddingMode::Same,
            0,
            1,
            1,
            &client,
        )
        .unwrap();
        assert_eq!(out.tensor().shape(), &[1, 1, 14]);
    }
}
