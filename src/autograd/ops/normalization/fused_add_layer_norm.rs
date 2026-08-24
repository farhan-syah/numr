//! Backward implementation for Fused Add + Layer Normalization

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_mean, var_mul, var_sub, var_sum};
use crate::error::Result;
use crate::ops::{BinaryOps, NormalizationOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for Fused Add + Layer Normalization:
/// pre_norm = x + residual, output = layer_norm(pre_norm, weight, bias, eps)
///
/// Gradients:
/// - d_input_residual = shared gradient for both x and residual
/// - d_weight = sum(grad_out * x_norm, batch_dims)
/// - d_bias = sum(grad_out, batch_dims)
pub struct FusedAddLayerNormBackward<R: Runtime> {
    input_ids: [TensorId; 4],
    saved_tensors: Vec<Tensor<R>>, // [pre_norm, weight, bias]
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 4],
}

impl<R: Runtime> FusedAddLayerNormBackward<R> {
    /// Create a new FusedAddLayerNormBackward
    pub fn new(
        x_id: TensorId,
        residual_id: TensorId,
        weight_id: TensorId,
        bias_id: TensorId,
        pre_norm: Tensor<R>,
        weight: Tensor<R>,
        bias: Tensor<R>,
        eps: f32,
        x_grad_fn: Option<Arc<dyn GradFn<R>>>,
        residual_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [x_id, residual_id, weight_id, bias_id],
            saved_tensors: vec![pre_norm, weight, bias],
            eps,
            input_grad_fns: [x_grad_fn, residual_grad_fn, weight_grad_fn, bias_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for FusedAddLayerNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // One fused kernel produces all four gradients in a single pass, so no
        // slot has a cost of its own to skip. Guarding would need the kernel
        // itself to take the mask.
        let client = R::default_client(grad_output.device());
        let pre_norm = &self.saved_tensors[0];
        let weight = &self.saved_tensors[1];
        let bias = &self.saved_tensors[2];

        let (d_input_residual, d_weight, d_bias) =
            client.fused_add_layer_norm_bwd(grad_output, pre_norm, weight, bias, self.eps)?;

        Ok(vec![
            Some(d_input_residual.clone()),
            Some(d_input_residual),
            Some(d_weight),
            Some(d_bias),
        ])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + UnaryOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let pre_norm = &self.saved_tensors[0];
        let weight = &self.saved_tensors[1];
        let ndim = pre_norm.ndim();
        let last_dim = ndim - 1;

        // Recompute rstd and x_norm from pre_norm (treat as constants)
        let mu = client.mean(pre_norm, &[last_dim], true)?;
        let x_centered = client.sub(pre_norm, &mu)?;
        let x_centered_sq = client.mul(&x_centered, &x_centered)?;
        let variance = client.mean(&x_centered_sq, &[last_dim], true)?;
        let variance_eps = client.add_scalar(&variance, self.eps as f64)?;
        let std = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&std)?;
        let x_norm = client.mul(&x_centered, &rstd)?;

        // Wrap as non-differentiable Vars
        let rstd_var = Var::new(rstd, false);
        let x_norm_var = Var::new(x_norm, false);
        let weight_var = Var::new(weight.clone(), false);

        // d_input_residual = rstd * (gw - mean(gw) - x_norm * mean(gw * x_norm))
        let gw = var_mul(grad_output, &weight_var, &client)?;
        let mean_gw = var_mean(&gw, &[last_dim], true, &client)?;
        let gw_xn = var_mul(&gw, &x_norm_var, &client)?;
        let mean_gw_xn = var_mean(&gw_xn, &[last_dim], true, &client)?;
        let xn_mean_gw_xn = var_mul(&x_norm_var, &mean_gw_xn, &client)?;
        let inner = var_sub(&gw, &mean_gw, &client)?;
        let inner = var_sub(&inner, &xn_mean_gw_xn, &client)?;
        let d_input_residual = var_mul(&inner, &rstd_var, &client)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = var_mul(grad_output, &x_norm_var, &client)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            var_sum(&g_xn, &batch_dims, false, &client)?
        };

        // d_bias = sum(grad_output, batch_dims)
        let d_bias = if batch_dims.is_empty() {
            grad_output.clone()
        } else {
            var_sum(grad_output, &batch_dims, false, &client)?
        };

        Ok(vec![
            Some(d_input_residual.clone()),
            Some(d_input_residual),
            Some(d_weight),
            Some(d_bias),
        ])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "FusedAddLayerNormBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_fused_add_layer_norm_backward_basic() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let pre_norm =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device)
                .unwrap();
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();
        let bias =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap();
        let eps = 1e-5f32;

        let grad_out =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 4], &device)
                .unwrap();

        let (d_input_residual, d_weight, d_bias) = client
            .fused_add_layer_norm_bwd(&grad_out, &pre_norm, &weight, &bias, eps)
            .unwrap();

        let di: Vec<f32> = d_input_residual.to_vec();
        let dw: Vec<f32> = d_weight.to_vec();
        let db: Vec<f32> = d_bias.to_vec();

        // d_input for uniform grad through layer norm should sum to ~0
        let sum: f32 = di.iter().sum();
        assert!(
            sum.abs() < 1e-5,
            "d_input_residual sum should be ~0, got {sum}"
        );

        for val in dw.iter().chain(db.iter()) {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_fused_add_layer_norm_backward_shared_gradient() {
        let device = CpuDevice::new();

        let pre_norm =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap();
        let weight =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap();
        let bias =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0, 0.0], &[3], &device).unwrap();

        let grad_out =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[1, 3], &device).unwrap();

        let backward = FusedAddLayerNormBackward::<CpuRuntime>::new(
            TensorId::new(),
            TensorId::new(),
            weight.id(),
            bias.id(),
            pre_norm,
            weight,
            bias,
            1e-5,
            None,
            None,
            None,
            None,
        );
        let grads = backward.backward_all(&grad_out).unwrap();

        assert_eq!(grads.len(), 4);
        let d_x: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let d_r: Vec<f32> = grads[1].as_ref().unwrap().to_vec();
        for (a, b) in d_x.iter().zip(d_r.iter()) {
            assert!((a - b).abs() < 1e-10, "x and residual grads must match");
        }
    }
}
