//! Backward implementation for Fused Add + RMS Normalization

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_mean, var_mul, var_sub, var_sum};
use crate::error::Result;
use crate::ops::{BinaryOps, NormalizationOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for Fused Add + RMS Normalization: pre_norm = x + residual, output = rms_norm(pre_norm, weight, eps)
///
/// Gradients:
/// - d_input_residual = shared gradient for both x and residual (since d(x+r)/dx = d(x+r)/dr = 1)
/// - d_weight = sum(grad_out * x_norm, batch_dims)
pub struct FusedAddRmsNormBackward<R: Runtime> {
    input_ids: [TensorId; 3],
    saved_tensors: Vec<Tensor<R>>, // [pre_norm, weight]
    eps: f32,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime> FusedAddRmsNormBackward<R> {
    /// Create a new FusedAddRmsNormBackward
    pub fn new(
        x_id: TensorId,
        residual_id: TensorId,
        weight_id: TensorId,
        pre_norm: Tensor<R>,
        weight: Tensor<R>,
        eps: f32,
        x_grad_fn: Option<Arc<dyn GradFn<R>>>,
        residual_grad_fn: Option<Arc<dyn GradFn<R>>>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [x_id, residual_id, weight_id],
            saved_tensors: vec![pre_norm, weight],
            eps,
            input_grad_fns: [x_grad_fn, residual_grad_fn, weight_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for FusedAddRmsNormBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + ReduceOps<R> + UnaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // One fused kernel produces all three gradients in a single pass, so no
        // slot has a cost of its own to skip. Guarding would need the kernel
        // itself to take the mask.
        let client = R::default_client(grad_output.device());
        let pre_norm = &self.saved_tensors[0];
        let weight = &self.saved_tensors[1];

        let (d_input_residual, d_weight) =
            client.fused_add_rms_norm_bwd(grad_output, pre_norm, weight, self.eps)?;

        // x and residual share the same gradient
        Ok(vec![
            Some(d_input_residual.clone()),
            Some(d_input_residual),
            Some(d_weight),
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
        let x_sq = client.mul(pre_norm, pre_norm)?;
        let mean_x_sq = client.mean(&x_sq, &[last_dim], true)?;
        let variance_eps = client.add_scalar(&mean_x_sq, self.eps as f64)?;
        let rms = client.sqrt(&variance_eps)?;
        let rstd = client.recip(&rms)?;
        let x_norm = client.mul(pre_norm, &rstd)?;

        // Wrap as non-differentiable Vars
        let rstd_var = Var::new(rstd, false);
        let x_norm_var = Var::new(x_norm, false);
        let weight_var = Var::new(weight.clone(), false);

        // d_input_residual = rstd * (grad_output * weight - x_norm * mean(grad_output * weight * x_norm))
        let gw = var_mul(grad_output, &weight_var, &client)?;
        let gw_xn = var_mul(&gw, &x_norm_var, &client)?;
        let mean_gw_xn = var_mean(&gw_xn, &[last_dim], true, &client)?;
        let correction = var_mul(&x_norm_var, &mean_gw_xn, &client)?;
        let inner = var_sub(&gw, &correction, &client)?;
        let d_input_residual = var_mul(&inner, &rstd_var, &client)?;

        // d_weight = sum(grad_output * x_norm, batch_dims)
        let g_xn = var_mul(grad_output, &x_norm_var, &client)?;
        let batch_dims: Vec<usize> = (0..last_dim).collect();
        let d_weight = if batch_dims.is_empty() {
            g_xn
        } else {
            var_sum(&g_xn, &batch_dims, false, &client)?
        };

        Ok(vec![
            Some(d_input_residual.clone()),
            Some(d_input_residual),
            Some(d_weight),
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
        "FusedAddRmsNormBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_fused_add_rms_norm_backward_basic() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let pre_norm = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 4], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device);
        let eps = 1e-5f32;

        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 0.0], &[1, 4], &device);

        let (d_input_residual, d_weight) = client
            .fused_add_rms_norm_bwd(&grad_out, &pre_norm, &weight, eps)
            .unwrap();

        let di: Vec<f32> = d_input_residual.to_vec();
        let dw: Vec<f32> = d_weight.to_vec();

        for val in &di {
            assert!(val.is_finite(), "d_input_residual should be finite");
        }
        for val in &dw {
            assert!(val.is_finite(), "d_weight should be finite");
        }
    }

    #[test]
    fn test_fused_add_rms_norm_backward_shared_gradient() {
        let device = CpuDevice::new();

        let pre_norm = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device);

        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 3], &device);

        let backward = FusedAddRmsNormBackward::<CpuRuntime>::new(
            TensorId::new(),
            TensorId::new(),
            weight.id(),
            pre_norm,
            weight,
            1e-5,
            None,
            None,
            None,
        );
        let grads = backward.backward_all(&grad_out).unwrap();

        assert_eq!(grads.len(), 3);
        // x and residual gradients should be identical
        let d_x: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let d_r: Vec<f32> = grads[1].as_ref().unwrap().to_vec();
        for (a, b) in d_x.iter().zip(d_r.iter()) {
            assert!((a - b).abs() < 1e-10, "x and residual grads must match");
        }
    }
}
