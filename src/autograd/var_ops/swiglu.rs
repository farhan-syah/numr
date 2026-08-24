//! Fused SwiGLU activation with gradient support
//!
//! SwiGLU(gate, up) = silu(gate) * up
//!
//! Fused version saves one intermediate tensor vs composing var_silu + var_mul:
//! - Composed: stores gate, silu(gate), up (3 tensors)
//! - Fused: stores gate, up, output (3 tensors but recomputes sigmoid in backward)
//!
//! More importantly, the fused backward computes gradients in fewer ops.

use crate::autograd::Var;
use crate::autograd::var_ops::var_mul;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, BinaryOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Fused SwiGLU: output = silu(gate) * up
///
/// # Arguments
/// * `gate` - Gate input (will have silu applied)
/// * `up` - Up projection (multiplied element-wise with activated gate)
/// * `client` - Runtime client
///
/// # Returns
/// The SwiGLU output with autograd tracking.
pub fn var_swiglu<R, C>(gate: &Var<R>, up: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R>,
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    // Forward: output = silu(gate) * up (fused single-pass kernel)
    let silu_gate = client.silu(gate.tensor())?;
    let output = client.silu_mul(gate.tensor(), up.tensor())?;

    if gate.requires_grad() || up.requires_grad() {
        let grad_fn = SwiGLUBackward::<R>::new(
            gate.id(),
            up.id(),
            gate.tensor().clone(),
            up.tensor().clone(),
            silu_gate,
            gate.grad_fn().cloned(),
            up.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for fused SwiGLU: output = silu(gate) * up
///
/// Gradients:
/// - d_gate = grad_output * up * silu'(gate)
///   = grad_output * up * (sigmoid(gate) * (1 + gate - silu(gate)))
/// - d_up   = grad_output * silu(gate)
pub struct SwiGLUBackward<R: Runtime> {
    input_ids: [crate::tensor::TensorId; 2],
    saved_gate: crate::tensor::Tensor<R>,
    saved_up: crate::tensor::Tensor<R>,
    saved_silu_gate: crate::tensor::Tensor<R>,
    gate_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    up_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> SwiGLUBackward<R> {
    pub fn new(
        gate_id: crate::tensor::TensorId,
        up_id: crate::tensor::TensorId,
        gate: crate::tensor::Tensor<R>,
        up: crate::tensor::Tensor<R>,
        silu_gate: crate::tensor::Tensor<R>,
        gate_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        up_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [gate_id, up_id],
            saved_gate: gate,
            saved_up: up,
            saved_silu_gate: silu_gate,
            gate_grad_fn,
            up_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for SwiGLUBackward<R>
where
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // The two slots share only `grad_output`. d_gate costs five extra
        // elementwise passes that d_up never touches, so each is guarded.
        // Returned order is [gate, up].

        // d_gate = grad_output * up * silu'(gate)
        // silu'(x) = sigmoid(x) * (1 + x - silu(x))
        let d_gate = if needed[0] {
            let sigmoid_gate = client.sigmoid(&self.saved_gate)?;
            let one_plus_gate = client.add_scalar(&self.saved_gate, 1.0)?;
            let one_plus_gate_minus_silu = client.sub(&one_plus_gate, &self.saved_silu_gate)?;
            let silu_deriv = client.mul(&sigmoid_gate, &one_plus_gate_minus_silu)?;
            let grad_times_up = client.mul(grad_output, &self.saved_up)?;
            Some(client.mul(&grad_times_up, &silu_deriv)?)
        } else {
            None
        };

        // d_up = grad_output * silu(gate)
        let d_up = if needed[1] {
            Some(client.mul(grad_output, &self.saved_silu_gate)?)
        } else {
            None
        };

        Ok(vec![d_gate, d_up])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + ScalarOps<R> + BinaryOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // d_up = grad_output * silu(gate) [silu_gate is constant w.r.t. higher-order]
        let silu_var = Var::new(self.saved_silu_gate.clone(), false);
        let d_up = var_mul(grad_output, &silu_var, &client)?;

        // d_gate = grad_output * up * silu'(gate)
        let sigmoid_gate = client.sigmoid(&self.saved_gate)?;
        let one_plus_gate = client.add_scalar(&self.saved_gate, 1.0)?;
        let one_plus_gate_minus_silu = client.sub(&one_plus_gate, &self.saved_silu_gate)?;
        let silu_deriv = client.mul(&sigmoid_gate, &one_plus_gate_minus_silu)?;
        let silu_deriv_var = Var::new(silu_deriv, false);

        let up_var = Var::new(self.saved_up.clone(), false);
        let grad_times_up = var_mul(grad_output, &up_var, &client)?;
        let d_gate = var_mul(&grad_times_up, &silu_deriv_var, &client)?;

        Ok(vec![Some(d_gate), Some(d_up)])
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        vec![self.gate_grad_fn.clone(), self.up_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_gate)
    }

    fn name(&self) -> &'static str {
        "SwiGLUBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_swiglu_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let gate = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 1.0, -1.0], &[3], &device).unwrap(),
            false,
        );
        let up = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap(),
            false,
        );

        let output = var_swiglu(&gate, &up, &client).unwrap();
        let data: Vec<f32> = output.tensor().to_vec();

        // silu(0) * 1 = 0 * 0.5 * 1 = 0
        assert!(data[0].abs() < 1e-6);
        // silu(1) * 2 = 0.7311 * 2 ≈ 1.4621
        let silu_1 = 1.0 / (1.0 + (-1.0f32).exp());
        assert!((data[1] - silu_1 * 2.0).abs() < 1e-4);
        // silu(-1) * 3 = -0.2689 * 3 ≈ -0.8067
        let silu_neg1 = -1.0 / (1.0 + 1.0f32.exp());
        assert!((data[2] - silu_neg1 * 3.0).abs() < 1e-4);
    }

    #[test]
    fn test_swiglu_backward_gate() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let gate = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, -1.0], &[2], &device).unwrap(),
            true,
        );
        let up = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
            true,
        );

        let output = var_swiglu(&gate, &up, &client).unwrap();
        let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let d_gate: Vec<f32> = grads.get(gate.id()).unwrap().to_vec();
        let d_up: Vec<f32> = grads.get(up.id()).unwrap().to_vec();

        // Verify d_up = silu(gate)
        for (i, &g) in [1.0f32, -1.0].iter().enumerate() {
            let expected_d_up = g * (1.0 / (1.0 + (-g).exp()));
            assert!(
                (d_up[i] - expected_d_up).abs() < 1e-5,
                "d_up[{i}]: got {}, expected {expected_d_up}",
                d_up[i]
            );
        }

        // Verify d_gate = up * silu'(gate)
        for (i, (&g, &u)) in [1.0f32, -1.0].iter().zip([2.0f32, 3.0].iter()).enumerate() {
            let sig = 1.0 / (1.0 + (-g).exp());
            let silu_g = g * sig;
            let silu_deriv = sig * (1.0 + g - silu_g);
            let expected = u * silu_deriv;
            assert!(
                (d_gate[i] - expected).abs() < 1e-4,
                "d_gate[{i}]: got {}, expected {expected}",
                d_gate[i]
            );
        }
    }

    #[test]
    fn test_swiglu_no_grad() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let gate = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1], &device).unwrap(),
            false,
        );
        let up = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            false,
        );

        let output = var_swiglu(&gate, &up, &client).unwrap();
        assert!(!output.requires_grad());
    }
}
