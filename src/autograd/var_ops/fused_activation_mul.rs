//! Fused activation-multiplication with gradient support
//!
//! Each function computes `activation(a) * b` in a single memory pass.
//! Backward computes:
//! - d_a = grad_output * b * activation'(a)
//! - d_b = grad_output * activation(a)

use crate::autograd::Var;
use crate::autograd::var_ops::var_mul;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, ScalarOps, TensorOps, UnaryOps,
};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Which fused activation-mul variant
#[derive(Clone, Copy)]
enum FusedKind {
    Silu,
    Gelu,
    Relu,
    Sigmoid,
}

/// Fused SiLU-Mul: output = silu(a) * b
pub fn var_silu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Silu)
}

/// Fused GELU-Mul: output = gelu(a) * b
pub fn var_gelu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Gelu)
}

/// Fused ReLU-Mul: output = relu(a) * b
pub fn var_relu_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Relu)
}

/// Fused Sigmoid-Mul: output = sigmoid(a) * b
pub fn var_sigmoid_mul<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    var_fused_activation_mul(a, b, client, FusedKind::Sigmoid)
}

/// Shared implementation for all fused activation-mul variants
fn var_fused_activation_mul<R, C>(
    a: &Var<R>,
    b: &Var<R>,
    client: &C,
    kind: FusedKind,
) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    // Forward: use fused kernel
    let output = match kind {
        FusedKind::Silu => client.silu_mul(a.tensor(), b.tensor())?,
        FusedKind::Gelu => client.gelu_mul(a.tensor(), b.tensor())?,
        FusedKind::Relu => client.relu_mul(a.tensor(), b.tensor())?,
        FusedKind::Sigmoid => client.sigmoid_mul(a.tensor(), b.tensor())?,
    };

    if a.requires_grad() || b.requires_grad() {
        // Compute activation(a) for backward (needed for d_b)
        let activation_a = match kind {
            FusedKind::Silu => client.silu(a.tensor())?,
            FusedKind::Gelu => client.gelu(a.tensor())?,
            FusedKind::Relu => client.relu(a.tensor())?,
            FusedKind::Sigmoid => client.sigmoid(a.tensor())?,
        };

        let grad_fn = FusedActivationMulBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            activation_a,
            kind,
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Backward for fused activation-mul: output = activation(a) * b
///
/// Gradients:
/// - d_b = grad_output * activation(a)
/// - d_a = grad_output * b * activation'(a)
///
/// Derivatives:
/// - silu'(x)    = sigmoid(x) * (1 + x - silu(x))
/// - gelu'(x)    = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*sqrt(2/π)*(1+3*0.044715*x²)
/// - relu'(x)    = 1 if x > 0, else 0
/// - sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
pub struct FusedActivationMulBackward<R: Runtime> {
    input_ids: [crate::tensor::TensorId; 2],
    saved_a: crate::tensor::Tensor<R>,
    saved_b: crate::tensor::Tensor<R>,
    saved_activation_a: crate::tensor::Tensor<R>,
    kind: FusedKind,
    a_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    b_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> FusedActivationMulBackward<R> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        a_id: crate::tensor::TensorId,
        b_id: crate::tensor::TensorId,
        a: crate::tensor::Tensor<R>,
        b: crate::tensor::Tensor<R>,
        activation_a: crate::tensor::Tensor<R>,
        kind: FusedKind,
        a_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_a: a,
            saved_b: b,
            saved_activation_a: activation_a,
            kind,
            a_grad_fn,
            b_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for FusedActivationMulBackward<R>
where
    R::Client: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        // One fused kernel returns both gradients together, so neither slot has
        // a cost of its own to skip. Guarding would need the kernel to take the
        // mask.
        let client = R::default_client(grad_output.device());

        // Delegate to fused backward trait method — allows backends (e.g. CUDA)
        // to provide a single fused kernel for the entire backward pass.
        let (d_a, d_b) = match self.kind {
            FusedKind::Silu => client.silu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Gelu => client.gelu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Relu => client.relu_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?,
            FusedKind::Sigmoid => {
                client.sigmoid_mul_bwd(grad_output, &self.saved_a, &self.saved_b)?
            }
        };

        Ok(vec![Some(d_a), Some(d_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ActivationOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + CompareOps<R>
            + ConditionalOps<R>
            + UnaryOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // d_b = grad_output * activation(a) (activation_a is constant w.r.t. higher-order)
        let act_var = Var::new(self.saved_activation_a.clone(), false);
        let d_b = var_mul(grad_output, &act_var, &client)?;

        // d_a = grad_output * b * activation'(a)
        let activation_deriv = compute_activation_derivative(
            &client,
            &self.saved_a,
            &self.saved_activation_a,
            self.kind,
        )?;
        let deriv_var = Var::new(activation_deriv, false);
        let b_var = Var::new(self.saved_b.clone(), false);
        let grad_times_b = var_mul(grad_output, &b_var, &client)?;
        let d_a = var_mul(&grad_times_b, &deriv_var, &client)?;

        Ok(vec![Some(d_a), Some(d_b)])
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        vec![self.a_grad_fn.clone(), self.b_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_a)
    }

    fn name(&self) -> &'static str {
        match self.kind {
            FusedKind::Silu => "SiluMulBackward",
            FusedKind::Gelu => "GeluMulBackward",
            FusedKind::Relu => "ReluMulBackward",
            FusedKind::Sigmoid => "SigmoidMulBackward",
        }
    }
}

/// Compute activation'(x) for the backward pass
fn compute_activation_derivative<R, C>(
    client: &C,
    a: &crate::tensor::Tensor<R>,
    activation_a: &crate::tensor::Tensor<R>,
    kind: FusedKind,
) -> Result<crate::tensor::Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: TensorOps<R>
        + ActivationOps<R>
        + ScalarOps<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + UnaryOps<R>,
{
    match kind {
        FusedKind::Silu => {
            // silu'(x) = sigmoid(x) * (1 + x - silu(x))
            let sigmoid_a = client.sigmoid(a)?;
            let one_plus_a = client.add_scalar(a, 1.0)?;
            let one_plus_a_minus_silu = client.sub(&one_plus_a, activation_a)?;
            client.mul(&sigmoid_a, &one_plus_a_minus_silu)
        }
        FusedKind::Gelu => {
            // gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*sech²(inner)*sqrt(2/π)*(1+3*0.044715*x²)
            // where inner = sqrt(2/π) * (x + 0.044715*x³)
            //
            // Simpler: d/dx gelu(x) = gelu(x)/x + x * pdf(x)
            // But that has x=0 issues. Use the direct form:
            //
            // Let's use: gelu(x) = 0.5*x*(1+tanh(inner))
            // gelu'(x) = 0.5*(1+tanh(inner)) + 0.5*x*(1-tanh²(inner))*inner'
            // inner' = sqrt(2/π)*(1 + 3*0.044715*x²)
            let x_sq = client.mul(a, a)?;
            let x_cu = client.mul(&x_sq, a)?;
            let coef_x_cu = client.mul_scalar(&x_cu, 0.044715)?;
            let inner_arg = client.add(a, &coef_x_cu)?;
            let sqrt_2_pi = 0.7978845608028654;
            let inner = client.mul_scalar(&inner_arg, sqrt_2_pi)?;

            // tanh(inner)
            let tanh_inner = {
                // Use exp to compute tanh: tanh(x) = (exp(2x)-1)/(exp(2x)+1)
                let two_inner = client.mul_scalar(&inner, 2.0)?;
                let exp_2 = client.exp(&two_inner)?;
                let num = client.add_scalar(&exp_2, -1.0)?;
                let den = client.add_scalar(&exp_2, 1.0)?;
                client.div(&num, &den)?
            };

            // 0.5*(1+tanh(inner))
            let one_plus_tanh = client.add_scalar(&tanh_inner, 1.0)?;
            let term1 = client.mul_scalar(&one_plus_tanh, 0.5)?;

            // sech²(inner) = 1 - tanh²(inner)
            let tanh_sq = client.mul(&tanh_inner, &tanh_inner)?;
            let sech_sq = client.add_scalar(&tanh_sq, -1.0)?;
            let sech_sq = client.neg(&sech_sq)?;

            // inner' = sqrt(2/π) * (1 + 3*0.044715*x²)
            let three_coef_x_sq = client.mul_scalar(&x_sq, 3.0 * 0.044715)?;
            let inner_deriv_unscaled = client.add_scalar(&three_coef_x_sq, 1.0)?;
            let inner_deriv = client.mul_scalar(&inner_deriv_unscaled, sqrt_2_pi)?;

            // term2 = 0.5 * x * sech²(inner) * inner'
            let x_sech_sq = client.mul(a, &sech_sq)?;
            let x_sech_sq_inner_d = client.mul(&x_sech_sq, &inner_deriv)?;
            let term2 = client.mul_scalar(&x_sech_sq_inner_d, 0.5)?;

            client.add(&term1, &term2)
        }
        FusedKind::Relu => {
            // relu'(x) = 1 if x > 0, else 0
            let zeros = crate::tensor::Tensor::<R>::zeros(a.shape(), a.dtype(), a.device());
            let ones = crate::tensor::Tensor::<R>::ones(a.shape(), a.dtype(), a.device());
            let mask = client.gt(a, &zeros)?;
            client.where_cond(&mask, &ones, &zeros)
        }
        FusedKind::Sigmoid => {
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            let sigmoid_a = client.sigmoid(a)?;
            let one_minus_sig = client.add_scalar(&sigmoid_a, -1.0)?;
            let one_minus_sig = client.neg(&one_minus_sig)?;
            client.mul(&sigmoid_a, &one_minus_sig)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_silu_mul_forward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0, -1.0], &[3], &device),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
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
                &Tensor::<CpuRuntime>::from_slice(&a_data, &[5], &device),
                &Tensor::<CpuRuntime>::from_slice(&b_data, &[5], &device),
            )
            .unwrap();

        // Separate
        let silu_a = client
            .silu(&Tensor::<CpuRuntime>::from_slice(&a_data, &[5], &device))
            .unwrap();
        let separate = client
            .mul(
                &silu_a,
                &Tensor::<CpuRuntime>::from_slice(&b_data, &[5], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0], &[2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 2.0], &[3], &device),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 5.0, 5.0], &[3], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
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
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            false,
        );

        let output = var_gelu_mul(&a, &b, &client).unwrap();
        assert!(!output.requires_grad());
    }
}
