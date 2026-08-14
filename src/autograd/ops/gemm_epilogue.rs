//! Backward implementation for fused GEMM + bias + activation

use super::broadcast::{reduce_grad_for_broadcast, reduce_var_for_broadcast};
use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_matmul, var_sum};
use crate::error::Result;
use crate::ops::{BinaryOps, GemmActivation, MatmulOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// Backward for fused GEMM + bias + activation: output = activation(A @ B + bias)
pub struct MatmulBiasActivationBackward<R: Runtime> {
    input_ids: [TensorId; 3],
    saved_tensors: Vec<Tensor<R>>, // [a, b, bias]
    activation: GemmActivation,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 3],
}

impl<R: Runtime> MatmulBiasActivationBackward<R> {
    /// Create a new MatmulBiasActivationBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        bias_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        bias: Tensor<R>,
        activation: GemmActivation,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
        bias_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id, bias_id],
            saved_tensors: vec![a, b, bias],
            activation,
            input_grad_fns: [a_grad_fn, b_grad_fn, bias_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for MatmulBiasActivationBackward<R>
where
    R::Client:
        TensorOps<R> + ScalarOps<R> + BinaryOps<R> + ReduceOps<R> + UnaryOps<R> + MatmulOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let a = &self.saved_tensors[0];
        let b = &self.saved_tensors[1];
        let bias = &self.saved_tensors[2];

        // Recompute pre_activation = A @ B + bias
        let matmul_out = client.matmul(a, b)?;
        let pre_act = client.add(&matmul_out, bias)?;

        // Compute activation gradient: grad_pre = grad_output * activation'(pre_act)
        let grad_pre = apply_activation_grad(&client, grad_output, &pre_act, self.activation)?;

        // d_a = grad_pre @ B^T
        let b_t = b.transpose(-2, -1)?;
        let d_a_full = client.matmul(&grad_pre, &b_t)?;

        // d_b = A^T @ grad_pre
        let a_t = a.transpose(-2, -1)?;
        let d_b_full = client.matmul(&a_t, &grad_pre)?;

        // Batch dims may have been broadcast during forward: sum each operand's
        // gradient back over the dims where that operand had extent 1.
        let d_a = reduce_grad_for_broadcast::<R>(&d_a_full, a.shape())?;
        let d_b = reduce_grad_for_broadcast::<R>(&d_b_full, b.shape())?;

        // d_bias = sum(grad_pre, batch_and_row_dims)
        let ndim = grad_output.ndim();
        let batch_dims: Vec<usize> = (0..ndim - 1).collect();
        let d_bias = if batch_dims.is_empty() {
            grad_pre
        } else {
            client.sum(&grad_pre, &batch_dims, false)?
        };

        Ok(vec![Some(d_a), Some(d_b), Some(d_bias)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R>
            + TensorOps<R>
            + ScalarOps<R>
            + BinaryOps<R>
            + ReduceOps<R>
            + UnaryOps<R>
            + MatmulOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let a = &self.saved_tensors[0];
        let b = &self.saved_tensors[1];
        let bias = &self.saved_tensors[2];

        // Recompute pre_activation from saved tensors
        let matmul_out = client.matmul(a, b)?;
        let pre_act = client.add(&matmul_out, bias)?;

        // Compute activation gradient as a constant tensor
        let ones = client.add_scalar(&client.mul_scalar(&pre_act, 0.0)?, 1.0)?;
        let act_grad = apply_activation_grad(&client, &ones, &pre_act, self.activation)?;

        // grad_pre = grad_output * activation'(pre_act)
        let act_grad_var = Var::new(act_grad, false);
        let grad_pre = crate::autograd::var_ops::var_mul(grad_output, &act_grad_var, &client)?;

        // d_a = grad_pre @ B^T
        let b_t = b.transpose(-2, -1)?;
        let b_t_var = Var::new(b_t, false);
        let d_a_full = var_matmul(&grad_pre, &b_t_var, &client)?;

        // d_b = A^T @ grad_pre
        let a_t = a.transpose(-2, -1)?;
        let a_t_var = Var::new(a_t, false);
        let d_b_full = var_matmul(&a_t_var, &grad_pre, &client)?;

        // Batch dims may have been broadcast during forward: sum each operand's
        // gradient back over the dims where that operand had extent 1.
        let d_a = reduce_var_for_broadcast(&d_a_full, a.shape(), &client)?;
        let d_b = reduce_var_for_broadcast(&d_b_full, b.shape(), &client)?;

        // d_bias = sum(grad_pre, batch_dims)
        let ndim = grad_output.tensor().ndim();
        let batch_dims: Vec<usize> = (0..ndim - 1).collect();
        let d_bias = if batch_dims.is_empty() {
            grad_pre
        } else {
            var_sum(&grad_pre, &batch_dims, false, &client)?
        };

        Ok(vec![Some(d_a), Some(d_b), Some(d_bias)])
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
        "MatmulBiasActivationBackward"
    }
}

/// Compute grad_output * activation'(pre_act) using only basic ops
fn apply_activation_grad<R: Runtime>(
    client: &R::Client,
    grad: &Tensor<R>,
    pre_act: &Tensor<R>,
    activation: GemmActivation,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    match activation {
        GemmActivation::None => {
            // Identity: derivative is 1, so just return grad
            Ok(grad.clone())
        }
        GemmActivation::ReLU => {
            // ReLU': 1 if x > 0, 0 if x <= 0
            // Approximate mask: clamp(sign(x), 0, 1) using: (x + |x|) / (2 * |x| + eps)
            // Simpler: use step = (sign(x) + 1) / 2 where sign uses abs
            let abs_x = client.abs(pre_act)?;
            // For x > 0: sign = x/|x| = 1, for x < 0: sign = -1, x=0: 0
            let abs_plus_eps = client.add_scalar(&abs_x, 1e-30)?;
            let sign = client.div(pre_act, &abs_plus_eps)?;
            // mask = (sign + 1) / 2: maps 1->1, -1->0, 0->0.5 (close enough)
            let mask = client.mul_scalar(&client.add_scalar(&sign, 1.0)?, 0.5)?;
            client.mul(grad, &mask)
        }
        GemmActivation::Sigmoid => {
            // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            // sigmoid(x) = 1 / (1 + exp(-x))
            let neg_x = client.neg(pre_act)?;
            let exp_neg = client.exp(&neg_x)?;
            let one_plus = client.add_scalar(&exp_neg, 1.0)?;
            let sig = client.recip(&one_plus)?;
            let one_minus_sig = client.rsub_scalar(&sig, 1.0)?;
            let deriv = client.mul(&sig, &one_minus_sig)?;
            client.mul(grad, &deriv)
        }
        GemmActivation::Tanh => {
            // tanh'(x) = 1 - tanh(x)^2
            let t = client.tanh(pre_act)?;
            let t_sq = client.mul(&t, &t)?;
            let deriv = client.rsub_scalar(&t_sq, 1.0)?;
            client.mul(grad, &deriv)
        }
        GemmActivation::SiLU => {
            // silu(x) = x * sigmoid(x)
            // silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            let neg_x = client.neg(pre_act)?;
            let exp_neg = client.exp(&neg_x)?;
            let one_plus = client.add_scalar(&exp_neg, 1.0)?;
            let sig = client.recip(&one_plus)?;
            let one_minus_sig = client.rsub_scalar(&sig, 1.0)?;
            let x_one_minus_sig = client.mul(pre_act, &one_minus_sig)?;
            let inner = client.add_scalar(&x_one_minus_sig, 1.0)?;
            let deriv = client.mul(&sig, &inner)?;
            client.mul(grad, &deriv)
        }
        GemmActivation::GELU => {
            // GELU(x) = 0.5 * x * (1 + tanh(k)), k = sqrt(2/pi) * (x + 0.044715 * x^3)
            // d/dx = 0.5 * (1 + tanh(k)) + 0.5 * x * sech²(k) * dk/dx
            // dk/dx = sqrt(2/pi) * (1 + 3*0.044715*x²)
            let sqrt_2_pi: f64 = (2.0f64 / std::f64::consts::PI).sqrt();
            let x_sq = client.mul(pre_act, pre_act)?;
            let x_cubed = client.mul(pre_act, &x_sq)?;
            let inner = client.add(pre_act, &client.mul_scalar(&x_cubed, 0.044715)?)?;
            let k = client.mul_scalar(&inner, sqrt_2_pi)?;
            let tanh_k = client.tanh(&k)?;

            // 0.5 * (1 + tanh(k))
            let term1 = client.mul_scalar(&client.add_scalar(&tanh_k, 1.0)?, 0.5)?;

            // sech²(k) = 1 - tanh²(k)
            let tanh_sq = client.mul(&tanh_k, &tanh_k)?;
            let sech_sq = client.rsub_scalar(&tanh_sq, 1.0)?;

            // dk/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x²)
            let dk_dx = client.mul_scalar(
                &client.add_scalar(&client.mul_scalar(&x_sq, 3.0 * 0.044715)?, 1.0)?,
                sqrt_2_pi,
            )?;

            // 0.5 * x * sech²(k) * dk/dx
            let term2 =
                client.mul_scalar(&client.mul(pre_act, &client.mul(&sech_sq, &dk_dx)?)?, 0.5)?;

            let deriv = client.add(&term1, &term2)?;
            client.mul(grad, &deriv)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    /// A: [2,2,1,2] @ B: [2,1,2,1] -> [2,2,1,1], B broadcast over the MIDDLE
    /// batch dim. d_b must be summed back to [2,1,2,1].
    ///
    /// With identity activation and grad_out = ones:
    ///   d_a[i,j] = B[i]^T          -> [10,20] / [30,40]
    ///   d_b[i]   = A[i,0]^T + A[i,1]^T -> [4,6] / [12,14]
    ///   d_bias   = sum(grad_out)   -> [4]
    fn broadcast_case() -> (
        CpuDevice,
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
        Tensor<CpuRuntime>,
    ) {
        let device = CpuDevice::new();
        let a = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 2, 1, 2],
            &device,
        );
        let b =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 1, 2, 1], &device);
        let bias = Tensor::<CpuRuntime>::zeros(&[1], DType::F32, &device);
        (device, a, b, bias)
    }

    #[test]
    fn test_gemm_epilogue_backward_broadcast_middle_batch_dim() {
        let (device, a, b, bias) = broadcast_case();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 2, 1, 1], DType::F32, &device);

        let backward = MatmulBiasActivationBackward::<CpuRuntime>::new(
            a.id(),
            b.id(),
            bias.id(),
            a.clone(),
            b.clone(),
            bias.clone(),
            GemmActivation::None,
            None,
            None,
            None,
        );
        let grads = backward.backward(&grad_out).unwrap();

        let d_a = grads[0].as_ref().unwrap();
        assert_eq!(d_a.shape(), &[2, 2, 1, 2]);
        let d_a_data: Vec<f32> = d_a.contiguous().unwrap().to_vec();
        assert_eq!(
            d_a_data,
            vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
        );

        let d_b = grads[1].as_ref().unwrap();
        assert_eq!(d_b.shape(), &[2, 1, 2, 1]);
        let d_b_data: Vec<f32> = d_b.contiguous().unwrap().to_vec();
        assert_eq!(d_b_data, vec![4.0, 6.0, 12.0, 14.0]);

        let d_bias = grads[2].as_ref().unwrap();
        let d_bias_data: Vec<f32> = d_bias.contiguous().unwrap().to_vec();
        assert_eq!(d_bias_data, vec![4.0]);
    }

    #[test]
    fn test_gemm_epilogue_backward_var_broadcast_middle_batch_dim() {
        let (device, a, b, bias) = broadcast_case();
        let grad_out = Var::new(
            Tensor::<CpuRuntime>::ones(&[2, 2, 1, 1], DType::F32, &device),
            true,
        );

        let backward = MatmulBiasActivationBackward::<CpuRuntime>::new(
            a.id(),
            b.id(),
            bias.id(),
            a.clone(),
            b.clone(),
            bias.clone(),
            GemmActivation::None,
            None,
            None,
            None,
        );
        let grads = backward.backward_var(&grad_out).unwrap();

        let d_a = grads[0].as_ref().unwrap();
        assert_eq!(d_a.shape(), &[2, 2, 1, 2]);
        let d_a_data: Vec<f32> = d_a.tensor().contiguous().unwrap().to_vec();
        assert_eq!(
            d_a_data,
            vec![10.0, 20.0, 10.0, 20.0, 30.0, 40.0, 30.0, 40.0]
        );

        let d_b = grads[1].as_ref().unwrap();
        assert_eq!(d_b.shape(), &[2, 1, 2, 1]);
        let d_b_data: Vec<f32> = d_b.tensor().contiguous().unwrap().to_vec();
        assert_eq!(d_b_data, vec![4.0, 6.0, 12.0, 14.0]);

        let d_bias = grads[2].as_ref().unwrap();
        let d_bias_data: Vec<f32> = d_bias.tensor().contiguous().unwrap().to_vec();
        assert_eq!(d_bias_data, vec![4.0]);
    }
}
