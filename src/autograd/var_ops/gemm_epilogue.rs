//! Fused GEMM + bias + activation var operations

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{GemmActivation, GemmEpilogueOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Fused GEMM + bias + activation: output = activation(A @ B + bias)
///
/// # Arguments
///
/// * `a` - Input variable of shape `[..., M, K]`
/// * `b` - Weight variable of shape `[..., K, N]`
/// * `bias` - Bias variable of shape `[N]`
/// * `activation` - Activation function to apply
/// * `client` - Runtime client
pub fn var_matmul_bias_activation<R, C>(
    a: &Var<R>,
    b: &Var<R>,
    bias: &Var<R>,
    activation: GemmActivation,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + GemmEpilogueOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output =
        client.matmul_bias_activation(a.tensor(), b.tensor(), bias.tensor(), activation)?;

    if a.requires_grad() || b.requires_grad() || bias.requires_grad() {
        let grad_fn = MatmulBiasActivationBackward::<R>::new(
            a.id(),
            b.id(),
            bias.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            bias.tensor().clone(),
            activation,
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_matmul_bias_activation_forward_none() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap(),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device).unwrap(),
            true,
        );

        let result =
            var_matmul_bias_activation(&a, &b, &bias, GemmActivation::None, &client).unwrap();
        let data: Vec<f32> = result.tensor().to_vec();

        // A @ B = [[1, 2], [3, 4]] @ [[1, 0], [0, 1]] = [[1, 2], [3, 4]]
        // + bias = [[1.1, 2.2], [3.1, 4.2]]
        assert!((data[0] - 1.1).abs() < 1e-5);
        assert!((data[1] - 2.2).abs() < 1e-5);
        assert!((data[2] - 3.1).abs() < 1e-5);
        assert!((data[3] - 4.2).abs() < 1e-5);
    }

    #[test]
    fn test_var_matmul_bias_activation_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap(),
            true,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap(),
            true,
        );

        let output =
            var_matmul_bias_activation(&a, &b, &bias, GemmActivation::None, &client).unwrap();
        let loss = crate::autograd::var_sum(&output, &[0, 1], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let ga: Vec<f32> = grads.get(a.id()).unwrap().to_vec();
        let gb: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        let gbias: Vec<f32> = grads.get(bias.id()).unwrap().to_vec();

        assert_eq!(ga.len(), 4);
        assert_eq!(gb.len(), 4);
        assert_eq!(gbias.len(), 2);

        for val in ga.iter().chain(gb.iter()).chain(gbias.iter()) {
            assert!(val.is_finite(), "gradient should be finite");
        }

        // d_bias should be sum over rows = [2.0, 2.0] (2 rows, each contributing 1.0)
        assert!((gbias[0] - 2.0).abs() < 1e-5);
        assert!((gbias[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_var_matmul_bias_activation_no_grad() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap(),
            false,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device).unwrap(),
            false,
        );
        let bias = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap(),
            false,
        );

        let result =
            var_matmul_bias_activation(&a, &b, &bias, GemmActivation::None, &client).unwrap();
        assert!(!result.requires_grad());
    }
}
