//! Binary arithmetic operations (add, sub, mul, div, pow)

use super::macros::*;
use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

impl_var_binary_op_shapes!(
    /// Add two variables: z = a + b
    ///
    /// Creates AddBackward for gradient computation.
    var_add, add, AddBackward
);

impl_var_binary_op_shapes!(
    /// Subtract two variables: z = a - b
    ///
    /// Creates SubBackward for gradient computation.
    var_sub, sub, SubBackward
);

impl_var_binary_op_tensors!(
    /// Multiply two variables: z = a * b
    ///
    /// Creates MulBackward for gradient computation.
    var_mul, mul, MulBackward
);

impl_var_binary_op_tensors!(
    /// Divide two variables: z = a / b
    ///
    /// Creates DivBackward for gradient computation.
    var_div, div, DivBackward
);

/// Power: z = a^b
///
/// Creates PowBackward for gradient computation.
/// Note: Has special backward constructor requiring output tensor.
pub fn var_pow<R, C>(a: &Var<R>, b: &Var<R>, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let output = client.pow(a.tensor(), b.tensor())?;

    if a.requires_grad() || b.requires_grad() {
        let grad_fn = PowBackward::<R>::new(
            a.id(),
            b.id(),
            a.tensor().clone(),
            b.tensor().clone(),
            output.clone(),
            a.grad_fn().cloned(),
            b.grad_fn().cloned(),
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
    fn test_var_mul_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_mul(&x, &y, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 3.0).abs() < 1e-6);
        assert!((grad_y[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_add_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_add(&x, &y, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_chain_rule() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = (x * y) + x = x*(y+1)
        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        let xy = var_mul(&x, &y, &client).unwrap();
        let z = var_add(&xy, &x, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![8.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();

        assert!((grad_x[0] - 4.0).abs() < 1e-6); // dz/dx = y + 1 = 4
        assert!((grad_y[0] - 2.0).abs() < 1e-6); // dz/dy = x = 2
    }
}
