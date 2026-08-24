//! Scalar operations (add_scalar, sub_scalar, mul_scalar, div_scalar, pow_scalar)

use super::macros::*;
use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

impl_var_scalar_op_id!(
    /// Add scalar: z = a + scalar
    var_add_scalar, add_scalar, AddScalarBackward
);

impl_var_scalar_op_id!(
    /// Subtract scalar: z = a - scalar
    var_sub_scalar, sub_scalar, SubScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Multiply by scalar: z = a * scalar
    var_mul_scalar, mul_scalar, MulScalarBackward
);

impl_var_scalar_op_scalar!(
    /// Divide by scalar: z = a / scalar
    var_div_scalar, div_scalar, DivScalarBackward
);

/// Power by scalar: z = a^scalar
///
/// Note: Requires both ScalarOps and TensorOps bounds.
pub fn var_pow_scalar<R, C>(a: &Var<R>, scalar: f64, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + TensorOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.pow_scalar(a.tensor(), scalar)?;

    if a.requires_grad() {
        let grad_fn =
            PowScalarBackward::<R>::new(a.id(), a.tensor().clone(), scalar, a.grad_fn().cloned());
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
    fn test_var_add_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_add_scalar(&x, 5.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![7.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_mul_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_mul_scalar(&x, 3.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![6.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_pow_scalar_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_pow_scalar(&x, 2.0, &client).unwrap();

        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![9.0]);

        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 6.0).abs() < 1e-6);
    }
}
