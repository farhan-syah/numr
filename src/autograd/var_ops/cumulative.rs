//! Cumulative operations (cumsum, cumprod)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{CumulativeOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Cumulative sum along a dimension: z = cumsum(a, dim)
pub fn var_cumsum<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + CumulativeOps<R>,
    R::Client: CumulativeOps<R>,
{
    let ndim = a.tensor().ndim();
    let resolved_dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    let output = client.cumsum(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn = CumsumBackward::<R>::new(a.id(), resolved_dim, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Cumulative product along a dimension: z = cumprod(a, dim)
pub fn var_cumprod<R, C>(a: &Var<R>, dim: isize, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + CumulativeOps<R> + TensorOps<R>,
    R::Client: CumulativeOps<R> + TensorOps<R>,
{
    let ndim = a.tensor().ndim();
    let resolved_dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    let output = client.cumprod(a.tensor(), dim)?;

    if a.requires_grad() {
        let grad_fn = CumprodBackward::<R>::new(
            a.id(),
            a.tensor().clone(),
            output.clone(),
            resolved_dim,
            a.grad_fn().cloned(),
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
    fn test_var_cumsum_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap(),
            true,
        );

        let z = var_cumsum(&x, 0, &client).unwrap();
        // z = [1, 3, 6, 10]
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![1.0, 3.0, 6.0, 10.0]);

        // Sum z to get a scalar for backward
        let loss = crate::autograd::var_ops::var_sum(&z, &[0], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        // dL/dx_i = sum_{j>=i} 1 = n - i
        // So grad = [4, 3, 2, 1]
        assert_eq!(grad_x, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_var_cumprod_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap(),
            true,
        );

        let z = var_cumprod(&x, 0, &client).unwrap();
        // z = [1, 2, 6]
        let z_data: Vec<f32> = z.tensor().to_vec();
        assert_eq!(z_data, vec![1.0, 2.0, 6.0]);

        let loss = crate::autograd::var_ops::var_sum(&z, &[0], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        // dL/dx_0 = dz_0/dx_0 + dz_1/dx_0 + dz_2/dx_0 = 1 + 2 + 6 = 9
        //         = (1*1 + 1*2 + 1*6) / 1 = 9
        // dL/dx_1 = dz_1/dx_1 + dz_2/dx_1 = 1 + 3 = 4
        //         = (1*2 + 1*6) / 2 = 4
        // dL/dx_2 = dz_2/dx_2 = 2
        //         = (1*6) / 3 = 2
        assert!((grad_x[0] - 9.0).abs() < 1e-5);
        assert!((grad_x[1] - 4.0).abs() < 1e-5);
        assert!((grad_x[2] - 2.0).abs() < 1e-5);
    }
}
