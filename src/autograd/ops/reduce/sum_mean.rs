//! Backward implementations for sum and mean reductions

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::var_div_scalar;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

use super::common::ensure_contiguous;

// ============================================================================
// SumBackward
// ============================================================================

/// Backward for sum reduction: z = sum(a, dims)
///
/// The gradient of sum is broadcast expansion.
/// For z = sum(a, dims), dL/da = broadcast(dL/dz, original_shape)
///
/// If keepdim=false, we need to unsqueeze the gradient before broadcasting.
pub struct SumBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SumBackward<R> {
    /// Create a new SumBackward
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for SumBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let mut grad = grad_output.clone();

        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }

        grad = ensure_contiguous(grad.broadcast_to(&self.input_shape)?)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let mut grad_tensor = grad_output.tensor().clone();

        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad_tensor = grad_tensor.unsqueeze(dim as isize)?;
            }
        }

        grad_tensor = ensure_contiguous(grad_tensor.broadcast_to(&self.input_shape)?)?;

        Ok(vec![Some(Var::new(grad_tensor, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

// ============================================================================
// MeanBackward
// ============================================================================

/// Backward for mean reduction: z = mean(a, dims)
///
/// For z = mean(a, dims), dL/da = broadcast(dL/dz, original_shape) / count
/// where count is the number of elements being averaged.
pub struct MeanBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MeanBackward<R> {
    /// Create a new MeanBackward
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for MeanBackward<R>
where
    R::Client: ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        let count: usize = self.dims.iter().map(|&d| self.input_shape[d]).product();
        let count_f64 = count as f64;

        let mut grad = grad_output.clone();

        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }

        grad = ensure_contiguous(grad.broadcast_to(&self.input_shape)?)?;

        let grad = client.div_scalar(&grad, count_f64)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + crate::ops::TensorOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let count: usize = self.dims.iter().map(|&d| self.input_shape[d]).product();
        let count_f64 = count as f64;

        let mut grad_tensor = grad_output.tensor().clone();

        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad_tensor = grad_tensor.unsqueeze(dim as isize)?;
            }
        }

        grad_tensor = ensure_contiguous(grad_tensor.broadcast_to(&self.input_shape)?)?;

        let grad_var = Var::new(grad_tensor, grad_output.requires_grad());
        let grad = var_div_scalar(&grad_var, count_f64, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_sum_backward_keepdim() {
        let device = CpuDevice::new();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device).unwrap();

        let backward = SumBackward::<CpuRuntime>::new(TensorId::new(), &[2, 3], &[1], true, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_backward_no_keepdim() {
        let device = CpuDevice::new();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2], DType::F32, &device).unwrap();

        let backward = SumBackward::<CpuRuntime>::new(TensorId::new(), &[2, 3], &[1], false, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_backward() {
        let device = CpuDevice::new();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device).unwrap();

        let backward = MeanBackward::<CpuRuntime>::new(TensorId::new(), &[2, 3], &[1], true, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        let expected = 1.0 / 3.0;
        for val in grad_data {
            assert!((val - expected).abs() < 1e-6);
        }
    }
}
