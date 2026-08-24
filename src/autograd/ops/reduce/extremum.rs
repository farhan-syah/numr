//! Backward implementations for max and min reductions

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::var_mul;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

use super::common::ensure_contiguous;

// ============================================================================
// MaxBackward
// ============================================================================

/// Backward for max reduction: z = max(a, dims)
///
/// The gradient flows only to the element(s) that had the maximum value.
/// For ties, the gradient is distributed equally among tied elements.
pub struct MaxBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MaxBackward<R> {
    /// Create a new MaxBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

/// Shared logic for extremum (max/min) backward pass
fn extremum_backward<R: Runtime>(
    saved_input: &Tensor<R>,
    grad_output: &Tensor<R>,
    dims: &[usize],
    keepdim: bool,
    is_max: bool,
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
{
    let client = R::default_client(grad_output.device());

    // Recompute extremum values
    let extremum_vals = if is_max {
        client.max(saved_input, dims, true)?
    } else {
        client.min(saved_input, dims, true)?
    };

    // Broadcast to input shape for comparison
    let extremum_broadcast = ensure_contiguous(extremum_vals.broadcast_to(saved_input.shape())?)?;

    // Create mask where input equals extremum (handles ties)
    let mask = client.eq(saved_input, &extremum_broadcast)?;

    // Count ties per reduction group
    let mask_sum = client.sum(&mask, dims, true)?;
    let mask_sum_broadcast = ensure_contiguous(mask_sum.broadcast_to(saved_input.shape())?)?;

    // Normalize mask by count
    let normalized_mask = client.div(&mask, &mask_sum_broadcast)?;

    // Broadcast grad_output to input shape
    let mut grad = grad_output.clone();
    if !keepdim {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();
        for &dim in &sorted_dims {
            grad = grad.unsqueeze(dim as isize)?;
        }
    }
    let grad_broadcast = ensure_contiguous(grad.broadcast_to(saved_input.shape())?)?;

    client.mul(&grad_broadcast, &normalized_mask)
}

/// Shared logic for extremum backward_var pass
fn extremum_backward_var<R: Runtime>(
    saved_input: &Tensor<R>,
    grad_output: &Var<R>,
    dims: &[usize],
    keepdim: bool,
    is_max: bool,
) -> Result<Var<R>>
where
    R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
{
    let client = R::default_client(grad_output.tensor().device());

    let extremum_vals = if is_max {
        client.max(saved_input, dims, true)?
    } else {
        client.min(saved_input, dims, true)?
    };

    let extremum_broadcast = ensure_contiguous(extremum_vals.broadcast_to(saved_input.shape())?)?;
    let mask = client.eq(saved_input, &extremum_broadcast)?;
    let mask_sum = client.sum(&mask, dims, true)?;
    let mask_sum_broadcast = ensure_contiguous(mask_sum.broadcast_to(saved_input.shape())?)?;
    let normalized_mask = client.div(&mask, &mask_sum_broadcast)?;

    let mut grad_tensor = grad_output.tensor().clone();
    if !keepdim {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort();
        for &dim in &sorted_dims {
            grad_tensor = grad_tensor.unsqueeze(dim as isize)?;
        }
    }
    let grad_broadcast = ensure_contiguous(grad_tensor.broadcast_to(saved_input.shape())?)?;

    let grad_var = Var::new(grad_broadcast, grad_output.requires_grad());
    let mask_var = Var::new(normalized_mask, false);

    var_mul(&grad_var, &mask_var, &client)
}

impl<R: Runtime> GradFn<R> for MaxBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let grad_input = extremum_backward(
            &self.saved_input,
            grad_output,
            &self.dims,
            self.keepdim,
            true,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
    {
        let grad_input = extremum_backward_var(
            &self.saved_input,
            grad_output,
            &self.dims,
            self.keepdim,
            true,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "MaxBackward"
    }
}

// ============================================================================
// MinBackward
// ============================================================================

/// Backward for min reduction: z = min(a, dims)
///
/// The gradient flows only to the element(s) that had the minimum value.
/// For ties, the gradient is distributed equally among tied elements.
pub struct MinBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MinBackward<R> {
    /// Create a new MinBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for MinBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let grad_input = extremum_backward(
            &self.saved_input,
            grad_output,
            &self.dims,
            self.keepdim,
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R> + ReduceOps<R>,
    {
        let grad_input = extremum_backward_var(
            &self.saved_input,
            grad_output,
            &self.dims,
            self.keepdim,
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "MinBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_max_backward() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 2.0, 4.0, 2.0, 5.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = MaxBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_min_backward() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        let a =
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 1.0, 2.0, 4.0, 2.0, 5.0], &[2, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 1], DType::F32, &device);

        let backward = MinBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[2, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert_eq!(grad_data, vec![0.0, 1.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_max_backward_with_ties() {
        let device = CpuDevice::new();
        let _client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0, 1.0], &[1, 3], &device);
        let grad_out = Tensor::<CpuRuntime>::ones(&[1, 1], DType::F32, &device);

        let backward = MaxBackward::<CpuRuntime>::new(a.id(), a.clone(), &[1], true, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a = grads[0].as_ref().unwrap();
        assert_eq!(grad_a.shape(), &[1, 3]);

        let grad_data: Vec<f32> = grad_a.to_vec();
        assert!((grad_data[0] - 0.5).abs() < 1e-6);
        assert!((grad_data[1] - 0.5).abs() < 1e-6);
        assert!((grad_data[2] - 0.0).abs() < 1e-6);
    }
}
