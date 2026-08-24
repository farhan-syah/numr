//! Backward implementations for variance and standard deviation reductions

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::var_mul;
use crate::error::Result;
use crate::ops::{BinaryOps, ReduceOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

use super::common::ensure_contiguous;

// ============================================================================
// VarBackward
// ============================================================================

/// Backward for variance reduction: z = var(a, dims, correction)
///
/// The gradient of variance is:
/// dL/da = dL/dz * 2 * (a - mean(a)) / (N - correction)
///
/// where N is the number of elements being reduced.
pub struct VarBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    correction: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> VarBackward<R> {
    /// Create a new VarBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            dims: dims.to_vec(),
            keepdim,
            correction,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for VarBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        let mean = client.mean(&self.saved_input, &self.dims, true)?;
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?)?;

        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        let scale = 2.0 / n_minus_corr;
        let grad_contrib = client.mul_scalar(&centered, scale)?;

        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?)?;

        let grad_input = client.mul(&grad_broadcast, &grad_contrib)?;

        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        let mean = client.mean(&self.saved_input, &self.dims, true)?;
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?)?;

        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        let scale = 2.0 / n_minus_corr;
        let grad_contrib = client.mul_scalar(&centered, scale)?;

        let mut grad_tensor = grad_output.tensor().clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad_tensor = grad_tensor.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast =
            ensure_contiguous(grad_tensor.broadcast_to(self.saved_input.shape())?)?;

        let grad_var = Var::new(grad_broadcast, grad_output.requires_grad());
        let contrib_var = Var::new(grad_contrib, false);

        let grad_input = var_mul(&grad_var, &contrib_var, &client)?;

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
        "VarBackward"
    }
}

// ============================================================================
// StdBackward
// ============================================================================

/// Backward for standard deviation reduction: z = std(a, dims, correction)
///
/// std = sqrt(var), so by chain rule:
/// dL/da = dL/dz * d(sqrt(var))/dvar * dvar/da
///       = dL/dz * 1/(2*std) * 2*(a - mean) / (N - correction)
///       = dL/dz * (a - mean) / ((N - correction) * std)
pub struct StdBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    saved_output: Tensor<R>,
    dims: Vec<usize>,
    keepdim: bool,
    correction: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> StdBackward<R> {
    /// Create a new StdBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        output: Tensor<R>,
        dims: &[usize],
        keepdim: bool,
        correction: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            saved_output: output,
            dims: dims.to_vec(),
            keepdim,
            correction,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for StdBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        let mean = client.mean(&self.saved_input, &self.dims, true)?;
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?)?;

        let std_for_broadcast = if self.keepdim {
            self.saved_output.clone()
        } else {
            let mut std_expanded = self.saved_output.clone();
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                std_expanded = std_expanded.unsqueeze(dim as isize)?;
            }
            std_expanded
        };
        let std_broadcast =
            ensure_contiguous(std_for_broadcast.broadcast_to(self.saved_input.shape())?)?;

        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        let denominator = client.mul_scalar(&std_broadcast, n_minus_corr)?;
        let grad_contrib = client.div(&centered, &denominator)?;

        let mut grad = grad_output.clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad = grad.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast = ensure_contiguous(grad.broadcast_to(self.saved_input.shape())?)?;

        let grad_input = client.mul(&grad_broadcast, &grad_contrib)?;

        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + ReduceOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let n: usize = self
            .dims
            .iter()
            .map(|&d| self.saved_input.shape()[d])
            .product();
        let n_minus_corr = (n - self.correction) as f64;

        let mean = client.mean(&self.saved_input, &self.dims, true)?;
        let mean_broadcast = ensure_contiguous(mean.broadcast_to(self.saved_input.shape())?)?;

        let std_for_broadcast = if self.keepdim {
            self.saved_output.clone()
        } else {
            let mut std_expanded = self.saved_output.clone();
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                std_expanded = std_expanded.unsqueeze(dim as isize)?;
            }
            std_expanded
        };
        let std_broadcast =
            ensure_contiguous(std_for_broadcast.broadcast_to(self.saved_input.shape())?)?;

        let centered = client.sub(&self.saved_input, &mean_broadcast)?;

        let denominator = client.mul_scalar(&std_broadcast, n_minus_corr)?;
        let grad_contrib = client.div(&centered, &denominator)?;

        let mut grad_tensor = grad_output.tensor().clone();
        if !self.keepdim {
            let mut sorted_dims = self.dims.clone();
            sorted_dims.sort();
            for &dim in &sorted_dims {
                grad_tensor = grad_tensor.unsqueeze(dim as isize)?;
            }
        }
        let grad_broadcast =
            ensure_contiguous(grad_tensor.broadcast_to(self.saved_input.shape())?)?;

        let grad_var = Var::new(grad_broadcast, grad_output.requires_grad());
        let contrib_var = Var::new(grad_contrib, false);

        let grad_input = var_mul(&grad_var, &contrib_var, &client)?;

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
        "StdBackward"
    }
}
