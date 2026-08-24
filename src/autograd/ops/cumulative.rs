//! Backward implementations for cumulative operations (cumsum, cumprod)

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::error::Result;
use crate::ops::{BinaryOps, CumulativeOps};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// CumsumBackward
// ============================================================================

/// Backward for cumulative sum: z = cumsum(a, dim)
///
/// The gradient of cumsum is a reverse cumsum:
/// dL/da_i = sum_{j>=i} dL/dz_j
pub struct CumsumBackward<R: Runtime> {
    input_id: TensorId,
    dim: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> CumsumBackward<R> {
    /// Constructs a new backward function for cumulative sum.
    pub fn new(input_id: TensorId, dim: usize, input_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            dim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for CumsumBackward<R>
where
    R::Client: CumulativeOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let flipped = grad_output.flip(self.dim as isize)?;
        let cumsum_flipped = client.cumsum(&flipped, self.dim as isize)?;
        let grad_input = cumsum_flipped.flip(self.dim as isize)?;
        Ok(vec![Some(grad_input.contiguous()?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let flipped = grad_output.tensor().flip(self.dim as isize)?;
        let cumsum_flipped = client.cumsum(&flipped, self.dim as isize)?;
        let grad_input = cumsum_flipped.flip(self.dim as isize)?;
        Ok(vec![Some(Var::new(grad_input.contiguous()?, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "CumsumBackward"
    }
}

// ============================================================================
// CumprodBackward
// ============================================================================

/// Backward for cumulative product: z = cumprod(a, dim)
///
/// dL/da_i = sum_{j>=i} dL/dz_j * z_j / a_i
/// = reverse_cumsum(grad_output * output) / input
pub struct CumprodBackward<R: Runtime> {
    input_id: TensorId,
    input: Tensor<R>,
    output: Tensor<R>,
    dim: usize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> CumprodBackward<R> {
    /// Constructs a new backward function for cumulative product.
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        output: Tensor<R>,
        dim: usize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input,
            output,
            dim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for CumprodBackward<R>
where
    R::Client: CumulativeOps<R> + BinaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let grad_times_output = client.mul(grad_output, &self.output)?;

        let flipped = grad_times_output.flip(self.dim as isize)?;
        let cumsum_flipped = client.cumsum(&flipped, self.dim as isize)?;
        let reverse_cumsum = cumsum_flipped.flip(self.dim as isize)?;

        let grad_input = client.div(&reverse_cumsum, &self.input)?;
        Ok(vec![Some(grad_input.contiguous()?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let grad_times_output = client.mul(grad_output.tensor(), &self.output)?;

        let flipped = grad_times_output.flip(self.dim as isize)?;
        let cumsum_flipped = client.cumsum(&flipped, self.dim as isize)?;
        let reverse_cumsum = cumsum_flipped.flip(self.dim as isize)?;

        let grad_input = client.div(&reverse_cumsum, &self.input)?;
        Ok(vec![Some(Var::new(grad_input.contiguous()?, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "CumprodBackward"
    }
}
