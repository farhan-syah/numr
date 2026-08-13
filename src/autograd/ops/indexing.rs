//! Backward implementations for indexing operations (gather, embedding lookup)

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{IndexingOps, ScatterReduceOp};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// GatherBackward
// ============================================================================

/// Backward for gather: z = gather(a, dim, index)
///
/// dL/da = zeros_like(a); scatter(zeros, dim, index, grad_output)
pub struct GatherBackward<R: Runtime> {
    input_id: TensorId,
    input_shape: Vec<usize>,
    dim: usize,
    index: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> GatherBackward<R> {
    /// Constructs a new backward function for gather.
    pub fn new(
        input_id: TensorId,
        input_shape: &[usize],
        dim: usize,
        index: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            input_shape: input_shape.to_vec(),
            dim,
            index,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for GatherBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let zeros =
            Tensor::<R>::zeros(&self.input_shape, grad_output.dtype(), grad_output.device());
        let grad_input = client.scatter(&zeros, self.dim, &self.index, grad_output)?;
        Ok(vec![Some(grad_input)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let zeros = Tensor::<R>::zeros(
            &self.input_shape,
            grad_output.tensor().dtype(),
            grad_output.tensor().device(),
        );
        let grad_input = client.scatter(&zeros, self.dim, &self.index, grad_output.tensor())?;
        Ok(vec![Some(Var::new(grad_input, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "GatherBackward"
    }
}

// ============================================================================
// EmbeddingLookupBackward
// ============================================================================

/// Backward for embedding lookup: z = embedding_lookup(weight, indices)
///
/// dL/dweight = scatter_add(zeros_like(weight), dim=0, indices, grad_output)
pub struct EmbeddingLookupBackward<R: Runtime> {
    weight_id: TensorId,
    weight_shape: Vec<usize>,
    indices: Tensor<R>,
    weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> EmbeddingLookupBackward<R> {
    /// Constructs a new backward function for embedding lookup.
    pub fn new(
        weight_id: TensorId,
        weight_shape: &[usize],
        indices: Tensor<R>,
        weight_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            weight_id,
            weight_shape: weight_shape.to_vec(),
            indices,
            weight_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> EmbeddingLookupBackward<R> {
    fn weight_grad(&self, grad_output: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: IndexingOps<R>,
    {
        if self.weight_shape.len() != 2 {
            return Err(Error::ShapeMismatch {
                expected: vec![0, 0], // Indicates 2D expected
                got: self.weight_shape.clone(),
            });
        }

        let num_indices = self.indices.numel();
        let embedding_dim = self.weight_shape[1];
        let expected_numel = num_indices * embedding_dim;
        if grad_output.numel() != expected_numel {
            return Err(Error::ShapeMismatch {
                expected: vec![num_indices, embedding_dim],
                got: grad_output.shape().to_vec(),
            });
        }

        let client = R::default_client(grad_output.device());
        let zeros = Tensor::<R>::zeros(
            &self.weight_shape,
            grad_output.dtype(),
            grad_output.device(),
        );
        let grad_rows = grad_output
            .contiguous()?
            .reshape(&[num_indices, embedding_dim])?;
        let row_indices = self
            .indices
            .contiguous()?
            .reshape(&[num_indices, 1])?
            .broadcast_to(&[num_indices, embedding_dim])?
            .contiguous()?;

        client.scatter_reduce(
            &zeros,
            0,
            &row_indices,
            &grad_rows,
            ScatterReduceOp::Sum,
            true,
        )
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for EmbeddingLookupBackward<R>
where
    R::Client: IndexingOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        Ok(vec![Some(self.weight_grad(grad_output)?)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let grad_weight = self.weight_grad(grad_output.tensor())?;
        Ok(vec![Some(Var::new(grad_weight, true))])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.weight_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.weight_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "EmbeddingLookupBackward"
    }
}
