//! Variable: tensor with gradient tracking

use super::GradFn;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

/// A tensor that tracks gradients for automatic differentiation
///
/// `Var` wraps a `Tensor` and optionally records how it was created
/// (via `grad_fn`), enabling reverse-mode autodiff.
///
/// # Lazy Autograd
///
/// During inference, `grad_fn` is `None` and there's zero overhead.
/// Gradient tracking only occurs during training when needed.
pub struct Var<R: Runtime> {
    /// The underlying tensor data
    tensor: Tensor<R>,

    /// Unique identifier for graph tracking
    id: TensorId,

    /// Whether this variable requires gradient computation
    requires_grad: bool,

    /// Function to compute gradients (None for leaf tensors)
    grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> Var<R> {
    /// Create a leaf variable (no gradient function)
    pub fn new(tensor: Tensor<R>, requires_grad: bool) -> Self {
        Self {
            id: tensor.id(),
            tensor,
            requires_grad,
            grad_fn: None,
        }
    }

    /// Create a leaf variable with a specific ID
    ///
    /// This is used for second-order differentiation where we need to
    /// preserve the original input IDs so gradients can be accumulated
    /// correctly across multiple backward passes.
    pub fn with_id(tensor: Tensor<R>, id: TensorId, requires_grad: bool) -> Self {
        Self {
            id,
            tensor,
            requires_grad,
            grad_fn: None,
        }
    }

    /// Create a variable with a specific ID and gradient function
    ///
    /// This is essential for second-order differentiation. When computing
    /// gradients of gradients, we need to preserve both:
    /// 1. The original input ID (so second-order gradients accumulate correctly)
    /// 2. The grad_fn chain (so the backward traversal can continue through
    ///    the original computation graph)
    ///
    /// Without the grad_fn, second-order backward would stop at saved tensors
    /// instead of continuing to the original inputs.
    pub fn with_id_and_grad_fn(
        tensor: Tensor<R>,
        id: TensorId,
        grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            id,
            tensor,
            requires_grad: true,
            grad_fn,
        }
    }

    /// Create from an operation result with a gradient function
    pub fn from_op(tensor: Tensor<R>, grad_fn: Arc<dyn GradFn<R>>) -> Self {
        Self {
            id: TensorId::new(),
            tensor,
            requires_grad: true,
            grad_fn: Some(grad_fn),
        }
    }

    /// Get the tensor ID
    #[inline]
    pub fn id(&self) -> TensorId {
        self.id
    }

    /// Access the underlying tensor
    #[inline]
    pub fn tensor(&self) -> &Tensor<R> {
        &self.tensor
    }

    /// Check if this variable requires gradients
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get the gradient function (if any)
    #[inline]
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn<R>>> {
        self.grad_fn.as_ref()
    }

    /// Identity alias: same tensor, same [`TensorId`], same `requires_grad`,
    /// same `grad_fn`.
    ///
    /// Use this — NOT [`Clone::clone`] — whenever a layer returns its input
    /// unchanged (e.g. `Dropout` in eval mode, a disabled residual scale).
    /// `clone` mints a FRESH `TensorId`, which silently breaks two things:
    ///
    /// 1. If the input is a leaf parameter, `backward` records its gradient
    ///    under the clone's new id, so `GradStore::get(param.id())` returns
    ///    `None` and the parameter never updates.
    /// 2. Optimizer state (AdamW) is keyed by `TensorId`, so a drifting id
    ///    orphans the moment estimates.
    pub fn alias(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: self.id,
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        }
    }

    /// Detach from the computation graph
    ///
    /// Returns a new variable that doesn't track gradients.
    pub fn detach(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: TensorId::new(),
            requires_grad: false,
            grad_fn: None,
        }
    }

    /// Set requires_grad flag
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self.grad_fn = None;
        }
    }

    // Delegate common methods to tensor

    /// Get the shape
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Get the number of elements
    #[inline]
    pub fn numel(&self) -> usize {
        self.tensor.numel()
    }

    /// Get the number of dimensions
    #[inline]
    pub fn ndim(&self) -> usize {
        self.tensor.ndim()
    }
}

impl<R: Runtime> Clone for Var<R> {
    fn clone(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            id: TensorId::new(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        }
    }
}

impl<R: Runtime> std::fmt::Debug for Var<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Var")
            .field("id", &self.id)
            .field("shape", &self.tensor.shape())
            .field("requires_grad", &self.requires_grad)
            .field("has_grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    /// `alias` preserves the id (so gradients and TensorId-keyed optimizer
    /// state still find the variable); `clone` deliberately does not.
    #[test]
    fn alias_preserves_id_but_clone_does_not() {
        let device = <CpuRuntime as Runtime>::default_device();
        let t = Tensor::<CpuRuntime>::try_zeros(&[2, 3], DType::F32, &device).unwrap();
        let v = Var::new(t, true);

        let aliased = v.alias();
        assert_eq!(aliased.id(), v.id(), "alias must keep the same TensorId");
        assert_eq!(aliased.requires_grad(), v.requires_grad());
        assert_eq!(aliased.shape(), v.shape());

        let cloned = v.clone();
        assert_ne!(
            cloned.id(),
            v.id(),
            "clone is documented to mint a fresh TensorId — if this ever \
             changes, the `alias` call sites can be simplified"
        );
    }

    /// A no-op layer path must not strip `requires_grad == false`.
    #[test]
    fn alias_preserves_requires_grad_false() {
        let device = <CpuRuntime as Runtime>::default_device();
        let t = Tensor::<CpuRuntime>::try_zeros(&[4], DType::F32, &device).unwrap();
        let v = Var::new(t, false);
        let aliased = v.alias();
        assert!(!aliased.requires_grad());
        assert_eq!(aliased.id(), v.id());
    }
}
