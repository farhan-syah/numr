//! Gradient function trait for automatic differentiation
//!
//! This module defines the core [`GradFn`] trait that all backward operations
//! must implement to participate in the autograd system.

use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

use super::Var;

/// Trait for computing gradients during backward pass
///
/// Each operation that participates in autograd has an associated
/// `GradFn` that knows how to compute gradients for its inputs.
///
/// # Implementation Guide
///
/// When implementing this trait, you **must** implement both `backward()` and
/// `backward_var()` if you want proper second-order differentiation support.
///
/// The `backward()` method is used for first-order gradients (standard backprop).
/// The `backward_var()` method is used for second-order gradients (Hessians, HVPs).
///
/// # Example
///
/// ```ignore
/// # use numr::prelude::*;
/// # use numr::autograd::{GradFn, Var};
/// # use numr::error::Result;
/// # use std::sync::Arc;
/// struct MyOpBackward<R: Runtime> {
///     saved_tensor: Tensor<R>,
/// }
///
/// impl<R: Runtime> GradFn<R> for MyOpBackward<R> {
///     fn backward(
///         &self,
///         grad_output: &Tensor<R>,
///         needed: &[bool],
///     ) -> Result<Vec<Option<Tensor<R>>>> {
///         // Skip any slot the caller will discard
///         if !needed[0] {
///             return Ok(vec![None]);
///         }
///         // Compute gradients using tensor ops
///         // let grad = client.mul(grad_output, &self.saved_tensor)?;
///         Ok(vec![Some(grad_output.clone())])
///     }
///
///     fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
///         // Compute gradients using var_ops to maintain computation graph
///         Ok(vec![Some(grad_output.clone())])
///     }
/// }
/// ```
pub trait GradFn<R: Runtime>: Send + Sync {
    /// Compute gradients for input tensors given the gradient of the output
    ///
    /// Returns a vector of optional gradients - one per input.
    /// `None` indicates no gradient is produced for that input.
    ///
    /// # The `needed` mask
    ///
    /// `needed[i]` corresponds to `self.inputs()[i]` and tells the op whether
    /// the driver will keep that slot's gradient. The driver guarantees
    /// `needed.len() == self.inputs().len()`, so indexing it by input position
    /// is always in bounds.
    ///
    /// The contract:
    ///
    /// - An op **must** return the correct gradient for every slot where
    ///   `needed[i]` is true.
    /// - An op **must** return `None` for any slot where `needed[i]` is false,
    ///   **or may** return `Some`. Returning `Some` for an unneeded slot is pure
    ///   waste: the driver discards it.
    ///
    /// Guard a slot when its gradient costs an independent kernel launch — the
    /// two matmuls of `MatmulBackward`, a conv weight gradient, a per-operand
    /// broadcast reduction. Where every slot falls out of one shared
    /// intermediate, guarding buys nothing and the mask is ignored.
    ///
    /// Under [`crate::autograd::backward`] every entry is true, so a masked op
    /// takes exactly the branches it took before the mask existed.
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>>;

    /// Compute every input gradient, as if the caller wanted all of them.
    ///
    /// Convenience wrapper over [`GradFn::backward`] with an all-true mask.
    /// Use it when there is no pruning context — tests, and re-entrant callers
    /// that pass gradients straight through.
    fn backward_all(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let needed = vec![true; self.inputs().len()];
        self.backward(grad_output, &needed)
    }

    /// Compute gradients as Vars for second-order differentiation
    ///
    /// This method enables higher-order derivatives by returning `Var`s
    /// that retain their computation history.
    ///
    /// # Important: Override for Second-Order Derivatives
    ///
    /// **The default implementation creates detached Vars with no gradient history.**
    /// This means second-order derivatives will NOT flow through operations that
    /// rely on the default implementation.
    ///
    /// If your operation needs to support second-order differentiation (Hessians,
    /// Hessian-vector products), you **must** override this method to:
    ///
    /// 1. Use `var_ops` (var_mul, var_add, etc.) instead of raw tensor operations
    /// 2. Use `Var::with_id_and_grad_fn()` for saved tensors to preserve the chain
    /// 3. Return Vars that maintain the computation graph
    ///
    /// # Default Behavior
    ///
    /// The default implementation calls `backward_all()` and wraps results in Vars
    /// with `requires_grad=true` but `grad_fn=None`. This is suitable for:
    ///
    /// - Operations that don't need second-order derivatives
    /// - Leaf operations where the gradient chain naturally terminates
    /// - Initial development before adding full second-order support
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient of the loss with respect to this op's output
    ///
    /// # Returns
    ///
    /// A vector of optional `Var`s - one per input. Each `Var` can be
    /// differentiated again to compute second-order derivatives.
    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Default: compute tensor gradients and wrap in detached Vars.
        // WARNING: This breaks second-order derivatives. Override this method
        // if you need Hessians or HVPs to work correctly through this operation.
        // Second-order traversal keeps every node, so ask for every gradient.
        let tensor_grads = self.backward_all(grad_output.tensor())?;
        Ok(tensor_grads
            .into_iter()
            .map(|opt| opt.map(|t| Var::new(t, true)))
            .collect())
    }

    /// Get the IDs of input tensors
    ///
    /// Used for topological sorting during backward pass.
    fn inputs(&self) -> &[TensorId];

    /// Get the grad_fns of input tensors for graph traversal
    ///
    /// Returns a vector of optional grad_fns - one per input.
    /// `None` indicates a leaf tensor.
    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![None; self.inputs().len()]
    }

    /// Get tensors saved during forward pass
    ///
    /// Some operations (like softmax) need forward outputs for backward.
    fn saved_tensors(&self) -> &[Tensor<R>] {
        &[]
    }

    /// Human-readable name for debugging
    fn name(&self) -> &'static str;
}
