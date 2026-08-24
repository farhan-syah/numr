//! Gradient storage for second-order differentiation
//!
//! Stores gradients as `Var`s instead of raw `Tensor`s, enabling
//! the gradient computation graph to be differentiated again.

use super::Var;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::collections::HashMap;

/// Storage for gradients as differentiable variables
///
/// Unlike [`GradStore`](super::GradStore) which stores raw tensors,
/// `VarGradStore` stores `Var`s that retain their computation history.
/// This enables second-order differentiation (Hessians, HVPs).
///
/// # Example
///
/// ```no_run
/// # use numr::prelude::*;
/// # use numr::autograd::{backward_with_graph, backward, Var, var_mul, var_sum};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// # let loss = Var::new(Tensor::try_from_slice(&[1.0f32], &[], &device)?, true);
/// // First backward pass - gradients are Vars with history
/// let var_grads = backward_with_graph(&loss, &client)?;
/// let grad_x = var_grads.get_var(loss.id()).unwrap();
///
/// // grad_x is a Var, so we can differentiate it again
/// let v = Var::new(Tensor::try_from_slice(&[1.0f32], &[], &device)?, true);
/// let hvp = var_mul(grad_x, &v, &client)?;
/// let second_grads = backward(&var_sum(&hvp, &[], false, &client)?, &client)?;
/// # Ok::<(), numr::error::Error>(())
/// ```
pub struct VarGradStore<R: Runtime> {
    grads: HashMap<TensorId, Var<R>>,
}

impl<R: Runtime> VarGradStore<R> {
    /// Create a new empty gradient store
    pub fn new() -> Self {
        Self {
            grads: HashMap::new(),
        }
    }

    /// Get the gradient Var for a tensor
    pub fn get_var(&self, id: TensorId) -> Option<&Var<R>> {
        self.grads.get(&id)
    }

    /// Get the gradient tensor for a tensor (convenience method)
    pub fn get(&self, id: TensorId) -> Option<&Tensor<R>> {
        self.grads.get(&id).map(|v| v.tensor())
    }

    /// Insert a gradient Var (overwrites if exists)
    pub fn insert(&mut self, id: TensorId, grad: Var<R>) {
        self.grads.insert(id, grad);
    }

    /// Check if a gradient exists
    pub fn contains(&self, id: TensorId) -> bool {
        self.grads.contains_key(&id)
    }

    /// Remove and return a gradient Var
    pub fn remove(&mut self, id: TensorId) -> Option<Var<R>> {
        self.grads.remove(&id)
    }

    /// Get all tensor IDs with gradients
    pub fn keys(&self) -> impl Iterator<Item = &TensorId> {
        self.grads.keys()
    }

    /// Number of stored gradients
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Clear all gradients
    pub fn clear(&mut self) {
        self.grads.clear();
    }

    /// Accumulate a gradient Var using a fallible addition function
    ///
    /// If no gradient exists for this tensor, stores the gradient.
    /// If a gradient already exists, adds them using the provided function.
    ///
    /// The addition function should use var_ops to maintain the computation graph.
    pub fn try_accumulate<F>(&mut self, id: TensorId, grad: Var<R>, add_fn: F) -> Result<()>
    where
        F: FnOnce(Var<R>, Var<R>) -> Result<Var<R>>,
    {
        use std::collections::hash_map::Entry;

        match self.grads.entry(id) {
            Entry::Occupied(entry) => {
                // Remove existing, accumulate with new, insert result
                let existing = entry.remove();
                let accumulated = add_fn(existing, grad)?;
                self.grads.insert(id, accumulated);
            }
            Entry::Vacant(entry) => {
                // No existing gradient, just insert
                entry.insert(grad);
            }
        }
        Ok(())
    }

    /// Convert to a regular GradStore by extracting tensors
    ///
    /// This detaches all gradients from the computation graph.
    pub fn to_grad_store(self) -> super::GradStore<R> {
        let mut store = super::GradStore::new();
        for (id, var) in self.grads {
            store.insert(id, var.tensor().clone());
        }
        store
    }

    /// Iterate over all (id, var) pairs
    pub fn iter(&self) -> impl Iterator<Item = (&TensorId, &Var<R>)> {
        self.grads.iter()
    }
}

impl<R: Runtime> Default for VarGradStore<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R: Runtime> IntoIterator for VarGradStore<R> {
    type Item = (TensorId, Var<R>);
    type IntoIter = std::collections::hash_map::IntoIter<TensorId, Var<R>>;

    fn into_iter(self) -> Self::IntoIter {
        self.grads.into_iter()
    }
}
