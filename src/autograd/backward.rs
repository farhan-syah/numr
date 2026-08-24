//! Backward pass implementation
//!
//! Implements reverse-mode automatic differentiation using topological sort
//! to traverse the computation graph and accumulate gradients.
//!
//! # First-Order vs Second-Order Differentiation
//!
//! This module provides two backward functions:
//!
//! - [`backward`]: Standard first-order differentiation. Returns raw tensors.
//!   Efficient for training neural networks.
//!
//! - [`backward_with_graph`]: Second-order capable differentiation. Returns
//!   `Var`s that retain their computation history, enabling Hessians and HVPs.

use super::{GradFn, GradStore, Var, VarGradStore, var_add};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::collections::HashSet;
use std::sync::Arc;

// ============================================================================
// Backward Hooks
// ============================================================================

/// Hook called during backward when a leaf variable's gradient is fully accumulated.
///
/// This enables overlapping gradient communication with backward computation
/// in distributed training scenarios (e.g., bucketed allreduce).
pub trait BackwardHook<R: Runtime>: Send {
    /// Called when a leaf variable's gradient is fully accumulated.
    ///
    /// At the point this is called, the gradient for `id` in the grad store
    /// is complete — all upstream contributions have been accumulated.
    fn on_leaf_grad_ready(&mut self, id: TensorId, grad: &Tensor<R>);
}

/// No-op backward hook for use when no hook behavior is needed.
pub struct NoOpHook;

impl<R: Runtime> BackwardHook<R> for NoOpHook {
    fn on_leaf_grad_ready(&mut self, _id: TensorId, _grad: &Tensor<R>) {}
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Validate that the loss tensor is suitable for backward pass
///
/// Checks:
/// 1. Loss is a scalar (numel == 1)
/// 2. Loss requires gradients
#[inline]
fn validate_loss<R: Runtime>(loss: &Var<R>, fn_name: &str) -> Result<()> {
    if loss.numel() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![1],
            got: loss.shape().to_vec(),
        });
    }

    if !loss.requires_grad() {
        return Err(Error::Internal(format!(
            "{}() called on tensor that doesn't require grad",
            fn_name
        )));
    }

    Ok(())
}

/// Create the initial gradient tensor for the loss (dL/dL = 1)
#[inline]
fn create_loss_gradient<R: Runtime<DType = DType>>(loss: &Var<R>) -> Result<Tensor<R>> {
    Tensor::<R>::try_ones(loss.shape(), loss.tensor().dtype(), loss.tensor().device())
}

/// Compute gradients via reverse-mode automatic differentiation
///
/// Starting from a scalar loss, traverses the computation graph in
/// reverse topological order, computing gradients for all tensors
/// that require them.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `GradStore` containing gradients for all tensors in the graph.
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::{Var, backward, var_mul};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // Create variables
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
/// let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
///
/// // Forward: z = x * y
/// let z = var_mul(&x, &y, &client)?;
///
/// // Backward
/// let grads = backward(&z, &client)?;
///
/// // dx = y = 3.0
/// let grad_x = grads.get(x.id()).unwrap();
/// # Ok::<(), numr::error::Error>(())
/// ```
pub fn backward<R, C>(loss: &Var<R>, client: &C) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    backward_with_hooks(loss, client, &mut NoOpHook)
}

/// Compute gradients only for the tensors in `wanted`, and for the nodes on a
/// path to them.
///
/// Identical to [`backward`] on every id in `wanted`, bit for bit: the reverse
/// topological order and therefore the float accumulation order of each retained
/// node is unchanged. The difference is what is NOT done — a node whose input
/// cone contains no wanted id has its `GradFn::backward()` skipped entirely, so
/// neither the intermediate tensor nor the work that produces it is ever
/// materialized.
///
/// Pruning also reaches inside a node that IS needed. Each `GradFn::backward()`
/// call receives a per-input `needed` mask, so an op with independently priced
/// gradients skips the ones nothing reads. A frozen `Linear` is the case that
/// matters: its matmul node is needed for the activation gradient, yet the
/// weight gradient `A^T @ dL/dC` is dead, and `MatmulBackward` no longer runs
/// that second matmul at all.
///
/// This matters for partial-finetuning graphs (LoRA, frozen backbones). There,
/// most graph nodes are only reachable from frozen weights, yet a plain
/// [`backward`] still computes and stores a full-size gradient for each of them
/// under an id nothing can read back.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `wanted` - Tensor ids whose gradients are required (typically the trainable
///   parameter ids)
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `GradStore<R>` holding a gradient for every wanted id that receives one,
/// plus the intermediate nodes on a path to them.
pub fn backward_wrt<R, C>(loss: &Var<R>, wanted: &[TensorId], client: &C) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
{
    backward_wrt_with_hooks(loss, wanted, client, &mut NoOpHook)
}

/// Compute gradients only for `wanted`, with leaf-ready hooks.
///
/// Combines [`backward_wrt`] and [`backward_with_hooks`]. The hook fires for a
/// leaf only when that leaf is in `wanted` — a pruned leaf produces no gradient,
/// so there is nothing to report.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `wanted` - Tensor ids whose gradients are required
/// * `client` - The runtime client for tensor operations
/// * `hooks` - Hook implementation called when each wanted leaf gradient is ready
pub fn backward_wrt_with_hooks<R, C, H>(
    loss: &Var<R>,
    wanted: &[TensorId],
    client: &C,
    hooks: &mut H,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    let wanted = Wanted::Set(wanted.iter().copied().collect());
    backward_driver(loss, &wanted, client, hooks, "backward_wrt_with_hooks")
}

/// Compute gradients with hooks that fire when leaf gradients are ready.
///
/// Identical to [`backward`], but calls `hooks.on_leaf_grad_ready(id, grad)`
/// after a leaf variable's gradient is fully accumulated. This enables
/// overlapping gradient communication with backward computation (e.g.,
/// bucketed allreduce in distributed training).
///
/// A leaf variable is one with no `grad_fn` (i.e., a model parameter or
/// input created with `requires_grad = true`). By the time the hook fires,
/// all upstream contributions to that leaf's gradient have been accumulated.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
/// * `hooks` - Hook implementation called when each leaf gradient is ready
///
/// # Returns
///
/// A `GradStore` containing gradients for all tensors in the graph.
pub fn backward_with_hooks<R, C, H>(
    loss: &Var<R>,
    client: &C,
    hooks: &mut H,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    // `backward()` delegates here, so the validation error text has always named
    // `backward_with_hooks`. Keep it that way.
    backward_driver(loss, &Wanted::All, client, hooks, "backward_with_hooks")
}

/// Which gradients the backward driver must produce.
enum Wanted {
    /// Every node in the graph — the historical [`backward`] behavior.
    All,
    /// Only these ids, plus whatever lies on a path to them.
    Set(HashSet<TensorId>),
}

/// Mark every node whose input cone contains a wanted id.
///
/// `topo_order` lists inputs before outputs, so a single forward sweep suffices:
/// `needed[n] = (n in wanted) || any(needed[i] for i in inputs(n))`.
///
/// Returns `None` under [`Wanted::All`], meaning "every node is needed" — no set
/// is built and no lookup is paid.
///
/// This works on `TensorId`s only; it touches no tensor data and is O(nodes + edges).
fn compute_needed<R: Runtime>(
    topo_order: &[TopoEntry<R>],
    wanted: &Wanted,
) -> Option<HashSet<TensorId>> {
    let wanted = match wanted {
        Wanted::All => return None,
        Wanted::Set(set) => set,
    };

    let mut needed: HashSet<TensorId> = HashSet::new();
    for (id, _, input_ids) in topo_order {
        if wanted.contains(id) || input_ids.iter().any(|input| needed.contains(input)) {
            needed.insert(*id);
        }
    }
    Some(needed)
}

/// Single reverse-mode traversal shared by every first-order backward entry point.
///
/// Pruning is a reachability pre-pass from `wanted`, not a `requires_grad` edge
/// check: a node is skipped only when no wanted id lies anywhere below it. That
/// is closed under contribution — if node `n` is needed and `n` is an input of
/// consumer `c`, then `needed[c]` holds by construction — so no accumulation term
/// that reaches a wanted id is ever dropped, and the accumulation order of every
/// retained node is untouched.
fn backward_driver<R, C, H>(
    loss: &Var<R>,
    wanted: &Wanted,
    client: &C,
    hooks: &mut H,
    fn_name: &str,
) -> Result<GradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    H: BackwardHook<R>,
{
    validate_loss(loss, fn_name)?;

    // Initialize gradient store with dL/dL = 1
    let mut grad_store = GradStore::new();
    grad_store.insert(loss.id(), create_loss_gradient(loss)?);

    // Build the computation graph and get topological order
    let topo_order = topological_sort(loss);

    // Reachability pre-pass: which nodes lie on a path to a wanted gradient
    let needed = compute_needed(&topo_order, wanted);
    let is_needed = |id: TensorId| match &needed {
        None => true,
        Some(set) => set.contains(&id),
    };

    // Traverse in reverse topological order (from output to inputs)
    for var_entry in topo_order.into_iter().rev() {
        let (var_id, grad_fn_opt, input_ids) = var_entry;

        // Nothing wanted lies below this node — skip before any tensor work runs
        if !is_needed(var_id) {
            continue;
        }

        // Get gradient for this node
        let grad_output = match grad_store.get(var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flowing to this node
        };

        // If this node has a grad_fn, compute gradients for its inputs
        if let Some(grad_fn) = grad_fn_opt {
            // Per-slot mask, one entry per input, in `inputs()` order. This is
            // the same predicate the accumulation loop below applies, handed to
            // the op up front so it can skip producing a gradient nobody reads.
            // Under `Wanted::All` every entry is true and the op takes exactly
            // the branches it took before the mask existed.
            let input_needed: Vec<bool> = input_ids.iter().map(|id| is_needed(*id)).collect();

            // Compute gradients for inputs
            let input_grads = grad_fn.backward(&grad_output, &input_needed)?;

            // Accumulate gradients for each input
            for (input_id, input_grad_opt) in input_ids.iter().zip(input_grads) {
                if !is_needed(*input_id) {
                    continue;
                }
                if let Some(input_grad) = input_grad_opt {
                    // Accumulate gradient using tensor addition
                    grad_store.try_accumulate(*input_id, input_grad, |existing, new| {
                        client.add(&existing, &new)
                    })?;
                }
            }
        } else {
            // Leaf node (no grad_fn) with a gradient — notify hook
            hooks.on_leaf_grad_ready(var_id, &grad_output);
        }
    }

    Ok(grad_store)
}

/// Compute gradients with graph retention for second-order differentiation
///
/// Like [`backward`], but returns `Var`s instead of raw tensors. The returned
/// gradients retain their computation history, enabling them to be differentiated
/// again for computing Hessians, Hessian-vector products (HVPs), and other
/// second-order derivatives.
///
/// # Arguments
///
/// * `loss` - The scalar loss tensor to differentiate
/// * `client` - The runtime client for tensor operations
///
/// # Returns
///
/// A `VarGradStore` containing gradient `Var`s for all tensors in the graph.
/// Each gradient can be differentiated again using [`backward`].
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # use numr::autograd::{Var, backward, backward_with_graph, var_mul, var_sum};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // Forward pass
/// let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device), true);
/// let y = var_mul(&x, &x, &client)?;  // y = x²
///
/// // First backward - get gradient as Var (not detached)
/// let grads = backward_with_graph(&y, &client)?;
/// let grad_x = grads.get_var(x.id()).unwrap();  // dy/dx = 2x = 4
///
/// // grad_x is a Var with history, so we can differentiate it
/// // Compute HVP: multiply by vector v, then differentiate again
/// let v = Var::new(Tensor::from_slice(&[1.0f32], &[1], &device), true);
/// let grad_v = var_mul(grad_x, &v, &client)?;
/// let hvp = backward(&var_sum(&grad_v, &[], false, &client)?, &client)?;
/// // hvp[x] = d²y/dx² * v = 2 * 1 = 2
/// # Ok::<(), numr::error::Error>(())
/// ```
///
/// # Performance Note
///
/// This function is slower and uses more memory than [`backward`] because it
/// builds a computation graph for the gradient computation itself. Only use it
/// when you actually need second-order derivatives.
pub fn backward_with_graph<R, C>(loss: &Var<R>, client: &C) -> Result<VarGradStore<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    validate_loss(loss, "backward_with_graph")?;

    // Initialize gradient store with dL/dL = 1 as a Var
    // This is a leaf Var (no grad_fn), but requires_grad = true so it can be differentiated
    let mut var_grad_store = VarGradStore::new();
    var_grad_store.insert(loss.id(), Var::new(create_loss_gradient(loss)?, true));

    // Build the computation graph and get topological order
    let topo_order = topological_sort(loss);

    // Traverse in reverse topological order (from output to inputs)
    for var_entry in topo_order.into_iter().rev() {
        let (var_id, grad_fn_opt, input_ids) = var_entry;

        // Get gradient Var for this node (borrow, don't remove)
        let grad_output = match var_grad_store.get_var(var_id) {
            Some(g) => g.clone(),
            None => continue, // No gradient flowing to this node
        };

        // If this node has a grad_fn, compute gradients for its inputs
        if let Some(grad_fn) = grad_fn_opt {
            // Compute gradients for inputs using backward_var (returns Vars)
            let input_grads = grad_fn.backward_var(&grad_output)?;

            // Accumulate gradients for each input using var_add (builds graph)
            for (input_id, input_grad_opt) in input_ids.iter().zip(input_grads) {
                if let Some(input_grad) = input_grad_opt {
                    // Accumulate gradient using var_add to maintain computation graph
                    var_grad_store.try_accumulate(*input_id, input_grad, |existing, new| {
                        var_add(&existing, &new, client)
                    })?;
                }
            }
        }
    }

    Ok(var_grad_store)
}

/// Entry for topological sort: (tensor_id, grad_fn, input_ids)
type TopoEntry<R> = (TensorId, Option<Arc<dyn GradFn<R>>>, Vec<TensorId>);

/// Build topological sort of computation graph using DFS post-order traversal
///
/// Returns nodes in topological order (inputs before outputs).
fn topological_sort<R: Runtime>(loss: &Var<R>) -> Vec<TopoEntry<R>> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();

    fn dfs<R: Runtime>(
        id: TensorId,
        grad_fn: Option<Arc<dyn GradFn<R>>>,
        visited: &mut HashSet<TensorId>,
        result: &mut Vec<TopoEntry<R>>,
    ) {
        if visited.contains(&id) {
            return;
        }
        visited.insert(id);

        let input_ids: Vec<TensorId> = grad_fn
            .as_ref()
            .map(|gf| gf.inputs().to_vec())
            .unwrap_or_default();

        // Get input grad_fns and visit inputs first (dependencies)
        if let Some(gf) = &grad_fn {
            for (input_id, input_grad_fn) in input_ids.iter().zip(gf.input_grad_fns()) {
                dfs(*input_id, input_grad_fn, visited, result);
            }
        }

        // Add this node after its inputs (post-order)
        result.push((id, grad_fn, input_ids));
    }

    dfs(
        loss.id(),
        loss.grad_fn().cloned(),
        &mut visited,
        &mut result,
    );
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    use std::cell::RefCell;
    use std::rc::Rc;

    /// Test hook that records leaf gradient notifications
    struct RecordingHook {
        leaf_ids: Rc<RefCell<Vec<TensorId>>>,
    }

    impl RecordingHook {
        fn new() -> (Self, Rc<RefCell<Vec<TensorId>>>) {
            let ids = Rc::new(RefCell::new(Vec::new()));
            (
                Self {
                    leaf_ids: ids.clone(),
                },
                ids,
            )
        }
    }

    // RecordingHook is not Send (due to Rc), so we wrap for single-threaded tests
    unsafe impl Send for RecordingHook {}

    impl BackwardHook<CpuRuntime> for RecordingHook {
        fn on_leaf_grad_ready(&mut self, id: TensorId, _grad: &Tensor<CpuRuntime>) {
            self.leaf_ids.borrow_mut().push(id);
        }
    }

    #[test]
    fn test_backward_with_hooks_matches_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // z = x * y
        let z1 = var_mul(&x, &y, &client).unwrap();
        let z2 = var_mul(&x, &y, &client).unwrap();

        let grads1 = backward(&z1, &client).unwrap();

        let (mut hook, leaf_ids) = RecordingHook::new();
        let grads2 = backward_with_hooks(&z2, &client, &mut hook).unwrap();

        // Gradients should match
        let gx1: Vec<f32> = grads1.get(x.id()).unwrap().to_vec();
        let gx2: Vec<f32> = grads2.get(x.id()).unwrap().to_vec();
        assert!((gx1[0] - gx2[0]).abs() < 1e-6);

        let gy1: Vec<f32> = grads1.get(y.id()).unwrap().to_vec();
        let gy2: Vec<f32> = grads2.get(y.id()).unwrap().to_vec();
        assert!((gy1[0] - gy2[0]).abs() < 1e-6);

        // Hook should have been called for both leaf variables
        let ids = leaf_ids.borrow();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&x.id()));
        assert!(ids.contains(&y.id()));
    }

    #[test]
    fn test_backward_with_hooks_no_hook_for_non_leaf() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device),
            true,
        );

        // y = sum(x * x) — intermediate x*x is NOT a leaf
        let x_sq = var_mul(&x, &x, &client).unwrap();
        let loss = var_sum(&x_sq, &[0], false, &client).unwrap();

        let (mut hook, leaf_ids) = RecordingHook::new();
        let _grads = backward_with_hooks(&loss, &client, &mut hook).unwrap();

        // Only x is a leaf, not x_sq or loss
        let ids = leaf_ids.borrow();
        assert_eq!(ids.len(), 1);
        assert!(ids.contains(&x.id()));
    }

    // ========================================================================
    // backward_wrt (pruned backward) tests
    // ========================================================================

    use crate::autograd::{var_matmul, var_transpose};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Raw bit pattern of every element — no tolerance, no rounding.
    fn bits(tensor: &Tensor<CpuRuntime>) -> Vec<u32> {
        tensor.to_vec::<f32>().iter().map(|v| v.to_bits()).collect()
    }

    /// GradFn that records how many times it is actually executed.
    struct CountingBackward<R: Runtime> {
        input_ids: Vec<TensorId>,
        input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
        calls: Arc<AtomicUsize>,
    }

    impl<R: Runtime> GradFn<R> for CountingBackward<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            assert_eq!(
                needed.len(),
                self.input_ids.len(),
                "driver must pass one mask entry per input"
            );
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Some(grad_output.clone()); self.input_ids.len()])
        }

        fn inputs(&self) -> &[TensorId] {
            &self.input_ids
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.input_grad_fns.clone()
        }

        fn name(&self) -> &'static str {
            "CountingBackward"
        }
    }

    /// GradFn that panics if the driver asks it to compute a slot the test
    /// knows nothing wants.
    ///
    /// This is what turns "the mask reached the op" into an observable failure:
    /// if the driver ever handed a true entry for `forbidden`, the op would be
    /// obliged to produce that gradient, and the panic fires instead.
    struct PanicIfNeeded<R: Runtime> {
        input_ids: Vec<TensorId>,
        input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
        forbidden: usize,
    }

    impl<R: Runtime> GradFn<R> for PanicIfNeeded<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            assert_eq!(
                needed.len(),
                self.input_ids.len(),
                "driver must pass one mask entry per input"
            );
            assert!(
                !needed[self.forbidden],
                "driver asked for slot {} that no wanted id depends on",
                self.forbidden
            );
            Ok(needed
                .iter()
                .map(|&want| want.then(|| grad_output.clone()))
                .collect())
        }

        fn inputs(&self) -> &[TensorId] {
            &self.input_ids
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.input_grad_fns.clone()
        }

        fn name(&self) -> &'static str {
            "PanicIfNeeded"
        }
    }

    /// GradFn that delegates to `inner` and records every mask it is handed.
    struct MaskSpy<R: Runtime> {
        inner: Arc<dyn GradFn<R>>,
        seen: Arc<std::sync::Mutex<Vec<Vec<bool>>>>,
    }

    impl<R: Runtime> GradFn<R> for MaskSpy<R> {
        fn backward(
            &self,
            grad_output: &Tensor<R>,
            needed: &[bool],
        ) -> Result<Vec<Option<Tensor<R>>>> {
            self.seen
                .lock()
                .expect("mask spy lock")
                .push(needed.to_vec());
            self.inner.backward(grad_output, needed)
        }

        fn inputs(&self) -> &[TensorId] {
            self.inner.inputs()
        }

        fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
            self.inner.input_grad_fns()
        }

        fn name(&self) -> &'static str {
            "MaskSpy"
        }
    }

    /// The driver hands every node a mask exactly as long as its input list.
    #[test]
    fn test_driver_mask_length_matches_inputs() {
        // CountingBackward and PanicIfNeeded both assert the length on entry;
        // this drives a graph mixing unary, binary and reduce nodes through the
        // real driver so those assertions actually run.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25], &[2], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 4.0], &[2], &device),
            true,
        );

        let calls = Arc::new(AtomicUsize::new(0));
        let counted = Var::from_op(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device),
            Arc::new(CountingBackward::<CpuRuntime> {
                input_ids: vec![x.id(), y.id()],
                input_grad_fns: vec![None, None],
                calls: calls.clone(),
            }),
        );

        let h = var_mul(&counted, &y, &client).unwrap();
        let loss = var_sum(&h, &[0], false, &client).unwrap();

        let _ = backward(&loss, &client).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        let _ = backward_wrt(&loss, &[x.id()], &client).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    /// The driver delivers `false` for a slot nothing wants.
    ///
    /// The node itself IS needed — `x` lies below it — so it is not skipped
    /// wholesale. Only its second slot is dead, which is exactly the
    /// frozen-weight shape this mask exists for.
    #[test]
    fn test_driver_masks_dead_slot_of_a_needed_node() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            true,
        );
        let frozen = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[7.0f32], &[1], &device),
            false,
        );

        let node = Var::from_op(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            Arc::new(PanicIfNeeded::<CpuRuntime> {
                input_ids: vec![x.id(), frozen.id()],
                input_grad_fns: vec![None, None],
                forbidden: 1,
            }),
        );
        let loss = var_sum(&node, &[0], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[x.id()], &client).unwrap();

        assert!(grads.contains(x.id()), "wanted gradient still produced");
        assert!(
            !grads.contains(frozen.id()),
            "dead slot produced no gradient"
        );
    }

    /// Frozen `Linear`, end to end through the real driver.
    ///
    /// `var_transpose(&w)` mints a fresh `TensorId` every step, so the matmul's
    /// second operand is an id nothing can ever look up. The spy shows the
    /// driver marks that slot false, and `MatmulBackward` returns `None` there
    /// — the `A^T @ dL/dC` GEMM never runs. `x` still gets the right gradient.
    #[test]
    fn test_frozen_linear_skips_weight_gradient_end_to_end() {
        use crate::autograd::ops::MatmulBackward;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Trainable activation
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            true,
        );
        // Frozen weight
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0, 6.0], &[2, 2], &device),
            false,
        );
        let w_t = var_transpose(&w).unwrap();

        // Same node `var_matmul` would build, wrapped so the mask is visible.
        let inner: Arc<dyn GradFn<CpuRuntime>> = Arc::new(MatmulBackward::<CpuRuntime>::new(
            x.id(),
            w_t.id(),
            x.tensor().clone(),
            w_t.tensor().clone(),
            x.grad_fn().cloned(),
            w_t.grad_fn().cloned(),
        ));
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        // Reuse `var_matmul` purely for the forward value; the graph edge comes
        // from the spy-wrapped node below.
        let y_tensor = var_matmul(&x, &w_t, &client).unwrap().tensor().clone();
        let y = Var::from_op(
            y_tensor,
            Arc::new(MaskSpy::<CpuRuntime> {
                inner,
                seen: seen.clone(),
            }),
        );
        let loss = var_sum(&y, &[0, 1], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[x.id()], &client).unwrap();

        let masks = seen.lock().expect("mask spy lock");
        assert_eq!(masks.len(), 1, "the matmul node runs exactly once");
        assert_eq!(
            masks[0],
            vec![true, false],
            "activation wanted, transposed frozen weight dead"
        );

        assert!(
            !grads.contains(w_t.id()),
            "no gradient stored for the throwaway transposed id"
        );
        assert!(!grads.contains(w.id()), "frozen weight gets no gradient");

        // w = [[3,4],[5,6]], w_t = [[3,5],[4,6]], dL/dy = ones[1,2].
        // dL/dx = dL/dy @ w_t^T = ones[1,2] @ w = column sums of w = [8, 10].
        let grad_x: Vec<f32> = grads
            .get(x.id())
            .expect("gradient for x")
            .contiguous()
            .unwrap()
            .to_vec();
        assert_eq!(grad_x, vec![8.0f32, 10.0]);
    }

    /// The all-true path is untouched: a branching multi-input graph containing
    /// a guarded matmul gives bit-identical gradients whether the driver builds
    /// an all-true mask (`backward`) or derives one from a full wanted set.
    #[test]
    fn test_all_true_mask_is_bit_identical_through_guarded_ops() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25, 0.75, 3.0], &[2, 2], &device),
            true,
        );
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.5f32, 1.25, -2.0, 3.5], &[2, 2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.1f32, -0.4, 2.25, 0.125], &[2, 2], &device),
            true,
        );

        // `h` branches into two consumers, so its gradient accumulates two terms.
        let build = || {
            let h = var_matmul(&x, &w, &client).unwrap();
            let p = var_mul(&h, &b, &client).unwrap();
            let q = var_matmul(&h, &b, &client).unwrap();
            let s = var_add(&p, &q, &client).unwrap();
            var_sum(&s, &[0, 1], false, &client).unwrap()
        };

        let full = backward(&build(), &client).unwrap();
        let all_ids = [x.id(), w.id(), b.id()];
        let pruned = backward_wrt(&build(), &all_ids, &client).unwrap();

        for id in all_ids {
            let want = full.get(id).expect("baseline grad");
            let got = pruned.get(id).expect("pruned grad");
            assert_eq!(bits(want), bits(got), "gradient differs for {id:?}");
        }
    }

    #[test]
    fn test_backward_wrt_drops_frozen_operand_gradient() {
        // Mirrors boostr's Linear on a frozen weight: `var_transpose(&w)` mints a
        // fresh TensorId every step, so MatmulBackward stores a full-size
        // gradient under an id no caller can ever look up.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Trainable activation (stands in for the LoRA-carrying input)
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device),
            true,
        );
        // Frozen weight
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0, 5.0, 6.0], &[2, 2], &device),
            false,
        );

        let w_t = var_transpose(&w).unwrap();
        let y = var_matmul(&x, &w_t, &client).unwrap();
        let loss = var_sum(&y, &[0, 1], false, &client).unwrap();

        // Baseline: plain backward keeps the throwaway id
        let full = backward(&loss, &client).unwrap();
        assert!(
            full.contains(w_t.id()),
            "baseline: backward stores a gradient under the transposed frozen weight's id"
        );

        // Pruned: the unreadable entry is never created
        let pruned = backward_wrt(&loss, &[x.id()], &client).unwrap();
        assert!(
            !pruned.contains(w_t.id()),
            "backward_wrt must not store a gradient for the frozen operand"
        );

        // Every wanted gradient is present and bit-identical
        let want = full.get(x.id()).expect("baseline grad for x");
        let got = pruned.get(x.id()).expect("pruned grad for x");
        assert_eq!(bits(want), bits(got));
    }

    #[test]
    fn test_backward_wrt_bit_identical_to_backward() {
        // Multi-layer graph with a branch: `h` feeds two consumers, so its
        // gradient is accumulated from two terms.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.25], &[2], &device),
            true,
        );
        let a = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.3f32, 0.7], &[2], &device),
            true,
        );
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.1f32, -0.4], &[2], &device),
            true,
        );

        let build = || {
            let h = var_mul(&x, &a, &client).unwrap();
            let p = var_mul(&h, &b, &client).unwrap();
            let q = var_mul(&h, &h, &client).unwrap();
            let s = var_add(&p, &q, &client).unwrap();
            var_sum(&s, &[0], false, &client).unwrap()
        };

        let full = backward(&build(), &client).unwrap();
        let all_ids = [x.id(), a.id(), b.id()];
        let pruned = backward_wrt(&build(), &all_ids, &client).unwrap();

        for id in all_ids {
            let want = full.get(id).expect("baseline grad");
            let got = pruned.get(id).expect("pruned grad");
            assert_eq!(bits(want), bits(got), "gradient differs for {id:?}");
        }
    }

    #[test]
    fn test_backward_wrt_accumulates_both_diamond_branches() {
        // Diamond: `w` reaches the loss through a short branch and a long one.
        // Both contributions must survive pruning.
        //   left  = w * c1 ; left2 = left * c3   (long branch)
        //   right = w * c2                       (short branch)
        //   loss  = sum(left2 + right)
        //   dL/dw = c1*c3 + c2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            true,
        );
        let c1 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            false,
        );
        let c2 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device),
            false,
        );
        let c3 = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            false,
        );

        let left = var_mul(&w, &c1, &client).unwrap();
        let left2 = var_mul(&left, &c3, &client).unwrap();
        let right = var_mul(&w, &c2, &client).unwrap();
        let s = var_add(&left2, &right, &client).unwrap();
        let loss = var_sum(&s, &[0], false, &client).unwrap();

        let grads = backward_wrt(&loss, &[w.id()], &client).unwrap();
        let grad_w: Vec<f32> = grads.get(w.id()).expect("grad for w").to_vec();

        // 2*3 + 5 = 11. Losing the long branch would give 5.
        assert_eq!(grad_w, vec![11.0f32]);
    }

    #[test]
    fn test_backward_still_returns_non_parameter_gradients() {
        // backward() keeps its old meaning: a gradient for every graph node,
        // including intermediates nothing will ever step.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device),
            true,
        );
        let frozen = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device),
            false,
        );

        let h = var_mul(&x, &frozen, &client).unwrap();
        let loss = var_sum(&h, &[0], false, &client).unwrap();

        let grads = backward(&loss, &client).unwrap();

        assert!(grads.contains(loss.id()), "loss seed dL/dL");
        assert!(grads.contains(h.id()), "intermediate node");
        assert!(grads.contains(frozen.id()), "frozen operand");
        assert!(grads.contains(x.id()), "parameter");
    }

    #[test]
    fn test_backward_wrt_skips_execution_not_just_storage() {
        // A pruned node's GradFn must never run. Storing-then-discarding would
        // still pay for the tensor work; this asserts the work is not done.
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let build = |calls: &Arc<AtomicUsize>| {
            let trainable = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
                true,
            );
            let frozen = Var::new(
                Tensor::<CpuRuntime>::from_slice(&[7.0f32], &[1], &device),
                false,
            );
            let counted = Var::from_op(
                Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
                Arc::new(CountingBackward::<CpuRuntime> {
                    input_ids: vec![frozen.id()],
                    input_grad_fns: vec![None],
                    calls: calls.clone(),
                }),
            );
            let loss = var_add(&trainable, &counted, &client).unwrap();
            (trainable.id(), loss)
        };

        let full_calls = Arc::new(AtomicUsize::new(0));
        let (_, loss_full) = build(&full_calls);
        let _ = backward(&loss_full, &client).unwrap();
        assert_eq!(
            full_calls.load(Ordering::SeqCst),
            1,
            "backward executes the frozen-only node"
        );

        let pruned_calls = Arc::new(AtomicUsize::new(0));
        let (trainable_id, loss_pruned) = build(&pruned_calls);
        let _ = backward_wrt(&loss_pruned, &[trainable_id], &client).unwrap();
        assert_eq!(
            pruned_calls.load(Ordering::SeqCst),
            0,
            "backward_wrt must not execute a node with no wanted id below it"
        );
    }

    #[test]
    fn test_backward_requires_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Non-scalar tensor should fail
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let var = Var::new(tensor, true);

        let result = backward(&var, &client);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_leaf_variable() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple scalar leaf variable
        let tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);
        let var = Var::new(tensor, true);

        let grads = backward(&var, &client).unwrap();

        // Gradient of loss w.r.t. itself should be 1
        let grad = grads.get(var.id()).unwrap();
        let grad_data: Vec<f32> = grad.to_vec();
        assert_eq!(grad_data, vec![1.0f32]);
    }

    // ========================================================================
    // backward_with_graph tests
    // ========================================================================

    #[test]
    fn test_backward_with_graph_requires_scalar() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Non-scalar tensor should fail
        let tensor = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let var = Var::new(tensor, true);

        let result = backward_with_graph(&var, &client);
        assert!(result.is_err());
    }

    #[test]
    fn test_backward_with_graph_leaf_variable() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple scalar leaf variable
        let tensor = Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device);
        let var = Var::new(tensor, true);

        let grads = backward_with_graph(&var, &client).unwrap();

        // Gradient of loss w.r.t. itself should be 1
        let grad_var = grads.get_var(var.id()).unwrap();
        let grad_data: Vec<f32> = grad_var.tensor().to_vec();
        assert_eq!(grad_data, vec![1.0f32]);

        // The gradient Var should require grad (for second-order)
        assert!(grad_var.requires_grad());
    }

    #[test]
    fn test_backward_with_graph_simple_mul() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // y = x * x = x²
        // dy/dx = 2x
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = var_mul(&x, &x, &client).unwrap();

        let grads = backward_with_graph(&y, &client).unwrap();

        // dy/dx = 2x = 6
        let grad_x = grads.get_var(x.id()).unwrap();
        let grad_data: Vec<f32> = grad_x.tensor().to_vec();
        assert!((grad_data[0] - 6.0).abs() < 1e-6);

        // grad_x should require grad for second-order differentiation
        assert!(grad_x.requires_grad());
    }

    #[test]
    fn test_backward_with_graph_matches_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Test that backward_with_graph produces same numerical results as backward
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // z = x * y
        let z1 = var_mul(&x, &y, &client).unwrap();
        let z2 = var_mul(&x, &y, &client).unwrap();

        let grads1 = backward(&z1, &client).unwrap();
        let grads2 = backward_with_graph(&z2, &client).unwrap();

        // Compare gradients
        let grad_x1: Vec<f32> = grads1.get(x.id()).unwrap().to_vec();
        let grad_x2: Vec<f32> = grads2.get(x.id()).unwrap().to_vec();
        assert!((grad_x1[0] - grad_x2[0]).abs() < 1e-6);

        let grad_y1: Vec<f32> = grads1.get(y.id()).unwrap().to_vec();
        let grad_y2: Vec<f32> = grads2.get(y.id()).unwrap().to_vec();
        assert!((grad_y1[0] - grad_y2[0]).abs() < 1e-6);
    }

    #[test]
    fn test_backward_with_graph_to_grad_store() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = var_mul(&x, &x, &client).unwrap();

        let var_grads = backward_with_graph(&y, &client).unwrap();

        // Convert to regular GradStore
        let grad_store = var_grads.to_grad_store();

        // Should still have the gradient
        let grad_x: Vec<f32> = grad_store.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 4.0).abs() < 1e-6); // dy/dx = 2x = 4
    }

    #[test]
    fn test_second_order_derivative_x_squared() {
        // Test true second-order differentiation
        // f(x) = x², f'(x) = 2x, f''(x) = 2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // y = x²
        let y = var_mul(&x, &x, &client).unwrap();

        // First backward with graph
        let grads = backward_with_graph(&y, &client).unwrap();
        let grad_x = grads.get_var(x.id()).unwrap();

        // grad_x = 2x = 6
        let first_deriv: Vec<f32> = grad_x.tensor().to_vec();
        assert!((first_deriv[0] - 6.0).abs() < 1e-6);

        // Now differentiate grad_x to get second derivative
        // We need to sum grad_x to get a scalar for backward
        let grad_x_sum = var_sum(grad_x, &[], false, &client).unwrap();

        // Second backward
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²y/dx² = 2
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 2.0).abs() < 1e-5,
            "Expected 2.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_hessian_vector_product() {
        // Test HVP: H @ v where H is the Hessian of f(x) = x²
        // H = [[2]], so H @ [1] = [2]
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // f(x) = x²
        let y = var_mul(&x, &x, &client).unwrap();

        // First backward with graph
        let grads = backward_with_graph(&y, &client).unwrap();
        let grad_x = grads.get_var(x.id()).unwrap();

        // Vector v for HVP
        let v = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            false, // v doesn't need grad
        );

        // Compute grad_x · v
        let grad_v = var_mul(grad_x, &v, &client).unwrap();
        let grad_v_sum = var_sum(&grad_v, &[], false, &client).unwrap();

        // Differentiate to get HVP
        let hvp_grads = backward(&grad_v_sum, &client).unwrap();

        // HVP = H @ v = 2 * 1 = 2
        let hvp: Vec<f32> = hvp_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (hvp[0] - 2.0).abs() < 1e-5,
            "Expected HVP = 2.0, got {}",
            hvp[0]
        );
    }

    #[test]
    fn test_second_order_add() {
        // f(x, y) = x + y, d²f/dx² = 0, d²f/dy² = 0
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = crate::autograd::var_add(&x, &y, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&z, &client).unwrap();

        // df/dx = 1, df/dy = 1
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - 1.0).abs() < 1e-6);

        // Second derivative of constant gradient is 0
        // (The gradient is 1, which doesn't depend on x or y)
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 0 (gradient of 1 w.r.t. x is 0)
        // x shouldn't have a gradient in second_grads because grad_x doesn't depend on x
        assert!(
            second_grads.get(x.id()).is_none(),
            "Expected no second-order gradient for add"
        );
    }

    #[test]
    fn test_second_order_sub() {
        // f(x, y) = x - y
        // df/dx = 1, df/dy = -1
        // d²f/dx² = 0, d²f/dy² = 0
        use crate::autograd::var_sub;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = var_sub(&x, &y, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&z, &client).unwrap();

        // df/dx = 1, df/dy = -1
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let grad_y: Vec<f32> = grads.get(y.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
        assert!((grad_y[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_second_order_div() {
        // f(x) = 1/x = x^(-1)
        // df/dx = -1/x² = -x^(-2)
        // d²f/dx² = 2/x³ = 2x^(-3)
        // At x = 2: d²f/dx² = 2/8 = 0.25
        use crate::autograd::var_div;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let one = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device),
            false,
        );
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = 1/x
        let y = var_div(&one, &x, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = -1/x² = -0.25
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - (-0.25)).abs() < 1e-5,
            "Expected -0.25, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 2/x³ = 2/8 = 0.25
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 0.25).abs() < 1e-4,
            "Expected 0.25, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_sum() {
        // f(x) = sum(x²)
        // For x = [a, b]: f = a² + b²
        // df/da = 2a, df/db = 2b
        // d²f/da² = 2, d²f/db² = 2
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device),
            true,
        );

        // f(x) = sum(x * x)
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_sum(&x_squared, &[0], false, &client).unwrap(); // dim 0 to reduce all

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 2x = [6, 8]
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 6.0).abs() < 1e-5,
            "Expected 6.0, got {}",
            grad_x[0]
        );
        assert!(
            (grad_x[1] - 8.0).abs() < 1e-5,
            "Expected 8.0, got {}",
            grad_x[1]
        );

        // Second backward - differentiate sum(grad_x)
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap(); // dim 0 to reduce all
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 2 for each element
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            second_deriv[0]
        );
        assert!(
            (second_deriv[1] - 2.0).abs() < 1e-4,
            "Expected 2.0, got {}",
            second_deriv[1]
        );
    }

    #[test]
    fn test_second_order_through_mean() {
        // f(x) = mean(x²)
        // For x = [a, b]: f = (a² + b²) / 2
        // df/da = a, df/db = b
        // d²f/da² = 1, d²f/db² = 1
        use crate::autograd::var_mean;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device),
            true,
        );

        // f(x) = mean(x * x)
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_mean(&x_squared, &[0], false, &client).unwrap(); // dim 0 to reduce all

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = x (due to mean dividing by 2)
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 3.0).abs() < 1e-5,
            "Expected 3.0, got {}",
            grad_x[0]
        );
        assert!(
            (grad_x[1] - 4.0).abs() < 1e-5,
            "Expected 4.0, got {}",
            grad_x[1]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap(); // dim 0 to reduce all
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 1 for each element
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 1.0).abs() < 1e-4,
            "Expected 1.0, got {}",
            second_deriv[0]
        );
        assert!(
            (second_deriv[1] - 1.0).abs() < 1e-4,
            "Expected 1.0, got {}",
            second_deriv[1]
        );
    }

    #[test]
    fn test_second_order_through_mul_scalar() {
        // f(x) = sum(3 * x²)
        // df/dx = 6x
        // d²f/dx² = 6
        use crate::autograd::var_mul_scalar;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = 3 * x²
        let x_squared = var_mul(&x, &x, &client).unwrap();
        let y = var_mul_scalar(&x_squared, 3.0, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 6x = 12
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 12.0).abs() < 1e-5,
            "Expected 12.0, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 6
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 6.0).abs() < 1e-4,
            "Expected 6.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_pow_scalar() {
        // f(x) = x³
        // df/dx = 3x²
        // d²f/dx² = 6x
        // At x = 2: d²f/dx² = 12
        use crate::autograd::var_pow_scalar;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        // f(x) = x³
        let y = var_pow_scalar(&x, 3.0, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&y, &client).unwrap();

        // df/dx = 3x² = 12
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 12.0).abs() < 1e-5,
            "Expected 12.0, got {}",
            grad_x[0]
        );

        // Second backward
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²f/dx² = 6x = 12
        let second_deriv: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        assert!(
            (second_deriv[0] - 12.0).abs() < 1e-4,
            "Expected 12.0, got {}",
            second_deriv[0]
        );
    }

    #[test]
    fn test_second_order_through_broadcast() {
        // Test that second-order gradients work through broadcasting
        // Simpler test: f(x, b) = sum((x + b)²) where x is [2] and b is [2]
        // No actual broadcasting, but uses var_add to verify basic chain works
        //
        // Forward: y = (x + b)², then sum
        // First backward: dL/dx = 2(x + b), dL/db = 2(x + b)
        // Second backward: d²L/dx² = 2
        use crate::autograd::var_add;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // x is [2] vector
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device),
            true,
        );

        // b is [2] vector (same shape, no broadcast)
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2], &[2], &device),
            true,
        );

        // f(x, b) = sum((x + b)²)
        let x_plus_b = var_add(&x, &b, &client).unwrap();
        let squared = var_mul(&x_plus_b, &x_plus_b, &client).unwrap();
        let loss = var_sum(&squared, &[0], false, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&loss, &client).unwrap();

        // Verify first-order gradients exist
        assert!(grads.get(x.id()).is_some(), "Should have gradient for x");
        assert!(grads.get(b.id()).is_some(), "Should have gradient for b");

        // dL/dx = 2(x + b)
        // For x[0] = 1.0, b[0] = 0.1: grad = 2 * 1.1 = 2.2
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 2.2).abs() < 1e-5,
            "Expected 2.2, got {}",
            grad_x[0]
        );

        // Second backward through x
        let grad_x_var = grads.get_var(x.id()).unwrap();
        let grad_x_sum = var_sum(grad_x_var, &[0], false, &client).unwrap();
        let second_grads = backward(&grad_x_sum, &client).unwrap();

        // d²L/dx² = 2 for each element (since d/dx[2(x+b)] = 2)
        let second_deriv_x: Vec<f32> = second_grads.get(x.id()).unwrap().to_vec();
        for (i, &val) in second_deriv_x.iter().enumerate() {
            assert!(
                (val - 2.0).abs() < 1e-4,
                "Expected d²L/dx²[{}] = 2.0, got {}",
                i,
                val
            );
        }
    }

    #[test]
    fn test_second_order_through_broadcast_shapes() {
        // Test that second-order gradients work through actual broadcasting
        // f(x, b) = sum((x + b)²) where x is [2, 3] and b is [3] (broadcasts)
        //
        // This tests that reduce_var_for_broadcast properly maintains
        // the gradient chain via var_reshape.
        use crate::autograd::var_add;

        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // x is [2, 3] matrix
        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device),
            true,
        );

        // b is [3] vector that will broadcast to [2, 3]
        let b = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3], &[3], &device),
            true,
        );

        // f(x, b) = sum((x + b)²)
        let x_plus_b = var_add(&x, &b, &client).unwrap();
        let squared = var_mul(&x_plus_b, &x_plus_b, &client).unwrap();
        let loss = var_sum(&squared, &[0, 1], false, &client).unwrap();

        // First backward
        let grads = backward_with_graph(&loss, &client).unwrap();

        // Verify first-order gradients exist for both x and b
        assert!(grads.get(x.id()).is_some(), "Should have gradient for x");
        assert!(grads.get(b.id()).is_some(), "Should have gradient for b");

        // Verify gradient shapes
        assert_eq!(grads.get(x.id()).unwrap().shape(), &[2, 3]);
        assert_eq!(grads.get(b.id()).unwrap().shape(), &[3]);

        // dL/dx = 2(x + b)
        // For x[0,0] = 1.0, b[0] = 0.1: grad = 2 * 1.1 = 2.2
        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!(
            (grad_x[0] - 2.2).abs() < 1e-5,
            "Expected 2.2, got {}",
            grad_x[0]
        );

        // dL/db[0] = sum over rows of 2(x + b) at column 0
        // = 2*(1.0+0.1) + 2*(4.0+0.1) = 2.2 + 8.2 = 10.4
        let grad_b: Vec<f32> = grads.get(b.id()).unwrap().to_vec();
        assert!(
            (grad_b[0] - 10.4).abs() < 1e-4,
            "Expected 10.4, got {}",
            grad_b[0]
        );

        // Second backward through x - need to verify get_var works
        if let Some(grad_x_var) = grads.get_var(x.id()) {
            let grad_x_sum = var_sum(grad_x_var, &[0, 1], false, &client).unwrap();
            let second_grads = backward(&grad_x_sum, &client).unwrap();

            // d²L/dx² = 2 for each element
            if let Some(second_deriv_x) = second_grads.get(x.id()) {
                let second_deriv_x: Vec<f32> = second_deriv_x.to_vec();
                for (i, &val) in second_deriv_x.iter().enumerate() {
                    assert!(
                        (val - 2.0).abs() < 1e-4,
                        "Expected d²L/dx²[{}] = 2.0, got {}",
                        i,
                        val
                    );
                }
            }
        }
    }
}
