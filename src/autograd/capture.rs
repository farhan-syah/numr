//! Discovery of the values a checkpointed segment reads from outside itself.
//!
//! A checkpointed segment runs its forward on detached copies of the listed
//! inputs, so every operand it sees through `inputs` has `requires_grad = false`
//! and builds no graph node. A node therefore appears in the retained forward
//! graph only because the closure read some OTHER value that carries gradient —
//! a parameter or an activation the caller captured instead of listing. Those
//! captured values are real inputs of the segment, and this module finds them so
//! the segment can differentiate them.

use std::collections::HashSet;
use std::sync::Arc;

use super::{GradFn, Var};
use crate::runtime::Runtime;
use crate::tensor::TensorId;

/// Collect the leaves `output`'s graph reaches from outside the segment.
///
/// `boundary` is a [`TensorId`] minted immediately before the segment ran.
/// `TensorId::new` hands out strictly increasing values, so an id below it was
/// created before the segment and belongs to the caller, while an id above it is
/// something the segment itself made. `known` holds the ids the caller already
/// accounts for: the detached copies of the listed inputs, and the listed input
/// ids themselves.
///
/// The walk descends through every node that has a `grad_fn` and reports only
/// leaves. Stopping earlier — at a captured intermediate, keeping its `grad_fn`
/// for the outer pass — would double-count: the recompute differentiates the
/// captured intermediate's own history as well, so a leaf that the segment reads
/// directly AND reaches through that intermediate would collect its second path
/// twice. Reporting leaves only makes every capture terminal in both passes.
///
/// A frozen capture stays out of the result in the case that matters. A segment
/// reading only frozen values builds no node at all, because every operand has
/// `requires_grad = false`, so there is nothing to walk. A frozen value reached
/// through a node that some trainable capture created is reported, because the
/// retained graph records ids and never `requires_grad`. That costs nothing in
/// correctness: an unpruned `backward` gives such a value a gradient too, so the
/// segment still behaves as if it were not checkpointed.
pub(super) fn collect_captures<R: Runtime>(
    output: &Var<R>,
    boundary: TensorId,
    known: &HashSet<TensorId>,
) -> Vec<TensorId> {
    let mut visited: HashSet<TensorId> = HashSet::new();
    let mut captures: Vec<TensorId> = Vec::new();
    let mut stack: Vec<(TensorId, Option<Arc<dyn GradFn<R>>>)> =
        vec![(output.id(), output.grad_fn().cloned())];

    while let Some((id, grad_fn)) = stack.pop() {
        if !visited.insert(id) {
            continue;
        }

        let Some(gf) = grad_fn else {
            // A leaf. Older than the boundary means the caller owns it; newer
            // means the segment minted it this run, and its id changes on the
            // next one, so no gradient can be delivered under it.
            if id.raw() < boundary.raw() && !known.contains(&id) {
                captures.push(id);
            }
            continue;
        };

        for (input_id, input_grad_fn) in gf.inputs().iter().zip(gf.input_grad_fns()) {
            stack.push((*input_id, input_grad_fn));
        }
    }

    captures
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{var_add, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn leaf(device: &CpuDevice, value: f32, requires_grad: bool) -> Var<CpuRuntime> {
        Var::new(
            Tensor::<CpuRuntime>::from_slice(&[value], &[1], device).unwrap(),
            requires_grad,
        )
    }

    #[test]
    fn detached_only_segment_captures_nothing() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = leaf(&device, 3.0, false);
        let known: HashSet<TensorId> = [x.id()].into_iter().collect();

        let boundary = TensorId::new();
        let y = var_mul(&x, &x, &client).unwrap();

        assert!(collect_captures(&y, boundary, &known).is_empty());
    }

    #[test]
    fn trainable_capture_is_found_once() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = leaf(&device, 3.0, false);
        let w = leaf(&device, 2.0, true);
        let known: HashSet<TensorId> = [x.id()].into_iter().collect();

        let boundary = TensorId::new();
        // `w` is read twice, so the walk must still report it once.
        let a = var_mul(&x, &w, &client).unwrap();
        let b = var_mul(&a, &w, &client).unwrap();
        let out = var_sum(&b, &[], false, &client).unwrap();

        assert_eq!(collect_captures(&out, boundary, &known), vec![w.id()]);
    }

    #[test]
    fn captured_intermediate_resolves_to_its_leaves() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = leaf(&device, 2.0, true);
        // `h` is computed outside the segment. The walk must report `a`, not
        // `h`, so that every capture is terminal for both backward passes.
        let h = var_mul(&a, &a, &client).unwrap();

        let x = leaf(&device, 3.0, false);
        let known: HashSet<TensorId> = [x.id()].into_iter().collect();

        let boundary = TensorId::new();
        let out = var_mul(&x, &h, &client).unwrap();

        assert_eq!(collect_captures(&out, boundary, &known), vec![a.id()]);
    }

    #[test]
    fn leaves_minted_inside_the_segment_are_ignored() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = leaf(&device, 3.0, false);
        let w = leaf(&device, 2.0, true);
        let known: HashSet<TensorId> = [x.id()].into_iter().collect();

        let boundary = TensorId::new();
        // The constant is born inside the segment, so its id differs on the next
        // run and it must not claim a slot.
        let inside = leaf(&device, 1.0, false);
        let scaled = var_mul(&x, &w, &client).unwrap();
        let out = var_add(&scaled, &inside, &client).unwrap();

        assert_eq!(collect_captures(&out, boundary, &known), vec![w.id()]);
    }
}
