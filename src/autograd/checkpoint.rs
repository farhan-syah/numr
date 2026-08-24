//! Activation checkpointing for memory-efficient training.
//!
//! Discards intermediate activations during forward and recomputes them during
//! backward. Trades ~33% extra compute for dramatically less activation memory.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, backward, checkpoint, var_mul, var_sum};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! let x = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device), true);
//!
//! // Wrap computation in checkpoint — intermediates are dropped and recomputed
//! let y = checkpoint(|inputs, c| {
//!     let x_sq = var_mul(&inputs[0], &inputs[0], c)?;
//!     Ok(x_sq)
//! }, &[&x])?;
//!
//! let loss = var_sum(&y, &[], false, &client)?;
//! let grads = backward(&loss, &client)?;
//! // grad_x = 2 * 3 = 6
//! # Ok::<(), numr::error::Error>(())
//! ```

use std::sync::Arc;

use crate::autograd::{GradFn, Var, backward_wrt, var_mul, var_sum};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TensorOps;
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};

/// Run `f` on `inputs` with activation checkpointing.
///
/// During forward, `f` runs on detached copies of the inputs so no intermediate
/// graph nodes are retained. During backward, `f` is re-run with grad tracking
/// to reconstruct the graph and propagate gradients.
pub fn checkpoint<R, F>(f: F, inputs: &[&Var<R>]) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>,
    F: Fn(&[Var<R>], &R::Client) -> Result<Var<R>> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(crate::error::Error::Internal(
            "checkpoint requires at least one input".to_string(),
        ));
    }

    // Save original input info for backward
    let input_ids: Vec<TensorId> = inputs.iter().map(|v| v.id()).collect();
    let input_tensors: Vec<Tensor<R>> = inputs.iter().map(|v| v.tensor().clone()).collect();
    let input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>> =
        inputs.iter().map(|v| v.grad_fn().cloned()).collect();

    // Forward: run on detached inputs (no grad tracking inside the segment)
    let detached: Vec<Var<R>> = inputs
        .iter()
        .map(|v| Var::new(v.tensor().clone(), false))
        .collect();

    let device = inputs[0].tensor().device();
    let client = R::default_client(device);

    let output = f(&detached, &client)?;
    // output has no grad graph inside — all intermediates are already dropped

    let checkpoint_backward = CheckpointBackward {
        func: Arc::new(f),
        input_ids: input_ids.clone(),
        input_tensors,
        input_grad_fns,
    };

    Ok(Var::from_op(
        output.tensor().clone(),
        Arc::new(checkpoint_backward),
    ))
}

struct CheckpointBackward<R: Runtime> {
    func: Arc<dyn Fn(&[Var<R>], &R::Client) -> Result<Var<R>> + Send + Sync>,
    input_ids: Vec<TensorId>,
    input_tensors: Vec<Tensor<R>>,
    input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
}

impl<R> GradFn<R> for CheckpointBackward<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // The mask narrows the re-entrant pass below: only the wanted segment
        // inputs are asked for, so the recomputed graph prunes to them and the
        // rest of the segment's backward work is never run.
        if !needed.iter().any(|&n| n) {
            return Ok(vec![None; self.input_ids.len()]);
        }

        let client = R::default_client(grad_output.device());

        // Reconstruct input Vars as LEAF nodes with original IDs.
        // They have no grad_fn so backward stops here — the outer backward
        // pass handles continuing through input_grad_fns() returned below.
        let reconstructed: Vec<Var<R>> = self
            .input_ids
            .iter()
            .zip(self.input_tensors.iter())
            .map(|(id, tensor)| Var::with_id(tensor.clone(), *id, true))
            .collect();

        // Re-run forward WITH grad tracking — rebuilds the intermediate graph
        let recomputed_output = (self.func)(&reconstructed, &client)?;

        // Backprop grad_output through the recomputed graph.
        // loss = sum(recomputed * grad_output) is a scalar whose gradient w.r.t.
        // each input is exactly the VJP: sum_j(grad_output_j * d(output_j)/d(input_i))
        let grad_output_var = Var::new(grad_output.clone(), false);
        let product = var_mul(&recomputed_output, &grad_output_var, &client)?;
        let loss = var_sum(&product, &[], false, &client)?;

        // Only the wanted segment inputs are read back below, so prune the
        // re-entrant pass to exactly those — the recomputed intermediates never
        // enter the store, and an unwanted input's cone is never walked.
        let wanted: Vec<TensorId> = self
            .input_ids
            .iter()
            .zip(needed)
            .filter_map(|(id, &want)| want.then_some(*id))
            .collect();
        let grads = backward_wrt(&loss, &wanted, &client)?;

        Ok(self
            .input_ids
            .iter()
            .zip(needed)
            .map(|(id, &want)| want.then(|| grads.get(*id).cloned()).flatten())
            .collect())
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.clone()
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::{BackwardHook, backward, backward_with_hooks, var_add, var_mul, var_sum};
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn device_and_client() -> (CpuDevice, <CpuRuntime as Runtime>::Client) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_checkpoint_x_squared() {
        // f(x) = x^2, df/dx = 2x
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        // Without checkpoint
        let y_normal = var_mul(&x, &x, &client).unwrap();
        let loss_normal = var_sum(&y_normal, &[], false, &client).unwrap();
        let grads_normal = backward(&loss_normal, &client).unwrap();

        // With checkpoint
        let y_ckpt = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&x]).unwrap();
        let loss_ckpt = var_sum(&y_ckpt, &[], false, &client).unwrap();
        let grads_ckpt = backward(&loss_ckpt, &client).unwrap();

        let g_normal: Vec<f32> = grads_normal.get(x.id()).unwrap().to_vec();
        let g_ckpt: Vec<f32> = grads_ckpt.get(x.id()).unwrap().to_vec();

        assert!(
            (g_normal[0] - g_ckpt[0]).abs() < 1e-6,
            "normal={}, checkpoint={}",
            g_normal[0],
            g_ckpt[0]
        );
        assert!((g_ckpt[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_multi_input() {
        // f(x, y) = x * y
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device),
            true,
        );

        let out = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[1], c), &[&x, &y]).unwrap();

        let grads = backward(&out, &client).unwrap();

        // d(x*y)/dx = y = 5
        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 5.0).abs() < 1e-6);

        // d(x*y)/dy = x = 2
        let gy: Vec<f32> = grads.get(y.id()).unwrap().to_vec();
        assert!((gy[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_chained() {
        // checkpoint(f1) -> checkpoint(f2)
        // f1(x) = x^2, f2(z) = z^2, so total = x^4
        // d(x^4)/dx = 4x^3 = 4*8 = 32 at x=2
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device),
            true,
        );

        let z = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&x]).unwrap();

        let w = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&z]).unwrap();

        let loss = var_sum(&w, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 32.0).abs() < 1e-4, "expected 32.0, got {}", gx[0]);
    }

    #[test]
    fn test_checkpoint_matches_normal_complex() {
        // More complex: f(x) = (x + x) * x = 2x^2
        // df/dx = 4x = 12 at x=3
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let y = checkpoint(
            |inputs, c| {
                let sum = var_add(&inputs[0], &inputs[0], c)?;
                var_mul(&sum, &inputs[0], c)
            },
            &[&x],
        )
        .unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 12.0).abs() < 1e-5, "expected 12.0, got {}", gx[0]);
    }

    #[test]
    fn test_checkpoint_with_backward_hooks() {
        // Verify leaf hooks still fire through checkpointed segments
        use std::cell::RefCell;
        use std::rc::Rc;

        struct RecordingHook {
            leaf_ids: Rc<RefCell<Vec<TensorId>>>,
        }

        unsafe impl Send for RecordingHook {}

        impl BackwardHook<CpuRuntime> for RecordingHook {
            fn on_leaf_grad_ready(&mut self, id: TensorId, _grad: &Tensor<CpuRuntime>) {
                self.leaf_ids.borrow_mut().push(id);
            }
        }

        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device),
            true,
        );

        let y = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&x]).unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();

        let ids = Rc::new(RefCell::new(Vec::new()));
        let mut hook = RecordingHook {
            leaf_ids: ids.clone(),
        };
        let _grads = backward_with_hooks(&loss, &client, &mut hook).unwrap();

        let recorded = ids.borrow();
        assert!(
            recorded.contains(&x.id()),
            "leaf hook should have fired for x"
        );
    }

    #[test]
    fn test_checkpoint_vector_output() {
        // f(x) = x * x where x is a vector [2, 3]
        // loss = sum(f(x)) = 4 + 9 = 13
        // d(loss)/dx = [4, 6]
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device),
            true,
        );

        let y = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&x]).unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 4.0).abs() < 1e-6);
        assert!((gx[1] - 6.0).abs() < 1e-6);
    }
}
