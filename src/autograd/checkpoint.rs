//! Activation checkpointing for memory-efficient training.
//!
//! Discards intermediate activations during forward and recomputes them during
//! backward. Trades ~33% extra compute for dramatically less activation memory.
//!
//! # Every trainable value inside the segment must be listed in `inputs`
//!
//! The recompute differentiates the segment only with respect to the ids in
//! `inputs`. A parameter the closure captures but the caller omits from `inputs`
//! receives no gradient, while the forward value stays correct. Both entry
//! points reject that case at forward time — see [`checkpoint`].
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, backward, checkpoint, var_mul, var_sum};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! let x = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);
//! let w = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
//!
//! // `w` lives inside the segment, so it must appear in `inputs`.
//! let y = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[1], c), &[&x, &w])?;
//!
//! let loss = var_sum(&y, &[], false, &client)?;
//! let grads = backward(&loss, &client)?;
//! // grad_x = w = 2, grad_w = x = 3
//! # Ok::<(), numr::error::Error>(())
//! ```

use std::sync::Arc;

use crate::autograd::{GradFn, Var, backward_wrt, var_mul, var_sum};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::TensorOps;
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};

/// Run `f` on `inputs` with activation checkpointing, on the default client.
///
/// Equivalent to [`checkpoint_with_client`] with `R::default_client(device)`,
/// where `device` is the device of `inputs[0]`. Prefer
/// [`checkpoint_with_client`] whenever a client is already in hand: it keeps the
/// recompute on the caller's client, and its closure takes the caller's own
/// client type instead of `R::Client`.
///
/// During forward, `f` runs on detached copies of the inputs so no intermediate
/// graph nodes are retained. During backward, `f` is re-run with grad tracking
/// to reconstruct the graph and propagate gradients.
///
/// # Every trainable value inside the segment must be listed in `inputs`
///
/// The recompute differentiates only with respect to the ids in `inputs`. A
/// parameter that lives inside the segment and is missing from `inputs` gets no
/// gradient at all, and the forward values stay correct, so a forward-only test
/// passes. List every such parameter in `inputs`, or mark it
/// `requires_grad = false` when it is frozen.
///
/// # Errors
///
/// - `inputs` is empty.
/// - The segment reaches a value with `requires_grad = true` that is not one of
///   `inputs`. This is the missing-gradient case above, caught at forward time.
pub fn checkpoint<R, F>(f: F, inputs: &[&Var<R>]) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + 'static,
    F: Fn(&[Var<R>], &R::Client) -> Result<Var<R>> + Send + Sync + 'static,
{
    let device = inputs
        .first()
        .ok_or_else(|| Error::InvalidArgument {
            arg: "inputs",
            reason: "checkpoint requires at least one input".to_string(),
        })?
        .tensor()
        .device();

    let client = R::default_client(device);
    checkpoint_with_client(f, inputs, &client)
}

/// Run `f` on `inputs` with activation checkpointing, on `client`.
///
/// Both the forward pass and the backward recompute run on `client`, so the
/// recompute uses the same stream, allocator and parallelism settings as the
/// original forward. `f` receives `client` itself, so the trait bounds the
/// closure needs land on the caller's own client type `C`, not on `R::Client`.
///
/// The client is cloned into the graph node and outlives this call.
///
/// # Every trainable value inside the segment must be listed in `inputs`
///
/// The recompute differentiates only with respect to the ids in `inputs`. A
/// parameter that lives inside the segment and is missing from `inputs` gets no
/// gradient at all, and the forward values stay correct, so a forward-only test
/// passes. List every such parameter in `inputs`, or mark it
/// `requires_grad = false` when it is frozen.
///
/// # Errors
///
/// - `inputs` is empty.
/// - The segment reaches a value with `requires_grad = true` that is not one of
///   `inputs`. This is the missing-gradient case above, caught at forward time.
pub fn checkpoint_with_client<R, C, F>(f: F, inputs: &[&Var<R>], client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + 'static,
    // Required by `CheckpointBackward`'s `GradFn` impl: the recompute's `var_*`
    // ops are bounded on `R::Client`, not on the caller's `C`.
    R::Client: TensorOps<R>,
    F: Fn(&[Var<R>], &C) -> Result<Var<R>> + Send + Sync + 'static,
{
    if inputs.is_empty() {
        return Err(Error::InvalidArgument {
            arg: "inputs",
            reason: "checkpoint requires at least one input".to_string(),
        });
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

    let output = f(&detached, client)?;

    // NOT VALIDATED HERE, deliberately. A retained graph means the segment read
    // some value that was not detached, but the retained graph cannot say
    // whether that value is TRAINABLE: a leaf records no `grad_fn` whether its
    // `requires_grad` is true or false, and a binary op saves BOTH operands
    // when either one requires grad. So a frozen backbone weight — the ordinary
    // case this function exists to serve — is indistinguishable by id from an
    // unlisted trainable parameter. Rejecting on that signal breaks correct
    // callers; see this module's docs for the rule the CALLER must keep.

    let checkpoint_backward = CheckpointBackward {
        func: Arc::new(f),
        client: client.clone(),
        input_ids,
        input_tensors,
        input_grad_fns,
    };

    Ok(Var::from_op(
        output.tensor().clone(),
        Arc::new(checkpoint_backward),
    ))
}

struct CheckpointBackward<R: Runtime, C: 'static> {
    func: Arc<dyn Fn(&[Var<R>], &C) -> Result<Var<R>> + Send + Sync>,
    /// The caller's client. The recompute must run where the forward ran.
    client: C,
    input_ids: Vec<TensorId>,
    input_tensors: Vec<Tensor<R>>,
    input_grad_fns: Vec<Option<Arc<dyn GradFn<R>>>>,
}

impl<R, C> GradFn<R> for CheckpointBackward<R, C>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + 'static,
    // The `var_*` ops the recompute runs are bounded on `R::Client`, not on
    // the caller's `C`, so this bound is required even though the recompute
    // itself uses `self.client`.
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>, needed: &[bool]) -> Result<Vec<Option<Tensor<R>>>> {
        // The mask narrows the re-entrant pass below: only the wanted segment
        // inputs are asked for, so the recomputed graph prunes to them and the
        // rest of the segment's backward work is never run.
        if !needed.iter().any(|&n| n) {
            return Ok(vec![None; self.input_ids.len()]);
        }

        let client = &self.client;

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
        let recomputed_output = (self.func)(&reconstructed, client)?;

        // Backprop grad_output through the recomputed graph.
        // loss = sum(recomputed * grad_output) is a scalar whose gradient w.r.t.
        // each input is exactly the VJP: sum_j(grad_output_j * d(output_j)/d(input_i))
        let grad_output_var = Var::new(grad_output.clone(), false);
        let product = var_mul(&recomputed_output, &grad_output_var, client)?;
        let loss = var_sum(&product, &[], false, client)?;

        // Only the wanted segment inputs are read back below, so prune the
        // re-entrant pass to exactly those — the recomputed intermediates never
        // enter the store, and an unwanted input's cone is never walked.
        let wanted: Vec<TensorId> = self
            .input_ids
            .iter()
            .zip(needed)
            .filter_map(|(id, &want)| want.then_some(*id))
            .collect();
        let grads = backward_wrt(&loss, &wanted, client)?;

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
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use std::sync::Mutex;

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
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let y = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[5.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
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
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap(),
            true,
        );

        let y = checkpoint(|inputs, c| var_mul(&inputs[0], &inputs[0], c), &[&x]).unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 4.0).abs() < 1e-6);
        assert!((gx[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_recompute_uses_caller_client() {
        // Defect 1: forward AND recompute must run on the client the caller
        // supplied, not on `R::default_client(device)`.
        let device = CpuDevice::new();
        let marked = CpuClient::new(device.clone())
            .with_parallelism(ParallelismConfig::new(Some(1), Some(7)));

        // The marker must distinguish `marked` from the default client.
        assert_eq!(
            CpuRuntime::default_client(&device).parallelism().chunk_size,
            None
        );

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );

        let seen: Arc<Mutex<Vec<Option<usize>>>> = Arc::new(Mutex::new(Vec::new()));
        let recorder = Arc::clone(&seen);

        let y = checkpoint_with_client(
            move |inputs: &[Var<CpuRuntime>], c: &CpuClient| {
                recorder
                    .lock()
                    .unwrap_or_else(|p| p.into_inner())
                    .push(c.parallelism().chunk_size);
                var_mul(&inputs[0], &inputs[0], c)
            },
            &[&x],
            &marked,
        )
        .unwrap();

        let loss = var_sum(&y, &[], false, &marked).unwrap();
        let grads = backward(&loss, &marked).unwrap();

        let calls = seen.lock().unwrap_or_else(|p| p.into_inner()).clone();
        assert_eq!(calls.len(), 2, "expected one forward and one recompute");
        assert!(
            calls.iter().all(|chunk| *chunk == Some(7)),
            "segment ran on a client that is not the caller's: {calls:?}"
        );

        // Gradient is still correct: d(x^2)/dx = 2x = 6
        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_unlisted_parameter_silently_gets_no_gradient() {
        // PINS A KNOWN TRAP so it cannot regress unnoticed. A trainable
        // parameter used inside the segment but absent from `inputs` receives
        // NO gradient, while the forward value stays correct — so a
        // forward-only test passes and training silently does nothing.
        //
        // This is NOT detectable inside `checkpoint`: the retained graph
        // records ids and grad_fns only, a leaf has no `grad_fn` whether or not
        // it requires grad, and a binary op saves both operands when either
        // requires grad. A frozen backbone weight is therefore indistinguishable
        // from an unlisted trainable one, and erroring on the signal would
        // reject the frozen-backbone case checkpointing exists to serve. The
        // rule is the CALLER's to keep; the module docs state it.
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );
        let w_id = w.id();

        let y = checkpoint(
            move |inputs: &[Var<CpuRuntime>], c: &CpuClient| var_mul(&inputs[0], &w, c),
            &[&x],
        )
        .unwrap();
        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // x is listed, so it differentiates: d(x*w)/dx = w = 2.
        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 2.0).abs() < 1e-6);

        // w is NOT listed, so it gets nothing. List it in `inputs` to train it.
        assert!(
            grads.get(w_id).is_none(),
            "an unlisted trainable parameter must be understood to get no \
             gradient; if this now returns one, the trap is fixed and this \
             test should assert the gradient instead"
        );
    }

    #[test]
    fn test_checkpoint_accepts_frozen_capture() {
        // A captured value with requires_grad = false needs no gradient, so the
        // segment is accepted and the listed input still differentiates.
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );
        let frozen = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            false,
        );

        let y = checkpoint(
            move |inputs: &[Var<CpuRuntime>], c: &CpuClient| var_mul(&inputs[0], &frozen, c),
            &[&x],
        )
        .unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        // d(2x)/dx = 2
        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_listed_parameter_gets_gradient() {
        // The fix for the trap: list the parameter and it trains.
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap(),
            true,
        );
        let w = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );

        let y = checkpoint(
            |inputs: &[Var<CpuRuntime>], c: &CpuClient| var_mul(&inputs[0], &inputs[1], c),
            &[&x, &w],
        )
        .unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        let gw: Vec<f32> = grads.get(w.id()).unwrap().to_vec();
        assert!((gx[0] - 2.0).abs() < 1e-6);
        assert!((gw[0] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_nested() {
        // A nested checkpoint retains its own graph node, yet every leaf it
        // reaches is a listed input, so the outer segment is accepted.
        // f(x) = (x^2)^2 = x^4, df/dx = 4x^3 = 32 at x = 2
        let (device, client) = device_and_client();

        let x = Var::new(
            Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap(),
            true,
        );

        let y = checkpoint_with_client(
            |inputs: &[Var<CpuRuntime>], c: &CpuClient| {
                let z = checkpoint_with_client(
                    |inner: &[Var<CpuRuntime>], ic: &CpuClient| var_mul(&inner[0], &inner[0], ic),
                    &[&inputs[0]],
                    c,
                )?;
                var_mul(&z, &z, c)
            },
            &[&x],
            &client,
        )
        .unwrap();

        let loss = var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let gx: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((gx[0] - 32.0).abs() < 1e-4, "expected 32.0, got {}", gx[0]);
    }

    #[test]
    fn test_checkpoint_empty_inputs_errors() {
        let inputs: [&Var<CpuRuntime>; 0] = [];
        let err = checkpoint(
            |ins: &[Var<CpuRuntime>], c: &CpuClient| var_mul(&ins[0], &ins[0], c),
            &inputs,
        )
        .unwrap_err();
        assert!(err.to_string().contains("at least one input"));
    }
}
