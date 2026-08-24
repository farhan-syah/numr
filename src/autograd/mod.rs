//! Automatic differentiation (autograd)
//!
//! This module provides both reverse-mode and forward-mode automatic differentiation
//! for computing gradients of tensor computations.
//!
//! # Overview
//!
//! The autograd system consists of:
//!
//! ## Reverse-Mode AD (Backpropagation)
//! - [`Var`]: A tensor that tracks gradients
//! - [`GradFn`]: Trait for computing gradients in backward pass
//! - [`GradStore`]: Storage for accumulated gradients (first-order)
//! - [`VarGradStore`]: Storage for gradient Vars (second-order)
//! - [`backward`]: Function to compute gradients via reverse-mode AD
//! - [`backward_wrt`]: Backward that prunes to a wanted set of tensor ids
//! - [`backward_with_graph`]: Backward with graph retention for Hessians
//!
//! ## Forward-Mode AD (JVP)
//! - [`DualTensor`]: Tensor carrying both primal value and tangent
//! - [`jvp`]: Compute Jacobian-vector product in a single forward pass
//! - [`jvp_multi`]: JVP for functions with multiple outputs
//! - [`jacobian_forward`]: Compute full Jacobian using forward-mode
//! - `dual_ops`: Operations on dual tensors (dual_add, dual_mul, etc.)
//!
//! # When to Use Forward vs Reverse Mode
//!
//! - **Reverse-mode (VJP)**: Efficient when inputs >> outputs
//!   - Training neural networks (many params, scalar loss)
//!   - Computing gradients of scalar functions
//!
//! - **Forward-mode (JVP)**: Efficient when outputs >> inputs
//!   - Directional derivatives
//!   - Newton-Krylov methods (need J @ v without forming J)
//!   - Sensitivity analysis with few inputs
//!
//! # First-Order Example (Reverse-Mode)
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, backward, var_mul};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // Create leaf variables
//! let x = Var::new(Tensor::try_from_slice(&[2.0f32], &[1], &device)?, true);
//! let y = Var::new(Tensor::try_from_slice(&[3.0f32], &[1], &device)?, true);
//!
//! // Forward: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Backward
//! let grads = backward(&z, &client)?;
//!
//! // dx = y = 3.0, dy = x = 2.0
//! let grad_x = grads.get(x.id()).unwrap();
//! let grad_y = grads.get(y.id()).unwrap();
//! # Ok::<(), numr::error::Error>(())
//! ```
//!
//! # Forward-Mode Example (JVP)
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{DualTensor, jvp, dual_ops::*};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // f(x) = x² at x=3, tangent v=1 → df/dx in direction v
//! let x = Tensor::try_from_slice(&[3.0f32], &[1], &device)?;
//! let v = Tensor::try_from_slice(&[1.0f32], &[1], &device)?;
//!
//! let (y, dy) = jvp(
//!     |inputs, c| {
//!         let x = &inputs[0];
//!         dual_mul(x, x, c)
//!     },
//!     &[&x],
//!     &[&v],
//!     &client,
//! )?;
//! // y = 9.0, dy = 2*3*1 = 6.0
//! # Ok::<(), numr::error::Error>(())
//! ```
//!
//! # Second-Order Example (Hessian-Vector Product)
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, backward, backward_with_graph, var_mul, var_sum};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // f(x) = x²
//! let x = Var::new(Tensor::try_from_slice(&[3.0f32], &[1], &device)?, true);
//! let y = var_mul(&x, &x, &client)?;
//!
//! // First backward with graph retention
//! let grads = backward_with_graph(&y, &client)?;
//! let grad_x = grads.get_var(x.id()).unwrap();  // dy/dx = 2x = 6
//!
//! // Compute Hessian-vector product: H @ v where v = [1.0]
//! let v = Var::new(Tensor::try_from_slice(&[1.0f32], &[1], &device)?, false);
//! let grad_v = var_mul(grad_x, &v, &client)?;
//! let scalar = var_sum(&grad_v, &[], false, &client)?;
//!
//! // Second backward gives d²y/dx² * v = 2 * 1 = 2
//! let second_grads = backward(&scalar, &client)?;
//! # Ok::<(), numr::error::Error>(())
//! ```

// Reverse-mode AD
mod backward;
mod checkpoint;
mod grad_fn;
mod grad_store;
mod var;
mod var_grad_store;
pub mod var_ops;

// Forward-mode AD
mod dual;
pub mod dual_ops;
mod forward;

pub mod ops;

// Reverse-mode exports
pub use crate::tensor::id::TensorId;
pub use backward::{
    BackwardHook, NoOpHook, backward, backward_with_graph, backward_with_hooks, backward_wrt,
    backward_wrt_with_hooks,
};
pub use checkpoint::checkpoint;
pub use grad_fn::GradFn;
pub use grad_store::GradStore;
pub use var::Var;
pub use var_grad_store::VarGradStore;
pub use var_ops::var_dropout;
pub use var_ops::{
    var_abs, var_add, var_add_scalar, var_cast, var_cholesky, var_clamp, var_conv_transpose1d,
    var_conv1d, var_conv2d, var_cos, var_cumprod, var_cumsum, var_det, var_div, var_div_scalar,
    var_embedding_lookup, var_exp, var_fused_add_layer_norm, var_fused_add_rms_norm, var_gather,
    var_gelu_mul, var_group_norm, var_inverse, var_layer_norm, var_log, var_log_softmax,
    var_matmul, var_matmul_bias_activation, var_max, var_mean, var_min, var_mul, var_mul_scalar,
    var_neg, var_pow, var_pow_scalar, var_recip, var_relu, var_relu_mul, var_rms_norm, var_sigmoid,
    var_sigmoid_mul, var_silu, var_silu_mul, var_sin, var_softmax, var_softplus, var_solve,
    var_sqrt, var_square, var_std, var_sub, var_sub_scalar, var_sum, var_swiglu, var_tan, var_tanh,
    var_trace, var_var,
};

// Shape operation exports (re-exported via autograd::ops::*)
pub use self::ops::{
    var_broadcast_to, var_cat, var_narrow, var_permute, var_reshape, var_transpose,
};

// Forward-mode exports
pub use dual::DualTensor;
pub use forward::{hvp, jacobian_forward, jvp, jvp_multi};
