//! Operations on Var that build the computation graph
//!
//! These functions perform forward computation and create the appropriate
//! backward functions for gradient tracking.
//!
//! # Example
//!
//! ```
//! # use numr::prelude::*;
//! # use numr::autograd::{Var, var_mul, backward};
//! # let device = CpuDevice::new();
//! # let client = CpuRuntime::default_client(&device);
//! // Create leaf variables
//! let x = Var::new(Tensor::from_slice(&[2.0f32], &[1], &device)?, true);
//! let y = Var::new(Tensor::from_slice(&[3.0f32], &[1], &device)?, true);
//!
//! // Build computation graph: z = x * y
//! let z = var_mul(&x, &y, &client)?;
//!
//! // Compute gradients
//! let grads = backward(&z, &client)?;
//! # Ok::<(), numr::error::Error>(())
//! ```

mod macros;
pub mod ops;

mod activation;
mod arithmetic;
mod cast;
mod conv1d;
mod conv2d;
mod conv_common;
mod conv_transpose1d;
mod cumulative;
mod dropout;
mod fused_activation_mul;
mod gemm_epilogue;
mod indexing;

pub mod linalg;
mod matmul;
mod normalization;
pub mod reduce;
mod scalar;
mod stats;
mod swiglu;
mod unary;
mod utility;

// Re-export all public functions
pub use activation::{var_log_softmax, var_relu, var_sigmoid, var_silu, var_softmax, var_softplus};
pub use arithmetic::{var_add, var_div, var_mul, var_pow, var_sub};
pub use cast::var_cast;
pub use conv_transpose1d::var_conv_transpose1d;
pub use conv1d::var_conv1d;
pub use conv2d::var_conv2d;
pub use cumulative::{var_cumprod, var_cumsum};
pub use dropout::var_dropout;
pub use fused_activation_mul::{var_gelu_mul, var_relu_mul, var_sigmoid_mul, var_silu_mul};
pub use gemm_epilogue::var_matmul_bias_activation;
pub use indexing::{var_embedding_lookup, var_gather};
pub use linalg::{var_cholesky, var_det, var_inverse, var_solve, var_trace};
pub use matmul::var_matmul;
pub use normalization::{
    var_fused_add_layer_norm, var_fused_add_rms_norm, var_group_norm, var_layer_norm, var_rms_norm,
};
pub use reduce::{var_max, var_mean, var_min, var_sum};
pub use scalar::{var_add_scalar, var_div_scalar, var_mul_scalar, var_pow_scalar, var_sub_scalar};
pub use stats::{var_std, var_var};
pub use swiglu::var_swiglu;
pub use unary::{
    var_abs, var_cos, var_exp, var_log, var_neg, var_recip, var_sin, var_sqrt, var_square, var_tan,
    var_tanh,
};
pub use utility::var_clamp;
