//! Backward implementations for operations
//!
//! Each operation has a corresponding backward struct that implements
//! `GradFn` to compute gradients during the backward pass.
//!
//! # Structure
//!
//! - `arithmetic`: Binary operations (add, sub, mul, div, pow)
//! - `unary`: Unary operations (neg, exp, log, sqrt, tan, etc.)
//! - `matmul`: Matrix multiplication
//! - `reduce`: Reductions (sum, mean, max, min)
//! - `activation`: Activation functions (relu, sigmoid, softmax)
//! - `scalar`: Tensor-scalar operations (add_scalar, mul_scalar, etc.)
//! - `linalg`: Linear algebra operations (trace, inverse, det, solve, cholesky)
//! - `shape`: Shape operations (reshape, transpose, permute, broadcast)
//! - `broadcast`: Gradient reduction over broadcast dimensions

mod activation;
mod arithmetic;
mod broadcast;
mod cast;
mod common;
mod cumulative;
mod gemm_epilogue;
mod indexing;
mod linalg;
mod matmul;
mod normalization;
mod reduce;
mod scalar;
mod shape;
mod unary;

pub(crate) use common::ensure_contiguous;

pub use activation::*;
pub use arithmetic::*;
pub use cast::*;
pub use cumulative::*;
pub use gemm_epilogue::*;
pub use indexing::*;
pub use linalg::*;
pub use matmul::*;
pub use normalization::*;
pub use reduce::*;
pub use scalar::*;
pub use shape::*;
pub use unary::*;
