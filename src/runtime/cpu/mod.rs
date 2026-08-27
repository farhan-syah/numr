//! CPU runtime implementation
//!
//! The CPU runtime uses standard heap allocation and provides a reference
//! implementation for all tensor operations.
//!
//! # Broadcasting
//!
//! NumPy-style broadcasting is fully supported for binary arithmetic operations
//! (add, sub, mul, div, pow, max, min). Shapes are broadcast according to standard
//! rules: dimensions are right-aligned and expanded where one operand has size 1.
//!
//! Comparison operations (eq, ne, lt, le, gt, ge) also support broadcasting.
//!
//! # Non-contiguous Tensors
//!
//! Operations handle non-contiguous tensors via strided memory access. For
//! broadcasting, a strided kernel is used that correctly handles stride-0
//! dimensions (where a single value is broadcast across the dimension).

mod client;
mod device;
mod fft;
pub(crate) mod helpers;
pub mod jacobi;
mod kernel;
pub(crate) mod kernels;
mod linalg;
mod ops;
mod polynomial;
mod runtime;
pub(crate) mod sort;
#[cfg(feature = "sparse")]
pub(crate) mod sparse;
pub mod special;
pub(crate) mod statistics;
mod strided_copy;

pub use crate::tensor::Tensor;
pub use client::{CpuAllocator, CpuClient, ParallelismConfig};
pub use device::CpuDevice;
pub use runtime::CpuRuntime;
