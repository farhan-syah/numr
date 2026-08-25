//! CUDA runtime implementation
//!
//! This module provides GPU acceleration via NVIDIA CUDA using cudarc.
//!
//! # Features
//!
//! - `CudaDevice` - Represents a CUDA GPU device
//! - `CudaClient` - Manages GPU stream and context, launches kernels
//! - `CudaRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - CUDA-accelerated tensor operations using cuBLAS
//!
//! # Panics
//!
//! The following operations may panic on CUDA errors (allocation failures are
//! typically unrecoverable in GPU contexts):
//!
//! - `Runtime::allocate` - Panics if CUDA memory allocation fails
//! - `Runtime::copy_to_device` - Panics if host-to-device copy fails
//! - `Runtime::copy_from_device` - Panics if device-to-host copy fails
//! - `Runtime::copy_within_device` - Panics if device-to-device copy fails
//!
//! These panics follow CUDA best practices where allocation failures indicate
//! an unrecoverable out-of-memory condition.

mod allocator;
mod arena;
mod cache;
mod client;
#[cfg(feature = "nccl")]
mod communicator;
mod device;
mod env_config;
mod fft;
mod fft_bluestein;
mod graph;
pub mod kernels;
mod linalg;
mod ops;
mod polynomial;
mod runtime;
mod sobol_cache;
#[cfg(feature = "sparse")]
mod sparse;
mod special;

pub use crate::tensor::Tensor;
pub use allocator::CudaAllocator;
pub use client::{CudaClient, CudaRawHandle};
#[cfg(feature = "nccl")]
pub use communicator::NcclCommunicator;
pub use device::{CudaDevice, CudaError};
pub use graph::CudaGraph;
pub use runtime::{CudaRuntime, cuda_device, cuda_device_id, is_cuda_available};
