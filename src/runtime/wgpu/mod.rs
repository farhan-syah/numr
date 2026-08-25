//! WebGPU runtime implementation
//!
//! This module provides cross-platform GPU acceleration via WebGPU.
//!
//! # Features
//!
//! - `WgpuDevice` - Represents a WebGPU GPU adapter
//! - `WgpuClient` - Manages device and queue, dispatches operations
//! - `WgpuRuntime` - Implements the generic Runtime trait
//! - `TensorOps` - WebGPU-accelerated tensor operations
//!
//! # Backend Support
//!
//! WebGPU abstracts over multiple GPU APIs:
//! - Vulkan (Linux, Windows, Android)
//! - Metal (macOS, iOS)
//! - DirectX 12 (Windows)
//! - OpenGL (fallback)
//!
//! # Panics
//!
//! The following operations may panic on WebGPU errors:
//!
//! - `Runtime::allocate` - Panics if buffer creation fails
//! - `Runtime::copy_to_device` - Panics if write operation fails
//! - `Runtime::copy_from_device` - Panics if read operation fails

mod cache;
pub(crate) mod client;
mod device;
mod fft;
mod fft_bluestein;
mod linalg;
mod ops;
mod polynomial;
mod runtime;
pub mod shaders;
#[cfg(feature = "sparse")]
mod sparse;
mod special;
mod statistics;

pub use crate::tensor::Tensor;
pub use client::{WgpuAllocator, WgpuClient, WgpuRawHandle, get_buffer};
pub use device::{WgpuDevice, WgpuError};
pub use runtime::{WgpuRuntime, is_wgpu_available, wgpu_device, wgpu_device_id};
