//! Runtime backends for tensor computation
//!
//! This module defines the `Runtime` trait and provides implementations
//! for different compute backends (CPU, CUDA, WebGPU).
//!
//! # Architecture
//!
//! ```text
//! Runtime (backend identity)
//! ├── Device (identifies a specific GPU/CPU)
//! ├── Client (dispatches operations, owns stream/queue)
//! ├── Allocator (memory management with freeze support)
//! └── RawHandle (escape hatch for custom kernels)
//! ```

mod captured_graph;
pub(crate) mod common;
mod communicator;
pub mod traits;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "wgpu")]
pub mod wgpu;

// Common re-exports
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) use common::AllocGuard;
pub(crate) use common::DefaultAllocator;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) use common::compute_contiguous_strides;
pub use common::{AllocationStats, Allocator, Graph, NoOpGraph, TrackingAllocator};
pub(crate) use common::{
    cap_ipow_exponent, compute_broadcast_shape, ensure_contiguous, normalize_dim,
    pow_scalar_output_dtype, validate_arange, validate_binary_dtypes, validate_eye,
};

// Communicator re-exports
#[cfg(feature = "distributed-gpu")]
pub use communicator::HierarchicalCommunicator;
#[cfg(feature = "distributed")]
pub use communicator::NexarNetCommunicator;
pub use communicator::{
    Communicator, CommunicatorGroup, NoOpCommunicator, ParallelDim, ReduceOp, StreamSyncOps,
};
#[cfg(feature = "nccl")]
pub use cuda::NcclCommunicator;

// CapturedGraph re-export
pub use captured_graph::CapturedGraph;

// Trait re-exports
pub use traits::{Device, DeviceArch, DeviceCaps, DeviceProfile, Runtime, RuntimeClient};
