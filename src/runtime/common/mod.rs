pub(crate) mod helpers;
pub(crate) mod shape_ops;
pub(crate) mod statistics_common;

mod allocator;
mod graph;

#[cfg(feature = "sparse")]
pub(crate) mod sparse_utils;

// Allocator re-exports
#[cfg(any(feature = "cuda", feature = "wgpu"))]
pub(crate) use allocator::AllocGuard;
pub(crate) use allocator::DefaultAllocator;
pub use allocator::{AllocationStats, Allocator, TrackingAllocator};

// Graph re-exports
pub use graph::{Graph, NoOpGraph};

// Helper re-exports
pub(crate) use helpers::{
    cap_ipow_exponent, compute_broadcast_shape, ensure_contiguous, normalize_dim,
    pow_scalar_output_dtype, validate_arange, validate_binary_dtypes, validate_eye,
};

/// Compute contiguous (row-major) strides for a given shape.
#[cfg(any(feature = "cuda", feature = "wgpu"))]
#[inline]
pub(crate) fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
