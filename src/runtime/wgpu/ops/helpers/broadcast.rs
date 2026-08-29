//! Broadcast stride computation and dispatch.
//!
//! Broadcast dimensions get stride 0, so one strided kernel covers every
//! NumPy-style shape pairing. Arithmetic and compare share these buffers
//! because their broadcast shaders take the same layout.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::wgpu::WgpuClient;

use super::buffer::{create_params_buffer, create_storage_buffer};
use super::elementwise_params::BroadcastBinaryParams;

/// Compute broadcast strides for an input tensor relative to an output shape.
///
/// For each dimension in the output shape:
/// - If the input dimension matches, use the original stride
/// - If the input dimension is 1 (broadcast), use stride 0
/// - If the input doesn't have this dimension (prepended), use stride 0
pub fn compute_broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![0u32; output_shape.len()];
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    // Compute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Map input dimensions to output dimensions (right-aligned)
    let offset = output_ndim - input_ndim;
    for i in 0..output_ndim {
        if i < offset {
            // Dimension doesn't exist in input, broadcast with stride 0
            strides[i] = 0;
        } else {
            let input_idx = i - offset;
            if input_shape[input_idx] == 1 {
                // Broadcasting dimension, stride 0
                strides[i] = 0;
            } else {
                // Normal dimension, use input stride
                strides[i] = input_strides[input_idx] as u32;
            }
        }
    }

    strides
}

/// Stride and params buffers for a broadcast binary-shaped dispatch (shared by
/// arithmetic and compare, whose broadcast shaders take the same buffer layout).
pub(crate) struct BroadcastBuffers {
    pub(crate) a_strides: wgpu::Buffer,
    pub(crate) b_strides: wgpu::Buffer,
    pub(crate) out_strides: wgpu::Buffer,
    pub(crate) params: wgpu::Buffer,
}

/// Build the stride and params buffers for a broadcast dispatch over `out_shape`.
pub(crate) fn broadcast_buffers(
    client: &WgpuClient,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    numel: usize,
) -> BroadcastBuffers {
    let ndim = out_shape.len();

    // Broadcast dimensions get stride 0.
    let a_strides = compute_broadcast_strides(a_shape, out_shape);
    let b_strides = compute_broadcast_strides(b_shape, out_shape);

    // Output strides are row-major.
    let mut out_strides = vec![1u32; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1] as u32;
    }

    let params = BroadcastBinaryParams {
        numel: numel as u32,
        ndim: ndim as u32,
    };

    BroadcastBuffers {
        a_strides: create_storage_buffer(client, &a_strides),
        b_strides: create_storage_buffer(client, &b_strides),
        out_strides: create_storage_buffer(client, &out_strides),
        params: create_params_buffer(client, &params),
    }
}

/// Launch a broadcast dispatch of `op`, shared by arithmetic and compare: with
/// the compare mask carrying the input dtype, both call the same broadcast
/// shader entry points through the same buffer layout.
#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_broadcast(
    client: &WgpuClient,
    op: &'static str,
    a_buf: &wgpu::Buffer,
    b_buf: &wgpu::Buffer,
    out_buf: &wgpu::Buffer,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let bufs = broadcast_buffers(client, a_shape, b_shape, out_shape, numel);

    crate::runtime::wgpu::shaders::elementwise::launch_broadcast_binary_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        a_buf,
        b_buf,
        out_buf,
        &bufs.a_strides,
        &bufs.b_strides,
        &bufs.out_strides,
        &bufs.params,
        numel,
        dtype,
    )
}
