//! GEMV-BT WGSL kernel launchers: `C[M,N] = A[M,K] @ B^T` where B is `[N,K]`.
//!
//! Avoids the GPU-side contiguous copy of transposed weight matrices by
//! reading B in its native `[N,K]` layout. Each output element is a dot product
//! of contiguous A and B row vectors, computed via parallel reduction.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const GEMV_BT_SHADER: &str = include_str!("gemv_bt.wgsl");

/// Launch 2D GEMV-BT kernel.
///
/// Computes `C[M,N] = A[M,K] @ B^T` where B is stored as `[N,K]` row-major.
/// Dispatch: (N, M, 1) workgroups, each with 256 threads for K-reduction.
pub fn launch_gemv_bt(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b_nk: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "gemv_bt",
        });
    }

    let module = cache.get_or_create_module("gemv_bt", GEMV_BT_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("gemv_bt", "gemv_bt_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b_nk, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemv_bt"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemv_bt"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(n as u32, m as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch batched GEMV-BT kernel.
///
/// Computes `C[b,M,N] = A[b,M,K] @ B[b]^T` where each `B[b]` is stored `[N,K]`.
pub fn launch_batched_gemv_bt(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b_nk: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "batched_gemv_bt",
        });
    }

    let module = cache.get_or_create_module("gemv_bt", GEMV_BT_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("gemv_bt", "batched_gemv_bt_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b_nk, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_gemv_bt"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_gemv_bt"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(n as u32, m as u32, batch_size as u32);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
