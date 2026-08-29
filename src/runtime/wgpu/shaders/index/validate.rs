//! Index bounds validation kernel launcher.
//!
//! Runs ahead of a gather or scatter to count out-of-range indices on the GPU,
//! so the caller can raise an error without reading the index tensor back.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::error::Result;

const VALIDATE_INDICES_SHADER: &str = include_str!("../validate_indices.wgsl");

/// Launch index bounds validation kernel.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Returns the count of out-of-bounds indices in error_count buffer.
/// The error_count buffer must be initialized to 0 before calling.
pub fn launch_validate_indices(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    error_count: &Buffer,
    params_buffer: &Buffer,
    index_len: usize,
) -> Result<()> {
    if index_len == 0 {
        return Ok(());
    }

    let module = cache.get_or_create_module("validate_indices", VALIDATE_INDICES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("validate_indices", "validate_indices", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, error_count, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("validate_indices"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("validate_indices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(index_len), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
