//! Boolean-mask kernels: masked_fill writes in place, while masked_select runs
//! in three phases (count, exclusive prefix sum, gather) because the output
//! length is only known once the mask has been counted.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::Result;

/// Launch a masked_fill operation kernel.
///
/// Fills elements in output with fill_value where mask is non-zero.
pub fn launch_masked_fill(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    mask: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("masked_fill", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, mask, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_fill"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_fill"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch the masked_count phase of masked_select.
///
/// Counts the number of elements where mask is non-zero using atomic operations.
pub fn launch_masked_count(
    cache: &PipelineCache,
    queue: &Queue,
    mask: &Buffer,
    count_result: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("masked_count", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);

    // For count: mask (read), count_result (atomic), params
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[mask, count_result, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch the masked_prefix_sum phase of masked_select.
///
/// Computes exclusive prefix sum of mask for determining output positions.
pub fn launch_masked_prefix_sum(
    cache: &PipelineCache,
    queue: &Queue,
    mask: &Buffer,
    prefix_sum: &Buffer,
    params_buffer: &Buffer,
    _numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("masked_prefix_sum", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[mask, prefix_sum, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_prefix_sum"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_prefix_sum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Sequential kernel - only 1 workgroup needed
        pass.dispatch_workgroups(1, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch the masked_select gather phase.
///
/// Gathers elements from input where mask is non-zero into output.
pub fn launch_masked_select(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    mask: &Buffer,
    prefix_sum: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("masked_select", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);

    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, mask, prefix_sum, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("masked_select"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("masked_select"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
