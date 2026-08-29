//! Write-side index kernels: every launcher here overwrites positions of an
//! output tensor that already holds a copy of the destination data.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::Result;

/// Launch a copy operation kernel (for scatter initialization).
pub fn launch_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("copy", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a scatter operation kernel.
///
/// Scatters values from src to output at positions specified by indices along dim.
pub fn launch_scatter(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    src_total: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("scatter", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(src_total), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch an index_put operation kernel.
///
/// Puts values from src at positions specified by indices along the dimension.
/// Output should be pre-initialized with a copy of the input tensor.
pub fn launch_index_put(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    src: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("index_put", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, src, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_put"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("index_put"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a slice_assign operation kernel.
///
/// Overwrites a slice of the output tensor with src values along a dimension.
/// Output should already contain a copy of dst data.
pub fn launch_slice_assign(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("slice_assign", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("slice_assign"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("slice_assign"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
