//! Kernels that fill a freshly allocated tensor from scratch: arange, linspace
//! and eye. Each produces F32, I32 or U32; the integer variants carry the extra
//! WGSL helpers that keep their arithmetic exact.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::Result;

/// Launch an arange operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing ArangeParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output
pub fn launch_arange(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("arange", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arange"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("arange"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a linspace operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing LinspaceParams
/// * `steps` - Number of steps to generate
/// * `dtype` - Data type of output (F32, I32 or U32)
pub fn launch_linspace(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    steps: usize,
    dtype: DType,
) -> Result<()> {
    if steps == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("linspace", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("linspace"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("linspace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(steps), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch an eye (identity matrix) operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing EyeParams
/// * `numel` - Total elements (n * m)
/// * `dtype` - Data type of output
pub fn launch_eye(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("eye", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eye"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
