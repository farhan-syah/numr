//! Data-movement kernels that rearrange existing elements: cat, repeat, pad and
//! roll. Every one supports F32, I32 and U32 because none of them does math on
//! the values it copies.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::Result;

/// Launch a cat_copy operation kernel.
///
/// Copies data from a source tensor to the appropriate position in the
/// concatenated output tensor. This is called once per input tensor.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer (must be contiguous)
/// * `dst` - Destination tensor buffer (output)
/// * `params_buffer` - Uniform buffer containing CatParams
/// * `total_elements` - Total elements in source tensor
/// * `dtype` - Data type of tensors
#[allow(clippy::too_many_arguments)]
pub fn launch_cat_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("cat_copy", dtype)?;
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
            label: Some("cat_copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cat_copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a repeat operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing RepeatParams
/// * `total_elements` - Total elements in output tensor
/// * `dtype` - Data type of tensors
pub fn launch_repeat(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("repeat", dtype)?;
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
            label: Some("repeat"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repeat"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a pad operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing PadParams
/// * `total_elements` - Total elements in output tensor
/// * `dtype` - Data type of tensors
pub fn launch_pad(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("pad", dtype)?;
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
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pad") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pad"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a roll operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing RollParams
/// * `total_elements` - Total elements in tensor
/// * `dtype` - Data type of tensors
pub fn launch_roll(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("roll", dtype)?;
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
            label: Some("roll"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("roll"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
