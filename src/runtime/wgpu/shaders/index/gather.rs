//! Read-side index kernels: every launcher here collects elements out of an
//! input tensor at index-selected positions and writes a dense output.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::Result;

/// Launch an index_select operation kernel.
///
/// Selects elements from input along the specified dimension using indices.
/// Output shape is the same as input except the dimension size becomes index_len.
pub fn launch_index_select(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("index_select", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("index_select"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("index_select"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a gather operation kernel.
///
/// Gathers elements from input using indices along the specified dimension.
pub fn launch_gather(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("gather", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a gather_nd operation kernel.
///
/// Gathers slices from input using N-dimensional indices.
/// Input: input tensor, indices [num_slices, index_depth]
/// Output: output [num_slices, slice_size]
pub fn launch_gather_nd(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("gather_nd", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 0,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_nd"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_nd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch an embedding_lookup operation kernel.
///
/// Looks up embeddings from a 2D embedding table using indices.
/// Input: embeddings `[vocab_size, embedding_dim]`, indices `[num_indices]`
/// Output: output `[num_indices, embedding_dim]`
///
/// This is the industry-standard embedding lookup operation used in neural networks
/// for word embeddings, entity embeddings, etc.
pub fn launch_embedding_lookup(
    cache: &PipelineCache,
    queue: &Queue,
    embeddings: &Buffer,
    indices: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_indices: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("embedding_lookup", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[embeddings, indices, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("embedding_lookup"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("embedding_lookup"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_indices), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a gather_2d operation kernel.
///
/// Gathers elements from a 2D matrix at specific (row, col) positions.
/// Input: input `[nrows, ncols]`, rows `[num_indices]`, cols `[num_indices]`
/// Output: output `[num_indices]`
///
/// For each index i: `output[i] = input[rows[i], cols[i]]`
#[allow(clippy::too_many_arguments)]
pub fn launch_gather_2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    rows: &Buffer,
    cols: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_indices: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("gather_2d", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, rows, cols, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_indices), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
