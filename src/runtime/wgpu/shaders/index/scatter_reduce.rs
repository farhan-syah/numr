//! Combining scatter kernels: values landing on the same destination position
//! are folded with sum, prod, max, min or mean instead of overwriting.
//!
//! Mean runs as three dispatches -- sum, count, then divide -- because WGSL has
//! no atomic that can average in one pass.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::shader_registry::shader_info;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Launch a scatter_reduce operation kernel.
///
/// Scatters values with reduction (sum, max, min).
/// Uses atomic operations for thread-safe accumulation.
pub fn launch_scatter_reduce(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
    op: &str,
) -> Result<()> {
    // Get static kernel name based on op type
    let op_name: &'static str = match op {
        "sum" => "scatter_reduce_sum",
        "max" => "scatter_reduce_max",
        "min" => "scatter_reduce_min",
        _ => {
            return Err(Error::InvalidArgument {
                arg: "op",
                reason: format!("scatter_reduce op must be sum, max, or min, got {}", op),
            });
        }
    };

    let (shader, module_key, entry_point) = shader_info(op_name, dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, indices, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_reduce"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a scatter_reduce_prod operation kernel.
///
/// `items` is the number of threads to dispatch, which differs by dtype: the
/// float kernel owns one SOURCE element and combines with an atomic, while the
/// integer kernels own one DESTINATION element and scan their own lane (see
/// scatter_reduce_prod_i32.wgsl). Callers pass the source element count for
/// F32 and the destination element count for I32 and U32.
pub fn launch_scatter_reduce_prod(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    items: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("scatter_reduce_prod", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, indices, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce_prod"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_reduce_prod"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(items), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a scatter_reduce_count operation kernel.
///
/// Atomically counts scattered elements per destination position.
pub fn launch_scatter_reduce_count(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    count: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("scatter_reduce_count", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, count, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce_count"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_reduce_count"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_src), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch scatter_reduce_mean_div: `output[i] = sum[i] / count[i]`.
pub fn launch_scatter_reduce_mean_div(
    cache: &PipelineCache,
    queue: &Queue,
    sum_buf: &Buffer,
    count_buf: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = shader_info("scatter_reduce_mean_div", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[sum_buf, count_buf, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_reduce_mean_div"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_reduce_mean_div"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
