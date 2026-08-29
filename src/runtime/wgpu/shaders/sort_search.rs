//! Top-k and searchsorted WGSL kernel launchers. F32, I32, U32.
//!
//! Split from `sort.rs`, which was already past this crate's file-size limit.
//! Both ops read a sorted or sortable run, so they share the ordering helpers
//! that `sort_cmp.rs` also gives the sort shaders.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::sort_cmp::{
    sort_cmp_f32_wgsl, sort_rank_f32_wgsl, sort_rank_i32_wgsl, sort_rank_u32_wgsl,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

const TOPK_SHADER_F32: &str = concat!(
    sort_cmp_f32_wgsl!(),
    sort_rank_f32_wgsl!(),
    include_str!("topk_f32.wgsl")
);
const TOPK_SHADER_I32: &str = concat!(sort_rank_i32_wgsl!(), include_str!("topk_i32.wgsl"));
const TOPK_SHADER_U32: &str = concat!(sort_rank_u32_wgsl!(), include_str!("topk_u32.wgsl"));

const SEARCHSORTED_SHADER_F32: &str =
    concat!(sort_cmp_f32_wgsl!(), include_str!("searchsorted_f32.wgsl"));
const SEARCHSORTED_SHADER_I32: &str = include_str!("searchsorted_i32.wgsl");
const SEARCHSORTED_SHADER_U32: &str = include_str!("searchsorted_u32.wgsl");

/// Returns (shader, module_key, entry_point) for `op` at `dtype`.
fn search_shader_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    match (op, dtype) {
        ("topk", DType::F32) => Ok((TOPK_SHADER_F32, "topk_f32", "topk_f32")),
        ("topk", DType::I32) => Ok((TOPK_SHADER_I32, "topk_i32", "topk_i32")),
        ("topk", DType::U32) => Ok((TOPK_SHADER_U32, "topk_u32", "topk_u32")),
        ("searchsorted", DType::F32) => Ok((
            SEARCHSORTED_SHADER_F32,
            "searchsorted_f32",
            "searchsorted_f32",
        )),
        ("searchsorted", DType::I32) => Ok((
            SEARCHSORTED_SHADER_I32,
            "searchsorted_i32",
            "searchsorted_i32",
        )),
        ("searchsorted", DType::U32) => Ok((
            SEARCHSORTED_SHADER_U32,
            "searchsorted_u32",
            "searchsorted_u32",
        )),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Launch topk kernel
pub fn launch_topk(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    values_output: &Buffer,
    indices_output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = search_shader_info("topk", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[input, values_output, indices_output, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("topk"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("topk"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Searchsorted
// ============================================================================

/// Launch searchsorted kernel
pub fn launch_searchsorted(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_seq: &Buffer,
    values: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    num_values: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = search_shader_info("searchsorted", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[sorted_seq, values, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("searchsorted"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("searchsorted"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(num_values), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
