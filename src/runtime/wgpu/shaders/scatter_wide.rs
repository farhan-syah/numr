//! Wide-accumulator integer scatter launchers. I32, U32.
//!
//! Split from `index.rs`, which was already well past this crate's file-size
//! limit. Three kernels run in sequence for one integer `scatter_reduce` sum or
//! mean: seed the 64-bit accumulator from the destination, add every source
//! element into it atomically, then narrow (dividing first, for mean).
//!
//! See scatter_wide_i32.wgsl for why the accumulator has to be wider than the
//! element type.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const SEED_SHADER_I32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("scatter_wide_seed_i32.wgsl"),
);
const SEED_SHADER_U32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("scatter_wide_seed_u32.wgsl"),
);
const WIDE_SHADER_I32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("int_wide_div.wgsl"),
    include_str!("scatter_wide_i32.wgsl"),
);
const WIDE_SHADER_U32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("int_wide_div.wgsl"),
    include_str!("scatter_wide_u32.wgsl"),
);

/// Params for the seed and finalize kernels (matches ScatterWideParams in WGSL).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScatterWideParams {
    /// Destination element count.
    pub n: u32,
    /// Non-zero for `mean`, which divides by the per-slot contribution count.
    pub divide: u32,
    /// Padding to the 16-byte uniform-buffer alignment WGSL requires.
    pub _pad0: u32,
    /// Padding to the 16-byte uniform-buffer alignment WGSL requires.
    pub _pad1: u32,
}

/// Returns (shader, module_key, entry_point) for `op` at `dtype`.
fn wide_shader_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    match (op, dtype) {
        ("seed", DType::I32) => Ok((
            SEED_SHADER_I32,
            "scatter_wide_seed_i32",
            "scatter_wide_seed_i32",
        )),
        ("seed", DType::U32) => Ok((
            SEED_SHADER_U32,
            "scatter_wide_seed_u32",
            "scatter_wide_seed_u32",
        )),
        ("sum", DType::I32) => Ok((WIDE_SHADER_I32, "scatter_wide_i32", "scatter_wide_sum_i32")),
        ("sum", DType::U32) => Ok((WIDE_SHADER_U32, "scatter_wide_u32", "scatter_wide_sum_u32")),
        ("finalize", DType::I32) => Ok((
            WIDE_SHADER_I32,
            "scatter_wide_i32",
            "scatter_wide_finalize_i32",
        )),
        ("finalize", DType::U32) => Ok((
            WIDE_SHADER_U32,
            "scatter_wide_u32",
            "scatter_wide_finalize_u32",
        )),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "scatter_reduce (wide integer accumulator)",
        }),
    }
}

/// Run one kernel of the sequence over `buffers`, one workgroup per 256 items.
fn dispatch(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    label: &'static str,
    buffers: &[&Buffer],
    num_storage_buffers: u32,
    items: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = wide_shader_info(op, dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, buffers);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(items), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Widen `seed` (the already-seeded destination) into the 64-bit accumulator.
pub fn launch_scatter_wide_seed(
    cache: &PipelineCache,
    queue: &Queue,
    seed: &Buffer,
    acc: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "seed",
        "scatter_wide_seed",
        &[seed, acc, params_buffer],
        2,
        numel,
        dtype,
    )
}

/// Add every source element into the 64-bit accumulator.
pub fn launch_scatter_wide_sum(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    indices: &Buffer,
    acc: &Buffer,
    params_buffer: &Buffer,
    total_src: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "sum",
        "scatter_wide_sum",
        &[src, indices, acc, params_buffer],
        3,
        total_src,
        dtype,
    )
}

/// Narrow the accumulator into `out`, dividing by `count` first for `mean`.
pub fn launch_scatter_wide_finalize(
    cache: &PipelineCache,
    queue: &Queue,
    acc: &Buffer,
    count: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    dispatch(
        cache,
        queue,
        "finalize",
        "scatter_wide_finalize",
        &[acc, count, out, params_buffer],
        3,
        numel,
        dtype,
    )
}
