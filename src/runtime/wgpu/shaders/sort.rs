//! Sort operation WGSL kernel launchers.
//!
//! dtype policy:
//! - sort, sort_values_only, argsort: F32 / I32 / U32
//! - topk, searchsorted: F32 only
//! - unique, unique_with_counts: F32 / I32 / U32
//! - nonzero, flat_to_multi_index: F32 / I32 / U32

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use super::sort_cmp::{sort_cmp_f32_wgsl, sort_rank_f32_wgsl};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders — sort ops (F32 / I32 / U32)
// ============================================================================

// The f32 shaders share one total order (NaN-greatest, -0.0 == +0.0), prepended
// from sort_cmp.rs so a change to the ordering cannot miss one of them.
const SORT_SHADER_F32: &str = concat!(
    sort_cmp_f32_wgsl!(),
    sort_rank_f32_wgsl!(),
    include_str!("sort_f32.wgsl")
);
const SORT_SHADER_I32: &str = include_str!("sort_i32.wgsl");
const SORT_SHADER_U32: &str = include_str!("sort_u32.wgsl");
const GLOBAL_SORT_SHADER: &str = include_str!("sort_global.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalSortParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    padded_size: u32,
    segment_count: u32,
    dtype_tag: u32,
    descending: u32,
    k: u32,
    j: u32,
    total_padded: u32,
    padding_0: u32,
    padding_1: u32,
}

// ============================================================================
// Static shaders — topk/searchsorted (F32 only)
// ============================================================================

const TOPK_SHADER_F32: &str = concat!(
    sort_cmp_f32_wgsl!(),
    sort_rank_f32_wgsl!(),
    include_str!("topk_f32.wgsl")
);
const SEARCHSORTED_SHADER_F32: &str =
    concat!(sort_cmp_f32_wgsl!(), include_str!("searchsorted_f32.wgsl"));

// ============================================================================
// Static shaders — data-movement ops (F32 / I32 / U32)
// ============================================================================

const COUNT_NONZERO_SHADER_F32: &str = include_str!("count_nonzero_f32.wgsl");
const COUNT_NONZERO_SHADER_I32: &str = include_str!("count_nonzero_i32.wgsl");
const COUNT_NONZERO_SHADER_U32: &str = include_str!("count_nonzero_u32.wgsl");

const GATHER_NONZERO_SHADER_F32: &str = include_str!("gather_nonzero_f32.wgsl");
const GATHER_NONZERO_SHADER_I32: &str = include_str!("gather_nonzero_i32.wgsl");
const GATHER_NONZERO_SHADER_U32: &str = include_str!("gather_nonzero_u32.wgsl");

const FLAT_TO_MULTI_INDEX_SHADER: &str = include_str!("flat_to_multi_index.wgsl");

const UNIQUE_WITH_COUNTS_SHADER_F32: &str = include_str!("unique_with_counts_f32.wgsl");
const UNIQUE_WITH_COUNTS_SHADER_I32: &str = include_str!("unique_with_counts_i32.wgsl");
const UNIQUE_WITH_COUNTS_SHADER_U32: &str = include_str!("unique_with_counts_u32.wgsl");

const COUNT_UNIQUE_SHADER_F32: &str = include_str!("count_unique_f32.wgsl");
const COUNT_UNIQUE_SHADER_I32: &str = include_str!("count_unique_i32.wgsl");
const COUNT_UNIQUE_SHADER_U32: &str = include_str!("count_unique_u32.wgsl");

const EXTRACT_UNIQUE_SHADER_F32: &str = include_str!("extract_unique_f32.wgsl");
const EXTRACT_UNIQUE_SHADER_I32: &str = include_str!("extract_unique_i32.wgsl");
const EXTRACT_UNIQUE_SHADER_U32: &str = include_str!("extract_unique_u32.wgsl");

// ============================================================================
// Helpers
// ============================================================================

/// Returns (shader, module_key, entry_point) for sort ops.
/// Supports F32/I32/U32 for sort/sort_values_only/argsort, F32 only for topk/searchsorted.
fn sort_math_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    match op {
        "sort" | "sort_values_only" | "argsort" => {
            let (shader, module_key, _suffix) = match dtype {
                DType::F32 => (SORT_SHADER_F32, "sort_f32", "f32"),
                DType::I32 => (SORT_SHADER_I32, "sort_i32", "i32"),
                DType::U32 => (SORT_SHADER_U32, "sort_u32", "u32"),
                _ => return Err(Error::UnsupportedDType { dtype, op }),
            };
            let entry_point: &'static str = match (op, dtype) {
                ("sort", DType::F32) => "sort_f32",
                ("sort", DType::I32) => "sort_i32",
                ("sort", DType::U32) => "sort_u32",
                ("sort_values_only", DType::F32) => "sort_values_only_f32",
                ("sort_values_only", DType::I32) => "sort_values_only_i32",
                ("sort_values_only", DType::U32) => "sort_values_only_u32",
                ("argsort", DType::F32) => "argsort_f32",
                ("argsort", DType::I32) => "argsort_i32",
                ("argsort", DType::U32) => "argsort_u32",
                _ => unreachable!(),
            };
            Ok((shader, module_key, entry_point))
        }
        "topk" => {
            if dtype != DType::F32 {
                return Err(Error::UnsupportedDType { dtype, op });
            }
            Ok((TOPK_SHADER_F32, "topk_f32", "topk_f32"))
        }
        "searchsorted" => {
            if dtype != DType::F32 {
                return Err(Error::UnsupportedDType { dtype, op });
            }
            Ok((
                SEARCHSORTED_SHADER_F32,
                "searchsorted_f32",
                "searchsorted_f32",
            ))
        }
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Returns (shader, module_key, entry_point) for data-movement ops. F32/I32/U32.
fn sort_data_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match (op, dtype) {
        ("count_nonzero", DType::F32) => (
            COUNT_NONZERO_SHADER_F32,
            "count_nonzero_f32",
            "count_nonzero_f32",
        ),
        ("count_nonzero", DType::I32) => (
            COUNT_NONZERO_SHADER_I32,
            "count_nonzero_i32",
            "count_nonzero_i32",
        ),
        ("count_nonzero", DType::U32) => (
            COUNT_NONZERO_SHADER_U32,
            "count_nonzero_u32",
            "count_nonzero_u32",
        ),
        ("gather_nonzero", DType::F32) => (
            GATHER_NONZERO_SHADER_F32,
            "gather_nonzero_f32",
            "gather_nonzero_f32",
        ),
        ("gather_nonzero", DType::I32) => (
            GATHER_NONZERO_SHADER_I32,
            "gather_nonzero_i32",
            "gather_nonzero_i32",
        ),
        ("gather_nonzero", DType::U32) => (
            GATHER_NONZERO_SHADER_U32,
            "gather_nonzero_u32",
            "gather_nonzero_u32",
        ),
        ("unique_with_counts", DType::F32) => (
            UNIQUE_WITH_COUNTS_SHADER_F32,
            "unique_with_counts_f32",
            "mark_boundaries_f32",
        ),
        ("unique_with_counts", DType::I32) => (
            UNIQUE_WITH_COUNTS_SHADER_I32,
            "unique_with_counts_i32",
            "mark_boundaries_i32",
        ),
        ("unique_with_counts", DType::U32) => (
            UNIQUE_WITH_COUNTS_SHADER_U32,
            "unique_with_counts_u32",
            "mark_boundaries_u32",
        ),
        ("scatter_unique_with_counts", DType::F32) => (
            UNIQUE_WITH_COUNTS_SHADER_F32,
            "unique_with_counts_f32",
            "scatter_unique_with_counts_f32",
        ),
        ("scatter_unique_with_counts", DType::I32) => (
            UNIQUE_WITH_COUNTS_SHADER_I32,
            "unique_with_counts_i32",
            "scatter_unique_with_counts_i32",
        ),
        ("scatter_unique_with_counts", DType::U32) => (
            UNIQUE_WITH_COUNTS_SHADER_U32,
            "unique_with_counts_u32",
            "scatter_unique_with_counts_u32",
        ),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    })
}

fn check_data_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// Sort Operations
// ============================================================================

/// Launch the global-memory stable bitonic path used for sort dimensions above 512.
#[allow(clippy::too_many_arguments)]
pub fn launch_global_sort(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    values_output: Option<&Buffer>,
    indices_output: Option<&Buffer>,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
    dtype: DType,
) -> Result<()> {
    let dtype_tag = match dtype {
        DType::U32 => 0,
        DType::I32 => 1,
        DType::F32 => 2,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "global_sort",
            });
        }
    };
    let padded_size = sort_size.checked_next_power_of_two().ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "sort dimension is too large")
    })?;
    let segment_count = outer_size.checked_mul(inner_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "segment count overflows usize")
    })?;
    let total_padded = segment_count.checked_mul(padded_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "global workspace size overflows usize")
    })?;
    let logical_total = segment_count.checked_mul(sort_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "output size overflows usize")
    })?;

    let outer_size_u32 = u32::try_from(outer_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "outer dimension exceeds u32"))?;
    let sort_size_u32 = u32::try_from(sort_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "sort dimension exceeds u32"))?;
    let inner_size_u32 = u32::try_from(inner_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "inner dimension exceeds u32"))?;
    let padded_size_u32 = u32::try_from(padded_size).map_err(|_| {
        Error::backend_limitation("WebGPU", "sort", "padded sort dimension exceeds u32")
    })?;
    let segment_count_u32 = u32::try_from(segment_count)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "segment count exceeds u32"))?;
    let total_padded_u32 = u32::try_from(total_padded).map_err(|_| {
        Error::backend_limitation("WebGPU", "sort", "global workspace exceeds u32 elements")
    })?;
    let _logical_total_u32 = u32::try_from(logical_total)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "output exceeds u32 elements"))?;

    let scratch_bytes = (total_padded as u64).checked_mul(4).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "global workspace byte size overflows")
    })?;
    let limits = cache.device().limits();
    let binding_limit = limits.max_storage_buffer_binding_size;
    let allocation_limit = limits.max_buffer_size;
    let effective_limit = binding_limit.min(allocation_limit);
    if scratch_bytes > effective_limit {
        return Err(Error::backend_limitation(
            "WebGPU",
            "sort",
            format!(
                "global workspace binding requires {scratch_bytes} bytes, device limit is {effective_limit}"
            ),
        ));
    }

    let make_storage = |label: &'static str, size: u64| {
        cache.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let keys = make_storage("global_sort_keys", scratch_bytes);
    let values = make_storage("global_sort_values", scratch_bytes);
    let indices = make_storage("global_sort_indices", scratch_bytes);
    let step_dummy = make_storage("global_sort_step_dummy", 4);
    let logical_bytes = (logical_total as u64) * 4;
    let temporary_values = values_output
        .is_none()
        .then(|| make_storage("global_sort_temporary_values_output", logical_bytes));
    let temporary_indices = indices_output
        .is_none()
        .then(|| make_storage("global_sort_temporary_indices_output", logical_bytes));
    let values_output = values_output
        .or(temporary_values.as_ref())
        .expect("output exists");
    let indices_output = indices_output
        .or(temporary_indices.as_ref())
        .expect("output exists");

    let base_params = GlobalSortParams {
        outer_size: outer_size_u32,
        sort_size: sort_size_u32,
        inner_size: inner_size_u32,
        padded_size: padded_size_u32,
        segment_count: segment_count_u32,
        dtype_tag,
        descending: u32::from(descending),
        k: 0,
        j: 0,
        total_padded: total_padded_u32,
        padding_0: 0,
        padding_1: 0,
    };
    let mut stage_params = vec![base_params];
    // k <= 512 is fused into one shared-memory tile dispatch.
    let mut k = 1024u32;
    while k <= padded_size_u32 {
        let mut j = k >> 1;
        while j > 0 {
            stage_params.push(GlobalSortParams {
                k,
                j,
                ..base_params
            });
            j >>= 1;
        }
        k = k.checked_shl(1).unwrap_or(0);
        if k == 0 {
            break;
        }
    }

    let param_size = std::mem::size_of::<GlobalSortParams>();
    let alignment = limits.min_uniform_buffer_offset_alignment as usize;
    let stride = param_size.div_ceil(alignment) * alignment;
    let params_bytes_len = stride.checked_mul(stage_params.len()).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "parameter buffer size overflows")
    })?;
    let mut params_bytes = vec![0u8; params_bytes_len];
    for (stage_index, params) in stage_params.iter().enumerate() {
        let bytes = bytemuck::bytes_of(params);
        let offset = stage_index * stride;
        params_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("global_sort_params"),
        size: params_bytes_len as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, &params_bytes);

    let module = cache.get_or_create_module("global_sort", GLOBAL_SORT_SHADER);
    let layout = cache.get_or_create_dynamic_uniform_layout(4, 0);
    let pack_pipeline =
        cache.get_or_create_pipeline("global_sort_pack", "pack_global_sort", &module, &layout);
    let tile_pipeline =
        cache.get_or_create_pipeline("global_sort_tiles", "sort_global_tiles", &module, &layout);
    let step_pipeline =
        cache.get_or_create_pipeline("global_sort_step", "global_bitonic_step", &module, &layout);
    let scatter_pipeline = cache.get_or_create_pipeline(
        "global_sort_scatter",
        "scatter_global_sort",
        &module,
        &layout,
    );
    let uniform_binding_size = param_size as u64;
    let pack_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[input, &keys, &values, &indices],
        &params_buffer,
        uniform_binding_size,
    );
    let step_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[&keys, &values, &indices, &step_dummy],
        &params_buffer,
        uniform_binding_size,
    );
    let scatter_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[&values, &indices, values_output, indices_output],
        &params_buffer,
        uniform_binding_size,
    );

    let dispatch_grid = |items: usize| -> Result<(u32, u32)> {
        let groups = items.div_ceil(256);
        let x = groups.clamp(1, 65_535);
        let y = groups.div_ceil(x);
        if y > 65_535 {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort",
                "global sort dispatch exceeds WebGPU's 2-D dispatch grid",
            ));
        }
        Ok((x as u32, y as u32))
    };
    let (padded_x, padded_y) = dispatch_grid(total_padded)?;
    let (logical_x, logical_y) = dispatch_grid(logical_total)?;
    let tile_groups = total_padded / 512;
    let tile_x = tile_groups.clamp(1, 65_535);
    let tile_y = tile_groups.div_ceil(tile_x);
    if tile_y > 65_535 {
        return Err(Error::backend_limitation(
            "WebGPU",
            "sort",
            "global sort tile dispatch exceeds WebGPU's 2-D dispatch grid",
        ));
    }
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("global_sort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_pack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pack_pipeline);
        pass.set_bind_group(0, Some(&pack_bind_group), &[0]);
        pass.dispatch_workgroups(padded_x, padded_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_tiles"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipeline);
        pass.set_bind_group(0, Some(&step_bind_group), &[0]);
        pass.dispatch_workgroups(tile_x as u32, tile_y as u32, 1);
    }
    for stage_index in 1..stage_params.len() {
        let dynamic_offset = u32::try_from(stage_index * stride).map_err(|_| {
            Error::backend_limitation("WebGPU", "sort", "dynamic uniform offset exceeds u32")
        })?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_step"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&step_pipeline);
        pass.set_bind_group(0, Some(&step_bind_group), &[dynamic_offset]);
        pass.dispatch_workgroups(padded_x, padded_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scatter_pipeline);
        pass.set_bind_group(0, Some(&scatter_bind_group), &[0]);
        pass.dispatch_workgroups(logical_x, logical_y, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sort with indices kernel
pub fn launch_sort(
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
    let (shader, module_key, entry_point) = sort_math_info("sort", dtype)?;

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
            label: Some("sort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch sort values only kernel (no indices)
pub fn launch_sort_values_only(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = sort_math_info("sort_values_only", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    // Need a 4-buffer layout but only use 3 (input, output, dummy_indices, params)
    // Actually for values_only we need different layout
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    // Create dummy indices buffer for the binding
    let dummy_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_indices"),
        size: 4, // minimum
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group = cache.create_bind_group(&layout, &[input, output, &dummy_buf, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("sort_values_only"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sort_values_only"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(outer_size as u32, inner_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch argsort kernel (indices only)
pub fn launch_argsort(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices_output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    inner_size: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point) = sort_math_info("argsort", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    // Create dummy values buffer for the binding
    let dummy_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("dummy_values"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let bind_group =
        cache.create_bind_group(&layout, &[input, &dummy_buf, indices_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("argsort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("argsort"),
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
// Topk Operations
// ============================================================================

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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "topk (WebGPU)",
        });
    }

    let (shader, module_key, entry_point) = sort_math_info("topk", dtype)?;

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
// Searchsorted Operations
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
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "searchsorted (WebGPU)",
        });
    }

    let (shader, module_key, entry_point) = sort_math_info("searchsorted", dtype)?;

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

// ============================================================================
// Nonzero Operations (Two-phase)
// ============================================================================

/// Launch count_nonzero kernel (phase 1)
pub fn launch_count_nonzero(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_data_dtype(dtype, "count_nonzero")?;

    let (shader, module_key, entry_point) = sort_data_info("count_nonzero", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, count_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_nonzero"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_nonzero"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch gather_nonzero kernel (phase 2)
pub fn launch_gather_nonzero(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    indices_output: &Buffer,
    counter: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_data_dtype(dtype, "gather_nonzero")?;

    let (shader, module_key, entry_point) = sort_data_info("gather_nonzero", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, indices_output, counter, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gather_nonzero"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gather_nonzero"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch flat_to_multi_index kernel
pub fn launch_flat_to_multi_index(
    cache: &PipelineCache,
    queue: &Queue,
    flat_indices: &Buffer,
    multi_indices: &Buffer,
    params_buffer: &Buffer,
    nnz: usize,
) -> Result<()> {
    let module = cache.get_or_create_module("flat_to_multi_index", FLAT_TO_MULTI_INDEX_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "flat_to_multi_index",
        "flat_to_multi_index",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[flat_indices, multi_indices, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("flat_to_multi_index"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("flat_to_multi_index"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(nnz), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Unique Operations (Two-phase)
// ============================================================================

/// Launch count_unique kernel (phase 1 - on sorted input)
pub fn launch_count_unique(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    count_output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => (
            "count_unique_f32",
            COUNT_UNIQUE_SHADER_F32,
            "count_unique_f32",
        ),
        DType::I32 => (
            "count_unique_i32",
            COUNT_UNIQUE_SHADER_I32,
            "count_unique_i32",
        ),
        DType::U32 => (
            "count_unique_u32",
            COUNT_UNIQUE_SHADER_U32,
            "count_unique_u32",
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "count_unique",
            });
        }
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[sorted_input, count_output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("count_unique"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("count_unique"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch extract_unique kernel (phase 2 - on sorted input)
pub fn launch_extract_unique(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    unique_output: &Buffer,
    counter: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => (
            "extract_unique_f32",
            EXTRACT_UNIQUE_SHADER_F32,
            "extract_unique_f32",
        ),
        DType::I32 => (
            "extract_unique_i32",
            EXTRACT_UNIQUE_SHADER_I32,
            "extract_unique_i32",
        ),
        DType::U32 => (
            "extract_unique_u32",
            EXTRACT_UNIQUE_SHADER_U32,
            "extract_unique_u32",
        ),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "extract_unique",
            });
        }
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[sorted_input, unique_output, counter, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("extract_unique"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("extract_unique"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Unique With Counts Operations (Multi-phase)
// ============================================================================

/// Launch mark_boundaries kernel (marks where value changes in sorted array)
pub fn launch_mark_boundaries(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    boundary_flags: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_data_dtype(dtype, "unique_with_counts")?;

    let (shader, module_key, entry_point) = sort_data_info("unique_with_counts", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[sorted_input, boundary_flags, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("mark_boundaries"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("mark_boundaries"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch scatter_unique_with_counts kernel
pub fn launch_scatter_unique_with_counts(
    cache: &PipelineCache,
    queue: &Queue,
    sorted_input: &Buffer,
    prefix_sum: &Buffer,
    unique_output: &Buffer,
    inverse_indices: &Buffer,
    counts_output: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    check_data_dtype(dtype, "unique_with_counts")?;

    let (shader, module_key, entry_point) = sort_data_info("scatter_unique_with_counts", dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            sorted_input,
            prefix_sum,
            unique_output,
            inverse_indices,
            counts_output,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("scatter_unique_with_counts"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scatter_unique_with_counts"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
