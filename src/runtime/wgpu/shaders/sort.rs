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
