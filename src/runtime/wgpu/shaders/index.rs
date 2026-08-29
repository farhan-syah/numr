//! Index operation WGSL kernel launchers
//!
//! Provides launchers for indexing operations including:
//! - gather: Select elements along a dimension using an index tensor
//! - scatter: Scatter values to positions specified by an index tensor
//! - index_select: Select elements along a dimension using a 1D index tensor
//! - masked_select: Select elements where mask is true (returns flattened 1D tensor)
//! - masked_fill: Fill elements where mask is true with a scalar value
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders — data-movement ops (F32 / I32 / U32)
// ============================================================================

const INDEX_SELECT_SHADER_F32: &str = include_str!("index_select_f32.wgsl");
const INDEX_SELECT_SHADER_I32: &str = include_str!("index_select_i32.wgsl");
const INDEX_SELECT_SHADER_U32: &str = include_str!("index_select_u32.wgsl");

const INDEX_PUT_SHADER_F32: &str = include_str!("index_put_f32.wgsl");
const INDEX_PUT_SHADER_I32: &str = include_str!("index_put_i32.wgsl");
const INDEX_PUT_SHADER_U32: &str = include_str!("index_put_u32.wgsl");

const GATHER_SHADER_F32: &str = include_str!("gather_f32.wgsl");
const GATHER_SHADER_I32: &str = include_str!("gather_i32.wgsl");
const GATHER_SHADER_U32: &str = include_str!("gather_u32.wgsl");

const SCATTER_SHADER_F32: &str = include_str!("scatter_f32.wgsl");
const SCATTER_SHADER_I32: &str = include_str!("scatter_i32.wgsl");
const SCATTER_SHADER_U32: &str = include_str!("scatter_u32.wgsl");

const MASKED_FILL_SHADER_F32: &str = include_str!("masked_fill_f32.wgsl");
const MASKED_FILL_SHADER_I32: &str = include_str!("masked_fill_i32.wgsl");
const MASKED_FILL_SHADER_U32: &str = include_str!("masked_fill_u32.wgsl");

const MASKED_SELECT_SHADER_F32: &str = include_str!("masked_select_f32.wgsl");
const MASKED_SELECT_SHADER_I32: &str = include_str!("masked_select_i32.wgsl");
const MASKED_SELECT_SHADER_U32: &str = include_str!("masked_select_u32.wgsl");

const EMBEDDING_LOOKUP_SHADER_F32: &str = include_str!("embedding_lookup_f32.wgsl");
const EMBEDDING_LOOKUP_SHADER_I32: &str = include_str!("embedding_lookup_i32.wgsl");
const EMBEDDING_LOOKUP_SHADER_U32: &str = include_str!("embedding_lookup_u32.wgsl");

const GATHER_ND_SHADER_F32: &str = include_str!("gather_nd_f32.wgsl");
const GATHER_ND_SHADER_I32: &str = include_str!("gather_nd_i32.wgsl");
const GATHER_ND_SHADER_U32: &str = include_str!("gather_nd_u32.wgsl");

const SCATTER_REDUCE_SUM_SHADER_F32: &str = include_str!("scatter_reduce_sum_f32.wgsl");
const SCATTER_REDUCE_SUM_SHADER_I32: &str = include_str!("scatter_reduce_sum_i32.wgsl");
const SCATTER_REDUCE_SUM_SHADER_U32: &str = include_str!("scatter_reduce_sum_u32.wgsl");

const SCATTER_REDUCE_MAX_SHADER_F32: &str = include_str!("scatter_reduce_max_f32.wgsl");
const SCATTER_REDUCE_MAX_SHADER_I32: &str = include_str!("scatter_reduce_max_i32.wgsl");
const SCATTER_REDUCE_MAX_SHADER_U32: &str = include_str!("scatter_reduce_max_u32.wgsl");

const SCATTER_REDUCE_MIN_SHADER_F32: &str = include_str!("scatter_reduce_min_f32.wgsl");
const SCATTER_REDUCE_MIN_SHADER_I32: &str = include_str!("scatter_reduce_min_i32.wgsl");
const SCATTER_REDUCE_MIN_SHADER_U32: &str = include_str!("scatter_reduce_min_u32.wgsl");

const SCATTER_REDUCE_PROD_SHADER_F32: &str = include_str!("scatter_reduce_prod_f32.wgsl");

// The integer product saturates, so it needs the shared saturating helpers.
// WGSL has no include and no forward declarations, so the order is load-bearing.
const SCATTER_REDUCE_PROD_SHADER_I32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("scatter_reduce_prod_i32.wgsl"),
);
const SCATTER_REDUCE_PROD_SHADER_U32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("scatter_reduce_prod_u32.wgsl"),
);

const SCATTER_REDUCE_COUNT_SHADER: &str = include_str!("scatter_reduce_count.wgsl");
const SCATTER_REDUCE_MEAN_DIV_SHADER_F32: &str = include_str!("scatter_reduce_mean_div_f32.wgsl");

const SLICE_ASSIGN_SHADER_F32: &str = include_str!("slice_assign_f32.wgsl");
const SLICE_ASSIGN_SHADER_I32: &str = include_str!("slice_assign_i32.wgsl");
const SLICE_ASSIGN_SHADER_U32: &str = include_str!("slice_assign_u32.wgsl");

const GATHER_2D_SHADER_F32: &str = include_str!("gather_2d_f32.wgsl");
const GATHER_2D_SHADER_I32: &str = include_str!("gather_2d_i32.wgsl");
const GATHER_2D_SHADER_U32: &str = include_str!("gather_2d_u32.wgsl");

// ============================================================================
// Static shaders — dtype-agnostic ops
// ============================================================================

const VALIDATE_INDICES_SHADER: &str = include_str!("validate_indices.wgsl");
const BINCOUNT_UNWEIGHTED_SHADER: &str = include_str!("bincount_i32.wgsl");

// ============================================================================
// Static shaders — F32-only ops
// ============================================================================

const BINCOUNT_WEIGHTED_SHADER_F32: &str = include_str!("bincount_weighted_f32.wgsl");

// ============================================================================
// Helpers
// ============================================================================

/// Returns (shader, module_key, entry_point) for standard index/scatter/gather ops.
fn shader_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    Ok(match (op, dtype) {
        ("index_select", DType::F32) => (
            INDEX_SELECT_SHADER_F32,
            "index_select_f32",
            "index_select_f32",
        ),
        ("index_select", DType::I32) => (
            INDEX_SELECT_SHADER_I32,
            "index_select_i32",
            "index_select_i32",
        ),
        ("index_select", DType::U32) => (
            INDEX_SELECT_SHADER_U32,
            "index_select_u32",
            "index_select_u32",
        ),
        ("index_put", DType::F32) => (INDEX_PUT_SHADER_F32, "index_put_f32", "index_put_f32"),
        ("index_put", DType::I32) => (INDEX_PUT_SHADER_I32, "index_put_i32", "index_put_i32"),
        ("index_put", DType::U32) => (INDEX_PUT_SHADER_U32, "index_put_u32", "index_put_u32"),
        ("gather", DType::F32) => (GATHER_SHADER_F32, "gather_f32", "gather_f32"),
        ("gather", DType::I32) => (GATHER_SHADER_I32, "gather_i32", "gather_i32"),
        ("gather", DType::U32) => (GATHER_SHADER_U32, "gather_u32", "gather_u32"),
        ("scatter", DType::F32) => (SCATTER_SHADER_F32, "scatter_f32", "scatter_f32"),
        ("scatter", DType::I32) => (SCATTER_SHADER_I32, "scatter_i32", "scatter_i32"),
        ("scatter", DType::U32) => (SCATTER_SHADER_U32, "scatter_u32", "scatter_u32"),
        // copy shares the scatter shader module but uses a different entry point
        ("copy", DType::F32) => (SCATTER_SHADER_F32, "scatter_f32", "copy_f32"),
        ("copy", DType::I32) => (SCATTER_SHADER_I32, "scatter_i32", "copy_i32"),
        ("copy", DType::U32) => (SCATTER_SHADER_U32, "scatter_u32", "copy_u32"),
        ("masked_fill", DType::F32) => {
            (MASKED_FILL_SHADER_F32, "masked_fill_f32", "masked_fill_f32")
        }
        ("masked_fill", DType::I32) => {
            (MASKED_FILL_SHADER_I32, "masked_fill_i32", "masked_fill_i32")
        }
        ("masked_fill", DType::U32) => {
            (MASKED_FILL_SHADER_U32, "masked_fill_u32", "masked_fill_u32")
        }
        ("masked_select", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_select_f32",
        ),
        ("masked_select", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_select_i32",
        ),
        ("masked_select", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_select_u32",
        ),
        // masked_count and masked_prefix_sum share the masked_select shader module
        ("masked_count", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_count",
        ),
        ("masked_count", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_count",
        ),
        ("masked_count", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_count",
        ),
        ("masked_prefix_sum", DType::F32) => (
            MASKED_SELECT_SHADER_F32,
            "masked_select_f32",
            "masked_prefix_sum",
        ),
        ("masked_prefix_sum", DType::I32) => (
            MASKED_SELECT_SHADER_I32,
            "masked_select_i32",
            "masked_prefix_sum",
        ),
        ("masked_prefix_sum", DType::U32) => (
            MASKED_SELECT_SHADER_U32,
            "masked_select_u32",
            "masked_prefix_sum",
        ),
        ("embedding_lookup", DType::F32) => (
            EMBEDDING_LOOKUP_SHADER_F32,
            "embedding_lookup_f32",
            "embedding_lookup_f32",
        ),
        ("embedding_lookup", DType::I32) => (
            EMBEDDING_LOOKUP_SHADER_I32,
            "embedding_lookup_i32",
            "embedding_lookup_i32",
        ),
        ("embedding_lookup", DType::U32) => (
            EMBEDDING_LOOKUP_SHADER_U32,
            "embedding_lookup_u32",
            "embedding_lookup_u32",
        ),
        ("gather_nd", DType::F32) => (GATHER_ND_SHADER_F32, "gather_nd_f32", "gather_nd_f32"),
        ("gather_nd", DType::I32) => (GATHER_ND_SHADER_I32, "gather_nd_i32", "gather_nd_i32"),
        ("gather_nd", DType::U32) => (GATHER_ND_SHADER_U32, "gather_nd_u32", "gather_nd_u32"),
        ("scatter_reduce_sum", DType::F32) => (
            SCATTER_REDUCE_SUM_SHADER_F32,
            "scatter_reduce_sum_f32",
            "scatter_reduce_sum_f32",
        ),
        ("scatter_reduce_sum", DType::I32) => (
            SCATTER_REDUCE_SUM_SHADER_I32,
            "scatter_reduce_sum_i32",
            "scatter_reduce_sum_i32",
        ),
        ("scatter_reduce_sum", DType::U32) => (
            SCATTER_REDUCE_SUM_SHADER_U32,
            "scatter_reduce_sum_u32",
            "scatter_reduce_sum_u32",
        ),
        ("scatter_reduce_max", DType::F32) => (
            SCATTER_REDUCE_MAX_SHADER_F32,
            "scatter_reduce_max_f32",
            "scatter_reduce_max_f32",
        ),
        ("scatter_reduce_max", DType::I32) => (
            SCATTER_REDUCE_MAX_SHADER_I32,
            "scatter_reduce_max_i32",
            "scatter_reduce_max_i32",
        ),
        ("scatter_reduce_max", DType::U32) => (
            SCATTER_REDUCE_MAX_SHADER_U32,
            "scatter_reduce_max_u32",
            "scatter_reduce_max_u32",
        ),
        ("scatter_reduce_min", DType::F32) => (
            SCATTER_REDUCE_MIN_SHADER_F32,
            "scatter_reduce_min_f32",
            "scatter_reduce_min_f32",
        ),
        ("scatter_reduce_min", DType::I32) => (
            SCATTER_REDUCE_MIN_SHADER_I32,
            "scatter_reduce_min_i32",
            "scatter_reduce_min_i32",
        ),
        ("scatter_reduce_min", DType::U32) => (
            SCATTER_REDUCE_MIN_SHADER_U32,
            "scatter_reduce_min_u32",
            "scatter_reduce_min_u32",
        ),
        ("scatter_reduce_prod", DType::F32) => (
            SCATTER_REDUCE_PROD_SHADER_F32,
            "scatter_reduce_prod_f32",
            "scatter_reduce_prod_f32",
        ),
        ("scatter_reduce_prod", DType::I32) => (
            SCATTER_REDUCE_PROD_SHADER_I32,
            "scatter_reduce_prod_i32",
            "scatter_reduce_prod_i32",
        ),
        ("scatter_reduce_prod", DType::U32) => (
            SCATTER_REDUCE_PROD_SHADER_U32,
            "scatter_reduce_prod_u32",
            "scatter_reduce_prod_u32",
        ),
        // One count kernel for every value dtype: it reads only the index tensor.
        ("scatter_reduce_count", DType::F32 | DType::I32 | DType::U32) => (
            SCATTER_REDUCE_COUNT_SHADER,
            "scatter_reduce_count",
            "scatter_reduce_count",
        ),
        ("scatter_reduce_mean_div", DType::F32) => (
            SCATTER_REDUCE_MEAN_DIV_SHADER_F32,
            "scatter_reduce_mean_div_f32",
            "scatter_reduce_mean_div_f32",
        ),
        ("slice_assign", DType::F32) => (
            SLICE_ASSIGN_SHADER_F32,
            "slice_assign_f32",
            "slice_assign_f32",
        ),
        ("slice_assign", DType::I32) => (
            SLICE_ASSIGN_SHADER_I32,
            "slice_assign_i32",
            "slice_assign_i32",
        ),
        ("slice_assign", DType::U32) => (
            SLICE_ASSIGN_SHADER_U32,
            "slice_assign_u32",
            "slice_assign_u32",
        ),
        ("gather_2d", DType::F32) => (GATHER_2D_SHADER_F32, "gather_2d_f32", "gather_2d_f32"),
        ("gather_2d", DType::I32) => (GATHER_2D_SHADER_I32, "gather_2d_i32", "gather_2d_i32"),
        ("gather_2d", DType::U32) => (GATHER_2D_SHADER_U32, "gather_2d_u32", "gather_2d_u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    })
}

// ============================================================================
// Index Select Operation
// ============================================================================

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

// ============================================================================
// Index Put Operation
// ============================================================================

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

// ============================================================================
// Index Bounds Validation
// ============================================================================

/// Launch index bounds validation kernel.
///
/// Validates that all indices are within bounds [0, dim_size).
/// Returns the count of out-of-bounds indices in error_count buffer.
/// The error_count buffer must be initialized to 0 before calling.
pub fn launch_validate_indices(
    cache: &PipelineCache,
    queue: &Queue,
    indices: &Buffer,
    error_count: &Buffer,
    params_buffer: &Buffer,
    index_len: usize,
) -> Result<()> {
    if index_len == 0 {
        return Ok(());
    }

    let module = cache.get_or_create_module("validate_indices", VALIDATE_INDICES_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("validate_indices", "validate_indices", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[indices, error_count, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("validate_indices"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("validate_indices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(index_len), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Gather Operation
// ============================================================================

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

// ============================================================================
// Scatter Operation
// ============================================================================

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

// ============================================================================
// Masked Fill Operation
// ============================================================================

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

// ============================================================================
// Masked Select Operation (3-phase)
// ============================================================================

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

// ============================================================================
// Embedding Lookup Operation
// ============================================================================

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

// ============================================================================
// Bincount Operation
// ============================================================================

/// Launch a bincount operation kernel.
///
/// Counts occurrences of each value in an integer tensor.
/// Input: integer tensor with values in `[0, minlength)`
/// Output: count tensor of shape `[minlength]`
pub fn launch_bincount(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weights: Option<&Buffer>,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    weights_dtype: Option<DType>,
) -> Result<()> {
    let (name, shader) = if let Some(dtype) = weights_dtype {
        // bincount_weighted is F32 only (uses float atomics)
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bincount_weighted",
            });
        }
        ("bincount_weighted_f32", BINCOUNT_WEIGHTED_SHADER_F32)
    } else {
        ("bincount_i32", BINCOUNT_UNWEIGHTED_SHADER)
    };

    let module = cache.get_or_create_module(name, shader);

    let (layout, bind_group) = if let Some(weights_buf) = weights {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 2, // input and weights are read-only
        });
        let bind_group =
            cache.create_bind_group(&layout, &[input, weights_buf, output, params_buffer]);
        (layout, bind_group)
    } else {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 2,
            num_uniform_buffers: 1,
            num_readonly_storage: 1, // input is read-only
        });
        let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);
        (layout, bind_group)
    };

    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bincount"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bincount"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Scatter Reduce Operation
// ============================================================================

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

// ============================================================================
// Scatter Reduce Prod Operation
// ============================================================================

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

// ============================================================================
// Scatter Reduce Count Operation (for Mean)
// ============================================================================

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

// ============================================================================
// Scatter Reduce Mean Divide Operation
// ============================================================================

/// Launch scatter_reduce_mean_div: output[i] = sum[i] / count[i].
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

// ============================================================================
// Embedding Lookup Operation
// ============================================================================

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

// ============================================================================
// Slice Assign Operation
// ============================================================================

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

// ============================================================================
// Gather 2D Operation
// ============================================================================

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
