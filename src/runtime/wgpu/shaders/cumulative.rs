//! Cumulative operation WGSL kernel launchers
//!
//! - `cumsum` - F32, I32, U32
//! - `cumprod` - F32, I32, U32
//! - `logsumexp` - F32 only

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const CUMSUM_F32_SHADER: &str = include_str!("cumsum_f32.wgsl");
// The I32 kernel needs the shared wide accumulator (`NumrI64`) and the U32
// kernel needs the shared saturating add, both defined once in
// int_saturate.wgsl and prepended here - the same file the `pow` shaders in
// elementwise.rs share.
const CUMSUM_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumsum_i32.wgsl")
);
const CUMSUM_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumsum_u32.wgsl")
);

const CUMSUM_STRIDED_F32_SHADER: &str = include_str!("cumsum_strided_f32.wgsl");
const CUMSUM_STRIDED_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumsum_strided_i32.wgsl")
);
const CUMSUM_STRIDED_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumsum_strided_u32.wgsl")
);

const CUMPROD_F32_SHADER: &str = include_str!("cumprod_f32.wgsl");
// The integer kernels track an exact magnitude plus a sign, using the overflow
// check and the two store helpers in int_saturate.wgsl, prepended here.
const CUMPROD_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumprod_i32.wgsl")
);
const CUMPROD_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumprod_u32.wgsl")
);

const CUMPROD_STRIDED_F32_SHADER: &str = include_str!("cumprod_strided_f32.wgsl");
const CUMPROD_STRIDED_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumprod_strided_i32.wgsl")
);
const CUMPROD_STRIDED_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("cumprod_strided_u32.wgsl")
);

const LOGSUMEXP_SHADER: &str = include_str!("logsumexp_f32.wgsl");
const LOGSUMEXP_STRIDED_SHADER: &str = include_str!("logsumexp_strided_f32.wgsl");

fn check_f32(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

fn check_f32_i32_u32(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 | DType::I32 | DType::U32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Cumulative Sum
// ============================================================================

/// Launch cumsum operation kernel (contiguous data). Supports F32, I32, U32.
pub fn launch_cumsum(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    check_f32_i32_u32(dtype, "cumsum")?;

    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => ("cumsum_f32", CUMSUM_F32_SHADER, "cumsum_f32"),
        DType::I32 => ("cumsum_i32", CUMSUM_I32_SHADER, "cumsum_i32"),
        DType::U32 => ("cumsum_u32", CUMSUM_U32_SHADER, "cumsum_u32"),
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumsum"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumsum"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided cumsum operation kernel. Supports F32, I32, U32.
pub fn launch_cumsum_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    check_f32_i32_u32(dtype, "cumsum_strided")?;

    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => (
            "cumsum_strided_f32",
            CUMSUM_STRIDED_F32_SHADER,
            "cumsum_strided_f32",
        ),
        DType::I32 => (
            "cumsum_strided_i32",
            CUMSUM_STRIDED_I32_SHADER,
            "cumsum_strided_i32",
        ),
        DType::U32 => (
            "cumsum_strided_u32",
            CUMSUM_STRIDED_U32_SHADER,
            "cumsum_strided_u32",
        ),
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumsum_strided"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumsum_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Cumulative Product
// ============================================================================

/// Launch cumprod operation kernel (contiguous data). Supports F32, I32, U32.
pub fn launch_cumprod(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    check_f32_i32_u32(dtype, "cumprod")?;

    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => ("cumprod_f32", CUMPROD_F32_SHADER, "cumprod_f32"),
        DType::I32 => ("cumprod_i32", CUMPROD_I32_SHADER, "cumprod_i32"),
        DType::U32 => ("cumprod_u32", CUMPROD_U32_SHADER, "cumprod_u32"),
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumprod"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumprod"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided cumprod operation kernel. Supports F32, I32, U32.
pub fn launch_cumprod_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    check_f32_i32_u32(dtype, "cumprod_strided")?;

    let (module_key, shader, entry_point) = match dtype {
        DType::F32 => (
            "cumprod_strided_f32",
            CUMPROD_STRIDED_F32_SHADER,
            "cumprod_strided_f32",
        ),
        DType::I32 => (
            "cumprod_strided_i32",
            CUMPROD_STRIDED_I32_SHADER,
            "cumprod_strided_i32",
        ),
        DType::U32 => (
            "cumprod_strided_u32",
            CUMPROD_STRIDED_U32_SHADER,
            "cumprod_strided_u32",
        ),
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cumprod_strided"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cumprod_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Log-Sum-Exp
// ============================================================================

/// Launch logsumexp operation kernel (contiguous data).
pub fn launch_logsumexp(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    outer_size: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "logsumexp")?;

    let module = cache.get_or_create_module("logsumexp_f32", LOGSUMEXP_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("logsumexp_f32", "logsumexp_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logsumexp"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logsumexp"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(outer_size), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch strided logsumexp operation kernel.
pub fn launch_logsumexp_strided(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_inner: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "logsumexp_strided")?;

    let module = cache.get_or_create_module("logsumexp_strided_f32", LOGSUMEXP_STRIDED_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "logsumexp_strided_f32",
        "logsumexp_strided_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("logsumexp_strided"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("logsumexp_strided"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_inner), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
