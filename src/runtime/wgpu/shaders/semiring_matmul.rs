//! Semiring matrix multiplication WGSL kernel launchers. F32 and I32.
//!
//! U32 is absent by design, not by omission: `SemiringOp::validate_dtype` admits
//! only F32, F64, I32 and I64, so no U32 call can reach here. `OrAnd` is absent
//! for the mirror reason - it admits only Bool and U8, neither of which WebGPU
//! carries.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::semiring::SemiringOp;

const SR_MIN_PLUS_SHADER: &str = include_str!("semiring_matmul_min_plus_f32.wgsl");
const SR_MAX_PLUS_SHADER: &str = include_str!("semiring_matmul_max_plus_f32.wgsl");
const SR_MAX_MIN_SHADER: &str = include_str!("semiring_matmul_max_min_f32.wgsl");
const SR_MIN_MAX_SHADER: &str = include_str!("semiring_matmul_min_max_f32.wgsl");
const SR_OR_AND_SHADER: &str = include_str!("semiring_matmul_or_and_f32.wgsl");
const SR_PLUS_MAX_SHADER: &str = include_str!("semiring_matmul_plus_max_f32.wgsl");

// The I32 shaders read the saturated reduce identities NUMR_I32_MAX/MIN from
// `int_saturate.wgsl`. WGSL has no include, so the single copy in the repository
// is prepended here, and it must come first: WGSL has no forward declarations.
const SR_MIN_PLUS_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("semiring_matmul_min_plus_i32.wgsl")
);
const SR_MAX_PLUS_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("semiring_matmul_max_plus_i32.wgsl")
);
const SR_MAX_MIN_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("semiring_matmul_max_min_i32.wgsl")
);
const SR_MIN_MAX_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("semiring_matmul_min_max_i32.wgsl")
);
const SR_PLUS_MAX_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("semiring_matmul_plus_max_i32.wgsl")
);

const TILE_SIZE: u32 = 16;

fn semiring_shader_info(
    op: SemiringOp,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str, &'static str)> {
    let unsupported = || Error::UnsupportedDType {
        dtype,
        op: "semiring_matmul (WebGPU)",
    };
    if dtype == DType::I32 {
        return Ok(match op {
            SemiringOp::MinPlus => (
                SR_MIN_PLUS_I32_SHADER,
                "sr_min_plus_i32",
                "semiring_matmul_min_plus_i32",
                "batched_semiring_matmul_min_plus_i32",
            ),
            SemiringOp::MaxPlus => (
                SR_MAX_PLUS_I32_SHADER,
                "sr_max_plus_i32",
                "semiring_matmul_max_plus_i32",
                "batched_semiring_matmul_max_plus_i32",
            ),
            SemiringOp::MaxMin => (
                SR_MAX_MIN_I32_SHADER,
                "sr_max_min_i32",
                "semiring_matmul_max_min_i32",
                "batched_semiring_matmul_max_min_i32",
            ),
            SemiringOp::MinMax => (
                SR_MIN_MAX_I32_SHADER,
                "sr_min_max_i32",
                "semiring_matmul_min_max_i32",
                "batched_semiring_matmul_min_max_i32",
            ),
            SemiringOp::PlusMax => (
                SR_PLUS_MAX_I32_SHADER,
                "sr_plus_max_i32",
                "semiring_matmul_plus_max_i32",
                "batched_semiring_matmul_plus_max_i32",
            ),
            // Unreachable through `SemiringOp::validate_dtype`, which refuses
            // OrAnd on anything but Bool and U8.
            SemiringOp::OrAnd => return Err(unsupported()),
        });
    }
    if dtype != DType::F32 {
        return Err(unsupported());
    }
    Ok(match op {
        SemiringOp::MinPlus => (
            SR_MIN_PLUS_SHADER,
            "sr_min_plus_f32",
            "semiring_matmul_min_plus_f32",
            "batched_semiring_matmul_min_plus_f32",
        ),
        SemiringOp::MaxPlus => (
            SR_MAX_PLUS_SHADER,
            "sr_max_plus_f32",
            "semiring_matmul_max_plus_f32",
            "batched_semiring_matmul_max_plus_f32",
        ),
        SemiringOp::MaxMin => (
            SR_MAX_MIN_SHADER,
            "sr_max_min_f32",
            "semiring_matmul_max_min_f32",
            "batched_semiring_matmul_max_min_f32",
        ),
        SemiringOp::MinMax => (
            SR_MIN_MAX_SHADER,
            "sr_min_max_f32",
            "semiring_matmul_min_max_f32",
            "batched_semiring_matmul_min_max_f32",
        ),
        SemiringOp::OrAnd => (
            SR_OR_AND_SHADER,
            "sr_or_and_f32",
            "semiring_matmul_or_and_f32",
            "batched_semiring_matmul_or_and_f32",
        ),
        SemiringOp::PlusMax => (
            SR_PLUS_MAX_SHADER,
            "sr_plus_max_f32",
            "semiring_matmul_plus_max_f32",
            "batched_semiring_matmul_plus_max_f32",
        ),
    })
}

/// Launch semiring matrix multiplication kernel.
pub fn launch_semiring_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    op: SemiringOp,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, entry_point, _) = semiring_shader_info(op, dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("semiring_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("semiring_matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let num_groups_x = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(num_groups_x, num_groups_y, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch batched semiring matrix multiplication kernel.
pub fn launch_batched_semiring_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    op: SemiringOp,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key, _, batched_entry_point) = semiring_shader_info(op, dtype)?;

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, batched_entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_semiring_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_semiring_matmul"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let num_groups_x = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let num_groups_y = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(num_groups_x, num_groups_y, batch_size as u32);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
