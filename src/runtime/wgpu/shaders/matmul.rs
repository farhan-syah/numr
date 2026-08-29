//! Matrix multiplication WGSL kernel launchers. F32, I32, U32.
//!
//! `matvec` stays F32-only: nothing dispatches it for an integer dtype.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const MATMUL_SHADER: &str = include_str!("matmul.wgsl");
const MATMUL_BIAS_SHADER: &str = include_str!("matmul_bias_f32.wgsl");
// The integer shaders share one wide accumulator, prepended to each module that
// uses it. WGSL has no include, so this is where the single copy in the
// repository reaches them. `int_saturate.wgsl` must come first: it defines the
// `NumrI64` and range constants `int_matmul_acc.wgsl` builds on, and WGSL has no
// forward declarations.
const MATMUL_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("matmul_i32.wgsl")
);
const MATMUL_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("matmul_u32.wgsl")
);
const MATMUL_BIAS_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("matmul_bias_i32.wgsl")
);
const MATMUL_BIAS_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("matmul_bias_u32.wgsl")
);

/// Tile size for tiled matrix multiplication (must match shader constant)
const TILE_SIZE: u32 = 16;

// ============================================================================
// DType Dispatch
// ============================================================================

/// The shader module and entry points serving one matmul dtype.
struct MatmulShader {
    module_key: &'static str,
    source: &'static str,
    tiled: &'static str,
    batched: &'static str,
    simple: &'static str,
}

/// The shader module and entry points serving one fused matmul+bias dtype.
struct MatmulBiasShader {
    module_key: &'static str,
    source: &'static str,
    tiled: &'static str,
    batched: &'static str,
}

static MATMUL_F32: MatmulShader = MatmulShader {
    module_key: "matmul",
    source: MATMUL_SHADER,
    tiled: "matmul_f32",
    batched: "batched_matmul_f32",
    simple: "matmul_simple_f32",
};

static MATMUL_I32: MatmulShader = MatmulShader {
    module_key: "matmul_i32",
    source: MATMUL_I32_SHADER,
    tiled: "matmul_i32",
    batched: "batched_matmul_i32",
    simple: "matmul_simple_i32",
};

static MATMUL_U32: MatmulShader = MatmulShader {
    module_key: "matmul_u32",
    source: MATMUL_U32_SHADER,
    tiled: "matmul_u32",
    batched: "batched_matmul_u32",
    simple: "matmul_simple_u32",
};

static MATMUL_BIAS_F32: MatmulBiasShader = MatmulBiasShader {
    module_key: "matmul_bias_f32",
    source: MATMUL_BIAS_SHADER,
    tiled: "matmul_bias_f32",
    batched: "batched_matmul_bias_f32",
};

static MATMUL_BIAS_I32: MatmulBiasShader = MatmulBiasShader {
    module_key: "matmul_bias_i32",
    source: MATMUL_BIAS_I32_SHADER,
    tiled: "matmul_bias_i32",
    batched: "batched_matmul_bias_i32",
};

static MATMUL_BIAS_U32: MatmulBiasShader = MatmulBiasShader {
    module_key: "matmul_bias_u32",
    source: MATMUL_BIAS_U32_SHADER,
    tiled: "matmul_bias_u32",
    batched: "batched_matmul_bias_u32",
};

fn matmul_shader(dtype: DType, op: &'static str) -> Result<&'static MatmulShader> {
    match dtype {
        DType::F32 => Ok(&MATMUL_F32),
        DType::I32 => Ok(&MATMUL_I32),
        DType::U32 => Ok(&MATMUL_U32),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

fn matmul_bias_shader(dtype: DType, op: &'static str) -> Result<&'static MatmulBiasShader> {
    match dtype {
        DType::F32 => Ok(&MATMUL_BIAS_F32),
        DType::I32 => Ok(&MATMUL_BIAS_I32),
        DType::U32 => Ok(&MATMUL_BIAS_U32),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// 2D Matrix Multiplication
// ============================================================================

/// Launch tiled matrix multiplication kernel.
///
/// Computes C = A @ B where A is [M, K] and B is [K, N].
pub fn launch_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let shader = matmul_shader(dtype, "matmul")?;

    let module = cache.get_or_create_module(shader.module_key, shader.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(shader.module_key, shader.tiled, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul"),
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

/// Launch simple (non-tiled) matrix multiplication kernel.
///
/// For small matrices where tiling overhead isn't worth it.
pub fn launch_matmul_simple(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let shader = matmul_shader(dtype, "matmul_simple")?;

    let module = cache.get_or_create_module(shader.module_key, shader.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(shader.module_key, shader.simple, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_simple"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_simple"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let total = m * n;
        let num_groups = (total as u32 + 255) / 256;
        pass.dispatch_workgroups(num_groups, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Batched Matrix Multiplication
// ============================================================================

/// Launch batched matrix multiplication kernel.
///
/// Computes `C[b] = A[b] @ B[b]` for each batch b.
pub fn launch_batched_matmul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    let shader = matmul_shader(dtype, "batched_matmul")?;

    let module = cache.get_or_create_module(shader.module_key, shader.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline(shader.module_key, shader.batched, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_matmul"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_matmul"),
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

// ============================================================================
// Matrix-Vector Multiplication
// ============================================================================

/// Launch matrix-vector multiplication kernel.
///
/// Computes y = A @ x where A is `[M, N]` and x is `[N]`.
pub fn launch_matvec(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    x: &Buffer,
    y: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    dtype: DType,
) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "matvec",
        });
    }

    let module = cache.get_or_create_module("matmul", MATMUL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline("matmul", "matvec_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, x, y, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matvec"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matvec"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(m as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Fused Matrix Multiplication with Bias
// ============================================================================

/// Launch tiled matrix multiplication with fused bias addition.
///
/// Computes C = A @ B + bias where bias is `[N]` (broadcast across rows).
pub fn launch_matmul_bias(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    let shader = matmul_bias_shader(dtype, "matmul_bias")?;

    let module = cache.get_or_create_module(shader.module_key, shader.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // a, b, bias, c
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(shader.module_key, shader.tiled, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_bias"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("matmul_bias"),
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

/// Launch batched matrix multiplication with fused bias addition.
///
/// Computes `C[b] = A[b] @ B[b] + bias` for each batch b.
pub fn launch_batched_matmul_bias(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    let shader = matmul_bias_shader(dtype, "batched_matmul_bias")?;

    let module = cache.get_or_create_module(shader.module_key, shader.source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4, // a, b, bias, c
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline(shader.module_key, shader.batched, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("batched_matmul_bias"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("batched_matmul_bias"),
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
