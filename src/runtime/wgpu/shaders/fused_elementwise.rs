//! Fused elementwise WGSL kernel launchers. F32, I32, U32.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const TERNARY_SHADER_F32: &str = include_str!("fused_elementwise.wgsl");
const TERNARY_SHADER_I32: &str = include_str!("fused_elementwise_i32.wgsl");
const TERNARY_SHADER_U32: &str = include_str!("fused_elementwise_u32.wgsl");

const SCALAR_SHADER_F32: &str = include_str!("fused_elementwise_scalar.wgsl");
const SCALAR_SHADER_I32: &str = include_str!("fused_elementwise_scalar_i32.wgsl");
const SCALAR_SHADER_U32: &str = include_str!("fused_elementwise_scalar_u32.wgsl");

/// Returns (shader, module_key) for the ternary fused ops.
fn ternary_module(dtype: DType, op: &'static str) -> Result<(&'static str, &'static str)> {
    match dtype {
        DType::F32 => Ok((TERNARY_SHADER_F32, "fused_elementwise_f32")),
        DType::I32 => Ok((TERNARY_SHADER_I32, "fused_elementwise_i32")),
        DType::U32 => Ok((TERNARY_SHADER_U32, "fused_elementwise_u32")),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// The `&'static str` entry-point name for `stem` at `dtype`.
///
/// The pipeline cache keys on `&'static str`, so the suffixed names are spelled
/// out rather than formatted.
fn entry_point_for(stem: &'static str, dtype: DType) -> &'static str {
    match (stem, dtype) {
        ("fused_mul_add", DType::I32) => "fused_mul_add_i32",
        ("fused_mul_add", DType::U32) => "fused_mul_add_u32",
        ("fused_add_mul", DType::I32) => "fused_add_mul_i32",
        ("fused_add_mul", DType::U32) => "fused_add_mul_u32",
        ("fused_mul_add_scalar", DType::I32) => "fused_mul_add_scalar_i32",
        ("fused_mul_add_scalar", DType::U32) => "fused_mul_add_scalar_u32",
        ("fused_add_mul", _) => "fused_add_mul_f32",
        ("fused_mul_add_scalar", _) => "fused_mul_add_scalar_f32",
        _ => "fused_mul_add_f32",
    }
}

/// Params for ternary ops (matches TernaryParams in WGSL)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct TernaryParams {
    numel: u32,
}

/// Params for scalar FMA (matches ScalarFmaParams in WGSL).
///
/// `scale_bits` and `bias_bits` hold each scalar re-encoded per dtype, not a
/// plain `f32`: the f32/i32/u32 shaders read these same two 4-byte fields but
/// declare them as their own type.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ScalarFmaParams {
    numel: u32,
    scale_bits: u32,
    bias_bits: u32,
    _pad: u32,
}

impl ScalarFmaParams {
    /// Encode both scalars the way the CPU backend converts them: an `as` cast
    /// to the element type, which saturates for integers.
    fn new(numel: usize, scale: f64, bias: f64, dtype: DType) -> Self {
        let encode = |v: f64| -> u32 {
            match dtype {
                DType::I32 => u32::from_ne_bytes((v as i32).to_ne_bytes()),
                DType::U32 => v as u32,
                _ => u32::from_ne_bytes((v as f32).to_ne_bytes()),
            }
        };
        Self {
            numel: numel as u32,
            scale_bits: encode(scale),
            bias_bits: encode(bias),
            _pad: 0,
        }
    }
}

fn launch_ternary(
    cache: &PipelineCache,
    queue: &Queue,
    op_name: &'static str,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (shader, module_key) = ternary_module(dtype, op_name)?;
    let entry_point = entry_point_for(op_name, dtype);

    let params = TernaryParams {
        numel: numel as u32,
    };
    let params_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_elem_params"),
        size: std::mem::size_of::<TernaryParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, b, c, out, &params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(op_name),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op_name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch fused_mul_add: out = a * b + c. F32, I32, U32.
pub fn launch_fused_mul_add(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_ternary(cache, queue, "fused_mul_add", a, b, c, out, numel, dtype)
}

/// Launch fused_add_mul: out = (a + b) * c. F32, I32, U32.
pub fn launch_fused_add_mul(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    c: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    launch_ternary(cache, queue, "fused_add_mul", a, b, c, out, numel, dtype)
}

/// Launch fused_mul_add_scalar: out = a * scale + bias. F32, I32, U32.
pub fn launch_fused_mul_add_scalar(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    numel: usize,
    dtype: DType,
    scale: f64,
    bias: f64,
) -> Result<()> {
    let (shader, module_key) = match dtype {
        DType::F32 => (SCALAR_SHADER_F32, "fused_elementwise_scalar_f32"),
        DType::I32 => (SCALAR_SHADER_I32, "fused_elementwise_scalar_i32"),
        DType::U32 => (SCALAR_SHADER_U32, "fused_elementwise_scalar_u32"),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "fused_mul_add_scalar",
            });
        }
    };
    let entry_point = entry_point_for("fused_mul_add_scalar", dtype);

    let params = ScalarFmaParams::new(numel, scale, bias, dtype);
    let params_buf = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fused_elem_scalar_params"),
        size: std::mem::size_of::<ScalarFmaParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, &params_buf]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_mul_add_scalar"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_mul_add_scalar"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
