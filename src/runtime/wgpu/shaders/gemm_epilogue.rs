//! WGSL kernel launchers for GEMM epilogue operations. F32 only.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::GemmActivation;

const GEMM_EPILOGUE_SHADER: &str = include_str!("gemm_epilogue_f32.wgsl");
const GEMM_EPILOGUE_RESIDUAL_SHADER: &str = include_str!("gemm_epilogue_residual_f32.wgsl");

const TILE_SIZE: u32 = 16;

fn activation_to_u32(act: GemmActivation) -> u32 {
    // The mapping lives on the enum: a launcher and a kernel that
    // disagreed about a code would silently apply the wrong activation.
    act.code()
}

/// Params struct for the activation shader (8 u32s for alignment).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemmEpilogueParams {
    /// Number of rows of A / output.
    pub m: u32,
    /// Inner dimension (columns of A, rows of B).
    pub k: u32,
    /// Number of columns of B / output.
    pub n: u32,
    /// Number of batches (1 for non-batched).
    pub batch_size: u32,
    /// Activation function index (0=None, 1=ReLU, 2=GELU, 3=SiLU, 4=Sigmoid, 5=Tanh).
    pub activation_type: u32,
    /// Padding for 32-byte alignment.
    pub _pad0: u32,
    /// Padding for 32-byte alignment.
    pub _pad1: u32,
    /// Padding for 32-byte alignment.
    pub _pad2: u32,
}

/// Params struct for the residual shader.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemmResidualParams {
    /// Number of rows of A / output.
    pub m: u32,
    /// Inner dimension (columns of A, rows of B).
    pub k: u32,
    /// Number of columns of B / output.
    pub n: u32,
    /// Number of batches (1 for non-batched).
    pub batch_size: u32,
}

fn check_f32(dtype: DType, op: &'static str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Launch fused GEMM + bias + activation (2D).
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm_bias_act(
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
    check_f32(dtype, "gemm_bias_act")?;

    let module = cache.get_or_create_module("gemm_epilogue_f32", GEMM_EPILOGUE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("gemm_bias_act_f32", "gemm_bias_act_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_bias_act"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bias_act"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let gx = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let gy = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(gx, gy, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch batched fused GEMM + bias + activation (3D).
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm_bias_act_batched(
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
    check_f32(dtype, "gemm_bias_act_batched")?;

    let module = cache.get_or_create_module("gemm_epilogue_f32", GEMM_EPILOGUE_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "gemm_bias_act_batched_f32",
        "gemm_bias_act_batched_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_bias_act_batched"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bias_act_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let gx = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let gy = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(gx, gy, batch_size as u32);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch fused GEMM + bias + residual (2D).
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm_bias_residual(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    residual: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "gemm_bias_residual")?;

    let module =
        cache.get_or_create_module("gemm_epilogue_residual_f32", GEMM_EPILOGUE_RESIDUAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "gemm_bias_residual_f32",
        "gemm_bias_residual_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, residual, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_bias_residual"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bias_residual"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let gx = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let gy = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(gx, gy, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch batched fused GEMM + bias + residual (3D).
#[allow(clippy::too_many_arguments)]
pub fn launch_gemm_bias_residual_batched(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    b: &Buffer,
    bias: &Buffer,
    residual: &Buffer,
    c: &Buffer,
    params_buffer: &Buffer,
    m: usize,
    n: usize,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_f32(dtype, "gemm_bias_residual_batched")?;

    let module =
        cache.get_or_create_module("gemm_epilogue_residual_f32", GEMM_EPILOGUE_RESIDUAL_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "gemm_bias_residual_batched_f32",
        "gemm_bias_residual_batched_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(&layout, &[a, b, bias, residual, c, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("gemm_bias_residual_batched"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gemm_bias_residual_batched"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        let gx = (n as u32 + TILE_SIZE - 1) / TILE_SIZE;
        let gy = (m as u32 + TILE_SIZE - 1) / TILE_SIZE;
        pass.dispatch_workgroups(gx, gy, batch_size as u32);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Create a uniform buffer for the activation params.
pub fn create_epilogue_params_buffer(
    cache: &PipelineCache,
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
    activation: GemmActivation,
) -> Buffer {
    let params = GemmEpilogueParams {
        m,
        k,
        n,
        batch_size,
        activation_type: activation_to_u32(activation),
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    use wgpu::util::DeviceExt;
    cache
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm_epilogue_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}

/// Create a uniform buffer for the residual params.
pub fn create_residual_params_buffer(
    cache: &PipelineCache,
    m: u32,
    k: u32,
    n: u32,
    batch_size: u32,
) -> Buffer {
    let params = GemmResidualParams {
        m,
        k,
        n,
        batch_size,
    };
    use wgpu::util::DeviceExt;
    cache
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gemm_residual_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        })
}
