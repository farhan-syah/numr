//! Convolution WGSL kernel launchers (F32 only on WebGPU)
//!
//! Provides launchers for convolution operations:
//! - 1D convolution (conv1d)
//! - 2D convolution (conv2d)
//! - Depthwise 2D convolution (depthwise_conv2d)
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const CONV1D_SHADER: &str = include_str!("conv1d_f32.wgsl");
// entry point: "conv1d_f32"

const CONV_TRANSPOSE1D_SHADER: &str = include_str!("conv_transpose1d_f32.wgsl");
// entry point: "conv_transpose1d_f32"

const CONV2D_SHADER: &str = include_str!("conv2d_f32.wgsl");
// entry point: "conv2d_f32"

const DEPTHWISE_CONV2D_SHADER: &str = include_str!("depthwise_conv2d_f32.wgsl");
// entry point: "depthwise_conv2d_f32"

fn check_dtype_f32(dtype: DType, op: &'static str) -> Result<()> {
    match dtype {
        DType::F32 => Ok(()),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Conv1d
// ============================================================================

/// Launch conv1d kernel.
///
/// Performs 1D convolution with optional groups support.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C_in, L)
/// * `weight` - Weight tensor (C_out, C_in/groups, K)
/// * `bias` - Optional bias tensor (C_out)
/// * `output` - Output tensor (N, C_out, L_out)
/// * `params_buffer` - Uniform buffer with Conv1dParams
/// * `total_output` - Total number of output elements
pub fn launch_conv1d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32(dtype, "conv1d")?;

    let module = cache.get_or_create_module("conv1d_f32", CONV1D_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline("conv1d_f32", "conv1d_f32", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv1d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv1d"),
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
// ConvTranspose1d
// ============================================================================

/// Launch conv_transpose1d kernel.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C_in, L)
/// * `weight` - Weight tensor (C_in, C_out/groups, K) — input channels lead
/// * `bias` - Optional bias tensor (C_out)
/// * `output` - Output tensor (N, C_out, L_out)
/// * `params_buffer` - Uniform buffer with Conv1dParams (`padding` = resolved
///   LEFT padding, which TRIMS the output for a transposed convolution)
/// * `total_output` - Total number of output elements
pub fn launch_conv_transpose1d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32(dtype, "conv_transpose1d")?;

    let module = cache.get_or_create_module("conv_transpose1d_f32", CONV_TRANSPOSE1D_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline(
        "conv_transpose1d_f32",
        "conv_transpose1d_f32",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv_transpose1d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv_transpose1d"),
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
// Conv2d
// ============================================================================

/// Launch conv2d kernel.
///
/// Performs 2D convolution with optional groups support.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C_in, H, W)
/// * `weight` - Weight tensor (C_out, C_in/groups, K_h, K_w)
/// * `bias` - Optional bias tensor (C_out)
/// * `output` - Output tensor (N, C_out, H_out, W_out)
/// * `params_buffer` - Uniform buffer with Conv2dParams
/// * `total_output` - Total number of output elements
pub fn launch_conv2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32(dtype, "conv2d")?;

    let module = cache.get_or_create_module("conv2d_f32", CONV2D_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline("conv2d_f32", "conv2d_f32", &module, &layout);

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("conv2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("conv2d"),
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
// Depthwise Conv2d
// ============================================================================

/// Launch depthwise conv2d kernel.
///
/// Performs depthwise 2D convolution where each channel is convolved independently.
///
/// # Arguments
///
/// * `input` - Input tensor (N, C, H, W)
/// * `weight` - Weight tensor (C, 1, K_h, K_w)
/// * `bias` - Optional bias tensor (C)
/// * `output` - Output tensor (N, C, H_out, W_out)
/// * `params_buffer` - Uniform buffer with DepthwiseConv2dParams
/// * `total_output` - Total number of output elements
pub fn launch_depthwise_conv2d(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    total_output: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32(dtype, "depthwise_conv2d")?;

    let module = cache.get_or_create_module("depthwise_conv2d_f32", DEPTHWISE_CONV2D_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 4,
        num_uniform_buffers: 1,
        num_readonly_storage: 3,
    });
    let pipeline = cache.get_or_create_pipeline(
        "depthwise_conv2d_f32",
        "depthwise_conv2d_f32",
        &module,
        &layout,
    );

    let bind_group =
        cache.create_bind_group(&layout, &[input, weight, bias, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("depthwise_conv2d"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("depthwise_conv2d"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_output), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
