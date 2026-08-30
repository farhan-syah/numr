//! Fused add + normalization WGSL kernel launchers
//!
//! Provides launchers for fused add+norm operations:
//! - Fused add + RMS normalization (forward and backward)
//! - Fused add + Layer normalization (forward and backward)
//! - Helper reduction kernel for backward passes
//!
//! All operations run entirely on GPU with no CPU fallback.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache};
use crate::dtype::DType;
use crate::error::{Error, Result};

const FUSED_ADD_NORM_SHADER: &str = include_str!("fused_add_norm.wgsl");

// ============================================================================
// Helper Macros
// ============================================================================

macro_rules! check_dtype_f32 {
    ($dtype:expr, $op:expr) => {
        if $dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: $dtype,
                op: $op,
            });
        }
    };
}

// ============================================================================
// Fused Add + RMS Normalization (Forward)
// ============================================================================

/// Launch fused add + RMS normalization kernel.
///
/// Computes: pre_norm = input + residual
///           output = pre_norm / sqrt(mean(pre_norm^2) + eps) * weight
pub fn launch_fused_add_rms_norm(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    residual: &Buffer,
    weight: &Buffer,
    output: &Buffer,
    pre_norm: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "fused_add_rms_norm");

    let module = cache.get_or_create_module("fused_add_norm", FUSED_ADD_NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fused_add_norm", "fused_add_rms_norm_f32", &module, &layout);

    let bind_group = cache.create_bind_group(
        &layout,
        &[input, residual, weight, output, pre_norm, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_add_rms_norm"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_add_rms_norm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Fused Add + Layer Normalization (Forward)
// ============================================================================

/// Launch fused add + layer normalization kernel.
///
/// Computes: pre_norm = input + residual
///           output = (pre_norm - mean) / sqrt(var + eps) * weight + bias
pub fn launch_fused_add_layer_norm(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    residual: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    output: &Buffer,
    pre_norm: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "fused_add_layer_norm");

    let module = cache.get_or_create_module("fused_add_norm", FUSED_ADD_NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "fused_add_norm",
        "fused_add_layer_norm_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            input,
            residual,
            weight,
            bias,
            output,
            pre_norm,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_add_layer_norm"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_add_layer_norm"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Fused Add + RMS Normalization (Backward)
// ============================================================================

/// Launch fused add + RMS normalization backward kernel.
///
/// Computes:
///   `d_input_residual = (grad * weight - pre_norm * coeff) * inv_rms`
///   `d_weight_scratch[batch_idx * hidden + i] = grad[batch_idx * hidden + i] * pre_norm[...] / rms`
///
/// Caller must launch reduce_sum_rows to sum d_weight_scratch across batch dimension.
pub fn launch_fused_add_rms_norm_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    pre_norm: &Buffer,
    weight: &Buffer,
    d_input_residual: &Buffer,
    d_weight_scratch: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "fused_add_rms_norm_bwd");

    let module = cache.get_or_create_module("fused_add_norm", FUSED_ADD_NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 5,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "fused_add_norm",
        "fused_add_rms_norm_bwd_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight_scratch,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_add_rms_norm_bwd"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_add_rms_norm_bwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Fused Add + Layer Normalization (Backward)
// ============================================================================

/// Launch fused add + layer normalization backward kernel.
///
/// Computes:
///   `d_input_residual = inv_std * (grad - mean_grad - normalized * mean_grad_normalized)`
///   `d_weight_scratch[batch_idx * hidden + i] = grad[...] * normalized`
///   `d_bias_scratch[batch_idx * hidden + i] = grad[...]`
///
/// Caller must launch reduce_sum_rows twice to sum d_weight_scratch and d_bias_scratch.
pub fn launch_fused_add_layer_norm_bwd(
    cache: &PipelineCache,
    queue: &Queue,
    grad: &Buffer,
    pre_norm: &Buffer,
    weight: &Buffer,
    bias: &Buffer,
    d_input_residual: &Buffer,
    d_weight_scratch: &Buffer,
    d_bias_scratch: &Buffer,
    params_buffer: &Buffer,
    batch_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "fused_add_layer_norm_bwd");

    let module = cache.get_or_create_module("fused_add_norm", FUSED_ADD_NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 7,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(
        "fused_add_norm",
        "fused_add_layer_norm_bwd_f32",
        &module,
        &layout,
    );

    let bind_group = cache.create_bind_group(
        &layout,
        &[
            grad,
            pre_norm,
            weight,
            bias,
            d_input_residual,
            d_weight_scratch,
            d_bias_scratch,
            params_buffer,
        ],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fused_add_layer_norm_bwd"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fused_add_layer_norm_bwd"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Reduce Sum Rows (Helper for backward)
// ============================================================================

/// Launch reduce sum rows kernel to sum a `[batch_size, hidden_size]` array across batch dimension.
///
/// Reduces input `[batch_size, hidden_size]` to output `[hidden_size]` by summing across batch.
pub fn launch_reduce_sum_rows(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params_buffer: &Buffer,
    hidden_size: usize,
    dtype: DType,
) -> Result<()> {
    check_dtype_f32!(dtype, "reduce_sum_rows");

    let module = cache.get_or_create_module("fused_add_norm", FUSED_ADD_NORM_SHADER);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline =
        cache.get_or_create_pipeline("fused_add_norm", "reduce_sum_rows_f32", &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("reduce_sum_rows"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("reduce_sum_rows"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // Dispatch enough workgroups to cover hidden_size elements
        let num_workgroups = (hidden_size as u32 + 255) / 256;
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
