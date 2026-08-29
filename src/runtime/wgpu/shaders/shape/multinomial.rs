//! Categorical sampling kernels, with and without replacement.
//!
//! Both are F32-only: they read a probability row and invert its CDF, so they
//! do not go through the shared shader table.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const MULTINOMIAL_WITH_REPLACEMENT_SHADER_F32: &str =
    include_str!("../multinomial_with_replacement_f32.wgsl");
const MULTINOMIAL_WITHOUT_REPLACEMENT_SHADER_F32: &str =
    include_str!("../multinomial_without_replacement_f32.wgsl");

/// Launch multinomial sampling with replacement kernel.
///
/// Samples indices from categorical distributions defined by probability rows.
/// Each thread independently samples one output using the CDF inversion method.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `probs` - Input probability buffer (num_distributions × num_categories)
/// * `out` - Output buffer for sampled indices (num_distributions × num_samples)
/// * `params_buffer` - Uniform buffer containing MultinomialWithReplacementParams
/// * `total_samples` - Total number of samples (num_distributions × num_samples)
/// * `input_dtype` - Data type of probability tensor (F32 only for WebGPU)
#[allow(clippy::too_many_arguments)]
pub fn launch_multinomial_with_replacement(
    cache: &PipelineCache,
    queue: &Queue,
    probs: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    total_samples: usize,
    input_dtype: DType,
) -> Result<()> {
    if total_samples == 0 {
        return Ok(());
    }

    // Only F32 input is supported for WebGPU multinomial
    if !matches!(input_dtype, DType::F32) {
        return Err(Error::UnsupportedDType {
            dtype: input_dtype,
            op: "multinomial",
        });
    }

    let name = "multinomial_with_replacement_f32";
    let module = cache.get_or_create_module(name, MULTINOMIAL_WITH_REPLACEMENT_SHADER_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[probs, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multinomial_with_replacement"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multinomial_with_replacement"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_samples), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch multinomial sampling without replacement kernel.
///
/// Samples indices from categorical distributions without replacement using
/// shared memory to track modified probabilities. Thread 0 does sequential
/// sampling within each workgroup (one workgroup per distribution).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `probs` - Input probability buffer (num_distributions × num_categories)
/// * `out` - Output buffer for sampled indices (num_distributions × num_samples)
/// * `params_buffer` - Uniform buffer containing MultinomialWithoutReplacementParams
/// * `num_distributions` - Number of distributions (one workgroup per distribution)
/// * `input_dtype` - Data type of probability tensor (F32 only for WebGPU)
#[allow(clippy::too_many_arguments)]
pub fn launch_multinomial_without_replacement(
    cache: &PipelineCache,
    queue: &Queue,
    probs: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    num_distributions: usize,
    input_dtype: DType,
) -> Result<()> {
    if num_distributions == 0 {
        return Ok(());
    }

    // Only F32 input is supported for WebGPU multinomial
    if !matches!(input_dtype, DType::F32) {
        return Err(Error::UnsupportedDType {
            dtype: input_dtype,
            op: "multinomial",
        });
    }

    let name = "multinomial_without_replacement_f32";
    let module = cache.get_or_create_module(name, MULTINOMIAL_WITHOUT_REPLACEMENT_SHADER_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[probs, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multinomial_without_replacement"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multinomial_without_replacement"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per distribution - thread 0 handles all samples
        pass.dispatch_workgroups(num_distributions as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
