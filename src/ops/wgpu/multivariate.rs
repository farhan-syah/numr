//! WebGPU implementation of multivariate random distribution operations.
//!
//! This module provides implementations that delegate to generic algorithms
//! while using WGSL shaders for multinomial sampling.
//!
//! # Backend Limitations
//!
//! WebGPU only supports F32 for linear algebra operations. F64 is not supported
//! by WGSL (WebGPU Shading Language).

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::impl_generic::{
    DTypeSupport, MultinomialSamplingOps, dirichlet_impl, multinomial_samples_impl,
    multivariate_normal_impl, wishart_impl,
};
use crate::ops::traits::multivariate::MultivariateRandomOps;
use crate::ops::{BinaryOps, CumulativeOps, RandomOps, ReduceOps};
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::shaders::{MultinomialCountParams, launch_multinomial_count};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl MultivariateRandomOps<WgpuRuntime> for WgpuClient {
    fn multivariate_normal(
        &self,
        mean: &Tensor<WgpuRuntime>,
        cov: &Tensor<WgpuRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        multivariate_normal_impl(self, mean, cov, n_samples, DTypeSupport::F32_ONLY)
    }

    fn wishart(
        &self,
        scale: &Tensor<WgpuRuntime>,
        df: usize,
        n_samples: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        wishart_impl(self, scale, df, n_samples, DTypeSupport::F32_ONLY)
    }

    fn dirichlet(
        &self,
        alpha: &Tensor<WgpuRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        dirichlet_impl(self, alpha, n_samples)
    }

    fn multinomial_samples(
        &self,
        probs: &Tensor<WgpuRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        multinomial_samples_impl(self, probs, n_trials, n_samples)
    }
}

/// WebGPU implementation of multinomial sampling kernel.
///
/// Uses WGSL compute shaders for CDF lookup and counting.
impl MultinomialSamplingOps<WgpuRuntime> for WgpuClient {
    fn multinomial_sample_kernel(
        &self,
        probs: &Tensor<WgpuRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = probs.dtype();
        let k = probs.shape()[0];

        // Validate dtype - WebGPU only supports F32
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multinomial_samples (WebGPU only supports F32)",
            });
        }

        // Step 1: Normalize probabilities (GPU tensor ops)
        let sum_probs = self.sum(probs, &[0], false)?;
        let normalized = self.div(probs, &sum_probs)?;

        // Step 2: Compute CDF using cumsum (GPU)
        let cdf = self.cumsum(&normalized, 0)?;

        // Step 3: Generate uniform samples [n_samples, n_trials] (GPU)
        let uniforms = self.rand(&[n_samples, n_trials], dtype)?;

        // Step 4: Dispatch WGSL shader for CDF lookup and counting
        dispatch_multinomial_count_shader(self, &cdf, &uniforms, n_samples, n_trials, k)
    }
}

/// Dispatch WGSL shader for multinomial counting.
fn dispatch_multinomial_count_shader(
    client: &WgpuClient,
    cdf: &Tensor<WgpuRuntime>,
    uniforms: &Tensor<WgpuRuntime>,
    n_samples: usize,
    n_trials: usize,
    k: usize,
) -> Result<Tensor<WgpuRuntime>> {
    use crate::runtime::wgpu::client::get_buffer;

    // Allocate output tensor [n_samples, k]
    let output = Tensor::<WgpuRuntime>::try_empty(&[n_samples, k], DType::F32, client.device())?;

    // Get buffers
    let cdf_buf =
        get_buffer(cdf.ptr()).ok_or_else(|| Error::Internal("CDF buffer not found".to_string()))?;
    let uniforms_buf = get_buffer(uniforms.ptr())
        .ok_or_else(|| Error::Internal("Uniforms buffer not found".to_string()))?;
    let output_buf = get_buffer(output.ptr())
        .ok_or_else(|| Error::Internal("Output buffer not found".to_string()))?;

    // Create params buffer
    let params = MultinomialCountParams {
        k: k as u32,
        n_trials: n_trials as u32,
        n_samples: n_samples as u32,
        _pad: 0,
    };
    let params_buf = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("multinomial_count_params"),
        size: std::mem::size_of::<MultinomialCountParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params));

    // Launch kernel using the proper launcher
    launch_multinomial_count(
        client.pipeline_cache(),
        &client.queue,
        &cdf_buf,
        &uniforms_buf,
        &output_buf,
        &params_buf,
        n_samples,
        DType::F32,
    )?;

    Ok(output)
}
