//! CUDA implementation of multivariate random distribution operations.
//!
//! This module provides implementations that delegate to generic algorithms
//! while using CUDA kernels for multinomial sampling.

use crate::error::Result;
use crate::ops::impl_generic::{
    DTypeSupport, MultinomialSamplingOps, dirichlet_impl, multinomial_samples_impl,
    multivariate_normal_impl, wishart_impl,
};
use crate::ops::traits::multivariate::MultivariateRandomOps;
use crate::ops::{BinaryOps, CumulativeOps, RandomOps, ReduceOps};
use crate::runtime::cuda::kernels::launch_multinomial_count;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;

impl MultivariateRandomOps<CudaRuntime> for CudaClient {
    fn multivariate_normal(
        &self,
        mean: &Tensor<CudaRuntime>,
        cov: &Tensor<CudaRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        multivariate_normal_impl(self, mean, cov, n_samples, DTypeSupport::FULL)
    }

    fn wishart(
        &self,
        scale: &Tensor<CudaRuntime>,
        df: usize,
        n_samples: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        wishart_impl(self, scale, df, n_samples, DTypeSupport::FULL)
    }

    fn dirichlet(
        &self,
        alpha: &Tensor<CudaRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        dirichlet_impl(self, alpha, n_samples)
    }

    fn multinomial_samples(
        &self,
        probs: &Tensor<CudaRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        multinomial_samples_impl(self, probs, n_trials, n_samples)
    }
}

/// CUDA implementation of multinomial sampling kernel.
///
/// Uses CUDA kernels for CDF computation and categorical sampling.
impl MultinomialSamplingOps<CudaRuntime> for CudaClient {
    fn multinomial_sample_kernel(
        &self,
        probs: &Tensor<CudaRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = probs.dtype();
        let k = probs.shape()[0];

        // Step 1: Normalize probabilities (GPU tensor ops)
        let sum_probs = self.sum(probs, &[0], false)?;
        let normalized = self.div(probs, &sum_probs)?;

        // Step 2: Compute CDF using cumsum (GPU)
        let cdf = self.cumsum(&normalized, 0)?;

        // Step 3: Generate uniform samples [n_samples, n_trials] (GPU)
        let uniforms = self.rand(&[n_samples, n_trials], dtype)?;

        // Step 4: Allocate output and launch CUDA kernel
        let output = Tensor::<CudaRuntime>::try_zeros(&[n_samples, k], dtype, &self.device)?;

        // Get device pointers
        let cdf_ptr = cdf.ptr();
        let uniforms_ptr = uniforms.ptr();
        let output_ptr = output.ptr();

        // Launch kernel
        unsafe {
            launch_multinomial_count(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                cdf_ptr,
                uniforms_ptr,
                output_ptr,
                k,
                n_trials,
                n_samples,
            )?;
        }

        Ok(output)
    }
}
