//! CPU implementation of multivariate random distribution operations.
//!
//! This module provides thin wrappers that delegate to the generic implementations
//! in `impl_generic/multivariate.rs` to ensure numerical parity across all backends.

use crate::dispatch_dtype;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::impl_generic::{
    DTypeSupport, MultinomialSamplingOps, dirichlet_impl, multinomial_samples_impl,
    multivariate_normal_impl, wishart_impl,
};
use crate::ops::traits::multivariate::MultivariateRandomOps;
use crate::ops::{BinaryOps, CumulativeOps, RandomOps, ReduceOps};
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl MultivariateRandomOps<CpuRuntime> for CpuClient {
    fn multivariate_normal(
        &self,
        mean: &Tensor<CpuRuntime>,
        cov: &Tensor<CpuRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        multivariate_normal_impl(self, mean, cov, n_samples, DTypeSupport::FULL)
    }

    fn wishart(
        &self,
        scale: &Tensor<CpuRuntime>,
        df: usize,
        n_samples: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        wishart_impl(self, scale, df, n_samples, DTypeSupport::FULL)
    }

    fn dirichlet(
        &self,
        alpha: &Tensor<CpuRuntime>,
        n_samples: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        dirichlet_impl(self, alpha, n_samples)
    }

    fn multinomial_samples(
        &self,
        probs: &Tensor<CpuRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        multinomial_samples_impl(self, probs, n_trials, n_samples)
    }
}

/// CPU implementation of multinomial sampling kernel.
///
/// For CPU, we can efficiently implement this using native operations
/// since CPU doesn't have the same kernel launch overhead as GPU.
impl MultinomialSamplingOps<CpuRuntime> for CpuClient {
    fn multinomial_sample_kernel(
        &self,
        probs: &Tensor<CpuRuntime>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        let dtype = probs.dtype();
        let k = probs.shape()[0];

        // Step 1: Normalize probabilities (on CPU, this is just tensor ops)
        let sum_probs = self.sum(probs, &[0], false)?;
        let normalized = self.div(probs, &sum_probs)?;

        // Step 2: Compute CDF using cumsum
        let cdf = self.cumsum(&normalized, 0)?;

        // Step 3: Generate uniform samples [n_samples, n_trials]
        let uniforms = self.rand(&[n_samples, n_trials], dtype)?;

        // Step 4: CDF lookup and counting
        // For CPU, we implement this efficiently in native code
        multinomial_count_kernel(&cdf, &uniforms, n_samples, n_trials, k, dtype, &self.device)
    }
}

/// Native CPU kernel for multinomial counting.
///
/// Takes CDF tensor [k] and uniform samples [n_samples, n_trials],
/// returns counts [n_samples, k].
fn multinomial_count_kernel(
    cdf: &Tensor<CpuRuntime>,
    uniforms: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_trials: usize,
    k: usize,
    dtype: DType,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    dispatch_dtype!(dtype, T => {
        multinomial_count_typed::<T>(cdf, uniforms, n_samples, n_trials, k, device)
    }, "multinomial_count")
}

/// Type-specific multinomial counting implementation.
///
/// Generic over float types to eliminate code duplication.
fn multinomial_count_typed<T>(
    cdf: &Tensor<CpuRuntime>,
    uniforms: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_trials: usize,
    k: usize,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>>
where
    T: crate::dtype::Element + PartialOrd,
{
    let cdf_data: Vec<T> = cdf.to_vec();
    let uniform_data: Vec<T> = uniforms.to_vec();
    let mut counts = vec![T::zero(); n_samples * k];

    for s in 0..n_samples {
        for t in 0..n_trials {
            let u = uniform_data[s * n_trials + t];
            // Binary search for category
            let category = binary_search_cdf(&cdf_data, u);
            counts[s * k + category] = counts[s * k + category] + T::one();
        }
    }

    Tensor::<CpuRuntime>::try_from_slice(&counts, &[n_samples, k], device)
}

/// Binary search to find the category for a uniform sample.
fn binary_search_cdf<T: PartialOrd>(cdf: &[T], u: T) -> usize {
    let mut lo = 0;
    let mut hi = cdf.len();
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if cdf[mid] <= u {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo.min(cdf.len() - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    fn get_client() -> CpuClient {
        let device = CpuRuntime::default_device();
        CpuRuntime::default_client(&device)
    }

    #[test]
    fn test_multivariate_normal_basic() {
        let client = get_client();
        let mean =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &client.device).unwrap();
        let cov =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &client.device)
                .unwrap();

        let samples = client
            .multivariate_normal(&mean, &cov, 100)
            .expect("multivariate_normal should succeed with valid inputs");
        assert_eq!(samples.shape(), &[100, 2]);

        // Verify samples have reasonable statistics
        let sample_data: Vec<f32> = samples.to_vec();
        let (mut mean_0, mut mean_1) = (0.0f64, 0.0f64);
        for i in 0..100 {
            mean_0 += sample_data[i * 2] as f64;
            mean_1 += sample_data[i * 2 + 1] as f64;
        }
        mean_0 /= 100.0;
        mean_1 /= 100.0;

        // With 100 samples from N(0,1), means should be within ~0.5 of 0
        assert!(mean_0.abs() < 0.5, "Mean 0 too far from 0: {}", mean_0);
        assert!(mean_1.abs() < 0.5, "Mean 1 too far from 0: {}", mean_1);
    }

    #[test]
    fn test_multivariate_normal_correlated() {
        let client = get_client();
        let mean =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 2.0], &[2], &client.device).unwrap();
        let cov =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 0.8, 0.8, 1.0], &[2, 2], &client.device)
                .unwrap();

        let samples = client
            .multivariate_normal(&mean, &cov, 1000)
            .expect("multivariate_normal should succeed with correlated covariance");
        assert_eq!(samples.shape(), &[1000, 2]);
    }

    #[test]
    fn test_multivariate_normal_invalid_cov() {
        let client = get_client();
        let mean =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &client.device).unwrap();
        // Not positive definite
        let cov =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 2.0, 1.0], &[2, 2], &client.device)
                .unwrap();

        let result = client.multivariate_normal(&mean, &cov, 100);
        assert!(
            result.is_err(),
            "Should fail with non-positive-definite cov"
        );
    }

    #[test]
    fn test_dirichlet_basic() {
        let client = get_client();
        let alpha = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &client.device)
            .unwrap();

        let samples = client
            .dirichlet(&alpha, 100)
            .expect("dirichlet should succeed with valid inputs");
        assert_eq!(samples.shape(), &[100, 3]);

        // Verify each row sums to 1
        let sample_data: Vec<f32> = samples.to_vec();
        for i in 0..100 {
            let row_sum: f32 = sample_data[i * 3..i * 3 + 3].iter().sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "Row {} sum is {}, expected 1.0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_dirichlet_concentrated() {
        let client = get_client();
        let alpha =
            Tensor::<CpuRuntime>::try_from_slice(&[100.0f64, 1.0, 1.0], &[3], &client.device)
                .unwrap();

        let samples = client
            .dirichlet(&alpha, 100)
            .expect("dirichlet should succeed with concentrated alpha");
        let sample_data: Vec<f64> = samples.to_vec();

        // First category should have most mass
        let mut mean_0 = 0.0;
        for i in 0..100 {
            mean_0 += sample_data[i * 3];
        }
        mean_0 /= 100.0;

        // Expected: 100 / (100 + 1 + 1) ≈ 0.98
        assert!(
            mean_0 > 0.9,
            "Expected first category mean > 0.9, got {}",
            mean_0
        );
    }

    #[test]
    fn test_multinomial_samples_basic() {
        let client = get_client();
        let probs =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32; 6], &[6], &client.device).unwrap();

        let samples = client
            .multinomial_samples(&probs, 60, 100)
            .expect("multinomial_samples should succeed with valid inputs");
        assert_eq!(samples.shape(), &[100, 6]);

        // Verify each row sums to n_trials
        let sample_data: Vec<f32> = samples.to_vec();
        for i in 0..100 {
            let row_sum: f32 = sample_data[i * 6..i * 6 + 6].iter().sum();
            assert!(
                (row_sum - 60.0).abs() < 1e-5,
                "Row {} sum is {}, expected 60.0",
                i,
                row_sum
            );
        }
    }

    #[test]
    fn test_multinomial_samples_biased() {
        let client = get_client();
        let probs =
            Tensor::<CpuRuntime>::try_from_slice(&[0.99f64, 0.01], &[2], &client.device).unwrap();

        let samples = client
            .multinomial_samples(&probs, 100, 50)
            .expect("multinomial_samples should succeed with biased probs");
        let sample_data: Vec<f64> = samples.to_vec();

        // First category should have most counts
        let mut mean_0 = 0.0;
        for i in 0..50 {
            mean_0 += sample_data[i * 2];
        }
        mean_0 /= 50.0;

        // Expected: ~99 out of 100 trials
        assert!(
            mean_0 > 90.0,
            "Expected first category mean > 90, got {}",
            mean_0
        );
    }

    #[test]
    fn test_wishart_basic() {
        let client = get_client();
        let scale =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &client.device)
                .unwrap();

        let samples = client
            .wishart(&scale, 5, 10)
            .expect("wishart should succeed with valid inputs");
        assert_eq!(samples.shape(), &[10, 2, 2]);

        // Verify samples are symmetric and positive definite
        let sample_data: Vec<f32> = samples.to_vec();
        for s in 0..10 {
            let offset = s * 4;
            let a00 = sample_data[offset];
            let a01 = sample_data[offset + 1];
            let a10 = sample_data[offset + 2];
            let a11 = sample_data[offset + 3];

            // Check symmetry
            assert!(
                (a01 - a10).abs() < 1e-4,
                "Sample {} not symmetric: a01={}, a10={}",
                s,
                a01,
                a10
            );

            // Check positive definiteness
            assert!(a00 > 0.0, "Sample {} has non-positive a00: {}", s, a00);
            assert!(a11 > 0.0, "Sample {} has non-positive a11: {}", s, a11);
            let det = a00 * a11 - a01 * a10;
            assert!(
                det > 0.0,
                "Sample {} has non-positive determinant: {}",
                s,
                det
            );
        }
    }

    #[test]
    fn test_wishart_f64() {
        let client = get_client();
        let scale =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], &client.device)
                .unwrap();

        let samples = client
            .wishart(&scale, 5, 5)
            .expect("wishart should succeed with F64");
        assert_eq!(samples.shape(), &[5, 2, 2]);
        assert_eq!(samples.dtype(), crate::dtype::DType::F64);
    }
}
