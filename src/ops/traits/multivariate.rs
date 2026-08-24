//! Multivariate random distribution operations.
//!
//! This module defines the `MultivariateRandomOps` trait for sampling from
//! multivariate probability distributions that require linear algebra operations.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Multivariate random distribution operations
///
/// Provides methods for sampling from multivariate probability distributions
/// that require linear algebra operations (Cholesky decomposition, matrix multiplication).
/// These distributions are fundamental for Bayesian inference, Gaussian mixture models,
/// and Monte Carlo methods.
///
/// # Implementation Note
///
/// All backends use the same algorithm (via `impl_generic/multivariate.rs`) to ensure
/// numerical parity:
/// - Multivariate normal: Cholesky decomposition + linear transform
/// - Wishart: Bartlett decomposition with batched random generation
/// - Dirichlet: Gamma sampling + normalization
/// - Multinomial samples: CDF-based categorical sampling with counting
///
/// # Backend Limitations
///
/// | Backend | Supported dtypes |
/// |---------|------------------|
/// | CPU     | F32, F64         |
/// | CUDA    | F32, F64         |
/// | WebGPU  | F32 only         |
///
/// WebGPU does not support F64 because WGSL (WebGPU Shading Language) lacks
/// native 64-bit floating point support.
pub trait MultivariateRandomOps<R: Runtime> {
    /// Sample from a multivariate normal distribution: X ~ N(μ, Σ)
    ///
    /// Generates samples from a multivariate normal distribution with the given
    /// mean vector and covariance matrix.
    ///
    /// # Algorithm
    ///
    /// Uses the Cholesky decomposition method:
    /// 1. Compute Cholesky decomposition: `Σ = L @ L^T`
    /// 2. Generate standard normal samples: `Z ~ N(0, I)` with shape `` `(n_samples, d)` ``
    /// 3. Transform: `X = μ + (L @ Z^T)^T = μ + Z @ L^T`
    ///
    /// This guarantees that X has the correct covariance:
    /// `Cov(X) = L @ Cov(Z) @ L^T = L @ I @ L^T = Σ`
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean vector `μ` with shape `` `(d,)` `` where `d` is the dimensionality
    /// * `cov` - Covariance matrix `Σ` with shape `` `(d, d)` ``, must be symmetric positive definite
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tensor with shape `` `(n_samples, d)` `` containing samples from `` `N(μ, Σ)` ``
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `mean` is not 1D
    /// - `cov` is not a 2D square matrix
    /// - `mean.len()` != `cov.shape()[0]`
    /// - `n_samples` is 0
    ///
    /// Returns `Error::Internal` if `cov` is not positive definite (Cholesky fails)
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not F32 or F64
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use numr::ops::MultivariateRandomOps;
    ///
    /// // 2D multivariate normal
    /// let mean = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    /// let cov = Tensor::<CpuRuntime>::from_slice(&[1.0, 0.5, 0.5, 1.0], &[2, 2], &device)?;
    /// let samples = client.multivariate_normal(&mean, &cov, 1000)?;
    /// // samples has shape [1000, 2]
    /// ```
    fn multivariate_normal(
        &self,
        mean: &Tensor<R>,
        cov: &Tensor<R>,
        n_samples: usize,
    ) -> Result<Tensor<R>> {
        let _ = (mean, cov, n_samples);
        Err(Error::NotImplemented {
            feature: "MultivariateRandomOps::multivariate_normal",
        })
    }

    /// Sample from a Wishart distribution: W ~ W(V, df)
    ///
    /// Generates samples from the Wishart distribution, which is the distribution
    /// of sample covariance matrices from a multivariate normal population.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// W ~ W(V, df) where:
    /// - V is the scale matrix (d × d, symmetric positive definite)
    /// - df is the degrees of freedom (df ≥ d)
    ///
    /// Mean = df × V
    /// ```
    ///
    /// # Algorithm (Bartlett Decomposition)
    ///
    /// For efficient sampling, uses the Bartlett decomposition:
    /// 1. Compute Cholesky: `V = L @ L^T`
    /// 2. Generate A (lower triangular):
    ///    - `A[i,i] ~ sqrt(χ²(df - i))` for `i = 0, ..., d-1`
    ///    - `A[i,j] ~ N(0, 1)` for `i > j`
    /// 3. Compute: `W = L @ A @ A^T @ L^T`
    ///
    /// # Arguments
    ///
    /// * `scale` - Scale matrix `V` with shape `(d, d)`, must be symmetric positive definite
    /// * `df` - Degrees of freedom, must be `≥ d` (matrix dimension)
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_samples, d, d)` containing symmetric positive definite matrices
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `scale` is not a 2D square matrix
    /// - `df` < d (degrees of freedom too small)
    /// - `n_samples` is 0
    ///
    /// Returns `Error::Internal` if `scale` is not positive definite
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not F32 or F64
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::MultivariateRandomOps;
    ///
    /// // 2x2 Wishart with identity scale
    /// let scale = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &device)?;
    /// let samples = client.wishart(&scale, 5, 100)?;
    /// // samples has shape [100, 2, 2]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn wishart(&self, scale: &Tensor<R>, df: usize, n_samples: usize) -> Result<Tensor<R>> {
        let _ = (scale, df, n_samples);
        Err(Error::NotImplemented {
            feature: "MultivariateRandomOps::wishart",
        })
    }

    /// Sample from a Dirichlet distribution: X ~ Dir(α)
    ///
    /// Generates samples from the Dirichlet distribution, which is a distribution
    /// over probability vectors (vectors that sum to 1).
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; α) = (1/B(α)) × ∏ᵢ xᵢ^(αᵢ - 1)
    ///
    /// where B(α) = ∏Γ(αᵢ) / Γ(∑αᵢ)
    ///
    /// Support: xᵢ ∈ (0, 1), ∑xᵢ = 1
    /// Mean[i] = αᵢ / ∑α
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship with Gamma distribution:
    /// 1. Generate Y_i ~ Gamma(α_i, 1) for i = 1, ..., k
    /// 2. Compute X_i = Y_i / sum(Y)
    ///
    /// # Arguments
    ///
    /// * `alpha` - Concentration parameters with shape `(k,)`, all α_i > 0
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_samples, k)` where each row sums to 1
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `alpha` is not 1D
    /// - Any α_i ≤ 0
    /// - `n_samples` is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::MultivariateRandomOps;
    ///
    /// // Symmetric Dirichlet with 3 categories
    /// let alpha = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device)?;
    /// let samples = client.dirichlet(&alpha, 1000)?;
    /// // samples has shape [1000, 3], each row sums to 1
    ///
    /// // Concentrated Dirichlet (favors first category)
    /// let alpha = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 1.0, 1.0], &[3], &device)?;
    /// let samples = client.dirichlet(&alpha, 1000)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn dirichlet(&self, alpha: &Tensor<R>, n_samples: usize) -> Result<Tensor<R>> {
        let _ = (alpha, n_samples);
        Err(Error::NotImplemented {
            feature: "MultivariateRandomOps::dirichlet",
        })
    }

    /// Sample from a multinomial distribution with counts: X ~ Multinomial(probs, n_trials)
    ///
    /// Generates samples representing the number of times each category was selected
    /// in n_trials independent draws from a categorical distribution.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = x) = n! / (x₁! × ... × xₖ!) × ∏ᵢ pᵢ^xᵢ
    ///
    /// where ∑xᵢ = n_trials
    ///
    /// Mean[i] = n_trials × pᵢ
    /// Variance[i] = n_trials × pᵢ × (1 - pᵢ)
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses repeated categorical sampling:
    /// 1. For each sample:
    ///    a. Initialize counts to zeros
    ///    b. For each of n_trials trials:
    ///       - Sample a category from probs
    ///       - Increment the count for that category
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability vector with shape `(k,)`, all p_i ≥ 0, will be normalized
    /// * `n_trials` - Number of trials per sample
    /// * `n_samples` - Number of samples to generate
    ///
    /// # Returns
    ///
    /// Tensor with shape `(n_samples, k)` containing integer counts (as floats)
    /// where each row sums to n_trials
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `probs` is not 1D
    /// - All probabilities are 0 or negative
    /// - `n_trials` is 0
    /// - `n_samples` is 0
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not floating point
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::MultivariateRandomOps;
    ///
    /// // Fair die: 6 categories, 100 rolls, 50 samples
    /// let probs = Tensor::<CpuRuntime>::from_slice(&[1.0f32; 6], &[6], &device)?;
    /// let samples = client.multinomial_samples(&probs, 100, 50)?;
    /// // samples has shape [50, 6], each row sums to 100
    ///
    /// // Biased coin: 2 categories, 10 flips
    /// let probs = Tensor::<CpuRuntime>::from_slice(&[0.7f32, 0.3], &[2], &device)?;
    /// let samples = client.multinomial_samples(&probs, 10, 1000)?;
    /// // samples has shape [1000, 2], each row sums to 10
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn multinomial_samples(
        &self,
        probs: &Tensor<R>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<R>> {
        let _ = (probs, n_trials, n_samples);
        Err(Error::NotImplemented {
            feature: "MultivariateRandomOps::multinomial_samples",
        })
    }
}
