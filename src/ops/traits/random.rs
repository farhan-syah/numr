//! Random number generation operations.
//!
//! This module defines the `RandomOps` trait for sampling from various probability distributions.

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Random number generation operations
///
/// Provides methods for sampling from various probability distributions.
/// All operations return a new tensor filled with random values.
pub trait RandomOps<R: Runtime> {
    // ===== Basic Random Generation =====

    /// Generate uniform random values in [0, 1)
    ///
    /// Creates a tensor filled with random values uniformly distributed in [0, 1).
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with uniform random values
    fn rand(&self, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>> {
        let _ = (shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::rand",
        })
    }

    /// Generate uniform random values in [0, 1) with a deterministic seed
    ///
    /// Same as `rand()` but uses the provided seed for reproducible output.
    /// Calling with the same seed and shape on the SAME BACKEND always
    /// produces the same tensor.
    ///
    /// # Reproducibility is per-backend, not across backends
    ///
    /// Each backend runs its own PRNG — the CPU uses xoshiro256, CUDA its own
    /// device kernel — so one seed yields DIFFERENT values on CPU and on CUDA.
    /// A seeded run is reproducible against itself and against other runs on
    /// the same backend, and is NOT expected to match a run of the same seed on
    /// another device. This matches PyTorch, where `manual_seed` likewise does
    /// not tie the CPU and CUDA streams together.
    ///
    /// A training baseline pinned to a loss value is therefore a per-device
    /// constant. Comparing a CPU number against a CUDA one measures the two
    /// PRNG streams, not the change under test.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    /// * `seed` - Deterministic seed for the PRNG
    fn rand_seeded(
        &self,
        shape: &[usize],
        dtype: crate::dtype::DType,
        seed: u64,
    ) -> Result<Tensor<R>> {
        let _ = (shape, dtype, seed);
        Err(Error::NotImplemented {
            feature: "RandomOps::rand_seeded",
        })
    }

    /// Generate standard normal random values (mean=0, std=1)
    ///
    /// Creates a tensor filled with random values from standard normal distribution N(0, 1).
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with normally distributed random values
    fn randn(&self, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>> {
        let _ = (shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::randn",
        })
    }

    /// Generate standard normal random values (mean=0, std=1) with a deterministic seed
    ///
    /// Same as `randn()` but uses the provided seed for reproducible output.
    /// Calling with the same seed and shape on the SAME BACKEND always
    /// produces the same tensor. Reproducibility is per-backend — see
    /// [`RandomOps::rand_seeded`] for why a CPU and a CUDA run of one seed
    /// differ, and what that means for a pinned baseline.
    ///
    /// # Arguments
    ///
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    /// * `seed` - Deterministic seed for the PRNG
    fn randn_seeded(
        &self,
        shape: &[usize],
        dtype: crate::dtype::DType,
        seed: u64,
    ) -> Result<Tensor<R>> {
        let _ = (shape, dtype, seed);
        Err(Error::NotImplemented {
            feature: "RandomOps::randn_seeded",
        })
    }

    /// Generate random integers in the range [low, high)
    ///
    /// Creates a tensor filled with random integers uniformly distributed in [low, high).
    /// The `high` value is exclusive (never included in the output).
    ///
    /// # Arguments
    ///
    /// * `low` - Lower bound (inclusive)
    /// * `high` - Upper bound (exclusive), must be > low
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be integer type: I8, I16, I32, I64, U8, U16, U32, U64)
    ///
    /// # Returns
    ///
    /// Tensor filled with random integers
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `high <= low`
    /// - The range `[low, high)` cannot be represented in the specified dtype
    ///
    /// Returns `Error::UnsupportedDType` if dtype is not an integer type.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::RandomOps;
    /// # use numr::dtype::DType;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Random integers in [0, 10)
    /// let a = client.randint(0, 10, &[3, 4], DType::I32)?;
    ///
    /// // Random bytes in [0, 256)
    /// let b = client.randint(0, 256, &[1024], DType::U8)?;
    ///
    /// // Random signed integers in [-100, 100)
    /// let c = client.randint(-100, 100, &[10], DType::I32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - For unsigned types, `low` must be >= 0
    /// - The distribution is uniform over the discrete values in [low, high)
    /// - Each call produces independent random values (not reproducible without seeding)
    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (low, high, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::randint",
        })
    }

    // ===== Discrete Distributions =====

    /// Sample from a multinomial (categorical) distribution
    ///
    /// Given a tensor of probabilities for each category, samples indices according
    /// to those probabilities. This is the fundamental operation for categorical
    /// sampling in machine learning, including LLM next-token selection.
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform sampling (CDF method):
    /// 1. Compute cumulative sum of probabilities (CDF)
    /// 2. For each sample, draw uniform random u ∈ `[0, 1)`
    /// 3. Find smallest index i where `` `CDF[i]` `` ≥ u (binary search)
    ///
    /// ```text
    /// probs:  `` `[0.1, 0.2, 0.3, 0.4]` ``
    /// CDF:    `` `[0.1, 0.3, 0.6, 1.0]` ``
    ///          ↑    ↑    ↑    ↑
    /// u=0.05 → 0    │    │    │   (u < 0.1)
    /// u=0.25 ──────→ 1   │    │   (0.1 ≤ u < 0.3)
    /// u=0.55 ─────────→ 2│    │   (0.3 ≤ u < 0.6)
    /// u=0.80 ──────────────→ 3 (0.6 ≤ u < 1.0)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `probs` - Probability tensor with shape `` `[..., num_categories]` ``
    ///   - Probabilities must be non-negative
    ///   - Probabilities are normalized automatically (do not need to sum to 1)
    ///   - Must be floating point dtype (F32, F64, F16, BF16)
    /// * `num_samples` - Number of samples to draw per distribution
    /// * `replacement` - Whether to sample with replacement
    ///   - `true`: Same category can be sampled multiple times
    ///   - `false`: Each category sampled at most once (requires num_samples ≤ num_categories)
    ///
    /// # Returns
    ///
    /// Tensor of sampled indices with shape `[..., num_samples]` and dtype I64 (I32 on WGPU).
    /// Each index is in the range `[0, num_categories)`.
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if:
    /// - `probs` is empty or has zero categories
    /// - `num_samples` is 0
    /// - `replacement` is false and `num_samples > num_categories`
    /// - `probs` contains negative values
    /// - All probabilities in a row are zero (no valid category to sample)
    ///
    /// Returns `Error::UnsupportedDType` if `probs` is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::RandomOps;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Sample one index from a 4-category distribution
    /// let probs = Tensor::<CpuRuntime>::from_slice(&[0.1f32, 0.2, 0.3, 0.4], &[4], &device)?;
    /// let sample = client.multinomial(&probs, 1, true)?;  // Shape: [1]
    ///
    /// // Sample 3 indices without replacement
    /// let samples = client.multinomial(&probs, 3, false)?;  // Shape: [3]
    ///
    /// // Batch sampling: 2 distributions, 5 samples each
    /// let batch_probs = Tensor::<CpuRuntime>::from_slice(
    ///     &[0.1f32, 0.9, 0.5, 0.5],  // 2 rows of 2 categories
    ///     &[2, 2], &device
    /// )?;
    /// let batch_samples = client.multinomial(&batch_probs, 5, true)?;  // Shape: [2, 5]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Notes
    ///
    /// - Input probabilities are normalized per distribution (per row)
    /// - Zero-probability categories are never sampled
    /// - When `replacement=false`, samples within each distribution are unique
    /// - Each call produces independent random samples (not reproducible without seeding)
    /// - This is the PyTorch `torch.multinomial` equivalent
    fn multinomial(
        &self,
        probs: &Tensor<R>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<R>> {
        let _ = (probs, num_samples, replacement);
        Err(Error::NotImplemented {
            feature: "RandomOps::multinomial",
        })
    }

    /// Sample from a Bernoulli distribution
    ///
    /// Creates a tensor where each element is 1 with probability p and 0 otherwise.
    /// This is the fundamental binary random variable.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = 1) = p
    /// P(X = 0) = 1 - p
    /// Mean = p
    /// Variance = p(1 - p)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `p` - Probability of success (1), must be in `` `[0, 1]` ``
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with 0s and 1s sampled from Bernoulli(p)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if p is not in `` `[0, 1]` ``.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::RandomOps;
    /// # use numr::dtype::DType;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Fair coin flips
    /// let flips = client.bernoulli(0.5, &[100], DType::F32)?;
    ///
    /// // Biased coin (70% heads)
    /// let biased = client.bernoulli(0.7, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn bernoulli(&self, p: f64, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>> {
        let _ = (p, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::bernoulli",
        })
    }

    /// Sample from a Poisson distribution
    ///
    /// Creates a tensor filled with random integer values from the Poisson distribution,
    /// which models the number of events in a fixed interval.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = k) = λ^k * e^(-λ) / k!
    ///
    /// Support: k ∈ {0, 1, 2, ...}
    /// Mean = λ
    /// Variance = λ
    /// ```
    ///
    /// # Algorithm
    ///
    /// - For small λ (< 30): Direct inversion method
    /// - For large λ (≥ 30): Normal approximation with continuity correction
    ///
    /// # Arguments
    ///
    /// * `lambda` - Rate parameter λ (λ > 0), the expected number of events
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point, stores integer counts)
    ///
    /// # Returns
    ///
    /// Tensor filled with non-negative integer values (stored as floats) from Poisson(λ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if lambda ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::RandomOps;
    /// # use numr::dtype::DType;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Average of 5 events per interval
    /// let counts = client.poisson(5.0, &[1000], DType::F32)?;
    ///
    /// // Rare events (average 0.1 per interval)
    /// let rare = client.poisson(0.1, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # Notes
    ///
    /// Output values are non-negative integers but stored in floating point dtype
    /// for compatibility with GPU operations. Cast to integer type if needed.
    fn poisson(
        &self,
        lambda: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (lambda, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::poisson",
        })
    }

    /// Sample from a Binomial distribution
    ///
    /// Creates a tensor filled with random values from the Binomial distribution,
    /// which models the number of successes in n independent Bernoulli trials.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// P(X = k) = C(n, k) * p^k * (1-p)^(n-k)
    ///
    /// Support: k ∈ {0, 1, 2, ..., n}
    /// Mean = n * p
    /// Variance = n * p * (1 - p)
    /// ```
    ///
    /// # Algorithm
    ///
    /// - For small n (< 25): Direct simulation (sum of Bernoulli trials)
    /// - For large n: BTRD algorithm (Binomial, Triangle, Rectangle, Decomposition)
    ///
    /// # Arguments
    ///
    /// * `n` - Number of trials (n > 0)
    /// * `p` - Probability of success per trial, must be in `` `[0, 1]` ``
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with integer values in [0, n] from Binomial(n, p)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if n ≤ 0 or p is not in `` `[0, 1]` ``.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::ops::RandomOps;
    /// # use numr::dtype::DType;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // 10 coin flips with fair coin
    /// let flips = client.binomial(10, 0.5, &[1000], DType::F32)?;
    ///
    /// // 100 trials with 20% success rate
    /// let trials = client.binomial(100, 0.2, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (n, p, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::binomial",
        })
    }

    // ===== Continuous Distributions =====

    /// Sample from a Beta distribution
    ///
    /// Creates a tensor filled with random values from the Beta distribution,
    /// which is commonly used as a prior for probabilities in Bayesian statistics.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; α, β) = x^(α-1) * (1-x)^(β-1) / B(α, β)
    /// where B(α, β) = Γ(α)Γ(β) / Γ(α+β)
    ///
    /// Support: x ∈ (0, 1)
    /// Mean = α / (α + β)
    /// Variance = αβ / ((α+β)²(α+β+1))
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: if X ~ Gamma(α, 1) and Y ~ Gamma(β, 1),
    /// then X / (X + Y) ~ Beta(α, β).
    ///
    /// # Arguments
    ///
    /// * `alpha` - First shape parameter (α > 0)
    /// * `beta` - Second shape parameter (β > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values in (0, 1) from Beta(α, β)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if alpha ≤ 0 or beta ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // Symmetric beta (same as uniform for α=β=1)
    /// let uniform_like = client.beta(1.0, 1.0, &[1000], DType::F32)?;
    ///
    /// // Skewed towards 0
    /// let left_skewed = client.beta(0.5, 5.0, &[1000], DType::F32)?;
    ///
    /// // Skewed towards 1
    /// let right_skewed = client.beta(5.0, 0.5, &[1000], DType::F32)?;
    ///
    /// // Bell-shaped in middle
    /// let bell = client.beta(5.0, 5.0, &[1000], DType::F32)?;
    /// ```
    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (alpha, beta, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::beta",
        })
    }

    /// Sample from a Gamma distribution
    ///
    /// Creates a tensor filled with random values from the Gamma distribution,
    /// which is used for modeling waiting times and is the basis for many other distributions.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; k, θ) = x^(k-1) * e^(-x/θ) / (θ^k * Γ(k))
    ///
    /// Support: x > 0
    /// Mean = k * θ
    /// Variance = k * θ²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses Marsaglia and Tsang's method for shape ≥ 1,
    /// with Ahrens-Dieter acceptance-rejection for shape < 1.
    ///
    /// # Arguments
    ///
    /// * `shape_param` - Shape parameter k (k > 0), also called α in some notations
    /// * `scale` - Scale parameter θ (θ > 0), also called 1/rate or 1/β
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from Gamma(k, θ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if shape_param ≤ 0 or scale ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // Exponential distribution is Gamma(1, θ)
    /// let exponential_like = client.gamma(1.0, 2.0, &[1000], DType::F32)?;
    ///
    /// // Chi-squared with df=5 is Gamma(2.5, 2)
    /// let chi2_like = client.gamma(2.5, 2.0, &[1000], DType::F32)?;
    /// ```
    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (shape_param, scale, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::gamma",
        })
    }

    /// Sample from an Exponential distribution
    ///
    /// Creates a tensor filled with random values from the Exponential distribution,
    /// which models the time between events in a Poisson process.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; λ) = λ * e^(-λx)
    ///
    /// Support: x ≥ 0
    /// Mean = 1/λ
    /// Variance = 1/λ²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform sampling: X = -ln(U) / λ where U ~ Uniform(0, 1).
    ///
    /// # Arguments
    ///
    /// * `rate` - Rate parameter λ (λ > 0), also called 1/scale
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with non-negative values from Exponential(λ)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if rate ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // Average wait time of 2 seconds (rate = 0.5)
    /// let wait_times = client.exponential(0.5, &[1000], DType::F32)?;
    ///
    /// // High rate = short wait times
    /// let fast_events = client.exponential(10.0, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn exponential(
        &self,
        rate: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (rate, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::exponential",
        })
    }

    /// Sample from a Laplace (double exponential) distribution
    ///
    /// Creates a tensor filled with random values from the Laplace distribution,
    /// which has heavier tails than the normal distribution.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; μ, b) = (1/2b) * e^(-|x - μ| / b)
    ///
    /// Support: x ∈ (-∞, +∞)
    /// Mean = μ
    /// Variance = 2b²
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses inverse transform: X = μ - b * sign(U - 0.5) * ln(1 - 2|U - 0.5|)
    /// where U ~ Uniform(0, 1).
    ///
    /// # Arguments
    ///
    /// * `loc` - Location parameter μ (mean)
    /// * `scale` - Scale parameter b (b > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values from Laplace(μ, b)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if scale ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // Standard Laplace (loc=0, scale=1)
    /// let standard = client.laplace(0.0, 1.0, &[1000], DType::F32)?;
    ///
    /// // Shifted and scaled
    /// let shifted = client.laplace(5.0, 2.0, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (loc, scale, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::laplace",
        })
    }

    /// Sample from a Chi-squared distribution
    ///
    /// Creates a tensor filled with random values from the Chi-squared distribution,
    /// which is the distribution of a sum of squared standard normal variables.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; k) = x^(k/2-1) * e^(-x/2) / (2^(k/2) * Γ(k/2))
    ///
    /// Support: x > 0
    /// Mean = k
    /// Variance = 2k
    /// ```
    ///
    /// # Algorithm
    ///
    /// Implemented as Gamma(k/2, 2), since χ²(k) = Gamma(k/2, 2).
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom k (k > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from χ²(k)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // Chi-squared with 5 degrees of freedom
    /// let chi2 = client.chi_squared(5.0, &[1000], DType::F32)?;
    ///
    /// // Chi-squared test statistic distribution
    /// let test_stats = client.chi_squared(10.0, &[10000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn chi_squared(
        &self,
        df: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (df, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::chi_squared",
        })
    }

    /// Sample from a Student's t distribution
    ///
    /// Creates a tensor filled with random values from Student's t distribution,
    /// which arises in estimating the mean of a normally distributed population
    /// when the sample size is small.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; ν) = Γ((ν+1)/2) / (√(νπ) Γ(ν/2)) * (1 + x²/ν)^(-(ν+1)/2)
    ///
    /// Support: x ∈ (-∞, +∞)
    /// Mean = 0 (for ν > 1), undefined for ν ≤ 1
    /// Variance = ν/(ν-2) (for ν > 2), infinite for 1 < ν ≤ 2
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: T = Z / √(V/ν) where Z ~ N(0,1) and V ~ χ²(ν).
    ///
    /// # Arguments
    ///
    /// * `df` - Degrees of freedom ν (ν > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with values from t(ν)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // t distribution with 10 degrees of freedom
    /// let t10 = client.student_t(10.0, &[1000], DType::F32)?;
    ///
    /// // Heavy tails with low df
    /// let heavy = client.student_t(2.0, &[1000], DType::F32)?;
    ///
    /// // Approaches normal as df → ∞
    /// let approx_normal = client.student_t(100.0, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn student_t(&self, df: f64, shape: &[usize], dtype: crate::dtype::DType) -> Result<Tensor<R>> {
        let _ = (df, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::student_t",
        })
    }

    /// Sample from an F distribution
    ///
    /// Creates a tensor filled with random values from the F distribution,
    /// which arises in the analysis of variance (ANOVA) and regression.
    ///
    /// # Mathematical Definition
    ///
    /// ```text
    /// PDF(x; d₁, d₂) = √[(d₁x)^d₁ * d₂^d₂ / (d₁x + d₂)^(d₁+d₂)] / (x * B(d₁/2, d₂/2))
    ///
    /// Support: x > 0
    /// Mean = d₂/(d₂-2) (for d₂ > 2)
    /// ```
    ///
    /// # Algorithm
    ///
    /// Uses the relationship: F = (X₁/d₁) / (X₂/d₂) where X₁ ~ χ²(d₁), X₂ ~ χ²(d₂).
    ///
    /// # Arguments
    ///
    /// * `df1` - Numerator degrees of freedom d₁ (d₁ > 0)
    /// * `df2` - Denominator degrees of freedom d₂ (d₂ > 0)
    /// * `shape` - Shape of the output tensor
    /// * `dtype` - Data type of the output tensor (must be floating point)
    ///
    /// # Returns
    ///
    /// Tensor filled with positive values from F(d₁, d₂)
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if df1 ≤ 0 or df2 ≤ 0.
    /// Returns `Error::UnsupportedDType` if dtype is not floating point.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    /// use numr::dtype::DType;
    ///
    /// // F distribution for ANOVA with 5 and 20 df
    /// let f_stat = client.f_distribution(5.0, 20.0, &[1000], DType::F32)?;
    ///
    /// // Equal degrees of freedom
    /// let f_equal = client.f_distribution(10.0, 10.0, &[1000], DType::F32)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: crate::dtype::DType,
    ) -> Result<Tensor<R>> {
        let _ = (df1, df2, shape, dtype);
        Err(Error::NotImplemented {
            feature: "RandomOps::f_distribution",
        })
    }

    /// Generate a random permutation of integers [0, n)
    ///
    /// Creates a 1D tensor containing a random permutation of the integers
    /// from 0 to n-1. Output dtype is I64.
    ///
    /// # Algorithm
    ///
    /// Uses Fisher-Yates shuffle:
    /// 1. Create array [0, 1, 2, ..., n-1]
    /// 2. For i from n-1 down to 1: swap `arr[i]` with `arr[random(0..=i)]`
    ///
    /// # Arguments
    ///
    /// * `n` - Length of permutation (must be > 0)
    ///
    /// # Returns
    ///
    /// 1D tensor of shape `[n]` with dtype I64 containing a random permutation
    ///
    /// # Errors
    ///
    /// Returns `Error::InvalidArgument` if n == 0.
    ///
    /// # Examples
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// use numr::ops::RandomOps;
    ///
    /// let perm = client.randperm(5)?;
    /// // perm might be [3, 0, 4, 1, 2] (random ordering of 0..5)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn randperm(&self, n: usize) -> Result<Tensor<R>> {
        let _ = n;
        Err(Error::NotImplemented {
            feature: "RandomOps::randperm",
        })
    }
}
