//! Random number generation for CUDA runtime
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::RandomOps;
#[cfg(feature = "fp8")]
use crate::ops::TypeConversionOps;
#[cfg(feature = "fp8")]
use crate::ops::UtilityOps;
use crate::runtime::cuda::kernels::{
    launch_bernoulli, launch_beta_dist, launch_binomial, launch_chi_squared, launch_exponential,
    launch_f_distribution, launch_gamma_dist, launch_laplace, launch_multinomial_with_replacement,
    launch_multinomial_without_replacement, launch_poisson, launch_rand, launch_randint,
    launch_randn, launch_student_t,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::tensor::Tensor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

impl RandomOps<CudaRuntime> for CudaClient {
    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // FP8: generate F32 rand, clamp below FP8's 1.0 bound, then cast down
        #[cfg(feature = "fp8")]
        if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            let f32_result = self.rand(shape, DType::F32)?;
            let clamped = clamp_below_one_for_narrowing(self, &f32_result, dtype)?;
            return self.cast(&clamped, dtype);
        }

        // Supported: F32, F64, F16, BF16
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA rand kernel
        unsafe {
            launch_rand(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn rand_seeded(&self, shape: &[usize], dtype: DType, seed: u64) -> Result<Tensor<CudaRuntime>> {
        #[cfg(feature = "fp8")]
        if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            let f32_result = self.rand_seeded(shape, DType::F32, seed)?;
            let clamped = clamp_below_one_for_narrowing(self, &f32_result, dtype)?;
            return self.cast(&clamped, dtype);
        }

        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rand_seeded",
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        unsafe {
            launch_rand(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        // FP8: generate F32 randn and cast down
        #[cfg(feature = "fp8")]
        if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            let f32_result = self.randn(shape, DType::F32)?;
            return self.cast(&f32_result, dtype);
        }

        // Supported: F32, F64, F16, BF16
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();

        // Launch native CUDA randn kernel (uses Box-Muller transform)
        unsafe {
            launch_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randn_seeded(
        &self,
        shape: &[usize],
        dtype: DType,
        seed: u64,
    ) -> Result<Tensor<CudaRuntime>> {
        // FP8: generate F32 randn_seeded and cast down
        #[cfg(feature = "fp8")]
        if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
            let f32_result = self.randn_seeded(shape, DType::F32, seed)?;
            return self.cast(&f32_result, dtype);
        }

        // Supported: F32, F64, F16, BF16
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "randn_seeded",
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Launch native CUDA randn kernel (uses Box-Muller transform) with the caller's seed
        unsafe {
            launch_randn(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        // Validate dtype is integer
        if !dtype.is_int() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "randint",
            });
        }

        // Validate range
        if high <= low {
            return Err(Error::InvalidArgument {
                arg: "high",
                reason: format!(
                    "randint requires high > low, got low={}, high={}",
                    low, high
                ),
            });
        }

        // Validate range fits in unsigned dtype
        if dtype.is_unsigned_int() && low < 0 {
            return Err(Error::InvalidArgument {
                arg: "low",
                reason: format!(
                    "randint with unsigned dtype {} requires low >= 0, got low={}",
                    dtype, low
                ),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            // Empty tensor - just allocate
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        // Allocate output tensor
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // Generate seed using atomic counter + time for better entropy
        let seed = generate_random_seed();
        let range = high - low;

        // Launch native CUDA randint kernel
        unsafe {
            launch_randint(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                low,
                range,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn multinomial(
        &self,
        probs: &Tensor<CudaRuntime>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = probs.dtype();

        // Validate probs is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multinomial",
            });
        }

        // Validate num_samples
        if num_samples == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: "num_samples must be > 0".to_string(),
            });
        }

        let shape = probs.shape();
        if shape.is_empty() {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "probs tensor must have at least 1 dimension".to_string(),
            });
        }

        let num_categories = *shape.last().unwrap();
        if num_categories == 0 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "probs tensor must have at least 1 category (last dim > 0)".to_string(),
            });
        }

        // Without replacement: can't sample more than we have
        if !replacement && num_samples > num_categories {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: format!(
                    "cannot sample {} items without replacement from {} categories",
                    num_samples, num_categories
                ),
            });
        }

        // Compute number of distributions (product of all dims except last).
        // Unclamped: a 1D input already products to 1 over the empty slice, so a
        // clamp would only fabricate a distribution for a genuinely zero leading dim
        // and write samples past the end of the output.
        let num_distributions: usize = shape[..shape.len() - 1].iter().product();

        // Ensure probs is contiguous
        let probs = crate::runtime::ensure_contiguous(probs)?;

        // Output shape: [..., num_samples]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(num_samples);
        if out_shape.is_empty() {
            out_shape.push(num_samples);
        }

        let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &self.device)?;

        // No distribution to sample from, or no sample asked for. The launchers take
        // their grid from `num_distributions` and `num_samples` without flooring
        // them, so a zero there is a launch error rather than a wrong answer.
        if out.numel() == 0 {
            return Ok(out);
        }

        // Generate seed
        let seed = generate_random_seed();

        // Launch CUDA kernel
        unsafe {
            if replacement {
                launch_multinomial_with_replacement(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    probs.ptr(),
                    out.ptr(),
                    seed,
                    num_distributions,
                    num_categories,
                    num_samples,
                )?;
            } else {
                launch_multinomial_without_replacement(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    probs.ptr(),
                    out.ptr(),
                    seed,
                    num_distributions,
                    num_categories,
                    num_samples,
                )?;
            }
        }

        Ok(out)
    }

    fn bernoulli(&self, p: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bernoulli",
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                arg: "p",
                reason: format!("bernoulli requires p in [0, 1], got {}", p),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_bernoulli(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                p,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "beta" });
        }
        if alpha <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "alpha",
                reason: format!("beta requires alpha > 0, got {}", alpha),
            });
        }
        if beta <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "beta",
                reason: format!("beta requires beta > 0, got {}", beta),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_beta_dist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                alpha,
                beta,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType { dtype, op: "gamma" });
        }
        if shape_param <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "shape_param",
                reason: format!("gamma requires shape_param > 0, got {}", shape_param),
            });
        }
        if scale <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "scale",
                reason: format!("gamma requires scale > 0, got {}", scale),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_gamma_dist(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                shape_param,
                scale,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn exponential(&self, rate: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "exponential",
            });
        }
        if rate <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "rate",
                reason: format!("exponential requires rate > 0, got {}", rate),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_exponential(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                rate,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn poisson(&self, lambda: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "poisson",
            });
        }
        if lambda <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "lambda",
                reason: format!("poisson requires lambda > 0, got {}", lambda),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_poisson(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                lambda,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "binomial",
            });
        }
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "binomial requires n > 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                arg: "p",
                reason: format!("binomial requires p in [0, 1], got {}", p),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_binomial(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                n,
                p,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "laplace",
            });
        }
        if scale <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "scale",
                reason: format!("laplace requires scale > 0, got {}", scale),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_laplace(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                loc,
                scale,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn chi_squared(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "chi_squared",
            });
        }
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("chi_squared requires df > 0, got {}", df),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_chi_squared(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn student_t(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "student_t",
            });
        }
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("student_t requires df > 0, got {}", df),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_student_t(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CudaRuntime>> {
        if !matches!(dtype, DType::F32 | DType::F64 | DType::F16 | DType::BF16) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "f_distribution",
            });
        }
        if df1 <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df1",
                reason: format!("f_distribution requires df1 > 0, got {}", df1),
            });
        }
        if df2 <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df2",
                reason: format!("f_distribution requires df2 > 0, got {}", df2),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?);
        }

        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;
        let seed = generate_random_seed();

        unsafe {
            launch_f_distribution(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                df1,
                df2,
                seed,
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }

    fn randperm(&self, n: usize) -> Result<Tensor<CudaRuntime>> {
        crate::ops::impl_generic::randperm_impl(self, n)
    }
}

// ============================================================================
// Random Seed Generation Helper
// ============================================================================

/// Global atomic counter for generating unique seeds
static SEED_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Generate a random seed combining atomic counter and system time.
///
/// This provides good entropy for parallel random number generation:
/// - Atomic counter ensures uniqueness across calls
/// - System time adds unpredictability
#[inline]
fn generate_random_seed() -> u64 {
    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_component = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    // Combine counter and time using splitmix64-style mixing
    let mut z = counter.wrapping_add(time_component);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Clamp an F32 uniform sample below the target FP8 dtype's largest value
/// under 1.0, so the F32->FP8 cast that follows can't round it up to 1.0.
///
/// The bound is exactly representable in both F32 and the FP8 dtype, so
/// clamping to it needs no further rounding. `dtype` is always FP8E4M3 or
/// FP8E5M2 at the call site, both of which have a bound, so `unwrap_or`
/// never actually falls back; 1.0 is a safe no-op clamp if it ever did.
#[cfg(feature = "fp8")]
fn clamp_below_one_for_narrowing(
    client: &CudaClient,
    f32_result: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<Tensor<CudaRuntime>> {
    let bound = dtype.largest_value_below_one().unwrap_or(1.0);
    client.clamp(f32_result, f64::MIN, bound)
}
