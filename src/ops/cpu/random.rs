//! CPU implementation of random operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::RandomOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{dispatch_dtype, ensure_contiguous},
    kernels,
};
use crate::tensor::Tensor;

/// RandomOps implementation for CPU runtime.
impl RandomOps<CpuRuntime> for CpuClient {
    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        // Handle empty tensor
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rand_uniform_kernel::<T>(out_ptr as *mut T, numel);
            }
        }, "rand");

        Ok(out)
    }

    fn rand_seeded(&self, shape: &[usize], dtype: DType, seed: u64) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rand_seeded",
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::xoshiro256_uniform_kernel::<T>(
                    out_ptr as *mut T, numel, seed,
                );
            }
        }, "rand_seeded");

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        // Handle empty tensor
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::rand_normal_kernel::<T>(out_ptr as *mut T, numel);
            }
        }, "randn");

        Ok(out)
    }

    fn randn_seeded(&self, shape: &[usize], dtype: DType, seed: u64) -> Result<Tensor<CpuRuntime>> {
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "randn_seeded",
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::xoshiro256_randn_kernel::<T>(
                    out_ptr as *mut T, numel, seed,
                );
            }
        }, "randn_seeded");

        Ok(out)
    }

    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
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

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();

        // Handle empty tensor
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::randint_kernel::<T>(out_ptr as *mut T, low, high, numel);
            }
        }, "randint");

        Ok(out)
    }

    fn multinomial(
        &self,
        probs: &Tensor<CpuRuntime>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<CpuRuntime>> {
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
        let ndim = shape.len();

        // Validate tensor is 1D or 2D (like PyTorch)
        if ndim == 0 || ndim > 2 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: format!(
                    "multinomial requires 1D or 2D probability tensor, got {}D",
                    ndim
                ),
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

        // Compute number of distributions (product of all dims except last)
        let num_distributions: usize = shape[..shape.len() - 1].iter().product();
        let num_distributions = num_distributions.max(1); // At least 1 for 1D input

        // Ensure probs is contiguous
        let probs = ensure_contiguous(probs)?;

        // Validate that probabilities sum to a positive value to prevent division by zero
        // Check the max value - if all values are <= 0, we cannot sample
        let max_prob: f64 = match dtype {
            DType::F32 => {
                let data: &[f32] =
                    unsafe { std::slice::from_raw_parts(probs.ptr() as *const f32, probs.numel()) };
                data.iter()
                    .cloned()
                    .fold(f64::NEG_INFINITY, |a, b| a.max(b as f64))
            }
            DType::F64 => {
                let data: &[f64] =
                    unsafe { std::slice::from_raw_parts(probs.ptr() as *const f64, probs.numel()) };
                data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            }
            _ => {
                // For F16/BF16, we still need to check but these are behind feature flags
                // For simplicity, skip validation for these rare cases (kernel will handle gracefully)
                f64::INFINITY
            }
        };
        if max_prob <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "probabilities must contain at least one positive value".to_string(),
            });
        }

        // Output shape: [..., num_samples]
        let mut out_shape = shape[..shape.len() - 1].to_vec();
        out_shape.push(num_samples);
        if out_shape.is_empty() {
            out_shape.push(num_samples);
        }

        let out = Tensor::<CpuRuntime>::empty(&out_shape, DType::I64, &self.device)?;
        let out_ptr = out.ptr() as *mut i64;
        let probs_ptr = probs.ptr();

        // Dispatch based on input dtype
        dispatch_dtype!(dtype, T => {
            unsafe {
                if replacement {
                    kernels::multinomial_kernel_with_replacement::<T>(
                        probs_ptr as *const T,
                        out_ptr,
                        num_distributions,
                        num_categories,
                        num_samples,
                    );
                } else {
                    kernels::multinomial_kernel_without_replacement::<T>(
                        probs_ptr as *const T,
                        out_ptr,
                        num_distributions,
                        num_categories,
                        num_samples,
                    );
                }
            }
        }, "multinomial");

        Ok(out)
    }

    fn bernoulli(&self, p: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bernoulli",
            });
        }

        // Validate probability
        if !(0.0..=1.0).contains(&p) {
            return Err(Error::InvalidArgument {
                arg: "p",
                reason: format!("bernoulli requires p in [0, 1], got {}", p),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::bernoulli_kernel::<T>(out_ptr as *mut T, p, numel); }
        }, "bernoulli");

        Ok(out)
    }

    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "beta" });
        }

        // Validate parameters
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

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::beta_kernel::<T>(out_ptr as *mut T, alpha, beta, numel); }
        }, "beta");

        Ok(out)
    }

    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType { dtype, op: "gamma" });
        }

        // Validate parameters
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

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::gamma_kernel::<T>(out_ptr as *mut T, shape_param, scale, numel); }
        }, "gamma");

        Ok(out)
    }

    fn exponential(&self, rate: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "exponential",
            });
        }

        // Validate parameter
        if rate <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "rate",
                reason: format!("exponential requires rate > 0, got {}", rate),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::exponential_kernel::<T>(out_ptr as *mut T, rate, numel); }
        }, "exponential");

        Ok(out)
    }

    fn poisson(&self, lambda: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "poisson",
            });
        }

        // Validate parameter
        if lambda <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "lambda",
                reason: format!("poisson requires lambda > 0, got {}", lambda),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::poisson_kernel::<T>(out_ptr as *mut T, lambda, numel); }
        }, "poisson");

        Ok(out)
    }

    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "binomial",
            });
        }

        // Validate parameters
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

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::binomial_kernel::<T>(out_ptr as *mut T, n, p, numel); }
        }, "binomial");

        Ok(out)
    }

    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "laplace",
            });
        }

        // Validate parameter
        if scale <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "scale",
                reason: format!("laplace requires scale > 0, got {}", scale),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::laplace_kernel::<T>(out_ptr as *mut T, loc, scale, numel); }
        }, "laplace");

        Ok(out)
    }

    fn chi_squared(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "chi_squared",
            });
        }

        // Validate parameter
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("chi_squared requires df > 0, got {}", df),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::chi_squared_kernel::<T>(out_ptr as *mut T, df, numel); }
        }, "chi_squared");

        Ok(out)
    }

    fn student_t(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "student_t",
            });
        }

        // Validate parameter
        if df <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "df",
                reason: format!("student_t requires df > 0, got {}", df),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::student_t_kernel::<T>(out_ptr as *mut T, df, numel); }
        }, "student_t");

        Ok(out)
    }

    fn randperm(&self, n: usize) -> Result<Tensor<CpuRuntime>> {
        if n == 0 {
            return Err(Error::InvalidArgument {
                arg: "n",
                reason: "randperm requires n > 0".to_string(),
            });
        }

        let out = Tensor::<CpuRuntime>::empty(&[n], DType::I64, &self.device)?;
        let out_ptr = out.ptr() as *mut i64;

        unsafe {
            kernels::randperm_kernel(out_ptr, n);
        }

        Ok(out)
    }

    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate dtype is floating point
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "f_distribution",
            });
        }

        // Validate parameters
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

        let out = Tensor::<CpuRuntime>::empty(shape, dtype, &self.device)?;
        let numel = out.numel();
        if numel == 0 {
            return Ok(out);
        }

        let out_ptr = out.ptr();
        dispatch_dtype!(dtype, T => {
            unsafe { kernels::f_distribution_kernel::<T>(out_ptr as *mut T, df1, df2, numel); }
        }, "f_distribution");

        Ok(out)
    }
}
