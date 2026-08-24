//! Random operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::RandomOps;
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    BernoulliParams, BetaDistParams, BinomialParams, ChiSquaredParams, ExponentialParams,
    FDistributionParams, GammaDistParams, LaplaceParams, MultinomialWithReplacementParams,
    MultinomialWithoutReplacementParams, PoissonParams, RandParams, RandintParamsI32,
    RandintParamsU32, RandnParams, StudentTParams, alloc_output, create_params_buffer,
    generate_wgpu_seed, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::{distributions, shape};
use crate::tensor::Tensor;

impl RandomOps<WgpuRuntime> for WgpuClient {
    fn rand(&self, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU rand only supports F32
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType { dtype, op: "rand" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed
        // Note: WGSL doesn't support u64 natively, so we use u32 seed (truncated from timestamp).
        // This limits the seed space but is sufficient for most use cases.
        // For reproducible results, users should use explicit seeding (future API).
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let params = RandParams {
            numel: numel as u32,
            seed,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_rand(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn rand_seeded(&self, shape: &[usize], dtype: DType, seed: u64) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rand_seeded",
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Truncate u64 seed to u32 — WGSL has no native u64 support.
        // Determinism is still guaranteed: same seed → same u32 → same output.
        let seed = seed as u32;

        let params = RandParams {
            numel: numel as u32,
            seed,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        shape::launch_rand(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn randn(&self, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU randn only supports F32
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType { dtype, op: "randn" });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed (see rand() for seed design notes)
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let params = RandnParams {
            numel: numel as u32,
            seed,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_randn(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn randint(
        &self,
        low: i64,
        high: i64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU randint only supports I32 and U32
        if !matches!(dtype, DType::I32 | DType::U32) {
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

        // For unsigned types, validate low >= 0
        if matches!(dtype, DType::U32) && low < 0 {
            return Err(Error::InvalidArgument {
                arg: "low",
                reason: format!(
                    "randint with unsigned dtype requires low >= 0, got low={}",
                    low
                ),
            });
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Tensor::empty(shape, dtype, self.device());
        }

        // Allocate output
        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params with random seed (see rand() for seed design notes)
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(12345u32);
        let seed = time_seed.wrapping_add(counter);

        let range = (high - low) as u32;

        // Use dtype-specific param struct to ensure correct type handling
        // I32 uses signed low (i32), U32 uses unsigned low (u32)
        let params_buf = match dtype {
            DType::I32 => {
                let params = RandintParamsI32 {
                    numel: numel as u32,
                    low: low as i32, // Preserve sign for negative bounds
                    range,
                    seed,
                };
                create_params_buffer(self, &params)
            }
            DType::U32 => {
                let params = RandintParamsU32 {
                    numel: numel as u32,
                    low: low as u32,
                    range,
                    seed,
                };
                create_params_buffer(self, &params)
            }
            _ => unreachable!("randint only supports I32 and U32, validated above"),
        };

        // Launch kernel
        shape::launch_randint(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn multinomial(
        &self,
        probs: &Tensor<WgpuRuntime>,
        num_samples: usize,
        replacement: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Validate input dtype - must be float for probabilities
        let dtype = probs.dtype();
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multinomial (WebGPU only supports F32 probabilities)",
            });
        }

        // Validate num_samples > 0
        if num_samples == 0 {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: "multinomial requires num_samples > 0".to_string(),
            });
        }

        let shape = probs.shape();
        let ndim = shape.len();

        // Determine shape: 1D tensor [K] or 2D tensor [N, K]
        let (num_distributions, num_categories) = match ndim {
            1 => (1usize, shape[0]),
            2 => (shape[0], shape[1]),
            _ => {
                return Err(Error::InvalidArgument {
                    arg: "probs",
                    reason: format!(
                        "multinomial requires 1D or 2D probability tensor, got {}D",
                        ndim
                    ),
                });
            }
        };

        // Validate num_categories > 0
        if num_categories == 0 {
            return Err(Error::InvalidArgument {
                arg: "probs",
                reason: "multinomial requires at least 1 category".to_string(),
            });
        }

        // For without replacement, num_samples must not exceed num_categories
        if !replacement && num_samples > num_categories {
            return Err(Error::InvalidArgument {
                arg: "num_samples",
                reason: format!(
                    "multinomial without replacement: num_samples ({}) cannot exceed num_categories ({})",
                    num_samples, num_categories
                ),
            });
        }

        // Check category limit for without replacement (shared memory limit)
        const MAX_CATEGORIES_WITHOUT_REPLACEMENT: usize = 1024;
        if !replacement && num_categories > MAX_CATEGORIES_WITHOUT_REPLACEMENT {
            return Err(Error::backend_limitation(
                "WebGPU",
                "multinomial",
                format!(
                    "without replacement supports max {} categories, got {}",
                    MAX_CATEGORIES_WITHOUT_REPLACEMENT, num_categories
                ),
            ));
        }

        // Ensure input is contiguous
        let probs_contig = if probs.is_contiguous() {
            probs.clone()
        } else {
            probs.contiguous()?
        };

        // Output dtype: I32 for WebGPU (no I64 support in WGSL)
        let out_dtype = DType::I32;
        let out_shape = if ndim == 1 {
            vec![num_samples]
        } else {
            vec![num_distributions, num_samples]
        };

        // Allocate output
        let out = alloc_output(self, &out_shape, out_dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let probs_buf = get_tensor_buffer(&probs_contig)?;

        // Generate random seed
        use std::sync::atomic::{AtomicU32, Ordering};
        static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);
        let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
        let time_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(54321u32);
        let seed = time_seed.wrapping_add(counter);

        if replacement {
            // With replacement: parallel sampling
            let params = MultinomialWithReplacementParams {
                num_distributions: num_distributions as u32,
                num_categories: num_categories as u32,
                num_samples: num_samples as u32,
                seed,
            };
            let params_buf = create_params_buffer(self, &params);

            let total_samples = num_distributions * num_samples;
            shape::launch_multinomial_with_replacement(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &probs_buf,
                &out_buf,
                &params_buf,
                total_samples,
                dtype,
            )?;
        } else {
            // Without replacement: sequential sampling with shared memory
            let params = MultinomialWithoutReplacementParams {
                num_distributions: num_distributions as u32,
                num_categories: num_categories as u32,
                num_samples: num_samples as u32,
                seed,
            };
            let params_buf = create_params_buffer(self, &params);

            shape::launch_multinomial_without_replacement(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &probs_buf,
                &out_buf,
                &params_buf,
                num_distributions,
                dtype,
            )?;
        }

        Ok(out)
    }

    fn bernoulli(&self, p: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bernoulli (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BernoulliParams {
            numel: numel as u32,
            seed,
            p: p as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_bernoulli(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn beta(
        &self,
        alpha: f64,
        beta: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "beta (WebGPU only supports F32)",
            });
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BetaDistParams {
            numel: numel as u32,
            seed,
            alpha: alpha as f32,
            beta: beta as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_beta_dist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn gamma(
        &self,
        shape_param: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "gamma (WebGPU only supports F32)",
            });
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = GammaDistParams {
            numel: numel as u32,
            seed,
            shape: shape_param as f32,
            scale: scale as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_gamma_dist(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn exponential(&self, rate: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "exponential (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = ExponentialParams {
            numel: numel as u32,
            seed,
            rate: rate as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_exponential(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn poisson(&self, lambda: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "poisson (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = PoissonParams {
            numel: numel as u32,
            seed,
            lambda: lambda as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_poisson(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn binomial(
        &self,
        n: u64,
        p: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "binomial (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = BinomialParams {
            numel: numel as u32,
            seed,
            n_trials: n as u32, // WebGPU doesn't support u64, truncate to u32
            p: p as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_binomial(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn laplace(
        &self,
        loc: f64,
        scale: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "laplace (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = LaplaceParams {
            numel: numel as u32,
            seed,
            loc: loc as f32,
            scale: scale as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_laplace(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn chi_squared(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "chi_squared (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = ChiSquaredParams {
            numel: numel as u32,
            seed,
            df: df as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_chi_squared(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn student_t(&self, df: f64, shape: &[usize], dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "student_t (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = StudentTParams {
            numel: numel as u32,
            seed,
            df: df as f32,
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_student_t(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn randperm(&self, n: usize) -> Result<Tensor<WgpuRuntime>> {
        crate::ops::impl_generic::randperm_impl(self, n)
    }

    fn f_distribution(
        &self,
        df1: f64,
        df2: f64,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "f_distribution (WebGPU only supports F32)",
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
            return Tensor::empty(shape, dtype, self.device());
        }

        let out = alloc_output(self, shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let seed = generate_wgpu_seed();

        let params = FDistributionParams {
            numel: numel as u32,
            seed,
            df1: df1 as f32,
            df2: df2 as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        distributions::launch_f_distribution(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }
}
