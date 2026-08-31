//! CUDA kernel launchers for distribution sampling operations

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

/// Launch a Bernoulli sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_bernoulli(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    p: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("bernoulli", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&p);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA bernoulli kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Beta distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_beta_dist(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    alpha: f64,
    beta: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("beta", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&alpha);
        builder.arg(&beta);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA beta kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Gamma distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_gamma_dist(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    shape_param: f64,
    scale: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("gamma", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&shape_param);
        builder.arg(&scale);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA gamma kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch an Exponential distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_exponential(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    rate: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("exponential", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&rate);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA exponential kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Poisson distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_poisson(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    lambda: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("poisson", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&lambda);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA poisson kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Binomial distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_binomial(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    n_trials: u64,
    p: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("binomial", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let count = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n_trials);
        builder.arg(&p);
        builder.arg(&seed);
        builder.arg(&count);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA binomial kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Laplace distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_laplace(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    loc: f64,
    scale: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("laplace", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&loc);
        builder.arg(&scale);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA laplace kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Chi-squared distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_chi_squared(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    df: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("chi_squared", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&df);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA chi_squared kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a Student's t distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_student_t(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    df: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("student_t", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&df);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA student_t kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch an F distribution sampling kernel.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `numel` elements
pub unsafe fn launch_f_distribution(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    df1: f64,
    df2: f64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("f_distribution", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&df1);
        builder.arg(&df2);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA f_distribution kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a multinomial count kernel.
///
/// Performs CDF lookup for uniform samples and counts occurrences per category.
/// Used for multinomial sampling: given uniform samples and a CDF, counts how
/// many samples fall into each category.
///
/// # Arguments
/// * `cdf_ptr` - Device pointer to CDF array [k]
/// * `uniforms_ptr` - Device pointer to uniform samples [n_samples, n_trials]
/// * `out_ptr` - Device pointer for output counts [n_samples, k]
/// * `k` - Number of categories
/// * `n_trials` - Number of trials per sample
/// * `n_samples` - Number of samples
///
/// # Safety
/// - All pointers must be valid device pointers with correct sizes
pub unsafe fn launch_multinomial_count(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    cdf_ptr: u64,
    uniforms_ptr: u64,
    out_ptr: u64,
    k: usize,
    n_trials: usize,
    n_samples: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::DISTRIBUTIONS_MODULE)?;
    let func_name = kernel_name("multinomial_count", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // Grid: one block per sample
    // Block: min(n_trials, 256) threads
    let block_size = n_trials.min(256) as u32;
    let grid = (n_samples as u32, 1, 1);
    let block = (block_size, 1, 1);

    // Shared memory: k * sizeof(unsigned int) for counting
    let shared_mem_bytes = (k * std::mem::size_of::<u32>()) as u32;
    let cfg = launch_config(grid, block, shared_mem_bytes);

    let k_param = k as u32;
    let n_trials_param = n_trials as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cdf_ptr);
        builder.arg(&uniforms_ptr);
        builder.arg(&out_ptr);
        builder.arg(&k_param);
        builder.arg(&n_trials_param);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA multinomial_count kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}
