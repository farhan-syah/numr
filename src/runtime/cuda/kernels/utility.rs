//! Utility CUDA kernel launchers
//!
//! Provides launchers for utility operations:
//! - `fill` - Initialize tensor with constant value
//! - `rand` - Generate uniform random values in [0, 1)
//! - `randn` - Generate normal random values (mean=0, std=1)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Value representation for fill operations.
///
/// This enum allows passing fill values of different types through a unified interface
/// while maintaining type safety at the kernel boundary.
#[derive(Debug, Clone, Copy)]
pub enum FillValue {
    /// 32-bit float fill value.
    F32(f32),
    /// 64-bit float fill value.
    F64(f64),
    /// 32-bit signed integer fill value.
    I32(i32),
    /// 64-bit signed integer fill value.
    I64(i64),
    /// 16-bit signed integer fill value.
    I16(i16),
    /// 8-bit signed integer fill value.
    I8(i8),
    /// 64-bit unsigned integer fill value.
    U64(u64),
    /// 32-bit unsigned integer fill value.
    U32(u32),
    /// 16-bit unsigned integer fill value.
    U16(u16),
    /// 8-bit unsigned integer fill value (also used for Bool).
    U8(u8),
    /// 16-bit float fill value (raw bits for __half).
    #[cfg(feature = "f16")]
    F16(u16),
    /// 16-bit bfloat fill value (raw bits for __nv_bfloat16).
    #[cfg(feature = "f16")]
    BF16(u16),
    /// FP8 E4M3 fill value (raw bits).
    FP8E4M3(u8),
    /// FP8 E5M2 fill value (raw bits).
    FP8E5M2(u8),
}

impl FillValue {
    /// Create a FillValue from an f64, converting to the appropriate type for the given dtype.
    pub fn from_f64(value: f64, dtype: DType) -> Self {
        match dtype {
            DType::F32 => FillValue::F32(value as f32),
            DType::F64 => FillValue::F64(value),
            DType::I32 => FillValue::I32(value as i32),
            DType::I64 => FillValue::I64(value as i64),
            DType::I16 => FillValue::I16(value as i16),
            DType::I8 => FillValue::I8(value as i8),
            DType::U64 => FillValue::U64(value as u64),
            DType::U32 => FillValue::U32(value as u32),
            DType::U16 => FillValue::U16(value as u16),
            DType::U8 | DType::Bool => FillValue::U8(value as u8),
            #[cfg(feature = "f16")]
            DType::F16 => FillValue::F16(half::f16::from_f64(value).to_bits()),
            #[cfg(feature = "f16")]
            DType::BF16 => FillValue::BF16(half::bf16::from_f64(value).to_bits()),
            DType::FP8E4M3 => {
                FillValue::FP8E4M3(crate::dtype::fp8::FP8E4M3::from_f64(value).to_bits())
            }
            DType::FP8E5M2 => {
                FillValue::FP8E5M2(crate::dtype::fp8::FP8E5M2::from_f64(value).to_bits())
            }
            // Complex is the only remaining case, and it has no fill kernel.
            // F64 is the widest scalar available, and launch_fill reports the
            // missing kernel by name rather than writing the wrong width.
            _ => FillValue::F64(value),
        }
    }

    /// Get the dtype this value corresponds to for kernel dispatch.
    fn kernel_dtype(&self) -> DType {
        match self {
            FillValue::F32(_) => DType::F32,
            FillValue::F64(_) => DType::F64,
            FillValue::I32(_) => DType::I32,
            FillValue::I64(_) => DType::I64,
            FillValue::I16(_) => DType::I16,
            FillValue::I8(_) => DType::I8,
            FillValue::U64(_) => DType::U64,
            FillValue::U32(_) => DType::U32,
            FillValue::U16(_) => DType::U16,
            FillValue::U8(_) => DType::U8,
            #[cfg(feature = "f16")]
            FillValue::F16(_) => DType::F16,
            #[cfg(feature = "f16")]
            FillValue::BF16(_) => DType::BF16,
            FillValue::FP8E4M3(_) => DType::FP8E4M3,
            FillValue::FP8E5M2(_) => DType::FP8E5M2,
        }
    }
}

/// Launch a fill kernel for any supported dtype.
///
/// Fills the output tensor with a constant value. This is the unified entry point
/// that dispatches to the appropriate typed kernel.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements of the given dtype
/// - The `value` dtype must match the actual data type at `out_ptr`
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the output tensor
/// * `value` - Value to fill with (will be converted to appropriate type)
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
///
/// # Example
///
/// ```ignore
/// // Fill with f32
/// unsafe {
///     launch_fill(ctx, stream, 0, DType::F32, FillValue::F32(1.0), ptr, 1024)?;
/// }
///
/// // Fill with automatic conversion from f64
/// unsafe {
///     launch_fill(ctx, stream, 0, DType::I32, FillValue::from_f64(42.0, DType::I32), ptr, 1024)?;
/// }
/// ```
pub unsafe fn launch_fill(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    _dtype: DType,
    value: FillValue,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_MODULE)?;
    let func_name = kernel_name("fill", value.kernel_dtype());
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    // Build and launch inside each match arm to ensure value lives long enough
    // SAFETY: All launch calls use valid kernel arguments with correct types
    let launch_result = match value {
        FillValue::F32(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::F64(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I32(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I64(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I16(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::I8(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::U64(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::U32(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::U16(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::U8(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        #[cfg(feature = "f16")]
        FillValue::F16(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        #[cfg(feature = "f16")]
        FillValue::BF16(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::FP8E4M3(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
        FillValue::FP8E5M2(v) => {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&out_ptr);
            builder.arg(&v);
            builder.arg(&n);
            unsafe { builder.launch(cfg) }
        }
    };

    launch_result.map_err(|e| {
        Error::Internal(format!(
            "CUDA fill kernel '{}' launch failed: {:?}",
            func_name, e
        ))
    })?;

    Ok(())
}

/// Convenience function: Launch a fill kernel from an f64 value.
///
/// Automatically converts the f64 value to the appropriate type for the given dtype.
///
/// # Safety
///
/// Same requirements as [`launch_fill`].
pub unsafe fn launch_fill_with_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    value: f64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    // SAFETY: Caller must ensure out_ptr is valid device memory
    unsafe {
        launch_fill(
            context,
            stream,
            device_index,
            dtype,
            FillValue::from_f64(value, dtype),
            out_ptr,
            numel,
        )
    }
}

// ============================================================================
// Random Number Generation Kernels
// ============================================================================

/// Launch a uniform random kernel: generates values in [0, 1).
///
/// Uses xorshift128+ PRNG with per-element seeding based on global thread index.
/// This ensures reproducibility for a given seed.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements
/// - Supports F32, F64, F16, BF16 dtypes
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type (must be floating point)
/// * `seed` - Random seed for reproducibility
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_rand(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_RANDOM_MODULE)?;
    let func_name = kernel_name("rand", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA rand kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a normal random kernel: generates values from N(0, 1).
///
/// Uses Box-Muller transform with xorshift128+ PRNG.
/// Each thread generates a pair of normal random values for efficiency.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements
/// - Supports F32, F64, F16, BF16 dtypes
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type (must be floating point)
/// * `seed` - Random seed for reproducibility
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_randn(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_RANDOM_MODULE)?;
    let func_name = kernel_name("randn", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // Box-Muller processes pairs, so we launch half the threads (rounded up)
    let thread_count = (numel + 1) / 2;
    let grid = elementwise_launch_config(thread_count)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA randn kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a random integer kernel: generates integers in [low, high).
///
/// Uses xorshift128+ PRNG with modulo for uniform distribution.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements
/// - Supports all integer dtypes: I8, I16, I32, I64, U8, U16, U32, U64
/// - `range` must be positive (high - low)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type (must be integer type)
/// * `low` - Lower bound (inclusive)
/// * `range` - Range size (high - low, must be > 0)
/// * `seed` - Random seed for reproducibility
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_randint(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    low: i64,
    range: i64,
    seed: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_RANDOM_MODULE)?;
    let func_name = kernel_name("randint", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&low);
        builder.arg(&range);
        builder.arg(&seed);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA randint kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Arange Kernel
// ============================================================================

/// Launch an arange kernel: generates evenly spaced values in [start, stop).
///
/// Values are generated using: start + step * i for i in 0..numel
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `numel` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the output
/// * `start` - Start of the interval
/// * `step` - Step between values
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_arange(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    start: f64,
    step: f64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_MODULE)?;
    let func_name = kernel_name("arange", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    // Every dtype's kernel takes the f64 start and step and narrows once at the
    // store, which is what the CPU kernel does. Narrowing the scalars here
    // instead would build each value in f32: a precision loss for F32 output,
    // a double rounding for F16/BF16/FP8, and a wrap for an integer dtype that
    // cannot hold the start or step.
    match dtype {
        DType::F32 | DType::F64 => {}
        #[cfg(feature = "f16")]
        DType::F16 | DType::BF16 => {}
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 | DType::FP8E5M2 => {}
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8 => {}
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "arange",
            });
        }
    }

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&start);
        builder.arg(&step);
        builder.arg(&n);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA arange kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Linspace Kernel
// ============================================================================

/// Launch a linspace kernel: generates evenly spaced values from start to stop (inclusive).
///
/// Values are generated using: start + (stop - start) * i / (steps - 1)
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `steps` elements
/// - `steps` must be >= 2
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the output (supports float and integer types)
/// * `start` - Start of the interval
/// * `stop` - End of the interval (inclusive)
/// * `out_ptr` - Device pointer to output tensor
/// * `steps` - Number of values to generate
pub unsafe fn launch_linspace(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    start: f64,
    stop: f64,
    out_ptr: u64,
    steps: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_MODULE)?;
    let func_name = kernel_name("linspace", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(steps)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = steps as u32;
    let cfg = launch_config(grid, block, 0);

    // Same widening as arange: the kernel takes f64 start and stop for every
    // dtype and narrows once at the store.
    match dtype {
        DType::F32 | DType::F64 => {}
        #[cfg(feature = "f16")]
        DType::F16 | DType::BF16 => {}
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 | DType::FP8E5M2 => {}
        DType::I64
        | DType::I32
        | DType::I16
        | DType::I8
        | DType::U64
        | DType::U32
        | DType::U16
        | DType::U8 => {}
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "linspace",
            });
        }
    }

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&start);
        builder.arg(&stop);
        builder.arg(&n);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA linspace kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Eye Kernel
// ============================================================================

/// Launch an eye kernel: generates identity matrix.
///
/// Creates a matrix with ones on the diagonal and zeros elsewhere.
///
/// # Safety
///
/// - `out_ptr` must be valid device memory with at least `n * m` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the output
/// * `n` - Number of rows
/// * `m` - Number of columns
/// * `out_ptr` - Device pointer to output tensor
pub unsafe fn launch_eye(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    n: usize,
    m: usize,
    out_ptr: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_MODULE)?;
    let func_name = kernel_name("eye", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let numel = n * m;
    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n_u32 = n as u32;
    let m_u32 = m as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n_u32);
        builder.arg(&m_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA eye kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Multinomial Sampling Kernels
// ============================================================================

/// Launch a multinomial sampling kernel with replacement.
///
/// Uses inverse transform sampling (CDF method):
/// 1. Compute cumulative sum of normalized probabilities
/// 2. For each sample, draw uniform random u ∈ [0, 1)
/// 3. Find smallest index i where CDF[i] ≥ u
///
/// # Safety
///
/// - `probs_ptr` must be valid device memory with at least `num_distributions * num_categories` elements
/// - `out_ptr` must be valid device memory with at least `num_distributions * num_samples` i64 elements
/// - Supports F32, F64, F16, BF16 dtypes
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of probabilities (must be floating point)
/// * `probs_ptr` - Device pointer to probability tensor
/// * `out_ptr` - Device pointer to output tensor (i64)
/// * `seed` - Random seed for reproducibility
/// * `num_distributions` - Number of independent distributions
/// * `num_categories` - Number of categories per distribution
/// * `num_samples` - Number of samples to draw per distribution
pub unsafe fn launch_multinomial_with_replacement(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    probs_ptr: u64,
    out_ptr: u64,
    seed: u64,
    num_distributions: usize,
    num_categories: usize,
    num_samples: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_RANDOM_MODULE)?;
    let func_name = format!("multinomial_with_replacement_{}", dtype_suffix(dtype)?);
    let func = get_kernel_function(&module, &func_name)?;

    let total = num_distributions * num_samples;
    let grid = elementwise_launch_config(total)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let num_distributions_u32 = num_distributions as u32;
    let num_categories_u32 = num_categories as u32;
    let num_samples_u32 = num_samples as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&probs_ptr);
        builder.arg(&out_ptr);
        builder.arg(&seed);
        builder.arg(&num_distributions_u32);
        builder.arg(&num_categories_u32);
        builder.arg(&num_samples_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA multinomial_with_replacement kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Launch a multinomial sampling kernel without replacement.
///
/// Uses sequential sampling within each distribution where each thread block
/// handles one distribution. Selected categories are zeroed out to prevent resampling.
///
/// # Safety
///
/// - `probs_ptr` must be valid device memory with at least `num_distributions * num_categories` elements
/// - `out_ptr` must be valid device memory with at least `num_distributions * num_samples` i64 elements
/// - `num_samples <= num_categories`
/// - Supports F32, F64, F16, BF16 dtypes
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of probabilities (must be floating point)
/// * `probs_ptr` - Device pointer to probability tensor
/// * `out_ptr` - Device pointer to output tensor (i64)
/// * `seed` - Random seed for reproducibility
/// * `num_distributions` - Number of independent distributions
/// * `num_categories` - Number of categories per distribution
/// * `num_samples` - Number of samples to draw per distribution
pub unsafe fn launch_multinomial_without_replacement(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    probs_ptr: u64,
    out_ptr: u64,
    seed: u64,
    num_distributions: usize,
    num_categories: usize,
    num_samples: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::UTILITY_RANDOM_MODULE)?;
    let func_name = format!("multinomial_without_replacement_{}", dtype_suffix(dtype)?);
    let func = get_kernel_function(&module, &func_name)?;

    // Each block handles one distribution
    let grid = (num_distributions as u32, 1, 1);
    let block = (BLOCK_SIZE, 1, 1);
    // Shared memory for probabilities array
    let shared_mem = num_categories * std::mem::size_of::<f64>();
    let cfg = launch_config(grid, block, shared_mem as u32);

    let num_distributions_u32 = num_distributions as u32;
    let num_categories_u32 = num_categories as u32;
    let num_samples_u32 = num_samples as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&probs_ptr);
        builder.arg(&out_ptr);
        builder.arg(&seed);
        builder.arg(&num_distributions_u32);
        builder.arg(&num_categories_u32);
        builder.arg(&num_samples_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA multinomial_without_replacement kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}

/// Helper function to get dtype suffix for kernel name
fn dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F64 => Ok("f64"),
        #[cfg(feature = "f16")]
        DType::F16 => Ok("f16"),
        #[cfg(feature = "f16")]
        DType::BF16 => Ok("bf16"),
        _ => Err(Error::UnsupportedDType {
            dtype,
            op: "multinomial",
        }),
    }
}
