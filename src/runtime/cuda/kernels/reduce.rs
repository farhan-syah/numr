//! Reduction CUDA kernel launchers
//!
//! Provides launchers for reduction operations (sum, max, min, and the fused
//! integer mean) that reduce tensors along specified dimensions.
//!
//! See [`AccumulationPrecision`] for documentation on accumulation precision options.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    dtype_suffix, get_kernel_function, get_or_load_module, kernel_name, kernel_names,
    launch_config, reduce_dim_launch_config, reduce_launch_config, reduce_module,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
// Re-export AccumulationPrecision from ops for convenience
pub(crate) use crate::ops::AccumulationPrecision;

/// Reduction ops whose F16/BF16 kernels ship an `_fp32acc` variant.
///
/// `reduce.cu` instantiates `_fp32acc` for the four accumulating dim
/// reductions only. `reduce_any_dim` and `reduce_all_dim` hold no accumulator
/// and exist in the native suffix alone, so asking for `_fp32acc` there would
/// name a kernel that does not exist.
fn has_fp32acc_variant(base_op: &str) -> bool {
    matches!(
        base_op,
        "reduce_sum_dim" | "reduce_max_dim" | "reduce_min_dim" | "reduce_prod_dim"
    )
}

/// Generate kernel name with accumulation precision suffix.
///
/// Kernel naming conventions:
/// - F16/BF16: `{op}_{dtype}_fp32acc` (FP32, and the default), `{op}_{dtype}_fp64acc` (FP64),
///   `{op}_{dtype}` (native, reachable only for ops with no `_fp32acc` variant)
/// - FP8: `{op}_{dtype}` (FP32 default), `{op}_{dtype}_bf16acc` (BF16), `{op}_{dtype}_fp64acc` (FP64)
/// - F32: `{op}_{dtype}` (native) or `{op}_{dtype}_fp64acc` (FP64)
/// - F64/integers: `{op}_{dtype}` (always native, ignore acc_precision)
fn reduce_kernel_name(base_op: &str, dtype: DType, acc_precision: AccumulationPrecision) -> String {
    let suffix = dtype_suffix(dtype);

    // Determine accumulation suffix based on dtype and requested precision
    let acc_suffix = match dtype {
        // F16/BF16: FP32 accumulation by default, _fp64acc for FP64.
        //
        // `Native` means "let the library choose", and the library never
        // chooses F16/BF16: a running sum held in 16 bits stops growing once
        // its spacing exceeds twice the increment, so the reduction stalls on
        // a constant regardless of the data. This matches the default stated
        // in `reduce.cu`.
        DType::F16 | DType::BF16 => match acc_precision {
            AccumulationPrecision::FP64 => Some("_fp64acc"),
            AccumulationPrecision::FP32
            | AccumulationPrecision::Native
            | AccumulationPrecision::BF16 => {
                if has_fp32acc_variant(base_op) {
                    Some("_fp32acc")
                } else {
                    None
                }
            }
        },
        // FP8: FP32 by default (no suffix), _bf16acc for BF16, _fp64acc for FP64
        DType::FP8E4M3 | DType::FP8E5M2 => match acc_precision {
            AccumulationPrecision::BF16 => Some("_bf16acc"),
            AccumulationPrecision::FP64 => Some("_fp64acc"),
            // Native and FP32 both map to FP32 accumulation for FP8
            AccumulationPrecision::Native | AccumulationPrecision::FP32 => None,
        },
        // F32: native by default, _fp64acc for maximum precision
        DType::F32 => match acc_precision {
            AccumulationPrecision::FP64 => Some("_fp64acc"),
            // Native, BF16, FP32 all use native f32 accumulation
            _ => None,
        },
        // F64/integers: always native, ignore acc_precision
        _ => None,
    };

    match acc_suffix {
        Some(s) => format!("{}_{}{}", base_op, suffix, s),
        None => format!("{}_{}", base_op, suffix),
    }
}

/// Launch a global reduction kernel.
///
/// Performs a parallel reduction across all elements, producing partial results
/// (one per block). For complete reduction, call multiple times until only one
/// element remains.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have at least `numel` elements
/// - `output_ptr` must have space for the number of blocks launched
///
/// # Returns
///
/// The number of blocks launched (equals the number of partial results).
#[allow(dead_code)] // Kept for potential future optimization of global reductions
pub unsafe fn launch_reduce_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<u32> {
    unsafe {
        let module = get_or_load_module(context, device_index, reduce_module(dtype))?;
        let func_name = kernel_name(&kernel_names::reduce_kernel(op), dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid_size, block_size) = reduce_launch_config(numel);
        let n = numel as u32;

        let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA reduce kernel '{}' launch failed: {:?}",
                op, e
            ))
        })?;

        Ok(grid_size)
    }
}

/// Launch a dimension-wise reduction kernel.
///
/// Reduces a tensor along a single dimension, preserving the outer and inner
/// dimensions. The tensor is conceptually reshaped to `[outer, reduce, inner]`
/// and reduced along the middle dimension.
///
/// # Accumulation Precision
///
/// For F16/BF16 dtypes, set `acc_precision` to control accumulation:
/// - `AccumulationPrecision::Native`: Use native dtype (faster, default)
/// - `AccumulationPrecision::FP32`: Use FP32 accumulation (more precise)
///
/// FP8 types always use FP32 accumulation regardless of this setting.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `output_ptr` must have `outer_size * inner_size` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Reduction operation ("sum", "max", or "min")
/// * `dtype` - Data type of the tensor
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
/// * `acc_precision` - Accumulation precision (affects F16/BF16 only)
pub unsafe fn launch_reduce_dim_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    acc_precision: AccumulationPrecision,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, reduce_module(dtype))?;
        let base_op = kernel_names::reduce_dim_kernel(op);
        let func_name = reduce_kernel_name(&base_op, dtype, acc_precision);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);
        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA reduce_dim kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch the fused integer `reduce_mean_dim` kernel along one dimension.
///
/// Only integer dtypes have this kernel (see `reduce_int.cu`): it sums the
/// reduced axis in a 128-bit accumulator, divides by `divisor` inside that
/// accumulator, and narrows to the element type exactly once. Unlike
/// [`launch_reduce_dim_op`], `divisor` is a caller-supplied kernel argument
/// rather than always `reduce_size` - a plain mean passes `reduce_size`
/// itself, but an unbiased variance's `n - correction` is not the reduced
/// axis length and needs the same divide-once guarantee.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `output_ptr` must have space for `outer_size * inner_size` elements
/// - `dtype` must be an integer dtype - the fused mean kernel has no float
///   instantiation
///
/// # Arguments
///
/// * `divisor` - value the finished sum is divided by; the kernel forces a
///   zero divisor to 1, mirroring the CPU epilogue's `count.max(1)`
pub unsafe fn launch_reduce_mean_dim_int_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
    divisor: u64,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, reduce_module(dtype))?;
        let func_name = kernel_name("reduce_mean_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);
        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);
        builder.arg(&divisor);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA reduce_mean_dim kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch an argmax kernel along a dimension.
///
/// Returns indices (I64) of maximum values along the specified dimension.
/// The tensor is conceptually reshaped to `[outer, reduce, inner]` and
/// argmax is computed along the middle dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `output_ptr` must have `outer_size * inner_size` I64 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the input tensor (output is always I64)
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor (I64 indices)
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
pub unsafe fn launch_argmax_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, reduce_module(dtype))?;
        let func_name = kernel_name("argmax_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);
        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA argmax_dim kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch an argmin kernel along a dimension.
///
/// Returns indices (I64) of minimum values along the specified dimension.
/// The tensor is conceptually reshaped to `[outer, reduce, inner]` and
/// argmin is computed along the middle dimension.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `input_ptr` must have `outer_size * reduce_size * inner_size` elements
/// - `output_ptr` must have `outer_size * inner_size` I64 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of the input tensor (output is always I64)
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor (I64 indices)
/// * `outer_size` - Product of dimensions before the reduction dimension
/// * `reduce_size` - Size of the dimension being reduced
/// * `inner_size` - Product of dimensions after the reduction dimension
pub unsafe fn launch_argmin_dim(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    reduce_size: usize,
    inner_size: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, reduce_module(dtype))?;
        let func_name = kernel_name("argmin_dim", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let (grid, block) = reduce_dim_launch_config(outer_size, inner_size);
        let outer = outer_size as u32;
        let reduce = reduce_size as u32;
        let inner = inner_size as u32;

        let cfg = launch_config(grid, (block, 1, 1), 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&outer);
        builder.arg(&reduce);
        builder.arg(&inner);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA argmin_dim kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}
