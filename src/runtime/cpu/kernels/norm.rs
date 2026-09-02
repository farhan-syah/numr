//! Normalization operation kernels
//!
//! Provides normalization operations with automatic SIMD dispatch.
//! On x86-64, f32 and f64 operations use AVX-512 or AVX2 when available.

use crate::dtype::Element;

/// RMS Normalization: output = input * rsqrt(mean(input^2) + eps) * weight
///
/// Normalizes over the last dimension. Used in LLaMA and other modern transformers.
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` must be valid pointer to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn rms_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                norm::rms_norm_f32(
                    input as *const f32,
                    weight as *const f32,
                    out as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::rms_norm_f64(
                    input as *const f64,
                    weight as *const f64,
                    out as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::rms_norm_f16(
                    input as *const half::f16,
                    weight as *const half::f16,
                    out as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::rms_norm_bf16(
                    input as *const half::bf16,
                    weight as *const half::bf16,
                    out as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    rms_norm_kernel_scalar(input, weight, out, batch_size, hidden_size, eps);
}

/// Scalar RMS norm for all Element types
#[inline]
unsafe fn rms_norm_kernel_scalar<T: Element>(
    input: *const T,
    weight: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let eps = eps as f64;

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Compute sum of squares
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            sum_sq += x * x;
        }

        // Compute RMS: sqrt(mean(x^2) + eps)
        let rms = (sum_sq / hidden_size as f64 + eps).sqrt();
        let inv_rms = 1.0 / rms;

        // Apply normalization and weight
        for (i, &w) in weight_slice.iter().enumerate() {
            let x = (*input.add(row_start + i)).to_f64();
            let result = x * inv_rms * w.to_f64();
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}

/// Layer Normalization: output = (input - mean) / sqrt(variance + eps) * weight + bias
///
/// Normalizes over the last dimension.
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration
/// - AVX2: 8 f32s or 4 f64s per iteration
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `input` - Input tensor data, shape [batch_size, hidden_size] flattened
/// * `weight` - Weight (gamma) tensor, shape [hidden_size]
/// * `bias` - Bias (beta) tensor, shape [hidden_size]
/// * `out` - Output tensor, same shape as input
/// * `batch_size` - Number of rows
/// * `hidden_size` - Size of each row (dimension to normalize over)
/// * `eps` - Small constant for numerical stability
///
/// # Safety
/// - `input` and `out` must be valid pointers to `batch_size * hidden_size` elements
/// - `weight` and `bias` must be valid pointers to `hidden_size` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn layer_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                norm::layer_norm_f32(
                    input as *const f32,
                    weight as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::layer_norm_f64(
                    input as *const f64,
                    weight as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::layer_norm_f16(
                    input as *const half::f16,
                    weight as *const half::f16,
                    bias as *const half::f16,
                    out as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::layer_norm_bf16(
                    input as *const half::bf16,
                    weight as *const half::bf16,
                    bias as *const half::bf16,
                    out as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    layer_norm_kernel_scalar(input, weight, bias, out, batch_size, hidden_size, eps);
}

/// Scalar layer norm for all Element types
#[inline]
unsafe fn layer_norm_kernel_scalar<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let bias_slice = std::slice::from_raw_parts(bias, hidden_size);
    let eps = eps as f64;

    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Every element is shifted by the row's first value before it is
        // accumulated. `x - mean` cancels catastrophically when a row sits far
        // from zero relative to its own spread, which is where the accumulator's
        // precision goes. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic, and the GPU kernels
        // shift by the same element.
        let reference = (*input.add(row_start)).to_f64();

        // Compute the mean of the shifted values
        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += (*input.add(row_start + i)).to_f64() - reference;
        }
        let shifted_mean = sum / hidden_size as f64;

        // Compute variance
        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let diff = (x - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / hidden_size as f64;

        // Compute inverse standard deviation
        let inv_std = 1.0 / (variance + eps).sqrt();

        // Apply normalization, weight, and bias
        for i in 0..hidden_size {
            let x = (*input.add(row_start + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let b = bias_slice[i].to_f64();
            let result = ((x - reference) - shifted_mean) * inv_std * w + b;
            *out.add(row_start + i) = T::from_f64(result);
        }
    }
}

/// Group Normalization kernel.
///
/// Input layout: `[batch, channels, spatial]` (contiguous).
/// For each (batch, group), computes mean/var over `channels_per_group * spatial` elements,
/// then applies per-channel weight and bias.
///
/// # Safety
/// - `input` and `out`: valid for `batch * channels * spatial` elements
/// - `weight` and `bias`: valid for `channels` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn group_norm_kernel<T: Element>(
    input: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    batch: usize,
    channels: usize,
    spatial: usize,
    num_groups: usize,
    channels_per_group: usize,
    eps: f32,
) {
    let eps = eps as f64;
    let group_size = channels_per_group * spatial;

    // An empty group normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if group_size == 0 {
        return;
    }

    for b in 0..batch {
        for g in 0..num_groups {
            let c_start = g * channels_per_group;

            // Every element is shifted by the group's first value before it is
            // accumulated. `x - mean` cancels catastrophically when a group sits
            // far from zero relative to its own spread, and for F64 there is no
            // wider accumulator to hide that in. Mean, variance and the
            // normalized value are all invariant under the shift in exact
            // arithmetic. The GPU kernels shift by the same element.
            let reference = (*input.add((b * channels + c_start) * spatial)).to_f64();

            // Compute the mean of the shifted values
            let mut sum = 0.0f64;
            for c in 0..channels_per_group {
                let ch = c_start + c;
                let offset = (b * channels + ch) * spatial;
                for s in 0..spatial {
                    sum += (*input.add(offset + s)).to_f64() - reference;
                }
            }
            let shifted_mean = sum / group_size as f64;

            // Compute variance over group
            let mut var_sum = 0.0f64;
            for c in 0..channels_per_group {
                let ch = c_start + c;
                let offset = (b * channels + ch) * spatial;
                for s in 0..spatial {
                    let diff = ((*input.add(offset + s)).to_f64() - reference) - shifted_mean;
                    var_sum += diff * diff;
                }
            }
            let inv_std = 1.0 / (var_sum / group_size as f64 + eps).sqrt();

            // Normalize and apply per-channel affine
            for c in 0..channels_per_group {
                let ch = c_start + c;
                let w = (*weight.add(ch)).to_f64();
                let bi = (*bias.add(ch)).to_f64();
                let offset = (b * channels + ch) * spatial;
                for s in 0..spatial {
                    let x = (*input.add(offset + s)).to_f64();
                    let result = ((x - reference) - shifted_mean) * inv_std * w + bi;
                    *out.add(offset + s) = T::from_f64(result);
                }
            }
        }
    }
}
