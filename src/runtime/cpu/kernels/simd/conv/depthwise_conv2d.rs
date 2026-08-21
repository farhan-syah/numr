//! Runtime `SimdLevel` dispatch for `depthwise_conv2d`.
//!
//! Depthwise convolution has 1 input channel per group, so no channel
//! vectorization. Instead, this vectorizes over spatial positions (`output_w`).
//! See `conv2d` for the channel-vs-position vectorization rationale shared by
//! `conv1d`, `conv_transpose1d` and `depthwise_conv2d`.

use super::scalar::{depthwise_conv2d_scalar_f32, depthwise_conv2d_scalar_f64};
use super::threshold::{SIMD_THRESHOLD_F32, SIMD_THRESHOLD_F64};
use crate::ops::conv_common::Conv2dParams;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

#[cfg(target_arch = "aarch64")]
use super::aarch64::neon;
#[cfg(target_arch = "x86_64")]
use super::{avx2, avx512};

/// SIMD depthwise conv2d for f32
///
/// Depthwise convolution has 1 input channel per group, so no channel vectorization.
/// Instead, we vectorize over spatial positions (output_w).
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[inline]
pub unsafe fn depthwise_conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let level = detect_simd();

    // Depthwise: vectorize over output width instead of channels
    if params.output_w < SIMD_THRESHOLD_F32 || level == SimdLevel::Scalar {
        depthwise_conv2d_scalar_f32(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::depthwise_conv2d_f32(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::depthwise_conv2d_f32(input, weight, bias, output, params),
        _ => depthwise_conv2d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::depthwise_conv2d_f32(input, weight, bias, output, params)
        }
        _ => depthwise_conv2d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    depthwise_conv2d_scalar_f32(input, weight, bias, output, params);
}

/// SIMD depthwise conv2d for f64
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the operation
#[inline]
pub unsafe fn depthwise_conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let level = detect_simd();

    if params.output_w < SIMD_THRESHOLD_F64 || level == SimdLevel::Scalar {
        depthwise_conv2d_scalar_f64(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::depthwise_conv2d_f64(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::depthwise_conv2d_f64(input, weight, bias, output, params),
        _ => depthwise_conv2d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::depthwise_conv2d_f64(input, weight, bias, output, params)
        }
        _ => depthwise_conv2d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    depthwise_conv2d_scalar_f64(input, weight, bias, output, params);
}
