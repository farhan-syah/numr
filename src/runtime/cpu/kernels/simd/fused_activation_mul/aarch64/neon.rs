//! NEON fused activation-mul function kernels for ARM64
//!
//! Provides vectorized implementations of fused activation * multiplication
//! using 128-bit NEON registers. Functions take two inputs (a, b) and compute
//! activation(a) * b in a single pass.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{exp_f32, exp_f64, tanh_f32, tanh_f64};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

// ============================================================================
// SiLU_mul: (x / (1 + exp(-x))) * y
// ============================================================================

/// NEON silu_mul for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn silu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let y = vld1q_f32(b.add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = exp_f32(neg_x);
        let activation = vdivq_f32(x, vaddq_f32(one, exp_neg_x));
        let result = vmulq_f32(activation, y);
        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::silu_mul_scalar_f32(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// NEON silu_mul for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn silu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let y = vld1q_f64(b.add(offset));
        let neg_x = vnegq_f64(x);
        let exp_neg_x = exp_f64(neg_x);
        let activation = vdivq_f64(x, vaddq_f64(one, exp_neg_x));
        let result = vmulq_f64(activation, y);
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::silu_mul_scalar_f64(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// GELU_mul: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) * y
// ============================================================================

/// NEON gelu_mul for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gelu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let sqrt_2_over_pi = vdupq_n_f32(0.7978845608);
    let tanh_coef = vdupq_n_f32(0.044715);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let y = vld1q_f32(b.add(offset));

        // x_cubed = x * x * x
        let x_sq = vmulq_f32(x, x);
        let x_cubed = vmulq_f32(x_sq, x);

        // inner = sqrt_2_over_pi * (x + tanh_coef * x_cubed)
        let tanh_coef_x_cubed = vmulq_f32(tanh_coef, x_cubed);
        let x_plus = vaddq_f32(x, tanh_coef_x_cubed);
        let inner = vmulq_f32(sqrt_2_over_pi, x_plus);

        // tanh_inner = tanh(inner)
        let tanh_inner = tanh_f32(inner);

        // activation = 0.5 * x * (1 + tanh_inner)
        let one_plus = vaddq_f32(one, tanh_inner);
        let x_times = vmulq_f32(x, one_plus);
        let activation = vmulq_f32(half, x_times);

        // result = activation * y
        let result = vmulq_f32(activation, y);
        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::gelu_mul_scalar_f32(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// NEON gelu_mul for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn gelu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let half = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);
    let sqrt_2_over_pi = vdupq_n_f64(0.7978845608028654);
    let tanh_coef = vdupq_n_f64(0.044715);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let y = vld1q_f64(b.add(offset));

        // x_cubed = x * x * x
        let x_sq = vmulq_f64(x, x);
        let x_cubed = vmulq_f64(x_sq, x);

        // inner = sqrt_2_over_pi * (x + tanh_coef * x_cubed)
        let tanh_coef_x_cubed = vmulq_f64(tanh_coef, x_cubed);
        let x_plus = vaddq_f64(x, tanh_coef_x_cubed);
        let inner = vmulq_f64(sqrt_2_over_pi, x_plus);

        // `(exp(2x) - 1)/(exp(2x) + 1)` is inf/inf once exp overflows, at
        // |inner| > 354.9. tanh_f64 saturates to +-1 there, as the AVX2 and
        // AVX-512 versions of this kernel already do.
        let tanh_inner = tanh_f64(inner);

        // activation = 0.5 * x * (1 + tanh_inner)
        let one_plus = vaddq_f64(one, tanh_inner);
        let x_times = vmulq_f64(x, one_plus);
        let activation = vmulq_f64(half, x_times);

        // result = activation * y
        let result = vmulq_f64(activation, y);
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::gelu_mul_scalar_f64(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// ReLU_mul: max(0, x) * y
// ============================================================================

/// NEON relu_mul for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn relu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let y = vld1q_f32(b.add(offset));
        let activation = vmaxq_f32(zero, x);
        let result = vmulq_f32(activation, y);
        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::relu_mul_scalar_f32(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// NEON relu_mul for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn relu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let zero = vdupq_n_f64(0.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let y = vld1q_f64(b.add(offset));
        let activation = vmaxq_f64(zero, x);
        let result = vmulq_f64(activation, y);
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::relu_mul_scalar_f64(a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// Sigmoid_mul: (1 / (1 + exp(-x))) * y
// ============================================================================

/// NEON sigmoid_mul for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let one = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = vld1q_f32(a.add(offset));
        let y = vld1q_f32(b.add(offset));
        let neg_x = vnegq_f32(x);
        let exp_neg_x = exp_f32(neg_x);
        let activation = vdivq_f32(one, vaddq_f32(one, exp_neg_x));
        let result = vmulq_f32(activation, y);
        vst1q_f32(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        super::super::sigmoid_mul_scalar_f32(
            a.add(offset),
            b.add(offset),
            out.add(offset),
            remainder,
        );
    }
}

/// NEON sigmoid_mul for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a`, `b`, and `out` must point to `len` valid elements
/// - Elements must not overlap
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn sigmoid_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let one = vdupq_n_f64(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = vld1q_f64(a.add(offset));
        let y = vld1q_f64(b.add(offset));
        let neg_x = vnegq_f64(x);
        let exp_neg_x = exp_f64(neg_x);
        let activation = vdivq_f64(one, vaddq_f64(one, exp_neg_x));
        let result = vmulq_f64(activation, y);
        vst1q_f64(out.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        super::super::sigmoid_mul_scalar_f64(
            a.add(offset),
            b.add(offset),
            out.add(offset),
            remainder,
        );
    }
}
