//! NEON mathematical function implementations for ARM64
//!
//! Provides vectorized transcendental functions using 128-bit NEON registers.
//! All algorithms match those in `common.rs` to ensure numerical consistency.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Relative Error |
//! |----------|-----|-----|----------------|
//! | exp      | 4   | 2   | < 1e-6 / 1e-12 |
//! | tanh     | 4   | 2   | < 1e-6 / 1e-12 |
//! | log      | 4   | 2   | < 1e-6 / 2 ulp |
//! | sin      | 4   | 2   | < 1e-6 / 4 ulp |
//! | cos      | 4   | 2   | < 1e-6 / 4 ulp |
//! | tan      | 4   | 2   | < 2e-4 / 4 ulp |
//! | atan     | 4   | 2   | see note / 2 ulp |
//! | asin     | 4   | 2   | see note / 2 ulp |
//! | acos     | 4   | 2   | see note / 2 ulp |
//!
//! Note: the f32 atan/asin/acos paths reduce with `atan(x) = π/2 - atan(1/x)`
//! and a Gregory series, whose truncation error grows toward the |x| = 1
//! boundary. The f64 paths use the multi-centre reduction in `common.rs`
//! instead and hold below 2 ulps everywhere.
//!
//! The same split applies to the log family: the f64 log/log2/log10/log1p
//! paths hold below 2 ulps, while their f32 counterparts still use a truncated
//! series and are far coarser than f32 epsilon.
//!
//! The f64 sin/cos/tan bounds hold for |x| <= 2^21 * π/2 (about 3.3e6), the
//! limit of the Cody-Waite reduction in `common.rs`, and for tan away from its
//! poles. Their f32 counterparts reduce with a single rounded π/2 and use
//! truncated Taylor series, so they are far coarser than f32 epsilon.
//!
//! # Safety
//!
//! All functions require NEON CPU features (always available on AArch64).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::common::{
    asin_coefficients, atan_coefficients, exp_coefficients, log_coefficients, tan_coefficients,
    trig_coefficients,
};

// ============================================================================
// Horizontal Reductions
// ============================================================================

/// Horizontal sum of 4 f32 values in a NEON register
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hsum_f32(v: float32x4_t) -> f32 {
    // NEON has efficient pairwise operations
    // Step 1: Add adjacent pairs: [a+b, c+d, a+b, c+d]
    let sum = vpadd_f32(vget_low_f32(v), vget_high_f32(v));
    // Step 2: Add the two remaining pairs
    vget_lane_f32::<0>(vpadd_f32(sum, sum))
}

/// Horizontal sum of 2 f64 values in a NEON register
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hsum_f64(v: float64x2_t) -> f64 {
    vgetq_lane_f64::<0>(v) + vgetq_lane_f64::<1>(v)
}

/// Horizontal maximum of 4 f32 values in a NEON register
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hmax_f32(v: float32x4_t) -> f32 {
    // NEON pairwise max
    let max = vpmax_f32(vget_low_f32(v), vget_high_f32(v));
    vget_lane_f32::<0>(vpmax_f32(max, max))
}

/// Horizontal maximum of 2 f64 values in a NEON register
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hmax_f64(v: float64x2_t) -> f64 {
    let a = vgetq_lane_f64::<0>(v);
    let b = vgetq_lane_f64::<1>(v);
    if a > b { a } else { b }
}

/// Horizontal minimum of 4 f32 values in a NEON register
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hmin_f32(v: float32x4_t) -> f32 {
    let min = vpmin_f32(vget_low_f32(v), vget_high_f32(v));
    vget_lane_f32::<0>(vpmin_f32(min, min))
}

/// Horizontal minimum of 2 f64 values in a NEON register
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn hmin_f64(v: float64x2_t) -> f64 {
    let a = vgetq_lane_f64::<0>(v);
    let b = vgetq_lane_f64::<1>(v);
    if a < b { a } else { b }
}

// ============================================================================
// Exponential function: exp(x)
// ============================================================================

/// Fast SIMD exp approximation for f32 using NEON
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp_f32(x: float32x4_t) -> float32x4_t {
    use exp_coefficients::*;

    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);

    let c0 = vdupq_n_f32(C0_F32);
    let c1 = vdupq_n_f32(C1_F32);
    let c2 = vdupq_n_f32(C2_F32);
    let c3 = vdupq_n_f32(C3_F32);
    let c4 = vdupq_n_f32(C4_F32);
    let c5 = vdupq_n_f32(C5_F32);
    let c6 = vdupq_n_f32(C6_F32);

    // Clamp input to avoid overflow/underflow
    let x = vmaxq_f32(x, vdupq_n_f32(MIN_F32));
    let x = vminq_f32(x, vdupq_n_f32(MAX_F32));

    // y = x * log2(e)
    let y = vmulq_f32(x, log2e);

    // n = round(y) - integer part
    let n = vrndnq_f32(y);

    // f = y - n - fractional part in [-0.5, 0.5]
    let f = vsubq_f32(y, n);

    // r = f * ln(2) - convert back to natural log scale
    let r = vmulq_f32(f, ln2);

    // Polynomial approximation using Horner's method with FMA
    let r2 = vmulq_f32(r, r);
    let r3 = vmulq_f32(r2, r);
    let r4 = vmulq_f32(r2, r2);
    let r5 = vmulq_f32(r4, r);
    let r6 = vmulq_f32(r4, r2);

    let mut poly = c0;
    poly = vfmaq_f32(poly, c1, r);
    poly = vfmaq_f32(poly, c2, r2);
    poly = vfmaq_f32(poly, c3, r3);
    poly = vfmaq_f32(poly, c4, r4);
    poly = vfmaq_f32(poly, c5, r5);
    poly = vfmaq_f32(poly, c6, r6);

    // Compute 2^n using IEEE 754 bit manipulation
    // 2^n = reinterpret((n + 127) << 23) for f32
    let n_i32 = vcvtq_s32_f32(n);
    let bias = vdupq_n_s32(127);
    let exp_bits = vshlq_n_s32::<23>(vaddq_s32(n_i32, bias));
    let pow2n = vreinterpretq_f32_s32(exp_bits);

    // Result = 2^n * exp(r)
    vmulq_f32(pow2n, poly)
}

/// Fast SIMD exp approximation for f64 using NEON
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp_f64(x: float64x2_t) -> float64x2_t {
    use exp_coefficients::*;

    let log2e = vdupq_n_f64(std::f64::consts::LOG2_E);
    let ln2_hi = vdupq_n_f64(LN2_HI_F64);
    let ln2_lo = vdupq_n_f64(LN2_LO_F64);

    let c0 = vdupq_n_f64(C0_F64);
    let c1 = vdupq_n_f64(C1_F64);
    let c2 = vdupq_n_f64(C2_F64);
    let c3 = vdupq_n_f64(C3_F64);
    let c4 = vdupq_n_f64(C4_F64);
    let c5 = vdupq_n_f64(C5_F64);
    let c6 = vdupq_n_f64(C6_F64);
    let c7 = vdupq_n_f64(C7_F64);
    let c8 = vdupq_n_f64(C8_F64);
    let c9 = vdupq_n_f64(C9_F64);
    let c10 = vdupq_n_f64(C10_F64);
    let c11 = vdupq_n_f64(C11_F64);
    let c12 = vdupq_n_f64(C12_F64);
    let c13 = vdupq_n_f64(C13_F64);

    // Clamp input
    let x = vmaxq_f64(x, vdupq_n_f64(MIN_F64));
    let x = vminq_f64(x, vdupq_n_f64(MAX_F64));

    let y = vmulq_f64(x, log2e);
    let n = vrndnq_f64(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F64 is exact
    let r = vfmsq_f64(x, n, ln2_hi);
    let r = vfmsq_f64(r, n, ln2_lo);

    // Horner: one rounding per term, and no r^k powers to lose bits in
    let mut poly = c13;
    poly = vfmaq_f64(c12, poly, r);
    poly = vfmaq_f64(c11, poly, r);
    poly = vfmaq_f64(c10, poly, r);
    poly = vfmaq_f64(c9, poly, r);
    poly = vfmaq_f64(c8, poly, r);
    poly = vfmaq_f64(c7, poly, r);
    poly = vfmaq_f64(c6, poly, r);
    poly = vfmaq_f64(c5, poly, r);
    poly = vfmaq_f64(c4, poly, r);
    poly = vfmaq_f64(c3, poly, r);
    poly = vfmaq_f64(c2, poly, r);
    poly = vfmaq_f64(c1, poly, r);
    poly = vfmaq_f64(c0, poly, r);

    // Compute 2^n using IEEE 754 bit manipulation for f64
    // 2^n = reinterpret((n + 1023) << 52) for f64
    let n_i64 = vcvtq_s64_f64(n);
    let bias = vdupq_n_s64(1023);
    let exp_bits = vshlq_n_s64::<52>(vaddq_s64(n_i64, bias));
    let pow2n = vreinterpretq_f64_s64(exp_bits);

    vmulq_f64(pow2n, poly)
}

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh approximation for f32 using NEON
///
/// Algorithm: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tanh_f32(x: float32x4_t) -> float32x4_t {
    let two = vdupq_n_f32(2.0);
    let one = vdupq_n_f32(1.0);

    let exp2x = exp_f32(vmulq_f32(two, x));
    let num = vsubq_f32(exp2x, one);
    let den = vaddq_f32(exp2x, one);

    vdivq_f32(num, den)
}

/// Fast SIMD tanh approximation for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tanh_f64(x: float64x2_t) -> float64x2_t {
    let two = vdupq_n_f64(2.0);
    let one = vdupq_n_f64(1.0);

    let exp2x = exp_f64(vmulq_f64(two, x));
    let num = vsubq_f64(exp2x, one);
    let den = vaddq_f64(exp2x, one);

    vdivq_f64(num, den)
}

// ============================================================================
// Natural logarithm: log(x)
// ============================================================================

/// Fast SIMD log approximation for f32 using NEON
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log_f32(x: float32x4_t) -> float32x4_t {
    use log_coefficients::*;

    let one = vdupq_n_f32(1.0);
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    let sqrt2 = vdupq_n_f32(std::f32::consts::SQRT_2);
    let half = vdupq_n_f32(0.5);

    let c1 = vdupq_n_f32(C1_F32);
    let c2 = vdupq_n_f32(C2_F32);
    let c3 = vdupq_n_f32(C3_F32);
    let c4 = vdupq_n_f32(C4_F32);
    let c5 = vdupq_n_f32(C5_F32);
    let c6 = vdupq_n_f32(C6_F32);
    let c7 = vdupq_n_f32(C7_F32);

    // Extract exponent: reinterpret as int, shift right by 23, subtract bias
    let x_bits = vreinterpretq_s32_f32(x);
    let exp_raw = vshrq_n_s32::<23>(x_bits);
    let exp_unbiased = vsubq_s32(exp_raw, vdupq_n_s32(EXP_BIAS_F32));
    let mut n = vcvtq_f32_s32(exp_unbiased);

    // Extract mantissa and set exponent to 0 (so mantissa is in [1, 2))
    let mantissa_mask = vdupq_n_s32(MANTISSA_MASK_F32);
    let exp_zero = vdupq_n_s32(EXP_ZERO_F32);
    let m_bits = vorrq_s32(vandq_s32(x_bits, mantissa_mask), exp_zero);
    let mut m = vreinterpretq_f32_s32(m_bits);

    // Normalize: if m > sqrt(2), divide by 2 and increment exponent
    let need_adjust = vcgtq_f32(m, sqrt2);
    m = vbslq_f32(need_adjust, vmulq_f32(m, half), m);
    n = vbslq_f32(need_adjust, vaddq_f32(n, one), n);

    // f = m - 1, so log(m) = log(1 + f)
    let f = vsubq_f32(m, one);

    // Horner's method: ((((((c7*f + c6)*f + c5)*f + c4)*f + c3)*f + c2)*f + c1)*f
    let mut poly = c7;
    poly = vfmaq_f32(c6, poly, f);
    poly = vfmaq_f32(c5, poly, f);
    poly = vfmaq_f32(c4, poly, f);
    poly = vfmaq_f32(c3, poly, f);
    poly = vfmaq_f32(c2, poly, f);
    poly = vfmaq_f32(c1, poly, f);
    poly = vmulq_f32(poly, f);

    // Result = n * ln(2) + log(m)
    vfmaq_f32(poly, n, ln2)
}

/// Split `x` into an exponent `n` and `log(m)`, where `m` is the mantissa
/// normalized to [sqrt(2)/2, sqrt(2)), so that `log(x) = n*ln(2) + log(m)`.
///
/// log, log2, log10 and log1p all share this reduction and differ only in how
/// they recombine the two parts. Special values are applied by the callers via
/// `log_special_f64`.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn log_reduce_f64(x: float64x2_t) -> (float64x2_t, float64x2_t) {
    use log_coefficients::*;

    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let half = vdupq_n_f64(0.5);
    let sqrt2_val = std::f64::consts::SQRT_2;

    // Subnormals carry no implicit leading 1, so the split below is only valid
    // after scaling them into the normal range.
    let is_sub = vcltq_f64(x, vdupq_n_f64(f64::MIN_POSITIVE));
    let x_norm = vbslq_f64(is_sub, vmulq_f64(x, vdupq_n_f64(SUBNORMAL_SCALE_F64)), x);
    let n_shift = vbslq_f64(is_sub, vdupq_n_f64(SUBNORMAL_SHIFT_F64), vdupq_n_f64(0.0));

    // Use SIMD for bit manipulation
    let x_bits = vreinterpretq_s64_f64(x_norm);

    // Extract exponent using 64-bit SIMD shift
    let exp_raw = vshrq_n_s64::<52>(x_bits);

    // Extract mantissa and set exponent to bias (so mantissa is in [1, 2))
    let mantissa_mask = vdupq_n_s64(MANTISSA_MASK_F64 as i64);
    let exp_zero = vdupq_n_s64(EXP_ZERO_F64 as i64);
    let m_bits = vorrq_s64(vandq_s64(x_bits, mantissa_mask), exp_zero);
    let m_initial = vreinterpretq_f64_s64(m_bits);

    // Extract for normalization (NEON lacks some 64-bit comparison intrinsics)
    let mut m_arr = [0.0f64; 2];
    let mut exp_arr = [0i64; 2];
    vst1q_f64(m_arr.as_mut_ptr(), m_initial);
    vst1q_s64(exp_arr.as_mut_ptr(), exp_raw);

    let mut n_arr = [0.0f64; 2];
    for i in 0..2 {
        let mut exp_unbiased = exp_arr[i] - EXP_BIAS_F64;
        let mut m = m_arr[i];

        if m > sqrt2_val {
            m *= 0.5;
            exp_unbiased += 1;
        }

        n_arr[i] = exp_unbiased as f64;
        m_arr[i] = m;
    }

    let n = vaddq_f64(vld1q_f64(n_arr.as_ptr()), n_shift);
    let m = vld1q_f64(m_arr.as_ptr());

    // s = f/(2+f) halves the argument and leaves only odd powers, which is what
    // lets seven terms reach f64 precision (see `log_coefficients`).
    let f = vsubq_f64(m, one);
    let s = vdivq_f64(f, vaddq_f64(two, f));
    let z = vmulq_f64(s, s);
    let w = vmulq_f64(z, z);

    let t1 = vmulq_f64(
        w,
        vfmaq_f64(
            vdupq_n_f64(LG2_F64),
            w,
            vfmaq_f64(vdupq_n_f64(LG4_F64), w, vdupq_n_f64(LG6_F64)),
        ),
    );
    let t2 = vmulq_f64(
        z,
        vfmaq_f64(
            vdupq_n_f64(LG1_F64),
            w,
            vfmaq_f64(
                vdupq_n_f64(LG3_F64),
                w,
                vfmaq_f64(vdupq_n_f64(LG5_F64), w, vdupq_n_f64(LG7_F64)),
            ),
        ),
    );
    let r = vaddq_f64(t2, t1);

    // log(m) = f - (hfsq - s*(hfsq + R)); keeping f outside the parentheses
    // stops the f² term from eating f's low bits when f is small.
    let hfsq = vmulq_f64(half, vmulq_f64(f, f));
    let logm = vsubq_f64(f, vsubq_f64(hfsq, vmulq_f64(s, vaddq_f64(hfsq, r))));

    (n, logm)
}

/// Apply the IEEE domain values shared by log, log2 and log10:
/// `log(0) = -inf`, `log(x < 0) = NaN`, `log(+inf) = +inf`, `log(NaN) = NaN`.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn log_special_f64(x: float64x2_t, r: float64x2_t) -> float64x2_t {
    let zero = vdupq_n_f64(0.0);

    // `vcgtq_f64` is false for NaN, so its complement catches NaN alongside the
    // non-positive inputs instead of feeding garbage through the polynomial.
    let positive = vcgtq_f64(x, zero);
    let is_zero = vceqq_f64(x, zero);
    let is_inf = vceqq_f64(x, vdupq_n_f64(f64::INFINITY));

    let out = vbslq_f64(positive, r, vdupq_n_f64(f64::NAN));
    let out = vbslq_f64(is_zero, vdupq_n_f64(f64::NEG_INFINITY), out);
    vbslq_f64(is_inf, vdupq_n_f64(f64::INFINITY), out)
}

/// Fast SIMD log approximation for f64 using NEON
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps over the whole positive range.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log_f64(x: float64x2_t) -> float64x2_t {
    use log_coefficients::{LN2_HI_F64, LN2_LO_F64};

    let (n, logm) = log_reduce_f64(x);

    // Split ln(2): the head is exact against every reachable n, the tail
    // restores the bits a single rounded ln(2) would drop.
    let lo = vfmaq_f64(logm, n, vdupq_n_f64(LN2_LO_F64));
    let r = vfmaq_f64(lo, n, vdupq_n_f64(LN2_HI_F64));

    log_special_f64(x, r)
}

// ============================================================================
// Trigonometric functions: sin, cos, tan
// ============================================================================

/// Fast SIMD sin approximation for f32 using NEON
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sin_f32(x: float32x4_t) -> float32x4_t {
    use trig_coefficients::*;

    let two_over_pi = vdupq_n_f32(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    let s1 = vdupq_n_f32(S1_F32);
    let s3 = vdupq_n_f32(S3_F32);
    let s5 = vdupq_n_f32(S5_F32);
    let s7 = vdupq_n_f32(S7_F32);

    let c0 = vdupq_n_f32(C0_F32);
    let c2 = vdupq_n_f32(C2_F32);
    let c4 = vdupq_n_f32(C4_F32);
    let c6 = vdupq_n_f32(C6_F32);

    // Range reduction: j = round(x * 2/π), y = x - j * π/2
    let j = vrndnq_f32(vmulq_f32(x, two_over_pi));
    let j_int = vcvtq_s32_f32(j);

    // y = x - j * (π/2) using FMA for precision
    let y = vfmsq_f32(x, j, pi_over_2);

    let y2 = vmulq_f32(y, y);
    let y3 = vmulq_f32(y2, y);
    let y4 = vmulq_f32(y2, y2);
    let y5 = vmulq_f32(y4, y);
    let y6 = vmulq_f32(y4, y2);
    let y7 = vmulq_f32(y4, y3);

    // sin(y) polynomial: s1*y + s3*y³ + s5*y⁵ + s7*y⁷
    let sin_y = vfmaq_f32(
        vfmaq_f32(vfmaq_f32(vmulq_f32(s1, y), s3, y3), s5, y5),
        s7,
        y7,
    );

    // cos(y) polynomial: c0 + c2*y² + c4*y⁴ + c6*y⁶
    let cos_y = vfmaq_f32(vfmaq_f32(vfmaq_f32(c0, c2, y2), c4, y4), c6, y6);

    // Select sin or cos based on j mod 4
    let j_mod_4 = vandq_s32(j_int, vdupq_n_s32(3));

    // Use cos when j mod 4 is 1 or 3
    let use_cos_mask = vceqq_s32(vandq_s32(j_mod_4, vdupq_n_s32(1)), vdupq_n_s32(1));

    // Negate when j mod 4 is 2 or 3
    let negate_mask = vceqq_s32(vandq_s32(j_mod_4, vdupq_n_s32(2)), vdupq_n_s32(2));

    let result = vbslq_f32(use_cos_mask, cos_y, sin_y);
    let negated = vnegq_f32(result);
    vbslq_f32(negate_mask, negated, result)
}

/// Cody-Waite reduction of `x` modulo π/2 for f64.
///
/// Returns the quadrant index `j` (as a double) and the reduced argument in
/// [-π/4, π/4]. See `common::_TRIG_ALGORITHM_DOC`; valid for
/// |x| <= 2^21 * π/2, past which `j * PIO2_k` stops being exact.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn trig_reduce_f64(x: float64x2_t) -> (float64x2_t, float64x2_t) {
    use trig_coefficients::{PIO2_1_F64, PIO2_2_F64, PIO2_3_F64, PIO2_3T_F64};

    let j = vrndnq_f64(vmulq_f64(x, vdupq_n_f64(std::f64::consts::FRAC_2_PI)));

    let y = vfmsq_f64(x, j, vdupq_n_f64(PIO2_1_F64));
    let y = vfmsq_f64(y, j, vdupq_n_f64(PIO2_2_F64));
    let y = vfmsq_f64(y, j, vdupq_n_f64(PIO2_3_F64));
    let y = vfmsq_f64(y, j, vdupq_n_f64(PIO2_3T_F64));

    (j, y)
}

/// Minimax sin kernel on the reduced argument, |y| <= π/4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sin_kernel_f64(y: float64x2_t) -> float64x2_t {
    use trig_coefficients::{SIN1_F64, SIN2_F64, SIN3_F64, SIN4_F64, SIN5_F64, SIN6_F64};

    let z = vmulq_f64(y, y);

    let mut p = vdupq_n_f64(SIN6_F64);
    p = vfmaq_f64(vdupq_n_f64(SIN5_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(SIN4_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(SIN3_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(SIN2_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(SIN1_F64), p, z);

    // y is added last, so a tiny y comes back unchanged and keeps its sign.
    vfmaq_f64(y, vmulq_f64(z, y), p)
}

/// Minimax cos kernel on the reduced argument, |y| <= π/4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn cos_kernel_f64(y: float64x2_t) -> float64x2_t {
    use trig_coefficients::{COS1_F64, COS2_F64, COS3_F64, COS4_F64, COS5_F64, COS6_F64};

    let one = vdupq_n_f64(1.0);
    let z = vmulq_f64(y, y);

    let mut p = vdupq_n_f64(COS6_F64);
    p = vfmaq_f64(vdupq_n_f64(COS5_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(COS4_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(COS3_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(COS2_F64), p, z);
    p = vfmaq_f64(vdupq_n_f64(COS1_F64), p, z);
    let r = vmulq_f64(vmulq_f64(z, z), p);

    // `1 - z/2` rounds; `(1 - w) - hz` is exact and returns the rounded bits.
    let hz = vmulq_f64(vdupq_n_f64(0.5), z);
    let w = vsubq_f64(one, hz);
    let correction = vsubq_f64(vsubq_f64(one, w), hz);
    vaddq_f64(w, vaddq_f64(correction, r))
}

/// Evaluate sin on quadrant `j + offset`, the shared core of sin and cos.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sin_quadrant_f64(x: float64x2_t, offset: i32) -> float64x2_t {
    let (j, y) = trig_reduce_f64(x);
    let sin_y = sin_kernel_f64(y);
    let cos_y = cos_kernel_f64(y);

    let mut j_arr = [0.0f64; 2];
    let mut sin_arr = [0.0f64; 2];
    let mut cos_arr = [0.0f64; 2];
    vst1q_f64(j_arr.as_mut_ptr(), j);
    vst1q_f64(sin_arr.as_mut_ptr(), sin_y);
    vst1q_f64(cos_arr.as_mut_ptr(), cos_y);

    let mut result = [0.0f64; 2];
    for i in 0..2 {
        let quadrant = (j_arr[i] as i32).wrapping_add(offset) & 3;
        result[i] = match quadrant {
            0 => sin_arr[i],
            1 => cos_arr[i],
            2 => -sin_arr[i],
            _ => -cos_arr[i],
        };
    }

    vld1q_f64(result.as_ptr())
}

/// Fast SIMD sin approximation for f64 using NEON
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 4 ulps for |x| <= 2^21 * π/2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sin_f64(x: float64x2_t) -> float64x2_t {
    let r = sin_quadrant_f64(x, 0);

    // sin(±0) = ±0. The reduction computes 0 - (-0 * π/2), which is +0 for both
    // signed zeros, so the input is restored here.
    let is_zero = vceqq_f64(x, vdupq_n_f64(0.0));
    vbslq_f64(is_zero, x, r)
}

/// Fast SIMD cos approximation for f32 using NEON
///
/// Implemented as: cos(x) = sin(x + π/2)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cos_f32(x: float32x4_t) -> float32x4_t {
    let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
    sin_f32(vaddq_f32(x, pi_over_2))
}

/// Fast SIMD cos approximation for f64 using NEON
///
/// Shifts the quadrant index by one rather than evaluating `sin(x + π/2)`,
/// which would round the sum before reduction and lose bits proportional to
/// |x|. Relative error stays below 4 ulps for |x| <= 2^21 * π/2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cos_f64(x: float64x2_t) -> float64x2_t {
    sin_quadrant_f64(x, 1)
}

/// Fast SIMD tan approximation for f32 using NEON
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tan_f32(x: float32x4_t) -> float32x4_t {
    use tan_coefficients::*;

    let two_over_pi = vdupq_n_f32(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    // Range reduction
    let j = vrndnq_f32(vmulq_f32(x, two_over_pi));
    let y = vfmsq_f32(x, j, pi_over_2);

    let t1 = vdupq_n_f32(T1_F32);
    let t3 = vdupq_n_f32(T3_F32);
    let t5 = vdupq_n_f32(T5_F32);
    let t7 = vdupq_n_f32(T7_F32);
    let t9 = vdupq_n_f32(T9_F32);
    let t11 = vdupq_n_f32(T11_F32);

    let y2 = vmulq_f32(y, y);

    // Horner's method
    let mut poly = t11;
    poly = vfmaq_f32(t9, poly, y2);
    poly = vfmaq_f32(t7, poly, y2);
    poly = vfmaq_f32(t5, poly, y2);
    poly = vfmaq_f32(t3, poly, y2);
    poly = vfmaq_f32(t1, poly, y2);
    let tan_y = vmulq_f32(y, poly);

    // For quadrants 1 and 3, tan(y + π/2) = -1/tan(y) = -cot(y)
    let j_int = vcvtq_s32_f32(j);
    let use_cot_mask = vceqq_s32(vandq_s32(j_int, vdupq_n_s32(1)), vdupq_n_s32(1));

    let neg_one = vdupq_n_f32(-1.0);
    let cot_y = vdivq_f32(neg_one, tan_y);

    vbslq_f32(use_cot_mask, cot_y, tan_y)
}

/// Fast SIMD tan approximation for f64 using NEON
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 4 ulps away from the poles, for
/// |x| <= 2^21 * π/2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tan_f64(x: float64x2_t) -> float64x2_t {
    let (j, y) = trig_reduce_f64(x);
    let sin_y = sin_kernel_f64(y);
    let cos_y = cos_kernel_f64(y);

    // Odd quadrant: tan(x) = -cot(y) = -cos(y)/sin(y). Swapping the ratio costs
    // one rounding, where inverting an already-rounded tan(y) costs two.
    let mut j_arr = [0.0f64; 2];
    let mut sin_arr = [0.0f64; 2];
    let mut cos_arr = [0.0f64; 2];
    vst1q_f64(j_arr.as_mut_ptr(), j);
    vst1q_f64(sin_arr.as_mut_ptr(), sin_y);
    vst1q_f64(cos_arr.as_mut_ptr(), cos_y);

    let mut num = [0.0f64; 2];
    let mut den = [0.0f64; 2];
    for i in 0..2 {
        if (j_arr[i] as i32) & 1 == 1 {
            num[i] = -cos_arr[i];
            den[i] = sin_arr[i];
        } else {
            num[i] = sin_arr[i];
            den[i] = cos_arr[i];
        }
    }

    let r = vdivq_f64(vld1q_f64(num.as_ptr()), vld1q_f64(den.as_ptr()));

    // tan(±0) = ±0; the reduction turns -0 into +0.
    let is_zero = vceqq_f64(x, vdupq_n_f64(0.0));
    vbslq_f64(is_zero, x, r)
}

// ============================================================================
// Inverse tangent function: atan(x)
// ============================================================================

/// Fast SIMD atan approximation for f32 using NEON
///
/// See `common::_ATAN_ALGORITHM_DOC` for algorithm details.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atan_f32(x: float32x4_t) -> float32x4_t {
    use atan_coefficients::*;

    let one = vdupq_n_f32(1.0);
    let pi_over_2 = vdupq_n_f32(std::f32::consts::FRAC_PI_2);

    // Save sign and work with absolute value
    let sign_mask = vdupq_n_u32(0x80000000);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let abs_x = vabsq_f32(x);

    // Range reduction: for |x| > 1, compute atan(1/x) then adjust
    let need_recip = vcgtq_f32(abs_x, one);
    let recip_x = vdivq_f32(one, abs_x);
    let y = vbslq_f32(need_recip, recip_x, abs_x);

    // Polynomial approximation for atan(y) where y in [0, 1]
    let a0 = vdupq_n_f32(A0_F32);
    let a2 = vdupq_n_f32(A2_F32);
    let a4 = vdupq_n_f32(A4_F32);
    let a6 = vdupq_n_f32(A6_F32);
    let a8 = vdupq_n_f32(A8_F32);
    let a10 = vdupq_n_f32(A10_F32);
    let a12 = vdupq_n_f32(A12_F32);

    let y2 = vmulq_f32(y, y);

    // Horner's method
    let mut poly = a12;
    poly = vfmaq_f32(a10, poly, y2);
    poly = vfmaq_f32(a8, poly, y2);
    poly = vfmaq_f32(a6, poly, y2);
    poly = vfmaq_f32(a4, poly, y2);
    poly = vfmaq_f32(a2, poly, y2);
    poly = vfmaq_f32(a0, poly, y2);
    let atan_y = vmulq_f32(y, poly);

    // Apply range reduction inverse: if |x| > 1, result = π/2 - atan(1/x)
    let adjusted = vsubq_f32(pi_over_2, atan_y);
    let result = vbslq_f32(need_recip, adjusted, atan_y);

    // Restore sign
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign))
}

/// Fast SIMD atan approximation for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atan_f64(x: float64x2_t) -> float64x2_t {
    use atan_coefficients::*;

    let one = vdupq_n_f64(1.0);
    let zero = vdupq_n_f64(0.0);
    let sign_mask = vdupq_n_u64(0x8000000000000000);
    let sign = vandq_u64(vreinterpretq_u64_f64(x), sign_mask);
    let ax = vabsq_f64(x);

    // Pick the reduction centre c and the matching atan(c) head/correction.
    // The breakpoint masks are nested, so blending from the widest bucket
    // inward leaves each lane holding its tightest match.
    let mut c = vdupq_n_f64(1.5);
    let mut hi = vdupq_n_f64(ATAN_HI2_F64);
    let mut lo = vdupq_n_f64(ATAN_LO2_F64);

    let in2 = vcltq_f64(ax, vdupq_n_f64(BREAK2_F64));
    c = vbslq_f64(in2, one, c);
    hi = vbslq_f64(in2, vdupq_n_f64(ATAN_HI1_F64), hi);
    lo = vbslq_f64(in2, vdupq_n_f64(ATAN_LO1_F64), lo);

    let in1 = vcltq_f64(ax, vdupq_n_f64(BREAK1_F64));
    c = vbslq_f64(in1, vdupq_n_f64(0.5), c);
    hi = vbslq_f64(in1, vdupq_n_f64(ATAN_HI0_F64), hi);
    lo = vbslq_f64(in1, vdupq_n_f64(ATAN_LO0_F64), lo);

    let in0 = vcltq_f64(ax, vdupq_n_f64(BREAK0_F64));
    c = vbslq_f64(in0, zero, c);
    hi = vbslq_f64(in0, zero, hi);
    lo = vbslq_f64(in0, zero, lo);

    // Past the last breakpoint the centre is at infinity: t = -1/|x|.
    // NaN fails every comparison and falls through to c = 1.5, which
    // propagates NaN through the division below.
    let big = vcgeq_f64(ax, vdupq_n_f64(BREAK3_F64));
    hi = vbslq_f64(big, vdupq_n_f64(ATAN_HI3_F64), hi);
    lo = vbslq_f64(big, vdupq_n_f64(ATAN_LO3_F64), lo);
    let num = vbslq_f64(big, vdupq_n_f64(-1.0), vsubq_f64(ax, c));
    let den = vbslq_f64(big, ax, vfmaq_f64(one, c, ax));

    // t in [-0.4375, 0.4375]; |x| = inf gives t = -0.0, so the result is π/2.
    let t = vdivq_f64(num, den);
    let z = vmulq_f64(t, t);
    let w = vmulq_f64(z, z);

    // Even- and odd-indexed coefficients evaluated as two independent Horner
    // chains in w, which shortens the dependency chain versus one chain in z.
    let mut s1 = vdupq_n_f64(AT10_F64);
    s1 = vfmaq_f64(vdupq_n_f64(AT8_F64), s1, w);
    s1 = vfmaq_f64(vdupq_n_f64(AT6_F64), s1, w);
    s1 = vfmaq_f64(vdupq_n_f64(AT4_F64), s1, w);
    s1 = vfmaq_f64(vdupq_n_f64(AT2_F64), s1, w);
    s1 = vfmaq_f64(vdupq_n_f64(AT0_F64), s1, w);
    s1 = vmulq_f64(s1, z);

    let mut s2 = vdupq_n_f64(AT9_F64);
    s2 = vfmaq_f64(vdupq_n_f64(AT7_F64), s2, w);
    s2 = vfmaq_f64(vdupq_n_f64(AT5_F64), s2, w);
    s2 = vfmaq_f64(vdupq_n_f64(AT3_F64), s2, w);
    s2 = vfmaq_f64(vdupq_n_f64(AT1_F64), s2, w);
    s2 = vmulq_f64(s2, w);

    // atan(x) = atan(c) + atan(t), grouped so the correction term lands
    // beside the polynomial residual rather than beside the head.
    let poly = vmulq_f64(t, vaddq_f64(s1, s2));
    let result = vsubq_f64(hi, vsubq_f64(vsubq_f64(poly, lo), t));

    // Restore sign
    vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(result), sign))
}

// ============================================================================
// Additional transcendental functions
// ============================================================================

/// Fast SIMD rsqrt (1/sqrt(x)) for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn rsqrt_f32(x: float32x4_t) -> float32x4_t {
    // NEON provides vrsqrteq_f32 with Newton-Raphson refinement
    let est = vrsqrteq_f32(x);
    let step1 = vmulq_f32(est, x);
    let step2 = vrsqrtsq_f32(step1, est);
    let refined = vmulq_f32(est, step2);
    let step3 = vmulq_f32(refined, x);
    let step4 = vrsqrtsq_f32(step3, refined);
    vmulq_f32(refined, step4)
}

/// Fast SIMD rsqrt (1/sqrt(x)) for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn rsqrt_f64(x: float64x2_t) -> float64x2_t {
    let est = vrsqrteq_f64(x);
    let step1 = vmulq_f64(est, x);
    let step2 = vrsqrtsq_f64(step1, est);
    let refined = vmulq_f64(est, step2);
    let step3 = vmulq_f64(refined, x);
    let step4 = vrsqrtsq_f64(step3, refined);
    vmulq_f64(refined, step4)
}

/// Fast SIMD exp2 (2^x) for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp2_f32(x: float32x4_t) -> float32x4_t {
    let ln2 = vdupq_n_f32(std::f32::consts::LN_2);
    exp_f32(vmulq_f32(x, ln2))
}

/// Fast SIMD exp2 (2^x) for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp2_f64(x: float64x2_t) -> float64x2_t {
    let ln2 = vdupq_n_f64(std::f64::consts::LN_2);
    exp_f64(vmulq_f64(x, ln2))
}

/// Fast SIMD expm1 (e^x - 1) for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn expm1_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let abs_x = vabsq_f32(x);

    // For small |x|, use Taylor series
    let x2 = vmulq_f32(x, x);
    let x3 = vmulq_f32(x2, x);
    let x4 = vmulq_f32(x2, x2);
    let c2 = vdupq_n_f32(0.5);
    let c3 = vdupq_n_f32(1.0 / 6.0);
    let c4 = vdupq_n_f32(1.0 / 24.0);
    let taylor = vfmaq_f32(vfmaq_f32(vfmaq_f32(x, c2, x2), c3, x3), c4, x4);

    // For large |x|, use exp(x) - 1
    let exp_result = vsubq_f32(exp_f32(x), one);

    // Blend based on |x| > 0.5
    let mask = vcgtq_f32(abs_x, half);
    vbslq_f32(mask, exp_result, taylor)
}

/// Fast SIMD expm1 (e^x - 1) for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn expm1_f64(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let half = vdupq_n_f64(0.5);
    let abs_x = vabsq_f64(x);

    let x2 = vmulq_f64(x, x);
    let x3 = vmulq_f64(x2, x);
    let x4 = vmulq_f64(x2, x2);
    let c2 = vdupq_n_f64(0.5);
    let c3 = vdupq_n_f64(1.0 / 6.0);
    let c4 = vdupq_n_f64(1.0 / 24.0);
    let taylor = vfmaq_f64(vfmaq_f64(vfmaq_f64(x, c2, x2), c3, x3), c4, x4);

    let exp_result = vsubq_f64(exp_f64(x), one);
    let mask = vcgtq_f64(abs_x, half);
    vbslq_f64(mask, exp_result, taylor)
}

/// Fast SIMD log2 for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log2_f32(x: float32x4_t) -> float32x4_t {
    let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
    vmulq_f32(log_f32(x), log2e)
}

/// Fast SIMD log2 for f64 using NEON
///
/// Scaling `log(x)` would fold the exponent through two roundings and miss
/// exact powers of two, so the exponent is added back untouched instead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log2_f64(x: float64x2_t) -> float64x2_t {
    let (n, logm) = log_reduce_f64(x);
    let r = vfmaq_f64(n, logm, vdupq_n_f64(std::f64::consts::LOG2_E));
    log_special_f64(x, r)
}

/// Fast SIMD log10 for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log10_f32(x: float32x4_t) -> float32x4_t {
    let log10e = vdupq_n_f32(std::f32::consts::LOG10_E);
    vmulq_f32(log_f32(x), log10e)
}

/// Fast SIMD log10 for f64 using NEON
///
/// `log10(x) = n*log10(2) + log(m)*log10(e)`, keeping the exact exponent out
/// of the mantissa's rounding for the same reason as `log2_f64`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log10_f64(x: float64x2_t) -> float64x2_t {
    let (n, logm) = log_reduce_f64(x);
    let scaled = vmulq_f64(logm, vdupq_n_f64(std::f64::consts::LOG10_E));
    let r = vfmaq_f64(scaled, n, vdupq_n_f64(std::f64::consts::LOG10_2));
    log_special_f64(x, r)
}

/// Fast SIMD log1p (log(1+x)) for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log1p_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let abs_x = vabsq_f32(x);

    // For small |x|, use Taylor series
    let x2 = vmulq_f32(x, x);
    let x3 = vmulq_f32(x2, x);
    let x4 = vmulq_f32(x2, x2);
    let c2 = vdupq_n_f32(-0.5);
    let c3 = vdupq_n_f32(1.0 / 3.0);
    let c4 = vdupq_n_f32(-0.25);
    let taylor = vfmaq_f32(vfmaq_f32(vfmaq_f32(x, c2, x2), c3, x3), c4, x4);

    // For large |x|, use log(1 + x)
    let log_result = log_f32(vaddq_f32(one, x));

    let mask = vcgtq_f32(abs_x, half);
    vbslq_f32(mask, log_result, taylor)
}

/// Fast SIMD log1p (log(1+x)) for f64 using NEON
///
/// `1 + x` alone rounds away the information log1p exists to keep, so the sum
/// is carried as an exact pair `u + c` and the residual is folded back in.
/// Relative error stays below 2 ulps, including for |x| down to the subnormal
/// range where log1p(x) == x.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log1p_f64(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let u = vaddq_f64(one, x);

    // Fast2Sum: 1 + x = u + c exactly, with the larger addend leading.
    let c_small = vsubq_f64(x, vsubq_f64(u, one));
    let c_large = vsubq_f64(one, vsubq_f64(u, x));
    let x_leads = vcltq_f64(vabsq_f64(x), one);
    let c = vbslq_f64(x_leads, c_small, c_large);

    // log(u + c) = log(u) + log1p(c/u), and |c/u| <= 2^-53, so the inner series
    // collapses to its first term.
    let r = vaddq_f64(log_f64(u), vdivq_f64(c, u));

    // u == 1 means x fell entirely off the end of the sum; log1p(x) is then x
    // to within half an ulp. This is also what carries signed zero through.
    let out = vbslq_f64(vceqq_f64(u, one), x, r);

    // x == -1 gives u == 0 and c/u = 0/0; x == +inf gives inf - inf.
    let is_neg_one = vceqq_f64(x, vdupq_n_f64(-1.0));
    let is_inf = vceqq_f64(x, vdupq_n_f64(f64::INFINITY));
    let out = vbslq_f64(is_neg_one, vdupq_n_f64(f64::NEG_INFINITY), out);
    vbslq_f64(is_inf, vdupq_n_f64(f64::INFINITY), out)
}

/// Fast SIMD sinh for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sinh_f32(x: float32x4_t) -> float32x4_t {
    let half = vdupq_n_f32(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(vnegq_f32(x));
    vmulq_f32(half, vsubq_f32(exp_x, exp_neg_x))
}

/// Fast SIMD sinh for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sinh_f64(x: float64x2_t) -> float64x2_t {
    let half = vdupq_n_f64(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(vnegq_f64(x));
    vmulq_f64(half, vsubq_f64(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosh_f32(x: float32x4_t) -> float32x4_t {
    let half = vdupq_n_f32(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(vnegq_f32(x));
    vmulq_f32(half, vaddq_f32(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosh_f64(x: float64x2_t) -> float64x2_t {
    let half = vdupq_n_f64(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(vnegq_f64(x));
    vmulq_f64(half, vaddq_f64(exp_x, exp_neg_x))
}

/// Fast SIMD asinh for f32 using NEON
/// asinh(x) = log(x + sqrt(x^2 + 1))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asinh_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let x2 = vmulq_f32(x, x);
    let sqrt_term = vsqrtq_f32(vaddq_f32(x2, one));
    log_f32(vaddq_f32(x, sqrt_term))
}

/// Fast SIMD asinh for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asinh_f64(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let x2 = vmulq_f64(x, x);
    let sqrt_term = vsqrtq_f64(vaddq_f64(x2, one));
    log_f64(vaddq_f64(x, sqrt_term))
}

/// Fast SIMD acosh for f32 using NEON
/// acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acosh_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let x2 = vmulq_f32(x, x);
    let sqrt_term = vsqrtq_f32(vsubq_f32(x2, one));
    log_f32(vaddq_f32(x, sqrt_term))
}

/// Fast SIMD acosh for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acosh_f64(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let x2 = vmulq_f64(x, x);
    let sqrt_term = vsqrtq_f64(vsubq_f64(x2, one));
    log_f64(vaddq_f64(x, sqrt_term))
}

/// Fast SIMD atanh for f32 using NEON
/// atanh(x) = 0.5 * log((1 + x) / (1 - x)) for |x| < 1
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atanh_f32(x: float32x4_t) -> float32x4_t {
    let half = vdupq_n_f32(0.5);
    let one = vdupq_n_f32(1.0);
    let one_plus_x = vaddq_f32(one, x);
    let one_minus_x = vsubq_f32(one, x);
    let ratio = vdivq_f32(one_plus_x, one_minus_x);
    vmulq_f32(half, log_f32(ratio))
}

/// Fast SIMD atanh for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atanh_f64(x: float64x2_t) -> float64x2_t {
    let half = vdupq_n_f64(0.5);
    let one = vdupq_n_f64(1.0);
    let one_plus_x = vaddq_f64(one, x);
    let one_minus_x = vsubq_f64(one, x);
    let ratio = vdivq_f64(one_plus_x, one_minus_x);
    vmulq_f64(half, log_f64(ratio))
}

/// Fast SIMD asin for f32 using NEON
/// asin(x) = atan(x / sqrt(1 - x^2))
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asin_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let x2 = vmulq_f32(x, x);
    let sqrt_term = vsqrtq_f32(vsubq_f32(one, x2));
    let ratio = vdivq_f32(x, sqrt_term);
    atan_f32(ratio)
}

/// Shared rational correction `R(t) = p(t)/q(t)` for f64 asin/acos.
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC`. Valid for t in [0, 0.5].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn asin_r_f64(t: float64x2_t) -> float64x2_t {
    use asin_coefficients::*;

    let mut p = vdupq_n_f64(PS5_F64);
    p = vfmaq_f64(vdupq_n_f64(PS4_F64), p, t);
    p = vfmaq_f64(vdupq_n_f64(PS3_F64), p, t);
    p = vfmaq_f64(vdupq_n_f64(PS2_F64), p, t);
    p = vfmaq_f64(vdupq_n_f64(PS1_F64), p, t);
    p = vfmaq_f64(vdupq_n_f64(PS0_F64), p, t);
    p = vmulq_f64(p, t);

    let mut q = vdupq_n_f64(QS4_F64);
    q = vfmaq_f64(vdupq_n_f64(QS3_F64), q, t);
    q = vfmaq_f64(vdupq_n_f64(QS2_F64), q, t);
    q = vfmaq_f64(vdupq_n_f64(QS1_F64), q, t);
    q = vfmaq_f64(vdupq_n_f64(1.0), q, t);

    vdivq_f64(p, q)
}

/// Fast SIMD asin for f64 using NEON
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asin_f64(x: float64x2_t) -> float64x2_t {
    use asin_coefficients::*;

    let one = vdupq_n_f64(1.0);
    let half = vdupq_n_f64(HALF_F64);
    let sign_mask = vdupq_n_u64(0x8000000000000000);
    let sign = vandq_u64(vreinterpretq_u64_f64(x), sign_mask);
    let ax = vabsq_f64(x);

    // |x| > 1 leaves the reflection argument negative, so sqrt yields NaN.
    // NaN input fails the comparison and takes the same reflection path.
    let small = vcltq_f64(ax, half);
    let t_refl = vmulq_f64(vsubq_f64(one, ax), half);
    let t = vbslq_f64(small, vmulq_f64(ax, ax), t_refl);
    let r = asin_r_f64(t);
    let s = vsqrtq_f64(t);

    let res_small = vfmaq_f64(ax, ax, r);

    // π/2 - 2*asin(sqrt(t)), with the low half of π/2 folded into the
    // subtracted term so the cancellation keeps the trailing bits.
    let s_sr = vfmaq_f64(s, s, r);
    let two_s = vaddq_f64(s_sr, s_sr);
    let res_refl = vsubq_f64(
        vdupq_n_f64(PIO2_HI_F64),
        vsubq_f64(two_s, vdupq_n_f64(PIO2_LO_F64)),
    );

    let result = vbslq_f64(small, res_small, res_refl);
    vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(result), sign))
}

/// Fast SIMD acos for f32 using NEON
/// acos(x) = pi/2 - asin(x)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acos_f32(x: float32x4_t) -> float32x4_t {
    let pi_half = vdupq_n_f32(std::f32::consts::FRAC_PI_2);
    vsubq_f32(pi_half, asin_f32(x))
}

/// Fast SIMD acos for f64 using NEON
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details. Built from the
/// reflection directly, not as π/2 - asin(x): that subtraction cancels away the
/// whole result as x approaches 1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acos_f64(x: float64x2_t) -> float64x2_t {
    use asin_coefficients::*;

    let one = vdupq_n_f64(1.0);
    let zero = vdupq_n_f64(0.0);
    let half = vdupq_n_f64(HALF_F64);
    let pio2_lo = vdupq_n_f64(PIO2_LO_F64);
    let ax = vabsq_f64(x);

    let small = vcltq_f64(ax, half);
    let t_refl = vmulq_f64(vsubq_f64(one, ax), half);
    let t = vbslq_f64(small, vmulq_f64(ax, ax), t_refl);
    let r = asin_r_f64(t);
    let s = vsqrtq_f64(t);

    // |x| <= 0.5: π/2 - asin(x), evaluated without forming asin(x) first.
    let res_small = vsubq_f64(
        vdupq_n_f64(PIO2_HI_F64),
        vaddq_f64(x, vsubq_f64(vmulq_f64(x, r), pio2_lo)),
    );

    // x >= 0.5: 2*asin(sqrt(t)), which is small and free of cancellation.
    let s_sr = vfmaq_f64(s, s, r);
    let res_pos = vaddq_f64(s_sr, s_sr);

    // x <= -0.5: π - 2*asin(sqrt(t)).
    let w = vaddq_f64(s, vsubq_f64(vmulq_f64(s, r), pio2_lo));
    let res_neg = vsubq_f64(vdupq_n_f64(PI_HI_F64), vaddq_f64(w, w));

    // NaN fails both comparisons and lands on the negative branch, where the
    // NaN sqrt argument propagates.
    let positive = vcgtq_f64(x, zero);
    let res_refl = vbslq_f64(positive, res_pos, res_neg);
    vbslq_f64(small, res_small, res_refl)
}

/// Fast SIMD cbrt (cube root) for f32 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cbrt_f32(x: float32x4_t) -> float32x4_t {
    // Handle sign separately
    let sign_mask = vdupq_n_u32(0x80000000);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let abs_x = vabsq_f32(x);

    let one_third = vdupq_n_f32(1.0 / 3.0);
    let bias = vdupq_n_f32(127.0);

    // Extract exponent
    let xi = vreinterpretq_s32_f32(abs_x);
    let exp_bits = vshrq_n_s32::<23>(xi);
    let exp_f = vcvtq_f32_s32(vsubq_s32(exp_bits, vdupq_n_s32(127)));

    // Initial guess: 2^(e/3)
    let new_exp = vmulq_f32(exp_f, one_third);
    let new_exp_i = vcvtq_s32_f32(vaddq_f32(new_exp, bias));
    let guess = vreinterpretq_f32_s32(vshlq_n_s32::<23>(new_exp_i));

    // Newton-Raphson: y = (2*y + x/y^2) / 3
    let two = vdupq_n_f32(2.0);
    let three = vdupq_n_f32(3.0);

    let y = guess;
    let y2 = vmulq_f32(y, y);
    let y_new = vdivq_f32(vfmaq_f32(vdivq_f32(abs_x, y2), two, y), three);

    let y2 = vmulq_f32(y_new, y_new);
    let result = vdivq_f32(vfmaq_f32(vdivq_f32(abs_x, y2), two, y_new), three);

    // Restore sign
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign))
}

/// Fast SIMD cbrt (cube root) for f64 using NEON
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cbrt_f64(x: float64x2_t) -> float64x2_t {
    let sign_mask = vdupq_n_u64(0x8000000000000000);
    let sign = vandq_u64(vreinterpretq_u64_f64(x), sign_mask);
    let abs_x = vabsq_f64(x);

    let one_third = vdupq_n_f64(1.0 / 3.0);

    // Initial guess: cbrt(x) ≈ exp(log(x) / 3)
    let log_x = log_f64(abs_x);
    let guess = exp_f64(vmulq_f64(log_x, one_third));

    let two = vdupq_n_f64(2.0);
    let three = vdupq_n_f64(3.0);

    let y = guess;
    let y2 = vmulq_f64(y, y);
    let y_new = vdivq_f64(vfmaq_f64(vdivq_f64(abs_x, y2), two, y), three);

    let y2 = vmulq_f64(y_new, y_new);
    let result = vdivq_f64(vfmaq_f64(vdivq_f64(abs_x, y2), two, y_new), three);

    vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(result), sign))
}
