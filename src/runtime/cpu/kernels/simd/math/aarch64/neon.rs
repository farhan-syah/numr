//! NEON mathematical function implementations for ARM64
//!
//! Provides vectorized transcendental functions using 128-bit NEON registers.
//! All algorithms match those in `common.rs` to ensure numerical consistency.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Relative Error |
//! |----------|-----|-----|----------------|
//! | exp      | 4   | 2   | 1 ulp / 1e-12 |
//! | tanh     | 4   | 2   | 2 ulp / 2 ulp |
//! | log      | 4   | 2   | 2 ulp / 2 ulp |
//! | sin      | 4   | 2   | 2 ulp / 4 ulp |
//! | cos      | 4   | 2   | 2 ulp / 4 ulp |
//! | tan      | 4   | 2   | 2 ulp / 4 ulp |
//! | atan     | 4   | 2   | 2 ulp / 2 ulp |
//! | asin     | 4   | 2   | 2 ulp / 2 ulp |
//! | acos     | 4   | 2   | 2 ulp / 2 ulp |
//!
//! The log family holds below 2 ulps in both precisions, subnormal inputs
//! included.
//!
//! The sin/cos/tan bounds hold for |x| <= 2^21 * π/2 (about 3.3e6) in f64 and
//! |x| <= 2^17 in f32, the limits of the Cody-Waite reductions in `common.rs`,
//! and for tan away from its poles.
//!
//! exp, exp2, expm1, cbrt, sinh, cosh, tanh, asinh, acosh and atanh hold below
//! 2 ulps in both precisions, each over its whole representable range.
//!
//! # Safety
//!
//! All functions require NEON CPU features (always available on AArch64).

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::common::{
    asin_coefficients, atan_coefficients, cbrt_constants, exp_coefficients, exp2_coefficients,
    hyperbolic_breakpoints, inv_hyperbolic_breakpoints, log_coefficients, trig_coefficients,
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

    // FMAX/FMIN propagate NaN, so the clamp needs no NaN repair here; the x86
    // paths order the operands to get the same behaviour out of maxps/minps.
    let xc = vmaxq_f32(x, vdupq_n_f32(MIN_F32));
    let xc = vminq_f32(xc, vdupq_n_f32(MAX_F32));

    let y = vmulq_f32(xc, vdupq_n_f32(std::f32::consts::LOG2_E));
    let n = vrndnq_f32(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F32 is exact
    let r = vfmsq_f32(xc, n, vdupq_n_f32(LN2_HI_F32));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO_F32));

    // Horner: one rounding per term, and no r^k powers to lose bits in
    let mut poly = vdupq_n_f32(C7_F32);
    poly = vfmaq_f32(vdupq_n_f32(C6_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C5_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C4_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C3_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C2_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C1_F32), poly, r);
    poly = vfmaq_f32(vdupq_n_f32(C0_F32), poly, r);

    // 2^n as two halved powers of two. 2^128 on its own is infinity, yet the
    // largest finite result needs it; splitting keeps both factors normal, so
    // an overflow reaches infinity in the second multiply and a subnormal
    // result takes exactly one rounding.
    let n_i32 = vcvtq_s32_f32(n);
    let n_hi = vshrq_n_s32::<1>(n_i32);
    let n_lo = vsubq_s32(n_i32, n_hi);
    let bias = vdupq_n_s32(127);
    let p_hi = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_hi, bias)));
    let p_lo = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_lo, bias)));

    vmulq_f32(vmulq_f32(poly, p_hi), p_lo)
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

    // FMAX/FMIN propagate NaN, so the clamp needs no NaN repair here; the x86
    // paths order the operands to get the same behaviour out of maxpd/minpd.
    let xc = vmaxq_f64(x, vdupq_n_f64(MIN_F64));
    let xc = vminq_f64(xc, vdupq_n_f64(MAX_F64));

    let y = vmulq_f64(xc, log2e);
    let n = vrndnq_f64(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F64 is exact
    let r = vfmsq_f64(xc, n, ln2_hi);
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

    // 2^n as two halved powers of two. 2^1024 on its own is infinity, yet the
    // largest finite result needs it, and 2^-1076 is not representable at all.
    // Splitting keeps both factors normal, so an overflow reaches infinity in
    // the second multiply and a subnormal result takes exactly one rounding
    // because the first multiply is exact.
    let n_hi = vrndq_f64(vmulq_f64(n, vdupq_n_f64(0.5)));
    let n_lo = vsubq_f64(n, n_hi);
    let bias = vdupq_n_s64(1023);
    let p_hi = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n_hi), bias)));
    let p_lo = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n_lo), bias)));

    vmulq_f64(vmulq_f64(poly, p_hi), p_lo)
}

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh for f32 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^2x - 1)/(e^2x + 1)` cancels the
/// whole numerator away as x approaches zero; `u/(u+2)` with `u = expm1(2|x|)`
/// never forms that difference.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tanh_f32(x: float32x4_t) -> float32x4_t {
    let a = vabsq_f32(x);
    let u = expm1_f32(vaddq_f32(a, a));

    let d = vdivq_f32(u, vaddq_f32(u, vdupq_n_f32(2.0)));
    // u saturates to infinity past |x| = 44.5; the limit of u/(u+2) there is 1,
    // whereas the quotient itself would be inf/inf.
    let is_inf = vceqq_f32(u, vdupq_n_f32(f32::INFINITY));
    let d = vbslq_f32(is_inf, vdupq_n_f32(1.0), d);

    // The sign rides the sign bit, so tanh(-0) is -0 and tanh(-inf) is -1.
    copy_sign_f32(d, x)
}

/// OR the sign bit of `source` into `magnitude`, the f32 counterpart of
/// `copy_sign_f64`. `magnitude` must be non-negative or NaN.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn copy_sign_f32(magnitude: float32x4_t, source: float32x4_t) -> float32x4_t {
    let sign_mask = vdupq_n_u32(0x8000_0000);
    let sign = vandq_u32(vreinterpretq_u32_f32(source), sign_mask);
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(magnitude), sign))
}

/// Fast SIMD tanh for f64 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^2x - 1)/(e^2x + 1)` cancels the
/// whole numerator away as x approaches zero; `u/(u+2)` with `u = expm1(2|x|)`
/// never forms that difference.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tanh_f64(x: float64x2_t) -> float64x2_t {
    let a = vabsq_f64(x);
    let u = expm1_f64(vaddq_f64(a, a));

    let d = vdivq_f64(u, vaddq_f64(u, vdupq_n_f64(2.0)));
    // u saturates to infinity past |x| = 355; the limit of u/(u+2) there is 1,
    // whereas the quotient itself would be inf/inf.
    let is_inf = vceqq_f64(u, vdupq_n_f64(f64::INFINITY));
    let d = vbslq_f64(is_inf, vdupq_n_f64(1.0), d);

    // The sign rides the sign bit, so tanh(-0) is -0 and tanh(-inf) is -1.
    copy_sign_f64(d, x)
}

/// OR the sign bit of `source` into `magnitude`, which must be non-negative or
/// NaN — every caller here computes it from `|x|`.
///
/// The odd hyperbolic functions all work on `|x|` and restore the sign here, so
/// that ±0 and ±inf come back with the sign they went in with.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn copy_sign_f64(magnitude: float64x2_t, source: float64x2_t) -> float64x2_t {
    let sign_mask = vdupq_n_u64(0x8000_0000_0000_0000);
    let sign = vandq_u64(vreinterpretq_u64_f64(source), sign_mask);
    vreinterpretq_f64_u64(vorrq_u64(vreinterpretq_u64_f64(magnitude), sign))
}

// ============================================================================
// Natural logarithm: log(x)
// ============================================================================

/// Split `x` into an exponent `n` and `log(m)`, where `m` is the mantissa
/// normalized to [sqrt(2)/2, sqrt(2)), so that `log(x) = n*ln(2) + log(m)`.
///
/// The f32 counterpart of `log_reduce_f64`; log, log2, log10 and log1p all
/// share it and differ only in how they recombine the two parts. NEON has full
/// 32-bit integer support, so unlike the f64 path this stays vectorized
/// throughout.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn log_reduce_f32(x: float32x4_t) -> (float32x4_t, float32x4_t) {
    use log_coefficients::*;

    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let half = vdupq_n_f32(0.5);
    let sqrt2 = vdupq_n_f32(std::f32::consts::SQRT_2);

    // Subnormals carry no implicit leading 1, so the exponent/mantissa split
    // below is only valid after scaling them into the normal range.
    let is_sub = vcltq_f32(x, vdupq_n_f32(f32::MIN_POSITIVE));
    let x_norm = vbslq_f32(is_sub, vmulq_f32(x, vdupq_n_f32(SUBNORMAL_SCALE_F32)), x);
    let n_shift = vbslq_f32(is_sub, vdupq_n_f32(SUBNORMAL_SHIFT_F32), vdupq_n_f32(0.0));

    let x_bits = vreinterpretq_s32_f32(x_norm);
    let exp_raw = vshrq_n_s32::<23>(x_bits);
    let exp_unbiased = vsubq_s32(exp_raw, vdupq_n_s32(EXP_BIAS_F32));
    let mut n = vcvtq_f32_s32(exp_unbiased);

    let mantissa_mask = vdupq_n_s32(MANTISSA_MASK_F32);
    let exp_zero = vdupq_n_s32(EXP_ZERO_F32);
    let m_bits = vorrq_s32(vandq_s32(x_bits, mantissa_mask), exp_zero);
    let mut m = vreinterpretq_f32_s32(m_bits);

    // Normalize: if m > sqrt(2), halve it and carry a 1 into the exponent, so
    // that f stays in [-0.2929, 0.4142].
    let need_adjust = vcgtq_f32(m, sqrt2);
    m = vbslq_f32(need_adjust, vmulq_f32(m, half), m);
    n = vbslq_f32(need_adjust, vaddq_f32(n, one), n);
    let n = vaddq_f32(n, n_shift);

    // s = f/(2+f) halves the argument and leaves only odd powers, which is what
    // lets four terms reach f32 precision (see `log_coefficients`).
    let f = vsubq_f32(m, one);
    let s = vdivq_f32(f, vaddq_f32(two, f));
    let z = vmulq_f32(s, s);

    let r = vmulq_f32(
        z,
        vfmaq_f32(
            vdupq_n_f32(LG1_F32),
            z,
            vfmaq_f32(
                vdupq_n_f32(LG2_F32),
                z,
                vfmaq_f32(vdupq_n_f32(LG3_F32), z, vdupq_n_f32(LG4_F32)),
            ),
        ),
    );

    // log(m) = f - (hfsq - s*(hfsq + R)); keeping f outside the parentheses
    // stops the f² term from eating f's low bits when f is small.
    let hfsq = vmulq_f32(half, vmulq_f32(f, f));
    let logm = vsubq_f32(f, vsubq_f32(hfsq, vmulq_f32(s, vaddq_f32(hfsq, r))));

    (n, logm)
}

/// Apply the IEEE domain values shared by the f32 log, log2 and log10:
/// `log(0) = -inf`, `log(x < 0) = NaN`, `log(+inf) = +inf`, `log(NaN) = NaN`.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn log_special_f32(x: float32x4_t, r: float32x4_t) -> float32x4_t {
    let zero = vdupq_n_f32(0.0);

    // `vcgtq_f32` is false for NaN, so its complement catches NaN alongside the
    // non-positive inputs instead of feeding garbage through the polynomial.
    let positive = vcgtq_f32(x, zero);
    let is_zero = vceqq_f32(x, zero);
    let is_inf = vceqq_f32(x, vdupq_n_f32(f32::INFINITY));

    let out = vbslq_f32(positive, r, vdupq_n_f32(f32::NAN));
    let out = vbslq_f32(is_zero, vdupq_n_f32(f32::NEG_INFINITY), out);
    vbslq_f32(is_inf, vdupq_n_f32(f32::INFINITY), out)
}

/// Fast SIMD log approximation for f32 using NEON
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps over the whole positive range, subnormals
/// included.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log_f32(x: float32x4_t) -> float32x4_t {
    use log_coefficients::{LN2_HI_F32, LN2_LO_F32};

    let (n, logm) = log_reduce_f32(x);

    // Split ln(2): the head is exact against every reachable n, the tail
    // restores the bits a single rounded ln(2) would drop.
    let lo = vfmaq_f32(logm, n, vdupq_n_f32(LN2_LO_F32));
    let r = vfmaq_f32(lo, n, vdupq_n_f32(LN2_HI_F32));

    log_special_f32(x, r)
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

/// Cody-Waite reduction of `x` modulo π/2 for f32.
///
/// Returns the quadrant index and the reduced argument in [-π/4, π/4]. See
/// `common::_TRIG_ALGORITHM_DOC`; the subtraction chain is exact for |j| below
/// 2^24, and the quadrant index itself stays exact for |x| <= 2^17.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn trig_reduce_f32(x: float32x4_t) -> (int32x4_t, float32x4_t) {
    use trig_coefficients::{PIO2_1_F32, PIO2_2_F32, PIO2_3_F32};

    let j = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(std::f32::consts::FRAC_2_PI)));

    let y = vfmsq_f32(x, j, vdupq_n_f32(PIO2_1_F32));
    let y = vfmsq_f32(y, j, vdupq_n_f32(PIO2_2_F32));
    let y = vfmsq_f32(y, j, vdupq_n_f32(PIO2_3_F32));

    (vcvtq_s32_f32(j), y)
}

/// Minimax sin kernel on the reduced argument, |y| <= π/4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sin_kernel_f32(y: float32x4_t) -> float32x4_t {
    use trig_coefficients::{SIN0_F32, SIN1_F32, SIN2_F32};

    let z = vmulq_f32(y, y);

    let mut p = vdupq_n_f32(SIN0_F32);
    p = vfmaq_f32(vdupq_n_f32(SIN1_F32), p, z);
    p = vfmaq_f32(vdupq_n_f32(SIN2_F32), p, z);

    // y is added last, so a tiny y comes back unchanged and keeps its sign.
    vfmaq_f32(y, vmulq_f32(z, y), p)
}

/// Minimax cos kernel on the reduced argument, |y| <= π/4.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn cos_kernel_f32(y: float32x4_t) -> float32x4_t {
    use trig_coefficients::{COS0_F32, COS1_F32, COS2_F32};

    let one = vdupq_n_f32(1.0);
    let z = vmulq_f32(y, y);

    let mut p = vdupq_n_f32(COS0_F32);
    p = vfmaq_f32(vdupq_n_f32(COS1_F32), p, z);
    p = vfmaq_f32(vdupq_n_f32(COS2_F32), p, z);
    let r = vmulq_f32(vmulq_f32(z, z), p);

    // `1 - z/2` rounds; `(1 - w) - hz` is exact and returns the rounded bits.
    let hz = vmulq_f32(vdupq_n_f32(0.5), z);
    let w = vsubq_f32(one, hz);
    let correction = vsubq_f32(vsubq_f32(one, w), hz);
    vaddq_f32(w, vaddq_f32(correction, r))
}

/// Evaluate sin on quadrant `j + offset`, the shared core of f32 sin and cos.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn sin_quadrant_f32(x: float32x4_t, offset: i32) -> float32x4_t {
    let (j, y) = trig_reduce_f32(x);
    let sin_y = sin_kernel_f32(y);
    let cos_y = cos_kernel_f32(y);

    // j mod 4 = 0: sin(y), 1: cos(y), 2: -sin(y), 3: -cos(y)
    let j_mod_4 = vandq_s32(vaddq_s32(j, vdupq_n_s32(offset)), vdupq_n_s32(3));

    let use_cos_mask = vceqq_s32(vandq_s32(j_mod_4, vdupq_n_s32(1)), vdupq_n_s32(1));
    let negate_mask = vceqq_s32(vandq_s32(j_mod_4, vdupq_n_s32(2)), vdupq_n_s32(2));

    let result = vbslq_f32(use_cos_mask, cos_y, sin_y);
    vbslq_f32(negate_mask, vnegq_f32(result), result)
}

/// Fast SIMD sin approximation for f32 using NEON
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps for |x| <= 2^17.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sin_f32(x: float32x4_t) -> float32x4_t {
    let r = sin_quadrant_f32(x, 0);

    // sin(±0) = ±0. The reduction computes 0 - (-0 * π/2), which is +0 for both
    // signed zeros, so the input is restored here.
    let is_zero = vceqq_f32(x, vdupq_n_f32(0.0));
    vbslq_f32(is_zero, x, r)
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
/// Shifts the quadrant index by one rather than evaluating `sin(x + π/2)`,
/// which would round the sum before reduction and lose bits proportional to
/// |x|. Relative error stays below 2 ulps for |x| <= 2^17.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cos_f32(x: float32x4_t) -> float32x4_t {
    sin_quadrant_f32(x, 1)
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
/// Relative error stays below 2 ulps away from the poles, for |x| <= 2^17.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn tan_f32(x: float32x4_t) -> float32x4_t {
    let (j, y) = trig_reduce_f32(x);
    let sin_y = sin_kernel_f32(y);
    let cos_y = cos_kernel_f32(y);

    // Odd quadrant: tan(x) = -cot(y) = -cos(y)/sin(y). Swapping the ratio costs
    // one rounding, where inverting an already-rounded tan(y) costs two.
    let odd_mask = vceqq_s32(vandq_s32(j, vdupq_n_s32(1)), vdupq_n_s32(1));
    let num = vbslq_f32(odd_mask, vnegq_f32(cos_y), sin_y);
    let den = vbslq_f32(odd_mask, sin_y, cos_y);
    let r = vdivq_f32(num, den);

    // tan(±0) = ±0; the reduction turns -0 into +0.
    let is_zero = vceqq_f32(x, vdupq_n_f32(0.0));
    vbslq_f32(is_zero, x, r)
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
    let sign_mask = vdupq_n_u32(0x8000_0000);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let ax = vabsq_f32(x);

    // Pick the reduction centre and the matching atan(c) head/correction. The
    // breakpoint masks are nested, so blending from the widest bucket inward
    // leaves each lane holding its tightest match.
    //
    // The widest bucket has the centre at infinity: t = -1/|x|. NaN fails every
    // comparison and stays there, which propagates NaN through the division.
    let mut t = vdivq_f32(vdupq_n_f32(-1.0), ax);
    let mut hi = vdupq_n_f32(ATAN_HI1_F32);
    let mut lo = vdupq_n_f32(ATAN_LO1_F32);

    let in1 = vcltq_f32(ax, vdupq_n_f32(BREAK1_F32));
    let t_mid = vdivq_f32(vsubq_f32(ax, one), vaddq_f32(ax, one));
    t = vbslq_f32(in1, t_mid, t);
    hi = vbslq_f32(in1, vdupq_n_f32(ATAN_HI0_F32), hi);
    lo = vbslq_f32(in1, vdupq_n_f32(ATAN_LO0_F32), lo);

    let zero = vdupq_n_f32(0.0);
    let in0 = vcltq_f32(ax, vdupq_n_f32(BREAK0_F32));
    t = vbslq_f32(in0, ax, t);
    hi = vbslq_f32(in0, zero, hi);
    lo = vbslq_f32(in0, zero, lo);

    // t in [-0.4143, 0.4143]; |x| = inf gives t = -0.0, so the result is π/2.
    let z = vmulq_f32(t, t);

    let mut p = vdupq_n_f32(AT0_F32);
    p = vfmaq_f32(vdupq_n_f32(AT1_F32), p, z);
    p = vfmaq_f32(vdupq_n_f32(AT2_F32), p, z);
    p = vfmaq_f32(vdupq_n_f32(AT3_F32), p, z);

    // atan(x) = atan(c) + atan(t), with the correction term added beside the
    // polynomial rather than beside the head.
    let poly = vfmaq_f32(t, vmulq_f32(z, t), p);
    let result = vaddq_f32(hi, vaddq_f32(lo, poly));

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
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. Handing `x * ln2` to `exp_f32`
/// rounds a value as large as 128 before the exponential, which the
/// exponential turns into 5e-6 of relative error — forty ulps.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp2_f32(x: float32x4_t) -> float32x4_t {
    use exp_coefficients::{C0_F32, C1_F32, C2_F32, C3_F32, C4_F32, C5_F32, C6_F32, C7_F32};
    use exp2_coefficients::{MAX_F32, MIN_F32};

    let xc = vmaxq_f32(x, vdupq_n_f32(MIN_F32));
    let xc = vminq_f32(xc, vdupq_n_f32(MAX_F32));

    // n and r are both exact, so the reduction itself costs nothing.
    let n = vrndnq_f32(xc);
    let r = vsubq_f32(xc, n);
    let rl = vmulq_f32(r, vdupq_n_f32(std::f32::consts::LN_2));

    let mut poly = vdupq_n_f32(C7_F32);
    poly = vfmaq_f32(vdupq_n_f32(C6_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C5_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C4_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C3_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C2_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C1_F32), poly, rl);
    poly = vfmaq_f32(vdupq_n_f32(C0_F32), poly, rl);

    // Two halved powers of two, as in `exp_f32`.
    let n_i32 = vcvtq_s32_f32(n);
    let n_hi = vshrq_n_s32::<1>(n_i32);
    let n_lo = vsubq_s32(n_i32, n_hi);
    let bias = vdupq_n_s32(127);
    let p_hi = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_hi, bias)));
    let p_lo = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(n_lo, bias)));

    vmulq_f32(vmulq_f32(poly, p_hi), p_lo)
}

/// Fast SIMD exp2 (2^x) for f64 using NEON
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. Borrowing `exp(x * ln2)` would
/// round the product once, and the exponential turns that absolute error into
/// a relative one — about 1e-13 near |x| = 1000.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn exp2_f64(x: float64x2_t) -> float64x2_t {
    use exp2_coefficients::*;

    let xc = vmaxq_f64(x, vdupq_n_f64(MIN_F64));
    let xc = vminq_f64(xc, vdupq_n_f64(MAX_F64));

    // Both n and r are exact, so nothing is lost before the polynomial.
    let n = vrndnq_f64(xc);
    let r = vsubq_f64(xc, n);

    let mut poly = vdupq_n_f64(C13_F64);
    poly = vfmaq_f64(vdupq_n_f64(C12_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C11_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C10_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C9_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C8_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C7_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C6_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C5_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C4_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C3_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C2_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C1_F64), poly, r);
    poly = vfmaq_f64(vdupq_n_f64(C0_F64), poly, r);

    // Split the power of two in half: both factors stay normal, an overflow
    // reaches infinity in the second multiply, and a subnormal result takes
    // exactly one rounding because the first multiply is exact.
    let n_hi = vrndq_f64(vmulq_f64(n, vdupq_n_f64(0.5)));
    let n_lo = vsubq_f64(n, n_hi);
    let bias = vdupq_n_s64(1023);
    let p_hi = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n_hi), bias)));
    let p_lo = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n_lo), bias)));
    let out = vmulq_f64(vmulq_f64(poly, p_hi), p_lo);

    // Restore NaN explicitly rather than relying on FMAX/FMIN propagating it,
    // so this path matches the x86 ones, where max/min return their second
    // operand and the clamp would swallow it.
    vbslq_f64(vceqq_f64(x, x), out, x)
}

/// Fast SIMD expm1 (e^x - 1) for f32 using NEON
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. A degree-4 Taylor series on
/// |x| <= 0.5 drops `x⁵/120`, which is 2.6e-4 at the interval edge — over two
/// thousand ulps — and `exp(x) - 1` above it cancels the whole result away.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn expm1_f32(x: float32x4_t) -> float32x4_t {
    use exp_coefficients::*;

    let xc = vmaxq_f32(x, vdupq_n_f32(EXPM1_MIN_F32));
    let xc = vminq_f32(xc, vdupq_n_f32(EXPM1_MAX_F32));

    let y = vmulq_f32(xc, vdupq_n_f32(std::f32::consts::LOG2_E));
    let n = vrndnq_f32(y);

    // Cody-Waite reduction, identical to exp_f32: r = x - n*ln2, |r| <= ln2/2.
    let r = vfmsq_f32(xc, n, vdupq_n_f32(LN2_HI_F32));
    let r = vfmsq_f32(r, n, vdupq_n_f32(LN2_LO_F32));

    // Q is the exp series from its r² term up, so expm1(r) = r + r²*Q(r) keeps
    // r itself outside the polynomial and never rounds against a leading 1.
    let mut q = vdupq_n_f32(C7_F32);
    q = vfmaq_f32(vdupq_n_f32(C6_F32), q, r);
    q = vfmaq_f32(vdupq_n_f32(C5_F32), q, r);
    q = vfmaq_f32(vdupq_n_f32(C4_F32), q, r);
    q = vfmaq_f32(vdupq_n_f32(C3_F32), q, r);
    q = vfmaq_f32(vdupq_n_f32(C2_F32), q, r);
    let e = vfmaq_f32(r, vmulq_f32(r, r), q);

    // 2^n*(1+E) - 1 = 2*(t*E + (t - 0.5)) with t = 2^(n-1). t and t - 0.5 are
    // both exact, and the halved scale keeps n = 128 representable, so an
    // overflow happens in the final doubling rather than in 2^n.
    let bias = vdupq_n_s32(127 - 1);
    let t = vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(vcvtq_s32_f32(n), bias)));
    let out = vmulq_f32(
        vdupq_n_f32(2.0),
        vfmaq_f32(vsubq_f32(t, vdupq_n_f32(0.5)), t, e),
    );

    // At n = 0 the scale is exactly 1 and the answer is E itself. Taking it
    // directly matters for a subnormal E, where the halved form's `0.5 * E`
    // rounds the last bit to even and loses the whole value.
    let zero = vdupq_n_f32(0.0);
    let out = vbslq_f32(vceqq_f32(n, zero), e, out);

    // expm1(±0) = ±0, which the reduction would otherwise return as +0.
    vbslq_f32(vceqq_f32(x, zero), x, out)
}

/// Fast SIMD expm1 (e^x - 1) for f64 using NEON
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. A degree-4 Taylor series on
/// |x| <= 0.5 drops `x⁵/120`, which is 2.6e-4 at the interval edge — twelve
/// decimal digits short of double precision.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn expm1_f64(x: float64x2_t) -> float64x2_t {
    use exp_coefficients::*;

    let xc = vmaxq_f64(x, vdupq_n_f64(EXPM1_MIN_F64));
    let xc = vminq_f64(xc, vdupq_n_f64(EXPM1_MAX_F64));

    let y = vmulq_f64(xc, vdupq_n_f64(std::f64::consts::LOG2_E));
    let n = vrndnq_f64(y);

    // Cody-Waite reduction, identical to exp_f64: r = x - n*ln2, |r| <= ln2/2.
    let r = vfmsq_f64(xc, n, vdupq_n_f64(LN2_HI_F64));
    let r = vfmsq_f64(r, n, vdupq_n_f64(LN2_LO_F64));

    // Q is the exp series from its r² term up, so expm1(r) = r + r²*Q(r) keeps
    // r itself outside the polynomial and never rounds against a leading 1.
    let mut q = vdupq_n_f64(C13_F64);
    q = vfmaq_f64(vdupq_n_f64(C12_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C11_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C10_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C9_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C8_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C7_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C6_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C5_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C4_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C3_F64), q, r);
    q = vfmaq_f64(vdupq_n_f64(C2_F64), q, r);
    let e = vfmaq_f64(r, vmulq_f64(r, r), q);

    // 2^n*(1+E) - 1 = 2*(t*E + (t - 0.5)) with t = 2^(n-1). t and t - 0.5 are
    // both exact, and at n = 0 they are 0.5 and 0, so the result is E itself.
    // The halved scale also keeps n = 1024 representable, so the overflow
    // happens in the final doubling rather than in 2^n.
    let bias = vdupq_n_s64(1023 - 1);
    let t = vreinterpretq_f64_s64(vshlq_n_s64::<52>(vaddq_s64(vcvtq_s64_f64(n), bias)));
    let out = vmulq_f64(
        vdupq_n_f64(2.0),
        vfmaq_f64(vsubq_f64(t, vdupq_n_f64(0.5)), t, e),
    );

    // At n = 0 the scale is exactly 1 and the answer is E itself. Taking it
    // directly matters for a subnormal E, where the halved form's `0.5 * E`
    // rounds the last bit to even and loses the whole value.
    let out = vbslq_f64(vceqq_f64(n, vdupq_n_f64(0.0)), e, out);

    // expm1(±0) = ±0, and the clamp above would otherwise silence NaN.
    let out = vbslq_f64(vceqq_f64(x, vdupq_n_f64(0.0)), x, out);
    vbslq_f64(vceqq_f64(x, x), out, x)
}

/// Fast SIMD log2 for f32 using NEON
///
/// Scaling `log(x)` would fold the exponent through two roundings and miss
/// exact powers of two, so the exponent is added back untouched instead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log2_f32(x: float32x4_t) -> float32x4_t {
    let (n, logm) = log_reduce_f32(x);
    let r = vfmaq_f32(n, logm, vdupq_n_f32(std::f32::consts::LOG2_E));
    log_special_f32(x, r)
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
///
/// `log10(x) = n*log10(2) + log(m)*log10(e)`, keeping the exact exponent out
/// of the mantissa's rounding for the same reason as `log2_f32`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log10_f32(x: float32x4_t) -> float32x4_t {
    let (n, logm) = log_reduce_f32(x);
    let scaled = vmulq_f32(logm, vdupq_n_f32(std::f32::consts::LOG10_E));
    let r = vfmaq_f32(scaled, n, vdupq_n_f32(std::f32::consts::LOG10_2));
    log_special_f32(x, r)
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
///
/// `1 + x` alone rounds away the information log1p exists to keep, so the sum
/// is carried as an exact pair `u + c` and the residual is folded back in. The
/// degree-4 Taylor series this replaces dropped `x⁵/5`, which is 6.1e-3 at
/// x = -0.5 — a hundredth of the result there.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn log1p_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let u = vaddq_f32(one, x);

    // Fast2Sum: 1 + x = u + c exactly, with the larger addend leading.
    let c_small = vsubq_f32(x, vsubq_f32(u, one));
    let c_large = vsubq_f32(one, vsubq_f32(u, x));
    let x_leads = vcltq_f32(vabsq_f32(x), one);
    let c = vbslq_f32(x_leads, c_small, c_large);

    // log(u + c) = log(u) + log1p(c/u), and |c/u| <= 2^-24, so the inner series
    // collapses to its first term.
    let r = vaddq_f32(log_f32(u), vdivq_f32(c, u));

    // u == 1 means x fell entirely off the end of the sum; log1p(x) is then x
    // to within half an ulp. This is also what carries signed zero through.
    let is_unit = vceqq_f32(u, one);
    let out = vbslq_f32(is_unit, x, r);

    // x == -1 gives u == 0 and c/u = 0/0; x == +inf gives inf - inf.
    let is_neg_one = vceqq_f32(x, vdupq_n_f32(-1.0));
    let is_inf = vceqq_f32(x, vdupq_n_f32(f32::INFINITY));
    let out = vbslq_f32(is_neg_one, vdupq_n_f32(f32::NEG_INFINITY), out);
    vbslq_f32(is_inf, vdupq_n_f32(f32::INFINITY), out)
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
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^x - e^-x)/2` subtracts two
/// values that both approach 1 as x approaches 0, so it keeps none of the
/// result; `(u + u/(1+u))/2` with `u = expm1(|x|)` keeps all of it.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sinh_f32(x: float32x4_t) -> float32x4_t {
    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(0.5);
    let a = vabsq_f32(x);
    let u = expm1_f32(a);

    let d = vdivq_f32(u, vaddq_f32(one, u));
    // u/(1+u) tends to 1 as u overflows, where the quotient itself is inf/inf.
    let is_inf = vceqq_f32(u, vdupq_n_f32(f32::INFINITY));
    let d = vbslq_f32(is_inf, one, d);
    let s = vmulq_f32(half, vaddq_f32(u, d));

    // expm1 overflows at ln(f32::MAX) = 88.7228 while sinh stays finite up to
    // 89.4159. Past the breakpoint sinh is 0.5*exp(|x|), built as in cosh_f32.
    let t = exp_f32(vmulq_f32(half, a));
    let far = vmulq_f32(vmulq_f32(half, t), t);
    let big = vcgtq_f32(a, vdupq_n_f32(hyperbolic_breakpoints::BIG_F32));
    let s = vbslq_f32(big, far, s);

    copy_sign_f32(s, x)
}

/// Fast SIMD sinh for f64 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^x - e^-x)/2` subtracts two
/// values that both approach 1 as x approaches 0, so it keeps none of the
/// result; `(u + u/(1+u))/2` with `u = expm1(|x|)` keeps all of it.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn sinh_f64(x: float64x2_t) -> float64x2_t {
    let one = vdupq_n_f64(1.0);
    let a = vabsq_f64(x);
    let u = expm1_f64(a);

    let d = vdivq_f64(u, vaddq_f64(one, u));
    // u/(1+u) tends to 1 as u overflows, where the quotient itself is inf/inf.
    let is_inf = vceqq_f64(u, vdupq_n_f64(f64::INFINITY));
    let d = vbslq_f64(is_inf, one, d);
    let s = vmulq_f64(vdupq_n_f64(0.5), vaddq_f64(u, d));

    // expm1 overflows at ln(f64::MAX) = 709.7827 while sinh stays finite up to
    // 710.4758. Past the breakpoint sinh is 0.5*exp(|x|), built as in cosh_f64.
    let half = vdupq_n_f64(0.5);
    let t = exp_f64(vmulq_f64(half, a));
    let far = vmulq_f64(vmulq_f64(half, t), t);
    let big = vcgtq_f64(a, vdupq_n_f64(hyperbolic_breakpoints::BIG_F64));
    let s = vbslq_f64(big, far, s);

    copy_sign_f64(s, x)
}

/// Fast SIMD cosh for f32 using NEON
///
/// See `common::hyperbolic_breakpoints`. `(e^x + e^-x)/2` returns infinity over
/// the whole band where exp has overflowed but cosh has not, [88.7228,
/// 89.4159], so |x| past the breakpoint takes the squared form instead.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosh_f32(x: float32x4_t) -> float32x4_t {
    let half = vdupq_n_f32(0.5);
    let a = vabsq_f32(x);

    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(vnegq_f32(x));
    let near = vmulq_f32(half, vaddq_f32(exp_x, exp_neg_x));

    // (0.5*t)*t with t = exp(|x|/2) is 0.5*exp(|x|) with no intermediate past
    // f32::MAX. Halving t first, not the product, is what keeps it finite.
    let t = exp_f32(vmulq_f32(half, a));
    let far = vmulq_f32(vmulq_f32(half, t), t);

    // NaN compares false and takes the near branch, where exp propagates it.
    let big = vcgtq_f32(a, vdupq_n_f32(hyperbolic_breakpoints::BIG_F32));
    vbslq_f32(big, far, near)
}

/// Fast SIMD cosh for f64 using NEON
///
/// See `common::hyperbolic_breakpoints`. `(e^x + e^-x)/2` returns infinity over
/// the whole band where exp has overflowed but cosh has not, [709.7827,
/// 710.4758], so |x| past the breakpoint takes the squared form instead.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cosh_f64(x: float64x2_t) -> float64x2_t {
    let half = vdupq_n_f64(0.5);
    let a = vabsq_f64(x);

    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(vnegq_f64(x));
    let near = vmulq_f64(half, vaddq_f64(exp_x, exp_neg_x));

    // (0.5*t)*t with t = exp(|x|/2) is 0.5*exp(|x|) with no intermediate past
    // f64::MAX. Halving t first, not the product, is what keeps it finite.
    let t = exp_f64(vmulq_f64(half, a));
    let far = vmulq_f64(vmulq_f64(half, t), t);

    // NaN compares false and takes the near branch, where exp propagates it.
    let big = vcgtq_f64(a, vdupq_n_f64(hyperbolic_breakpoints::BIG_F64));
    vbslq_f64(big, far, near)
}

/// Fast SIMD asinh for f32 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `log(x + sqrt(x²+1))` cancels for
/// every negative x, and `x²` overflows f32 past 1.8e19, so the sign is taken
/// out first and large |x| collapses to `log(|x|) + ln2`.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asinh_f32(x: float32x4_t) -> float32x4_t {
    use inv_hyperbolic_breakpoints::{BIG_F32, NEAR_F32};

    let one = vdupq_n_f32(1.0);
    let a = vabsq_f32(x);
    let t = vmulq_f32(a, a);
    let root = vsqrtq_f32(vaddq_f32(t, one));

    // a <= 2: a + a²/(1 + sqrt(1+a²)) is sqrt(1+a²) - 1 + a without the
    // subtraction, and log1p keeps its low bits down to the subnormal range.
    let near = log1p_f32(vaddq_f32(a, vdivq_f32(t, vaddq_f32(one, root))));
    // 2 < a <= 2^12: the same identity with the reciprocal written out.
    let recip = vdivq_f32(one, vaddq_f32(root, a));
    let mid = log_f32(vfmaq_f32(recip, vdupq_n_f32(2.0), a));
    // a > 2^12: sqrt(a²+1) equals a in single precision, so asinh is log(2a).
    let far = vaddq_f32(log_f32(a), vdupq_n_f32(std::f32::consts::LN_2));

    let r = vbslq_f32(vcgtq_f32(a, vdupq_n_f32(NEAR_F32)), mid, near);
    let r = vbslq_f32(vcgtq_f32(a, vdupq_n_f32(BIG_F32)), far, r);

    copy_sign_f32(r, x)
}

/// Fast SIMD asinh for f64 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `log(x + sqrt(x²+1))` cancels for
/// every negative x — at x = -49.6 the two addends agree to twelve digits —
/// so the sign is taken out first and the work is done on |x|.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asinh_f64(x: float64x2_t) -> float64x2_t {
    use inv_hyperbolic_breakpoints::{BIG_F64, NEAR_F64};

    let one = vdupq_n_f64(1.0);
    let a = vabsq_f64(x);
    let t = vmulq_f64(a, a);
    let root = vsqrtq_f64(vaddq_f64(t, one));

    // a <= 2: a + a²/(1 + sqrt(1+a²)) is sqrt(1+a²) - 1 + a without the
    // subtraction, and log1p keeps its low bits down to the subnormal range.
    let near = log1p_f64(vaddq_f64(a, vdivq_f64(t, vaddq_f64(one, root))));
    // 2 < a <= 2^28: the same identity with the reciprocal written out.
    let recip = vdivq_f64(one, vaddq_f64(root, a));
    let mid = log_f64(vfmaq_f64(recip, vdupq_n_f64(2.0), a));
    // a > 2^28: sqrt(a²+1) equals a in double, so asinh collapses to log(2a).
    let far = vaddq_f64(log_f64(a), vdupq_n_f64(std::f64::consts::LN_2));

    let r = vbslq_f64(vcgtq_f64(a, vdupq_n_f64(NEAR_F64)), mid, near);
    let r = vbslq_f64(vcgtq_f64(a, vdupq_n_f64(BIG_F64)), far, r);

    copy_sign_f64(r, x)
}

/// Fast SIMD acosh for f32 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. Forming `x² - 1` near x = 1 throws
/// away half the significant bits of `x - 1`, and past 1.8e19 it overflows f32
/// outright, which turned every input above that into `log(f32::INFINITY)`.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acosh_f32(x: float32x4_t) -> float32x4_t {
    use inv_hyperbolic_breakpoints::{BIG_F32, NEAR_F32};

    let one = vdupq_n_f32(1.0);
    let two = vdupq_n_f32(2.0);
    let t = vsubq_f32(x, one);

    // 1 <= x < 2: acosh(1+t) = log1p(t + sqrt(2t + t²)), which never forms a
    // difference of two nearly equal quantities.
    let disc = vsqrtq_f32(vfmaq_f32(vaddq_f32(t, t), t, t));
    let near = log1p_f32(vaddq_f32(t, disc));
    // 2 <= x <= 2^12.
    let root = vsqrtq_f32(vfmaq_f32(vdupq_n_f32(-1.0), x, x));
    let mid = log_f32(vsubq_f32(
        vmulq_f32(two, x),
        vdivq_f32(one, vaddq_f32(x, root)),
    ));
    // x > 2^12: sqrt(x²-1) equals x in single precision, so acosh is log(2x).
    let far = vaddq_f32(log_f32(x), vdupq_n_f32(std::f32::consts::LN_2));

    let r = vbslq_f32(vcgeq_f32(x, vdupq_n_f32(NEAR_F32)), mid, near);
    let r = vbslq_f32(vcgtq_f32(x, vdupq_n_f32(BIG_F32)), far, r);

    // acosh is undefined below 1. NaN fails the ordered compare and keeps the
    // NaN the log1p branch already produced.
    vbslq_f32(vcltq_f32(x, one), vdupq_n_f32(f32::NAN), r)
}

/// Fast SIMD acosh for f64 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. Forming `x² - 1` near x = 1 throws
/// away half the significant bits of `x - 1`, which is the whole result there.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acosh_f64(x: float64x2_t) -> float64x2_t {
    use inv_hyperbolic_breakpoints::{BIG_F64, NEAR_F64};

    let one = vdupq_n_f64(1.0);
    let two = vdupq_n_f64(2.0);
    let t = vsubq_f64(x, one);

    // 1 <= x < 2: acosh(1+t) = log1p(t + sqrt(2t + t²)), which never forms a
    // difference of two nearly equal quantities.
    let disc = vsqrtq_f64(vfmaq_f64(vaddq_f64(t, t), t, t));
    let near = log1p_f64(vaddq_f64(t, disc));
    // 2 <= x <= 2^28.
    let root = vsqrtq_f64(vfmaq_f64(vdupq_n_f64(-1.0), x, x));
    let mid = log_f64(vsubq_f64(
        vmulq_f64(two, x),
        vdivq_f64(one, vaddq_f64(x, root)),
    ));
    // x > 2^28: sqrt(x²-1) equals x in double, so acosh collapses to log(2x).
    let far = vaddq_f64(log_f64(x), vdupq_n_f64(std::f64::consts::LN_2));

    let r = vbslq_f64(vcgeq_f64(x, vdupq_n_f64(NEAR_F64)), mid, near);
    let r = vbslq_f64(vcgtq_f64(x, vdupq_n_f64(BIG_F64)), far, r);

    // acosh is undefined below 1. NaN fails the ordered compare and keeps the
    // NaN the log1p branch already produced.
    vbslq_f64(vcltq_f64(x, one), vdupq_n_f64(f64::NAN), r)
}

/// Fast SIMD atanh for f32 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `0.5*log((1+x)/(1-x))` rounds
/// `1 + x` before the log, which at small |x| discards every bit the result is
/// made of, and it has no domain handling at all: at x = 1 the quotient is
/// infinity, whose logarithm the old kernel reported as 88.72.
///
/// # Safety
/// Requires NEON (always available on AArch64)
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atanh_f32(x: float32x4_t) -> float32x4_t {
    use inv_hyperbolic_breakpoints::ATANH_SPLIT_F32;

    let one = vdupq_n_f32(1.0);
    let a = vabsq_f32(x);
    let t = vaddq_f32(a, a);
    let den = vsubq_f32(one, a);

    // a < 0.5: t + t*a/(1-a) is 2a/(1-a) written so the leading term stays
    // exact, which is what carries atanh(x) == x through the subnormal range.
    let small = log1p_f32(vaddq_f32(t, vdivq_f32(vmulq_f32(t, a), den)));
    // 0.5 <= a: at a = 1 the quotient is +inf and log1p returns +inf; past 1 it
    // is at most -2, so log1p of it is NaN.
    let large = log1p_f32(vdivq_f32(t, den));

    let picked = vbslq_f32(vcltq_f32(a, vdupq_n_f32(ATANH_SPLIT_F32)), small, large);
    let r = vmulq_f32(vdupq_n_f32(0.5), picked);

    copy_sign_f32(r, x)
}

/// Fast SIMD atanh for f64 using NEON
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `0.5*log((1+x)/(1-x))` rounds
/// `1 + x` before the log, which at x = 7e-4 discards every bit the result is
/// made of; log1p keeps them.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn atanh_f64(x: float64x2_t) -> float64x2_t {
    use inv_hyperbolic_breakpoints::ATANH_SPLIT_F64;

    let one = vdupq_n_f64(1.0);
    let a = vabsq_f64(x);
    let t = vaddq_f64(a, a);
    let den = vsubq_f64(one, a);

    // a < 0.5: t + t*a/(1-a) is 2a/(1-a) written so the leading term stays
    // exact, which is what carries atanh(x) == x through the subnormal range.
    let small = log1p_f64(vaddq_f64(t, vdivq_f64(vmulq_f64(t, a), den)));
    // 0.5 <= a: at a = 1 the quotient is +inf and log1p returns +inf; past 1 it
    // is at most -2, so log1p of it is NaN.
    let large = log1p_f64(vdivq_f64(t, den));

    let picked = vbslq_f64(vcltq_f64(a, vdupq_n_f64(ATANH_SPLIT_F64)), small, large);
    let r = vmulq_f64(vdupq_n_f64(0.5), picked);

    copy_sign_f64(r, x)
}

/// Shared polynomial correction `t * R(t)` for f32 asin/acos.
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC`. Valid for t in [0, 0.5]; the caller
/// multiplies by its own argument to form `y * t * R(t)`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
unsafe fn asin_r_f32(t: float32x4_t) -> float32x4_t {
    use asin_coefficients::*;

    let mut p = vdupq_n_f32(PS0_F32);
    p = vfmaq_f32(vdupq_n_f32(PS1_F32), p, t);
    p = vfmaq_f32(vdupq_n_f32(PS2_F32), p, t);
    p = vfmaq_f32(vdupq_n_f32(PS3_F32), p, t);
    p = vfmaq_f32(vdupq_n_f32(PS4_F32), p, t);

    vmulq_f32(t, p)
}

/// Fast SIMD asin for f32 using NEON
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn asin_f32(x: float32x4_t) -> float32x4_t {
    use asin_coefficients::*;

    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(HALF_F32);
    let sign_mask = vdupq_n_u32(0x8000_0000);
    let sign = vandq_u32(vreinterpretq_u32_f32(x), sign_mask);
    let ax = vabsq_f32(x);

    // |x| > 1 leaves the reflection argument negative, so sqrt yields NaN.
    // NaN input fails the comparison and takes the same reflection path.
    let small = vcltq_f32(ax, half);
    let t_refl = vmulq_f32(vsubq_f32(one, ax), half);
    let t = vbslq_f32(small, vmulq_f32(ax, ax), t_refl);
    let s = vsqrtq_f32(t);
    let v = vbslq_f32(small, ax, s);

    // w is asin(|x|) on the direct branch and asin(sqrt(t)) on the reflection.
    let w = vfmaq_f32(v, v, asin_r_f32(t));

    // π/2 - 2*asin(sqrt(t)), with the low half of π/2 folded into the
    // subtracted term so the cancellation keeps the trailing bits.
    let two_w = vaddq_f32(w, w);
    let res_refl = vsubq_f32(
        vdupq_n_f32(PIO2_HI_F32),
        vsubq_f32(two_w, vdupq_n_f32(PIO2_LO_F32)),
    );

    let result = vbslq_f32(small, w, res_refl);
    vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(result), sign))
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
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details. Built from the
/// reflection directly, not as π/2 - asin(x): that subtraction cancels away the
/// whole result as x approaches 1.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn acos_f32(x: float32x4_t) -> float32x4_t {
    use asin_coefficients::*;

    let one = vdupq_n_f32(1.0);
    let half = vdupq_n_f32(HALF_F32);
    let pio2_lo = vdupq_n_f32(PIO2_LO_F32);
    let ax = vabsq_f32(x);

    let small = vcltq_f32(ax, half);
    let t_refl = vmulq_f32(vsubq_f32(one, ax), half);
    let t = vbslq_f32(small, vmulq_f32(x, x), t_refl);
    let q = asin_r_f32(t);
    let s = vsqrtq_f32(t);

    // |x| <= 0.5: π/2 - asin(x), evaluated without forming asin(x) first.
    let px = vmulq_f32(x, q);
    let res_small = vsubq_f32(
        vdupq_n_f32(PIO2_HI_F32),
        vaddq_f32(x, vsubq_f32(px, pio2_lo)),
    );

    // x >= 0.5: 2*asin(sqrt(t)), which is small and free of cancellation.
    let ps = vmulq_f32(s, q);
    let s_sr = vaddq_f32(s, ps);
    let res_pos = vaddq_f32(s_sr, s_sr);

    // x <= -0.5: π - 2*asin(sqrt(t)).
    let w = vaddq_f32(s, vsubq_f32(ps, pio2_lo));
    let res_neg = vsubq_f32(vdupq_n_f32(PI_HI_F32), vaddq_f32(w, w));

    // NaN fails both comparisons and lands on the negative branch, where the
    // NaN sqrt argument propagates.
    let positive = vcgtq_f32(x, vdupq_n_f32(0.0));
    let res_refl = vbslq_f32(positive, res_pos, res_neg);
    vbslq_f32(small, res_small, res_refl)
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
///
/// See `common::cbrt_constants`. The seed divides the whole bit pattern of
/// |x| by three, mantissa included, which lands within 3.3%; two Halley steps
/// then cube that error twice over. Seeding from the exponent alone is off by
/// up to 37%, and the same two steps only bring that to 5e-2.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[inline]
pub unsafe fn cbrt_f32(x: float32x4_t) -> float32x4_t {
    use cbrt_constants::*;

    let sign = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x8000_0000));
    let abs_x = vabsq_f32(x);
    let one = vdupq_n_f32(1.0);

    // Move the two ends of the range into the middle: the iteration forms
    // x + 2*t³, which overflows near f32::MAX, and the seed assumes an
    // implicit leading 1, which a subnormal does not have. Both shifts are
    // 2^96, a power of two whose exponent is a multiple of three, so undoing
    // them on the result is exact.
    let big = vcgeq_f32(abs_x, vdupq_n_f32(BIG_F32));
    let small = vcltq_f32(abs_x, vdupq_n_f32(SMALL_F32));
    let s_in = vbslq_f32(small, vdupq_n_f32(SCALE_UP_F32), one);
    let s_in = vbslq_f32(big, vdupq_n_f32(SCALE_DOWN_F32), s_in);
    let s_out = vbslq_f32(small, vdupq_n_f32(UNSCALE_DOWN_F32), one);
    let s_out = vbslq_f32(big, vdupq_n_f32(UNSCALE_UP_F32), s_out);
    let a = vmulq_f32(abs_x, s_in);

    // Seed: t = bits(a)/3 + B1, read back as a float. The integer-to-float
    // conversion drops the low bits of the bit pattern, which is worth 1e-5
    // of relative error against a seed that is already only good to 3.3%.
    let a_bits = vcvtq_f32_s32(vreinterpretq_s32_f32(a));
    let q = vfmaq_f32(vdupq_n_f32(B1_F32), a_bits, vdupq_n_f32(1.0 / 3.0));
    let t = vreinterpretq_f32_s32(vcvtnq_s32_f32(q));

    // Halley: t*(2a + t³)/(a + 2t³), whose error cubes each step.
    let r = vmulq_f32(vmulq_f32(t, t), t);
    let num = vaddq_f32(vaddq_f32(a, a), r);
    let den = vaddq_f32(vaddq_f32(a, r), r);
    let t = vmulq_f32(t, vdivq_f32(num, den));

    let r = vmulq_f32(vmulq_f32(t, t), t);
    let num = vaddq_f32(vaddq_f32(a, a), r);
    let den = vaddq_f32(vaddq_f32(a, r), r);
    let t = vmulq_f32(t, vdivq_f32(num, den));

    let scaled = vreinterpretq_u32_f32(vmulq_f32(t, s_out));
    let out = vreinterpretq_f32_u32(vorrq_u32(scaled, sign));

    // ±0, ±inf and NaN are their own cube roots, and none of them survives the
    // iteration: zero divides zero and infinity subtracts infinity.
    let ordinary = vandq_u32(
        vcgtq_f32(abs_x, vdupq_n_f32(0.0)),
        vcltq_f32(abs_x, vdupq_n_f32(f32::INFINITY)),
    );
    vbslq_f32(ordinary, out, x)
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
