//! AVX2 mathematical function implementations
//!
//! Provides vectorized transcendental functions using 256-bit registers.
//! All algorithms and coefficients are documented in `common.rs`.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Relative Error |
//! |----------|-----|-----|----------------|
//! | exp      | ✓   | ✓   | 1 ulp / 1e-12 |
//! | tanh     | ✓   | ✓   | 2 ulp / 2 ulp |
//! | log      | ✓   | ✓   | 2 ulp / 2 ulp |
//! | sin      | ✓   | ✓   | 2 ulp / 4 ulp |
//! | cos      | ✓   | ✓   | 2 ulp / 4 ulp |
//! | tan      | ✓   | ✓   | 2 ulp / 4 ulp |
//! | atan     | ✓   | ✓   | 2 ulp / 2 ulp |
//! | asin     | ✓   | ✓   | 2 ulp / 2 ulp |
//! | acos     | ✓   | ✓   | 2 ulp / 2 ulp |
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
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::common::{
    asin_coefficients, atan_coefficients, cbrt_constants, exp_coefficients, exp2_coefficients,
    hyperbolic_breakpoints, inv_hyperbolic_breakpoints, log_coefficients, trig_coefficients,
};

// ============================================================================
// Exponential function: exp(x)
// ============================================================================

/// Fast SIMD exp approximation for f32 using AVX2+FMA
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp_f32(x: __m256) -> __m256 {
    use exp_coefficients::*;

    // The bound is the FIRST operand: maxps/minps return their second operand
    // when either is NaN, so this order lets a NaN input through instead of
    // replacing it with the clamp bound.
    let xc = _mm256_max_ps(_mm256_set1_ps(MIN_F32), x);
    let xc = _mm256_min_ps(_mm256_set1_ps(MAX_F32), xc);

    let y = _mm256_mul_ps(xc, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F32 is exact
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI_F32), xc);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO_F32), r);

    // Horner: one rounding per term, and no r^k powers to lose bits in
    let mut poly = _mm256_set1_ps(C7_F32);
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C6_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C5_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C4_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C3_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C2_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C1_F32));
    poly = _mm256_fmadd_ps(poly, r, _mm256_set1_ps(C0_F32));

    // 2^n as two halved powers of two. 2^128 on its own is infinity, yet the
    // largest finite result needs it; splitting keeps both factors normal, so
    // an overflow reaches infinity in the second multiply and a subnormal
    // result takes exactly one rounding.
    let n_i32 = _mm256_cvtps_epi32(n);
    let n_hi = _mm256_srai_epi32::<1>(n_i32);
    let n_lo = _mm256_sub_epi32(n_i32, n_hi);
    let bias = _mm256_set1_epi32(127);
    let p_hi = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_hi, bias)));
    let p_lo = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_lo, bias)));

    _mm256_mul_ps(_mm256_mul_ps(poly, p_hi), p_lo)
}

/// Fast SIMD exp approximation for f64 using AVX2+FMA
///
/// See `common::_EXP_ALGORITHM_DOC` for algorithm details.
///
/// # Note
/// AVX2 lacks native 64-bit integer <-> double conversion. This implementation
/// uses scalar extraction for the 2^n computation, which is the standard
/// workaround. The polynomial computation remains fully vectorized.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp_f64(x: __m256d) -> __m256d {
    use exp_coefficients::*;

    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    let ln2_hi = _mm256_set1_pd(LN2_HI_F64);
    let ln2_lo = _mm256_set1_pd(LN2_LO_F64);

    let c0 = _mm256_set1_pd(C0_F64);
    let c1 = _mm256_set1_pd(C1_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c3 = _mm256_set1_pd(C3_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c5 = _mm256_set1_pd(C5_F64);
    let c6 = _mm256_set1_pd(C6_F64);
    let c7 = _mm256_set1_pd(C7_F64);
    let c8 = _mm256_set1_pd(C8_F64);
    let c9 = _mm256_set1_pd(C9_F64);
    let c10 = _mm256_set1_pd(C10_F64);
    let c11 = _mm256_set1_pd(C11_F64);
    let c12 = _mm256_set1_pd(C12_F64);
    let c13 = _mm256_set1_pd(C13_F64);

    // The bound is the FIRST operand: maxpd/minpd return their second operand
    // when either is NaN, so this order lets a NaN input through instead of
    // replacing it with the clamp bound.
    let xc = _mm256_max_pd(_mm256_set1_pd(MIN_F64), x);
    let xc = _mm256_min_pd(_mm256_set1_pd(MAX_F64), xc);

    let y = _mm256_mul_pd(xc, log2e);
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F64 is exact
    let r = _mm256_fnmadd_pd(n, ln2_hi, xc);
    let r = _mm256_fnmadd_pd(n, ln2_lo, r);

    // Horner: one rounding per term, and no r^k powers to lose bits in
    let mut poly = c13;
    poly = _mm256_fmadd_pd(poly, r, c12);
    poly = _mm256_fmadd_pd(poly, r, c11);
    poly = _mm256_fmadd_pd(poly, r, c10);
    poly = _mm256_fmadd_pd(poly, r, c9);
    poly = _mm256_fmadd_pd(poly, r, c8);
    poly = _mm256_fmadd_pd(poly, r, c7);
    poly = _mm256_fmadd_pd(poly, r, c6);
    poly = _mm256_fmadd_pd(poly, r, c5);
    poly = _mm256_fmadd_pd(poly, r, c4);
    poly = _mm256_fmadd_pd(poly, r, c3);
    poly = _mm256_fmadd_pd(poly, r, c2);
    poly = _mm256_fmadd_pd(poly, r, c1);
    poly = _mm256_fmadd_pd(poly, r, c0);

    // AVX2 lacks _mm256_cvtpd_epi64, so the scale drops to scalar - polynomial
    // eval is still SIMD. The power of two is split in half: 2^1024 alone is
    // already infinity, yet the largest finite result needs it, and 2^-1076 is
    // not representable at all. Both halves stay normal, so an overflow reaches
    // infinity in the second multiply and a subnormal result takes exactly one
    // rounding because the first multiply is exact.
    let mut result = [0.0f64; 4];
    let mut n_arr = [0.0f64; 4];
    let mut poly_arr = [0.0f64; 4];

    _mm256_storeu_pd(n_arr.as_mut_ptr(), n);
    _mm256_storeu_pd(poly_arr.as_mut_ptr(), poly);

    for i in 0..4 {
        let n_i = n_arr[i] as i64;
        let hi = n_i / 2;
        let lo = n_i - hi;
        let p_hi = f64::from_bits(((hi + 1023) as u64) << 52);
        let p_lo = f64::from_bits(((lo + 1023) as u64) << 52);
        result[i] = (poly_arr[i] * p_hi) * p_lo;
    }

    _mm256_loadu_pd(result.as_ptr())
}

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh for f32 using AVX2+FMA
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^2x - 1)/(e^2x + 1)` cancels the
/// whole numerator away as x approaches zero; `u/(u+2)` with `u = expm1(2|x|)`
/// never forms that difference.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f32(x: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let a = _mm256_andnot_ps(sign_mask, x);
    let u = expm1_f32(_mm256_add_ps(a, a));

    let d = _mm256_div_ps(u, _mm256_add_ps(u, _mm256_set1_ps(2.0)));
    // u saturates to infinity past |x| = 44.5; the limit of u/(u+2) there is 1,
    // whereas the quotient itself would be inf/inf.
    let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(u, _mm256_set1_ps(f32::INFINITY));
    let d = _mm256_blendv_ps(d, _mm256_set1_ps(1.0), is_inf);

    // The sign rides the sign bit, so tanh(-0) is -0 and tanh(-inf) is -1.
    _mm256_or_ps(d, _mm256_and_ps(x, sign_mask))
}

/// Fast SIMD tanh for f64 using AVX2+FMA
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^2x - 1)/(e^2x + 1)` cancels the
/// whole numerator away as x approaches zero; `u/(u+2)` with `u = expm1(2|x|)`
/// never forms that difference.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f64(x: __m256d) -> __m256d {
    let sign_mask = _mm256_set1_pd(-0.0);
    let a = _mm256_andnot_pd(sign_mask, x);
    let u = expm1_f64(_mm256_add_pd(a, a));

    let d = _mm256_div_pd(u, _mm256_add_pd(u, _mm256_set1_pd(2.0)));
    // u saturates to infinity past |x| = 355; the limit of u/(u+2) there is 1,
    // whereas the quotient itself would be inf/inf.
    let is_inf = _mm256_cmp_pd::<_CMP_EQ_OQ>(u, _mm256_set1_pd(f64::INFINITY));
    let d = _mm256_blendv_pd(d, _mm256_set1_pd(1.0), is_inf);

    // The sign rides the sign bit, so tanh(-0) is -0 and tanh(-inf) is -1.
    _mm256_or_pd(d, _mm256_and_pd(x, sign_mask))
}

// ============================================================================
// Natural logarithm: log(x)
// ============================================================================

/// Split `x` into an exponent `n` and `log(m)`, where `m` is the mantissa
/// normalized to [sqrt(2)/2, sqrt(2)), so that `log(x) = n*ln(2) + log(m)`.
///
/// The f32 counterpart of `log_reduce_f64`; log, log2, log10 and log1p all
/// share it and differ only in how they recombine the two parts. AVX2 has full
/// 32-bit integer support, so unlike the f64 path this stays vectorized
/// throughout.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn log_reduce_f32(x: __m256) -> (__m256, __m256) {
    use log_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let half = _mm256_set1_ps(0.5);
    let sqrt2 = _mm256_set1_ps(std::f32::consts::SQRT_2);

    // Subnormals carry no implicit leading 1, so the exponent/mantissa split
    // below is only valid after scaling them into the normal range.
    let is_sub = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(f32::MIN_POSITIVE));
    let scaled = _mm256_mul_ps(x, _mm256_set1_ps(SUBNORMAL_SCALE_F32));
    let x_norm = _mm256_blendv_ps(x, scaled, is_sub);
    let n_shift = _mm256_blendv_ps(
        _mm256_setzero_ps(),
        _mm256_set1_ps(SUBNORMAL_SHIFT_F32),
        is_sub,
    );

    let x_bits = _mm256_castps_si256(x_norm);
    let exp_raw = _mm256_srli_epi32::<23>(x_bits);
    let exp_unbiased = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(EXP_BIAS_F32));
    let mut n = _mm256_cvtepi32_ps(exp_unbiased);

    let mantissa_mask = _mm256_set1_epi32(MANTISSA_MASK_F32);
    let exp_zero = _mm256_set1_epi32(EXP_ZERO_F32);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let mut m = _mm256_castsi256_ps(m_bits);

    // Normalize: if m > sqrt(2), halve it and carry a 1 into the exponent, so
    // that f stays in [-0.2929, 0.4142].
    let need_adjust = _mm256_cmp_ps::<_CMP_GT_OQ>(m, sqrt2);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), need_adjust);
    n = _mm256_blendv_ps(n, _mm256_add_ps(n, one), need_adjust);
    let n = _mm256_add_ps(n, n_shift);

    // s = f/(2+f) halves the argument and leaves only odd powers, which is what
    // lets four terms reach f32 precision (see `log_coefficients`).
    let f = _mm256_sub_ps(m, one);
    let s = _mm256_div_ps(f, _mm256_add_ps(two, f));
    let z = _mm256_mul_ps(s, s);

    let r = _mm256_mul_ps(
        z,
        _mm256_fmadd_ps(
            z,
            _mm256_fmadd_ps(
                z,
                _mm256_fmadd_ps(z, _mm256_set1_ps(LG4_F32), _mm256_set1_ps(LG3_F32)),
                _mm256_set1_ps(LG2_F32),
            ),
            _mm256_set1_ps(LG1_F32),
        ),
    );

    // log(m) = f - (hfsq - s*(hfsq + R)); keeping f outside the parentheses
    // stops the f² term from eating f's low bits when f is small.
    let hfsq = _mm256_mul_ps(half, _mm256_mul_ps(f, f));
    let logm = _mm256_sub_ps(
        f,
        _mm256_sub_ps(hfsq, _mm256_mul_ps(s, _mm256_add_ps(hfsq, r))),
    );

    (n, logm)
}

/// Apply the IEEE domain values shared by the f32 log, log2 and log10:
/// `log(0) = -inf`, `log(x < 0) = NaN`, `log(+inf) = +inf`, `log(NaN) = NaN`.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn log_special_f32(x: __m256, r: __m256) -> __m256 {
    let zero = _mm256_setzero_ps();

    // Unordered compare, so NaN inputs take the NaN branch instead of feeding
    // garbage mantissa bits through the polynomial.
    let not_positive = _mm256_cmp_ps::<_CMP_NGT_UQ>(x, zero);
    let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, zero);
    let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_set1_ps(f32::INFINITY));

    let out = _mm256_blendv_ps(r, _mm256_set1_ps(f32::NAN), not_positive);
    let out = _mm256_blendv_ps(out, _mm256_set1_ps(f32::NEG_INFINITY), is_zero);
    _mm256_blendv_ps(out, _mm256_set1_ps(f32::INFINITY), is_inf)
}

/// Fast SIMD log approximation for f32 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps over the whole positive range, subnormals
/// included.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f32(x: __m256) -> __m256 {
    use log_coefficients::{LN2_HI_F32, LN2_LO_F32};

    let (n, logm) = log_reduce_f32(x);

    // Split ln(2): the head is exact against every reachable n, the tail
    // restores the bits a single rounded ln(2) would drop.
    let lo = _mm256_fmadd_ps(n, _mm256_set1_ps(LN2_LO_F32), logm);
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(LN2_HI_F32), lo);

    log_special_f32(x, r)
}

/// Split `x` into an exponent `n` and `log(m)`, where `m` is the mantissa
/// normalized to [sqrt(2)/2, sqrt(2)), so that `log(x) = n*ln(2) + log(m)`.
///
/// log, log2, log10 and log1p all share this reduction and differ only in how
/// they recombine the two parts. Special values are applied by the callers via
/// `log_special_f64`.
///
/// # Implementation Note
/// AVX2 lacks 64-bit integer comparison and conversion, so the normalization
/// step drops to scalar. The polynomial stays fully vectorized.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn log_reduce_f64(x: __m256d) -> (__m256d, __m256d) {
    use log_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let two = _mm256_set1_pd(2.0);
    let half = _mm256_set1_pd(0.5);
    let sqrt2_val = std::f64::consts::SQRT_2;

    // Subnormals carry no implicit leading 1, so the split below is only valid
    // after scaling them into the normal range.
    let is_sub = _mm256_cmp_pd::<_CMP_LT_OQ>(x, _mm256_set1_pd(f64::MIN_POSITIVE));
    let scaled = _mm256_mul_pd(x, _mm256_set1_pd(SUBNORMAL_SCALE_F64));
    let x_norm = _mm256_blendv_pd(x, scaled, is_sub);
    let n_shift = _mm256_blendv_pd(
        _mm256_setzero_pd(),
        _mm256_set1_pd(SUBNORMAL_SHIFT_F64),
        is_sub,
    );

    let x_bits = _mm256_castpd_si256(x_norm);

    // Extract exponent using 64-bit SIMD shift
    let exp_raw = _mm256_srli_epi64::<52>(x_bits);

    // Extract mantissa and set exponent to bias (so mantissa is in [1, 2))
    let mantissa_mask = _mm256_set1_epi64x(MANTISSA_MASK_F64 as i64);
    let exp_zero = _mm256_set1_epi64x(EXP_ZERO_F64 as i64);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let m_initial = _mm256_castsi256_pd(m_bits);

    let mut m_arr = [0.0f64; 4];
    let mut exp_arr = [0i64; 4];
    _mm256_storeu_pd(m_arr.as_mut_ptr(), m_initial);
    _mm256_storeu_si256(exp_arr.as_mut_ptr() as *mut __m256i, exp_raw);

    let mut n_arr = [0.0f64; 4];
    for i in 0..4 {
        let mut exp_unbiased = exp_arr[i] - EXP_BIAS_F64;
        let mut m = m_arr[i];

        // Normalize: if m > sqrt(2), divide by 2 and increment exponent
        if m > sqrt2_val {
            m *= 0.5;
            exp_unbiased += 1;
        }

        n_arr[i] = exp_unbiased as f64;
        m_arr[i] = m;
    }

    let n = _mm256_add_pd(_mm256_loadu_pd(n_arr.as_ptr()), n_shift);
    let m = _mm256_loadu_pd(m_arr.as_ptr());

    // s = f/(2+f) halves the argument and leaves only odd powers, which is what
    // lets seven terms reach f64 precision (see `log_coefficients`).
    let f = _mm256_sub_pd(m, one);
    let s = _mm256_div_pd(f, _mm256_add_pd(two, f));
    let z = _mm256_mul_pd(s, s);
    let w = _mm256_mul_pd(z, z);

    let t1 = _mm256_mul_pd(
        w,
        _mm256_fmadd_pd(
            w,
            _mm256_fmadd_pd(w, _mm256_set1_pd(LG6_F64), _mm256_set1_pd(LG4_F64)),
            _mm256_set1_pd(LG2_F64),
        ),
    );
    let t2 = _mm256_mul_pd(
        z,
        _mm256_fmadd_pd(
            w,
            _mm256_fmadd_pd(
                w,
                _mm256_fmadd_pd(w, _mm256_set1_pd(LG7_F64), _mm256_set1_pd(LG5_F64)),
                _mm256_set1_pd(LG3_F64),
            ),
            _mm256_set1_pd(LG1_F64),
        ),
    );
    let r = _mm256_add_pd(t2, t1);

    // log(m) = f - (hfsq - s*(hfsq + R)); keeping f outside the parentheses
    // stops the f² term from eating f's low bits when f is small.
    let hfsq = _mm256_mul_pd(half, _mm256_mul_pd(f, f));
    let logm = _mm256_sub_pd(
        f,
        _mm256_sub_pd(hfsq, _mm256_mul_pd(s, _mm256_add_pd(hfsq, r))),
    );

    (n, logm)
}

/// Apply the IEEE domain values shared by log, log2 and log10:
/// `log(0) = -inf`, `log(x < 0) = NaN`, `log(+inf) = +inf`, `log(NaN) = NaN`.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn log_special_f64(x: __m256d, r: __m256d) -> __m256d {
    let zero = _mm256_setzero_pd();

    // Unordered compare, so NaN inputs take the NaN branch instead of feeding
    // garbage mantissa bits through the polynomial.
    let not_positive = _mm256_cmp_pd::<_CMP_NGT_UQ>(x, zero);
    let is_zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, zero);
    let is_inf = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_set1_pd(f64::INFINITY));

    let out = _mm256_blendv_pd(r, _mm256_set1_pd(f64::NAN), not_positive);
    let out = _mm256_blendv_pd(out, _mm256_set1_pd(f64::NEG_INFINITY), is_zero);
    _mm256_blendv_pd(out, _mm256_set1_pd(f64::INFINITY), is_inf)
}

/// Fast SIMD log approximation for f64 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps over the whole positive range.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f64(x: __m256d) -> __m256d {
    use log_coefficients::{LN2_HI_F64, LN2_LO_F64};

    let (n, logm) = log_reduce_f64(x);

    // Split ln(2): the head is exact against every reachable n, the tail
    // restores the bits a single rounded ln(2) would drop.
    let lo = _mm256_fmadd_pd(n, _mm256_set1_pd(LN2_LO_F64), logm);
    let r = _mm256_fmadd_pd(n, _mm256_set1_pd(LN2_HI_F64), lo);

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
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn trig_reduce_f32(x: __m256) -> (__m256i, __m256) {
    use trig_coefficients::{PIO2_1_F32, PIO2_2_F32, PIO2_3_F32};

    let j = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        _mm256_set1_ps(std::f32::consts::FRAC_2_PI),
    ));

    let y = _mm256_fnmadd_ps(j, _mm256_set1_ps(PIO2_1_F32), x);
    let y = _mm256_fnmadd_ps(j, _mm256_set1_ps(PIO2_2_F32), y);
    let y = _mm256_fnmadd_ps(j, _mm256_set1_ps(PIO2_3_F32), y);

    (_mm256_cvtps_epi32(j), y)
}

/// Minimax sin kernel on the reduced argument, |y| <= π/4.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sin_kernel_f32(y: __m256) -> __m256 {
    use trig_coefficients::{SIN0_F32, SIN1_F32, SIN2_F32};

    let z = _mm256_mul_ps(y, y);

    let mut p = _mm256_set1_ps(SIN0_F32);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(SIN1_F32));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(SIN2_F32));

    // y is added last, so a tiny y comes back unchanged and keeps its sign.
    _mm256_fmadd_ps(_mm256_mul_ps(z, y), p, y)
}

/// Minimax cos kernel on the reduced argument, |y| <= π/4.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn cos_kernel_f32(y: __m256) -> __m256 {
    use trig_coefficients::{COS0_F32, COS1_F32, COS2_F32};

    let one = _mm256_set1_ps(1.0);
    let z = _mm256_mul_ps(y, y);

    let mut p = _mm256_set1_ps(COS0_F32);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(COS1_F32));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(COS2_F32));
    let r = _mm256_mul_ps(_mm256_mul_ps(z, z), p);

    // `1 - z/2` rounds; `(1 - w) - hz` is exact and returns the rounded bits.
    let hz = _mm256_mul_ps(_mm256_set1_ps(0.5), z);
    let w = _mm256_sub_ps(one, hz);
    let correction = _mm256_sub_ps(_mm256_sub_ps(one, w), hz);
    _mm256_add_ps(w, _mm256_add_ps(correction, r))
}

/// Evaluate sin on quadrant `j + offset`, the shared core of f32 sin and cos.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sin_quadrant_f32(x: __m256, offset: i32) -> __m256 {
    let (j, y) = trig_reduce_f32(x);
    let sin_y = sin_kernel_f32(y);
    let cos_y = cos_kernel_f32(y);

    // j mod 4 = 0: sin(y), 1: cos(y), 2: -sin(y), 3: -cos(y)
    let j_mod_4 = _mm256_and_si256(
        _mm256_add_epi32(j, _mm256_set1_epi32(offset)),
        _mm256_set1_epi32(3),
    );

    let use_cos_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    ));
    let negate_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(2)),
        _mm256_set1_epi32(2),
    ));

    let result = _mm256_blendv_ps(sin_y, cos_y, use_cos_mask);
    let negated = _mm256_xor_ps(result, _mm256_set1_ps(-0.0));
    _mm256_blendv_ps(result, negated, negate_mask)
}

/// Fast SIMD sin approximation for f32 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps for |x| <= 2^17.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f32(x: __m256) -> __m256 {
    let r = sin_quadrant_f32(x, 0);

    // sin(±0) = ±0. The reduction computes 0 - (-0 * π/2), which is +0 for both
    // signed zeros, so the input is restored here.
    let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(r, x, is_zero)
}

/// Cody-Waite reduction of `x` modulo π/2 for f64.
///
/// Returns the quadrant index `j` (as a double) and the reduced argument in
/// [-π/4, π/4]. See `common::_TRIG_ALGORITHM_DOC`; valid for
/// |x| <= 2^21 * π/2, past which `j * PIO2_k` stops being exact.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn trig_reduce_f64(x: __m256d) -> (__m256d, __m256d) {
    use trig_coefficients::{PIO2_1_F64, PIO2_2_F64, PIO2_3_F64, PIO2_3T_F64};

    let j = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        _mm256_set1_pd(std::f64::consts::FRAC_2_PI),
    ));

    let y = _mm256_fnmadd_pd(j, _mm256_set1_pd(PIO2_1_F64), x);
    let y = _mm256_fnmadd_pd(j, _mm256_set1_pd(PIO2_2_F64), y);
    let y = _mm256_fnmadd_pd(j, _mm256_set1_pd(PIO2_3_F64), y);
    let y = _mm256_fnmadd_pd(j, _mm256_set1_pd(PIO2_3T_F64), y);

    (j, y)
}

/// Minimax sin kernel on the reduced argument, |y| <= π/4.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sin_kernel_f64(y: __m256d) -> __m256d {
    use trig_coefficients::{SIN1_F64, SIN2_F64, SIN3_F64, SIN4_F64, SIN5_F64, SIN6_F64};

    let z = _mm256_mul_pd(y, y);

    let mut p = _mm256_set1_pd(SIN6_F64);
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(SIN5_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(SIN4_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(SIN3_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(SIN2_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(SIN1_F64));

    // y is added last, so a tiny y comes back unchanged and keeps its sign.
    _mm256_fmadd_pd(_mm256_mul_pd(z, y), p, y)
}

/// Minimax cos kernel on the reduced argument, |y| <= π/4.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn cos_kernel_f64(y: __m256d) -> __m256d {
    use trig_coefficients::{COS1_F64, COS2_F64, COS3_F64, COS4_F64, COS5_F64, COS6_F64};

    let one = _mm256_set1_pd(1.0);
    let z = _mm256_mul_pd(y, y);

    let mut p = _mm256_set1_pd(COS6_F64);
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(COS5_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(COS4_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(COS3_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(COS2_F64));
    p = _mm256_fmadd_pd(p, z, _mm256_set1_pd(COS1_F64));
    let r = _mm256_mul_pd(_mm256_mul_pd(z, z), p);

    // `1 - z/2` rounds; `(1 - w) - hz` is exact and returns the rounded bits.
    let hz = _mm256_mul_pd(_mm256_set1_pd(0.5), z);
    let w = _mm256_sub_pd(one, hz);
    let correction = _mm256_sub_pd(_mm256_sub_pd(one, w), hz);
    _mm256_add_pd(w, _mm256_add_pd(correction, r))
}

/// Evaluate sin on quadrant `j + offset`, the shared core of sin and cos.
///
/// AVX2 has no 64-bit float-to-int conversion, so the quadrant table is applied
/// per lane.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn sin_quadrant_f64(x: __m256d, offset: i32) -> __m256d {
    let (j, y) = trig_reduce_f64(x);
    let sin_y = sin_kernel_f64(y);
    let cos_y = cos_kernel_f64(y);

    let mut j_arr = [0.0f64; 4];
    let mut sin_arr = [0.0f64; 4];
    let mut cos_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    _mm256_storeu_pd(sin_arr.as_mut_ptr(), sin_y);
    _mm256_storeu_pd(cos_arr.as_mut_ptr(), cos_y);

    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let quadrant = (j_arr[i] as i32).wrapping_add(offset) & 3;
        result[i] = match quadrant {
            0 => sin_arr[i],
            1 => cos_arr[i],
            2 => -sin_arr[i],
            _ => -cos_arr[i],
        };
    }

    _mm256_loadu_pd(result.as_ptr())
}

/// Fast SIMD sin approximation for f64 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 4 ulps for |x| <= 2^21 * π/2.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f64(x: __m256d) -> __m256d {
    let r = sin_quadrant_f64(x, 0);

    // sin(±0) = ±0. The reduction computes 0 - (-0 * π/2), which is +0 for both
    // signed zeros, so the input is restored here.
    let is_zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_setzero_pd());
    _mm256_blendv_pd(r, x, is_zero)
}

/// Fast SIMD cos approximation for f32 using AVX2+FMA
///
/// Shifts the quadrant index by one rather than evaluating `sin(x + π/2)`,
/// which would round the sum before reduction and lose bits proportional to
/// |x|. Relative error stays below 2 ulps for |x| <= 2^17.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f32(x: __m256) -> __m256 {
    sin_quadrant_f32(x, 1)
}

/// Fast SIMD cos approximation for f64 using AVX2+FMA
///
/// Shifts the quadrant index by one rather than evaluating `sin(x + π/2)`,
/// which would round the sum before reduction and lose bits proportional to
/// |x|. Relative error stays below 4 ulps for |x| <= 2^21 * π/2.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f64(x: __m256d) -> __m256d {
    sin_quadrant_f64(x, 1)
}

/// Fast SIMD tan approximation for f32 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 2 ulps away from the poles, for |x| <= 2^17.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f32(x: __m256) -> __m256 {
    let (j, y) = trig_reduce_f32(x);
    let sin_y = sin_kernel_f32(y);
    let cos_y = cos_kernel_f32(y);

    // Odd quadrant: tan(x) = -cot(y) = -cos(y)/sin(y). Swapping the ratio costs
    // one rounding, where inverting an already-rounded tan(y) costs two.
    let odd_mask = _mm256_castsi256_ps(_mm256_cmpeq_epi32(
        _mm256_and_si256(j, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    ));
    let neg_cos = _mm256_xor_ps(cos_y, _mm256_set1_ps(-0.0));
    let num = _mm256_blendv_ps(sin_y, neg_cos, odd_mask);
    let den = _mm256_blendv_ps(cos_y, sin_y, odd_mask);
    let r = _mm256_div_ps(num, den);

    // tan(±0) = ±0; the reduction turns -0 into +0.
    let is_zero = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_setzero_ps());
    _mm256_blendv_ps(r, x, is_zero)
}

/// Fast SIMD tan approximation for f64 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
/// Relative error stays below 4 ulps away from the poles, for
/// |x| <= 2^21 * π/2.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f64(x: __m256d) -> __m256d {
    let (j, y) = trig_reduce_f64(x);
    let sin_y = sin_kernel_f64(y);
    let cos_y = cos_kernel_f64(y);

    // Odd quadrant: tan(x) = -cot(y) = -cos(y)/sin(y). Swapping the ratio costs
    // one rounding, where inverting an already-rounded tan(y) costs two.
    // AVX2 lacks 64-bit int comparison, so the swap is applied per lane.
    let mut j_arr = [0.0f64; 4];
    let mut sin_arr = [0.0f64; 4];
    let mut cos_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    _mm256_storeu_pd(sin_arr.as_mut_ptr(), sin_y);
    _mm256_storeu_pd(cos_arr.as_mut_ptr(), cos_y);

    let mut num = [0.0f64; 4];
    let mut den = [0.0f64; 4];
    for i in 0..4 {
        if (j_arr[i] as i32) & 1 == 1 {
            num[i] = -cos_arr[i];
            den[i] = sin_arr[i];
        } else {
            num[i] = sin_arr[i];
            den[i] = cos_arr[i];
        }
    }

    let r = _mm256_div_pd(_mm256_loadu_pd(num.as_ptr()), _mm256_loadu_pd(den.as_ptr()));

    // tan(±0) = ±0; the reduction turns -0 into +0.
    let is_zero = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_setzero_pd());
    _mm256_blendv_pd(r, x, is_zero)
}

// ============================================================================
// Inverse tangent function: atan(x)
// ============================================================================

/// Fast SIMD atan approximation for f32 using AVX2+FMA
///
/// See `common::_ATAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atan_f32(x: __m256) -> __m256 {
    use atan_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_mask);
    let ax = _mm256_andnot_ps(sign_mask, x);

    // Pick the reduction centre and the matching atan(c) head/correction. The
    // breakpoint masks are nested, so blending from the widest bucket inward
    // leaves each lane holding its tightest match.
    //
    // The widest bucket has the centre at infinity: t = -1/|x|. NaN fails every
    // comparison and stays there, which propagates NaN through the division.
    let mut t = _mm256_div_ps(_mm256_set1_ps(-1.0), ax);
    let mut hi = _mm256_set1_ps(ATAN_HI1_F32);
    let mut lo = _mm256_set1_ps(ATAN_LO1_F32);

    let in1 = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, _mm256_set1_ps(BREAK1_F32));
    let t_mid = _mm256_div_ps(_mm256_sub_ps(ax, one), _mm256_add_ps(ax, one));
    t = _mm256_blendv_ps(t, t_mid, in1);
    hi = _mm256_blendv_ps(hi, _mm256_set1_ps(ATAN_HI0_F32), in1);
    lo = _mm256_blendv_ps(lo, _mm256_set1_ps(ATAN_LO0_F32), in1);

    let zero = _mm256_setzero_ps();
    let in0 = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, _mm256_set1_ps(BREAK0_F32));
    t = _mm256_blendv_ps(t, ax, in0);
    hi = _mm256_blendv_ps(hi, zero, in0);
    lo = _mm256_blendv_ps(lo, zero, in0);

    // t in [-0.4143, 0.4143]; |x| = inf gives t = -0.0, so the result is π/2.
    let z = _mm256_mul_ps(t, t);

    let mut p = _mm256_set1_ps(AT0_F32);
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(AT1_F32));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(AT2_F32));
    p = _mm256_fmadd_ps(p, z, _mm256_set1_ps(AT3_F32));

    // atan(x) = atan(c) + atan(t), with the correction term added beside the
    // polynomial rather than beside the head.
    let poly = _mm256_fmadd_ps(_mm256_mul_ps(z, t), p, t);
    let result = _mm256_add_ps(hi, _mm256_add_ps(lo, poly));

    // Restore sign
    _mm256_or_ps(result, sign)
}

/// Fast SIMD atan approximation for f64 using AVX2+FMA
///
/// See `common::_ATAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atan_f64(x: __m256d) -> __m256d {
    use atan_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let sign_mask = _mm256_set1_pd(-0.0);
    let sign = _mm256_and_pd(x, sign_mask);
    let ax = _mm256_andnot_pd(sign_mask, x);

    // Pick the reduction centre c and the matching atan(c) head/correction.
    // The breakpoint masks are nested, so blending from the widest bucket
    // inward leaves each lane holding its tightest match.
    let mut c = _mm256_set1_pd(1.5);
    let mut hi = _mm256_set1_pd(ATAN_HI2_F64);
    let mut lo = _mm256_set1_pd(ATAN_LO2_F64);

    let in2 = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, _mm256_set1_pd(BREAK2_F64));
    c = _mm256_blendv_pd(c, one, in2);
    hi = _mm256_blendv_pd(hi, _mm256_set1_pd(ATAN_HI1_F64), in2);
    lo = _mm256_blendv_pd(lo, _mm256_set1_pd(ATAN_LO1_F64), in2);

    let in1 = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, _mm256_set1_pd(BREAK1_F64));
    c = _mm256_blendv_pd(c, _mm256_set1_pd(0.5), in1);
    hi = _mm256_blendv_pd(hi, _mm256_set1_pd(ATAN_HI0_F64), in1);
    lo = _mm256_blendv_pd(lo, _mm256_set1_pd(ATAN_LO0_F64), in1);

    let zero = _mm256_setzero_pd();
    let in0 = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, _mm256_set1_pd(BREAK0_F64));
    c = _mm256_blendv_pd(c, zero, in0);
    hi = _mm256_blendv_pd(hi, zero, in0);
    lo = _mm256_blendv_pd(lo, zero, in0);

    // Past the last breakpoint the centre is at infinity: t = -1/|x|.
    // NaN fails every comparison and falls through to c = 1.5, which
    // propagates NaN through the division below.
    let big = _mm256_cmp_pd::<_CMP_GE_OQ>(ax, _mm256_set1_pd(BREAK3_F64));
    hi = _mm256_blendv_pd(hi, _mm256_set1_pd(ATAN_HI3_F64), big);
    lo = _mm256_blendv_pd(lo, _mm256_set1_pd(ATAN_LO3_F64), big);
    let num = _mm256_blendv_pd(_mm256_sub_pd(ax, c), _mm256_set1_pd(-1.0), big);
    let den = _mm256_blendv_pd(_mm256_fmadd_pd(c, ax, one), ax, big);

    // t in [-0.4375, 0.4375]; |x| = inf gives t = -0.0, so the result is π/2.
    let t = _mm256_div_pd(num, den);
    let z = _mm256_mul_pd(t, t);
    let w = _mm256_mul_pd(z, z);

    // Even- and odd-indexed coefficients evaluated as two independent Horner
    // chains in w, which shortens the dependency chain versus one chain in z.
    let mut s1 = _mm256_set1_pd(AT10_F64);
    s1 = _mm256_fmadd_pd(s1, w, _mm256_set1_pd(AT8_F64));
    s1 = _mm256_fmadd_pd(s1, w, _mm256_set1_pd(AT6_F64));
    s1 = _mm256_fmadd_pd(s1, w, _mm256_set1_pd(AT4_F64));
    s1 = _mm256_fmadd_pd(s1, w, _mm256_set1_pd(AT2_F64));
    s1 = _mm256_fmadd_pd(s1, w, _mm256_set1_pd(AT0_F64));
    s1 = _mm256_mul_pd(s1, z);

    let mut s2 = _mm256_set1_pd(AT9_F64);
    s2 = _mm256_fmadd_pd(s2, w, _mm256_set1_pd(AT7_F64));
    s2 = _mm256_fmadd_pd(s2, w, _mm256_set1_pd(AT5_F64));
    s2 = _mm256_fmadd_pd(s2, w, _mm256_set1_pd(AT3_F64));
    s2 = _mm256_fmadd_pd(s2, w, _mm256_set1_pd(AT1_F64));
    s2 = _mm256_mul_pd(s2, w);

    // atan(x) = atan(c) + atan(t), grouped so the correction term lands
    // beside the polynomial residual rather than beside the head.
    let poly = _mm256_mul_pd(t, _mm256_add_pd(s1, s2));
    let result = _mm256_sub_pd(hi, _mm256_sub_pd(_mm256_sub_pd(poly, lo), t));

    // Restore sign
    _mm256_or_pd(result, sign)
}

// ============================================================================
// Horizontal reductions
// ============================================================================

/// Horizontal maximum of 8 f32 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hmax_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(low, high);
    let shuf = _mm_movehdup_ps(max128);
    let max64 = _mm_max_ps(max128, shuf);
    let shuf2 = _mm_movehl_ps(max64, max64);
    let max32 = _mm_max_ss(max64, shuf2);
    _mm_cvtss_f32(max32)
}

/// Horizontal maximum of 4 f64 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hmax_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let max128 = _mm_max_pd(low, high);
    let shuf = _mm_unpackhi_pd(max128, max128);
    let max64 = _mm_max_sd(max128, shuf);
    _mm_cvtsd_f64(max64)
}

/// Horizontal sum of 8 f32 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hsum_f32(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Horizontal sum of 4 f64 values in an AVX2 register
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}

// ============================================================================
// Additional transcendental functions
// ============================================================================

/// Fast SIMD rsqrt (1/sqrt(x)) for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn rsqrt_f32(x: __m256) -> __m256 {
    // Use Newton-Raphson refinement on the fast approximation
    let approx = _mm256_rsqrt_ps(x);
    let half = _mm256_set1_ps(0.5);
    let three = _mm256_set1_ps(3.0);
    // One Newton-Raphson iteration: y = 0.5 * y * (3 - x * y * y)
    let x_approx2 = _mm256_mul_ps(x, _mm256_mul_ps(approx, approx));
    let factor = _mm256_sub_ps(three, x_approx2);
    _mm256_mul_ps(half, _mm256_mul_ps(approx, factor))
}

/// Fast SIMD rsqrt (1/sqrt(x)) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn rsqrt_f64(x: __m256d) -> __m256d {
    let sqrt_x = _mm256_sqrt_pd(x);
    _mm256_div_pd(_mm256_set1_pd(1.0), sqrt_x)
}

/// Fast SIMD exp2 (2^x) for f32 using AVX2
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. Handing `x * ln2` to `exp_f32`
/// rounds a value as large as 128 before the exponential, which the
/// exponential turns into 5e-6 of relative error — forty ulps.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f32(x: __m256) -> __m256 {
    use exp_coefficients::{C0_F32, C1_F32, C2_F32, C3_F32, C4_F32, C5_F32, C6_F32, C7_F32};
    use exp2_coefficients::{MAX_F32, MIN_F32};

    // The bound leads so that maxps/minps let NaN through; see `exp_f32`.
    let xc = _mm256_max_ps(_mm256_set1_ps(MIN_F32), x);
    let xc = _mm256_min_ps(_mm256_set1_ps(MAX_F32), xc);

    // n and r are both exact, so the reduction itself costs nothing.
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(xc);
    let r = _mm256_sub_ps(xc, n);
    let rl = _mm256_mul_ps(r, _mm256_set1_ps(std::f32::consts::LN_2));

    let mut poly = _mm256_set1_ps(C7_F32);
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C6_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C5_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C4_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C3_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C2_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C1_F32));
    poly = _mm256_fmadd_ps(poly, rl, _mm256_set1_ps(C0_F32));

    // Two halved powers of two, as in `exp_f32`.
    let n_i32 = _mm256_cvtps_epi32(n);
    let n_hi = _mm256_srai_epi32::<1>(n_i32);
    let n_lo = _mm256_sub_epi32(n_i32, n_hi);
    let bias = _mm256_set1_epi32(127);
    let p_hi = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_hi, bias)));
    let p_lo = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_lo, bias)));

    _mm256_mul_ps(_mm256_mul_ps(poly, p_hi), p_lo)
}

/// Fast SIMD exp2 (2^x) for f64 using AVX2
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. Borrowing `exp(x * ln2)` would
/// round the product once, and the exponential turns that absolute error into
/// a relative one — about 1e-13 near |x| = 1000.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f64(x: __m256d) -> __m256d {
    use exp2_coefficients::*;

    let xc = _mm256_max_pd(x, _mm256_set1_pd(MIN_F64));
    let xc = _mm256_min_pd(xc, _mm256_set1_pd(MAX_F64));

    // Both n and r are exact, so nothing is lost before the polynomial.
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(xc);
    let r = _mm256_sub_pd(xc, n);

    let mut poly = _mm256_set1_pd(C13_F64);
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C12_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C11_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C10_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C9_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C8_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C7_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C6_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C5_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C4_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C3_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C2_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C1_F64));
    poly = _mm256_fmadd_pd(poly, r, _mm256_set1_pd(C0_F64));

    // AVX2 lacks _mm256_cvtpd_epi64, so the scale drops to scalar, as in
    // exp_f64. The power of two is split in half: both factors stay normal, an
    // overflow reaches infinity in the second multiply, and a subnormal result
    // takes exactly one rounding because the first multiply is exact.
    let mut scaled = [0.0f64; 4];
    let mut n_arr = [0.0f64; 4];
    let mut poly_arr = [0.0f64; 4];

    _mm256_storeu_pd(n_arr.as_mut_ptr(), n);
    _mm256_storeu_pd(poly_arr.as_mut_ptr(), poly);

    for i in 0..4 {
        let n_i = n_arr[i] as i64;
        let hi = n_i / 2;
        let lo = n_i - hi;
        let p_hi = f64::from_bits(((hi + 1023) as u64) << 52);
        let p_lo = f64::from_bits(((lo + 1023) as u64) << 52);
        scaled[i] = (poly_arr[i] * p_hi) * p_lo;
    }

    // maxpd/minpd return their second operand for NaN, so the clamp above turns
    // a NaN input into -1075. Restore it.
    let out = _mm256_loadu_pd(scaled.as_ptr());
    _mm256_blendv_pd(x, out, _mm256_cmp_pd::<_CMP_ORD_Q>(x, x))
}

/// Fast SIMD expm1 (e^x - 1) for f32 using AVX2
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. A degree-4 Taylor series on
/// |x| <= 0.5 drops `x⁵/120`, which is 2.6e-4 at the interval edge — over two
/// thousand ulps — and `exp(x) - 1` above it cancels the whole result away.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f32(x: __m256) -> __m256 {
    use exp_coefficients::*;

    // The bound leads so that maxps/minps let NaN through; see `exp_f32`.
    let xc = _mm256_max_ps(_mm256_set1_ps(EXPM1_MIN_F32), x);
    let xc = _mm256_min_ps(_mm256_set1_ps(EXPM1_MAX_F32), xc);

    let y = _mm256_mul_ps(xc, _mm256_set1_ps(std::f32::consts::LOG2_E));
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // Cody-Waite reduction, identical to exp_f32: r = x - n*ln2, |r| <= ln2/2.
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_HI_F32), xc);
    let r = _mm256_fnmadd_ps(n, _mm256_set1_ps(LN2_LO_F32), r);

    // Q is the exp series from its r² term up, so expm1(r) = r + r²*Q(r) keeps
    // r itself outside the polynomial and never rounds against a leading 1.
    let mut q = _mm256_set1_ps(C7_F32);
    q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C6_F32));
    q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C5_F32));
    q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C4_F32));
    q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C3_F32));
    q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(C2_F32));
    let e = _mm256_fmadd_ps(_mm256_mul_ps(r, r), q, r);

    // 2^n*(1+E) - 1 = 2*(t*E + (t - 0.5)) with t = 2^(n-1). t and t - 0.5 are
    // both exact, and the halved scale keeps n = 128 representable, so an
    // overflow happens in the final doubling rather than in 2^n.
    let n_i32 = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127 - 1);
    let t = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(n_i32, bias)));
    let out = _mm256_mul_ps(
        _mm256_set1_ps(2.0),
        _mm256_fmadd_ps(t, e, _mm256_sub_ps(t, _mm256_set1_ps(0.5))),
    );

    // At n = 0 the scale is exactly 1 and the answer is E itself. Taking it
    // directly matters for a subnormal E, where the halved form's `0.5 * E`
    // rounds the last bit to even and loses the whole value.
    let out = _mm256_blendv_ps(out, e, _mm256_cmp_ps::<_CMP_EQ_OQ>(n, _mm256_setzero_ps()));

    // expm1(±0) = ±0, which the reduction would otherwise return as +0.
    _mm256_blendv_ps(out, x, _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_setzero_ps()))
}

/// Fast SIMD expm1 (e^x - 1) for f64 using AVX2
///
/// See `common::_EXP2_EXPM1_ALGORITHM_DOC`. A degree-4 Taylor series on
/// |x| <= 0.5 drops `x⁵/120`, which is 2.6e-4 at the interval edge — twelve
/// decimal digits short of double precision.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f64(x: __m256d) -> __m256d {
    use exp_coefficients::*;

    let xc = _mm256_max_pd(x, _mm256_set1_pd(EXPM1_MIN_F64));
    let xc = _mm256_min_pd(xc, _mm256_set1_pd(EXPM1_MAX_F64));

    let y = _mm256_mul_pd(xc, _mm256_set1_pd(std::f64::consts::LOG2_E));
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // Cody-Waite reduction, identical to exp_f64: r = x - n*ln2, |r| <= ln2/2.
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_HI_F64), xc);
    let r = _mm256_fnmadd_pd(n, _mm256_set1_pd(LN2_LO_F64), r);

    // Q is the exp series from its r² term up, so expm1(r) = r + r²*Q(r) keeps
    // r itself outside the polynomial and never rounds against a leading 1.
    let mut q = _mm256_set1_pd(C13_F64);
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C12_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C11_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C10_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C9_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C8_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C7_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C6_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C5_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C4_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C3_F64));
    q = _mm256_fmadd_pd(q, r, _mm256_set1_pd(C2_F64));
    let e = _mm256_fmadd_pd(_mm256_mul_pd(r, r), q, r);

    // 2^n*(1+E) - 1 = 2*(t*E + (t - 0.5)) with t = 2^(n-1). t and t - 0.5 are
    // both exact, and at n = 0 they are 0.5 and 0, so the result is E itself.
    let mut scaled = [0.0f64; 4];
    let mut n_arr = [0.0f64; 4];
    let mut e_arr = [0.0f64; 4];

    _mm256_storeu_pd(n_arr.as_mut_ptr(), n);
    _mm256_storeu_pd(e_arr.as_mut_ptr(), e);

    for i in 0..4 {
        let m = n_arr[i] as i64 - 1;
        let t = f64::from_bits(((m + 1023) as u64) << 52);
        scaled[i] = 2.0 * t.mul_add(e_arr[i], t - 0.5);
    }

    let out = _mm256_loadu_pd(scaled.as_ptr());

    // At n = 0 the scale is exactly 1 and the answer is E itself. Taking it
    // directly matters for a subnormal E, where the halved form's `0.5 * E`
    // rounds the last bit to even and loses the whole value.
    let out = _mm256_blendv_pd(out, e, _mm256_cmp_pd::<_CMP_EQ_OQ>(n, _mm256_setzero_pd()));

    // expm1(±0) = ±0, and the clamp above would otherwise silence NaN.
    let out = _mm256_blendv_pd(out, x, _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_setzero_pd()));
    _mm256_blendv_pd(x, out, _mm256_cmp_pd::<_CMP_ORD_Q>(x, x))
}

/// Fast SIMD log2 for f32 using AVX2
///
/// Scaling `log(x)` would fold the exponent through two roundings and miss
/// exact powers of two, so the exponent is added back untouched instead.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f32(x: __m256) -> __m256 {
    let (n, logm) = log_reduce_f32(x);
    let r = _mm256_fmadd_ps(logm, _mm256_set1_ps(std::f32::consts::LOG2_E), n);
    log_special_f32(x, r)
}

/// Fast SIMD log2 for f64 using AVX2
///
/// Scaling `log(x)` would fold the exponent through two roundings and miss
/// exact powers of two, so the exponent is added back untouched instead.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f64(x: __m256d) -> __m256d {
    let (n, logm) = log_reduce_f64(x);
    let r = _mm256_fmadd_pd(logm, _mm256_set1_pd(std::f64::consts::LOG2_E), n);
    log_special_f64(x, r)
}

/// Fast SIMD log10 for f32 using AVX2
///
/// `log10(x) = n*log10(2) + log(m)*log10(e)`, keeping the exact exponent out
/// of the mantissa's rounding for the same reason as `log2_f32`.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f32(x: __m256) -> __m256 {
    let (n, logm) = log_reduce_f32(x);
    let scaled = _mm256_mul_ps(logm, _mm256_set1_ps(std::f32::consts::LOG10_E));
    let r = _mm256_fmadd_ps(n, _mm256_set1_ps(std::f32::consts::LOG10_2), scaled);
    log_special_f32(x, r)
}

/// Fast SIMD log10 for f64 using AVX2
///
/// `log10(x) = n*log10(2) + log(m)*log10(e)`, keeping the exact exponent out
/// of the mantissa's rounding for the same reason as `log2_f64`.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f64(x: __m256d) -> __m256d {
    let (n, logm) = log_reduce_f64(x);
    let scaled = _mm256_mul_pd(logm, _mm256_set1_pd(std::f64::consts::LOG10_E));
    let r = _mm256_fmadd_pd(n, _mm256_set1_pd(std::f64::consts::LOG10_2), scaled);
    log_special_f64(x, r)
}

/// Fast SIMD log1p (log(1+x)) for f32 using AVX2
///
/// `1 + x` alone rounds away the information log1p exists to keep, so the sum
/// is carried as an exact pair `u + c` and the residual is folded back in. The
/// degree-4 Taylor series this replaces dropped `x⁵/5`, which is 6.1e-3 at
/// x = -0.5 — a hundredth of the result there.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let u = _mm256_add_ps(one, x);

    // Fast2Sum: 1 + x = u + c exactly, with the larger addend leading.
    let c_small = _mm256_sub_ps(x, _mm256_sub_ps(u, one));
    let c_large = _mm256_sub_ps(one, _mm256_sub_ps(u, x));
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);
    let x_leads = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_x, one);
    let c = _mm256_blendv_ps(c_large, c_small, x_leads);

    // log(u + c) = log(u) + log1p(c/u), and |c/u| <= 2^-24, so the inner series
    // collapses to its first term.
    let r = _mm256_add_ps(log_f32(u), _mm256_div_ps(c, u));

    // u == 1 means x fell entirely off the end of the sum; log1p(x) is then x
    // to within half an ulp. This is also what carries signed zero through.
    let is_unit = _mm256_cmp_ps::<_CMP_EQ_OQ>(u, one);
    let out = _mm256_blendv_ps(r, x, is_unit);

    // x == -1 gives u == 0 and c/u = 0/0; x == +inf gives inf - inf.
    let is_neg_one = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_set1_ps(-1.0));
    let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_set1_ps(f32::INFINITY));
    let out = _mm256_blendv_ps(out, _mm256_set1_ps(f32::NEG_INFINITY), is_neg_one);
    _mm256_blendv_ps(out, _mm256_set1_ps(f32::INFINITY), is_inf)
}

/// Fast SIMD log1p (log(1+x)) for f64 using AVX2
///
/// `1 + x` alone rounds away the information log1p exists to keep, so the sum
/// is carried as an exact pair `u + c` and the residual is folded back in.
/// Relative error stays below 2 ulps, including for |x| down to the subnormal
/// range where log1p(x) == x.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let u = _mm256_add_pd(one, x);

    // Fast2Sum: 1 + x = u + c exactly, with the larger addend leading.
    let c_small = _mm256_sub_pd(x, _mm256_sub_pd(u, one));
    let c_large = _mm256_sub_pd(one, _mm256_sub_pd(u, x));
    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    let x_leads = _mm256_cmp_pd::<_CMP_LT_OQ>(abs_x, one);
    let c = _mm256_blendv_pd(c_large, c_small, x_leads);

    // log(u + c) = log(u) + log1p(c/u), and |c/u| <= 2^-53, so the inner series
    // collapses to its first term.
    let r = _mm256_add_pd(log_f64(u), _mm256_div_pd(c, u));

    // u == 1 means x fell entirely off the end of the sum; log1p(x) is then x
    // to within half an ulp. This is also what carries signed zero through.
    let is_unit = _mm256_cmp_pd::<_CMP_EQ_OQ>(u, one);
    let out = _mm256_blendv_pd(r, x, is_unit);

    // x == -1 gives u == 0 and c/u = 0/0; x == +inf gives inf - inf.
    let is_neg_one = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_set1_pd(-1.0));
    let is_inf = _mm256_cmp_pd::<_CMP_EQ_OQ>(x, _mm256_set1_pd(f64::INFINITY));
    let out = _mm256_blendv_pd(out, _mm256_set1_pd(f64::NEG_INFINITY), is_neg_one);
    _mm256_blendv_pd(out, _mm256_set1_pd(f64::INFINITY), is_inf)
}

/// Fast SIMD sinh for f32 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^x - e^-x)/2` subtracts two
/// values that both approach 1 as x approaches 0, so it keeps none of the
/// result; `(u + u/(1+u))/2` with `u = expm1(|x|)` keeps all of it.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f32(x: __m256) -> __m256 {
    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let a = _mm256_andnot_ps(sign_mask, x);
    let u = expm1_f32(a);

    let d = _mm256_div_ps(u, _mm256_add_ps(one, u));
    // u/(1+u) tends to 1 as u overflows, where the quotient itself is inf/inf.
    let is_inf = _mm256_cmp_ps::<_CMP_EQ_OQ>(u, _mm256_set1_ps(f32::INFINITY));
    let d = _mm256_blendv_ps(d, one, is_inf);
    let s = _mm256_mul_ps(half, _mm256_add_ps(u, d));

    // expm1 overflows at ln(f32::MAX) = 88.7228 while sinh stays finite up to
    // 89.4159. Past the breakpoint sinh is 0.5*exp(|x|), built as in cosh_f32.
    let t = exp_f32(_mm256_mul_ps(half, a));
    let far = _mm256_mul_ps(_mm256_mul_ps(half, t), t);
    let big = _mm256_cmp_ps::<_CMP_GT_OQ>(a, _mm256_set1_ps(hyperbolic_breakpoints::BIG_F32));
    let s = _mm256_blendv_ps(s, far, big);

    _mm256_or_ps(s, _mm256_and_ps(x, sign_mask))
}

/// Fast SIMD sinh for f64 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `(e^x - e^-x)/2` subtracts two
/// values that both approach 1 as x approaches 0, so it keeps none of the
/// result; `(u + u/(1+u))/2` with `u = expm1(|x|)` keeps all of it.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f64(x: __m256d) -> __m256d {
    let sign_mask = _mm256_set1_pd(-0.0);
    let one = _mm256_set1_pd(1.0);
    let a = _mm256_andnot_pd(sign_mask, x);
    let u = expm1_f64(a);

    let d = _mm256_div_pd(u, _mm256_add_pd(one, u));
    // u/(1+u) tends to 1 as u overflows, where the quotient itself is inf/inf.
    let is_inf = _mm256_cmp_pd::<_CMP_EQ_OQ>(u, _mm256_set1_pd(f64::INFINITY));
    let d = _mm256_blendv_pd(d, one, is_inf);
    let s = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_add_pd(u, d));

    // expm1 overflows at ln(f64::MAX) = 709.7827 while sinh stays finite up to
    // 710.4758. Past the breakpoint sinh is 0.5*exp(|x|), built as in cosh_f64.
    let half = _mm256_set1_pd(0.5);
    let t = exp_f64(_mm256_mul_pd(half, a));
    let far = _mm256_mul_pd(_mm256_mul_pd(half, t), t);
    let big = _mm256_cmp_pd::<_CMP_GT_OQ>(a, _mm256_set1_pd(hyperbolic_breakpoints::BIG_F64));
    let s = _mm256_blendv_pd(s, far, big);

    _mm256_or_pd(s, _mm256_and_pd(x, sign_mask))
}

/// Fast SIMD cosh for f32 using AVX2
///
/// See `common::hyperbolic_breakpoints`. `(e^x + e^-x)/2` returns infinity over
/// the whole band where exp has overflowed but cosh has not, [88.7228,
/// 89.4159], so |x| past the breakpoint takes the squared form instead.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f32(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let a = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), x));
    let near = _mm256_mul_ps(half, _mm256_add_ps(exp_x, exp_neg_x));

    // (0.5*t)*t with t = exp(|x|/2) is 0.5*exp(|x|) with no intermediate past
    // f32::MAX. Halving t first, not the product, is what keeps it finite.
    let t = exp_f32(_mm256_mul_ps(half, a));
    let far = _mm256_mul_ps(_mm256_mul_ps(half, t), t);

    // NaN compares false and takes the near branch, where exp propagates it.
    let big = _mm256_cmp_ps::<_CMP_GT_OQ>(a, _mm256_set1_ps(hyperbolic_breakpoints::BIG_F32));
    _mm256_blendv_ps(near, far, big)
}

/// Fast SIMD cosh for f64 using AVX2
///
/// See `common::hyperbolic_breakpoints`. `(e^x + e^-x)/2` returns infinity over
/// the whole band where exp has overflowed but cosh has not, [709.7827,
/// 710.4758], so |x| past the breakpoint takes the squared form instead.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let a = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    let near = _mm256_mul_pd(half, _mm256_add_pd(exp_x, exp_neg_x));

    // (0.5*t)*t with t = exp(|x|/2) is 0.5*exp(|x|) with no intermediate past
    // f64::MAX. Halving t first, not the product, is what keeps it finite.
    let t = exp_f64(_mm256_mul_pd(half, a));
    let far = _mm256_mul_pd(_mm256_mul_pd(half, t), t);

    // NaN compares false and takes the near branch, where exp propagates it.
    let big = _mm256_cmp_pd::<_CMP_GT_OQ>(a, _mm256_set1_pd(hyperbolic_breakpoints::BIG_F64));
    _mm256_blendv_pd(near, far, big)
}

/// Fast SIMD asinh for f32 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `log(x + sqrt(x²+1))` cancels for
/// every negative x, and `x²` overflows f32 past 1.8e19, so the sign is taken
/// out first and large |x| collapses to `log(|x|) + ln2`.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f32(x: __m256) -> __m256 {
    use inv_hyperbolic_breakpoints::{BIG_F32, NEAR_F32};

    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let a = _mm256_andnot_ps(sign_mask, x);
    let t = _mm256_mul_ps(a, a);
    let root = _mm256_sqrt_ps(_mm256_add_ps(t, one));

    // a <= 2: a + a²/(1 + sqrt(1+a²)) is sqrt(1+a²) - 1 + a without the
    // subtraction, and log1p keeps its low bits down to the subnormal range.
    let near = log1p_f32(_mm256_add_ps(a, _mm256_div_ps(t, _mm256_add_ps(one, root))));
    // 2 < a <= 2^12: the same identity with the reciprocal written out.
    let mid = log_f32(_mm256_fmadd_ps(
        _mm256_set1_ps(2.0),
        a,
        _mm256_div_ps(one, _mm256_add_ps(root, a)),
    ));
    // a > 2^12: sqrt(a²+1) equals a in single precision, so asinh is log(2a).
    let far = _mm256_add_ps(log_f32(a), _mm256_set1_ps(std::f32::consts::LN_2));

    let r = _mm256_blendv_ps(
        near,
        mid,
        _mm256_cmp_ps::<_CMP_GT_OQ>(a, _mm256_set1_ps(NEAR_F32)),
    );
    let r = _mm256_blendv_ps(
        r,
        far,
        _mm256_cmp_ps::<_CMP_GT_OQ>(a, _mm256_set1_ps(BIG_F32)),
    );

    _mm256_or_ps(r, _mm256_and_ps(x, sign_mask))
}

/// Fast SIMD asinh for f64 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `log(x + sqrt(x²+1))` cancels for
/// every negative x — at x = -49.6 the two addends agree to twelve digits —
/// so the sign is taken out first and the work is done on |x|.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f64(x: __m256d) -> __m256d {
    use inv_hyperbolic_breakpoints::{BIG_F64, NEAR_F64};

    let sign_mask = _mm256_set1_pd(-0.0);
    let one = _mm256_set1_pd(1.0);
    let a = _mm256_andnot_pd(sign_mask, x);
    let t = _mm256_mul_pd(a, a);
    let root = _mm256_sqrt_pd(_mm256_add_pd(t, one));

    // a <= 2: a + a²/(1 + sqrt(1+a²)) is sqrt(1+a²) - 1 + a without the
    // subtraction, and log1p keeps its low bits down to the subnormal range.
    let near = log1p_f64(_mm256_add_pd(a, _mm256_div_pd(t, _mm256_add_pd(one, root))));
    // 2 < a <= 2^28: the same identity with the reciprocal written out.
    let mid = log_f64(_mm256_fmadd_pd(
        _mm256_set1_pd(2.0),
        a,
        _mm256_div_pd(one, _mm256_add_pd(root, a)),
    ));
    // a > 2^28: sqrt(a²+1) equals a in double, so asinh collapses to log(2a).
    let far = _mm256_add_pd(log_f64(a), _mm256_set1_pd(std::f64::consts::LN_2));

    let r = _mm256_blendv_pd(
        near,
        mid,
        _mm256_cmp_pd::<_CMP_GT_OQ>(a, _mm256_set1_pd(NEAR_F64)),
    );
    let r = _mm256_blendv_pd(
        r,
        far,
        _mm256_cmp_pd::<_CMP_GT_OQ>(a, _mm256_set1_pd(BIG_F64)),
    );

    _mm256_or_pd(r, _mm256_and_pd(x, sign_mask))
}

/// Fast SIMD acosh for f32 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. Forming `x² - 1` near x = 1 throws
/// away half the significant bits of `x - 1`, and past 1.8e19 it overflows f32
/// outright, which turned every input above that into `log(f32::INFINITY)`.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f32(x: __m256) -> __m256 {
    use inv_hyperbolic_breakpoints::{BIG_F32, NEAR_F32};

    let one = _mm256_set1_ps(1.0);
    let two = _mm256_set1_ps(2.0);
    let t = _mm256_sub_ps(x, one);

    // 1 <= x < 2: acosh(1+t) = log1p(t + sqrt(2t + t²)), which never forms a
    // difference of two nearly equal quantities.
    let disc = _mm256_sqrt_ps(_mm256_fmadd_ps(t, t, _mm256_mul_ps(two, t)));
    let near = log1p_f32(_mm256_add_ps(t, disc));
    // 2 <= x <= 2^12.
    let root = _mm256_sqrt_ps(_mm256_fmsub_ps(x, x, one));
    let mid = log_f32(_mm256_fmsub_ps(
        two,
        x,
        _mm256_div_ps(one, _mm256_add_ps(x, root)),
    ));
    // x > 2^12: sqrt(x²-1) equals x in single precision, so acosh is log(2x).
    let far = _mm256_add_ps(log_f32(x), _mm256_set1_ps(std::f32::consts::LN_2));

    let r = _mm256_blendv_ps(
        near,
        mid,
        _mm256_cmp_ps::<_CMP_GE_OQ>(x, _mm256_set1_ps(NEAR_F32)),
    );
    let r = _mm256_blendv_ps(
        r,
        far,
        _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(BIG_F32)),
    );

    // acosh is undefined below 1. NaN fails the ordered compare and keeps the
    // NaN the log1p branch already produced.
    _mm256_blendv_ps(
        r,
        _mm256_set1_ps(f32::NAN),
        _mm256_cmp_ps::<_CMP_LT_OQ>(x, one),
    )
}

/// Fast SIMD acosh for f64 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. Forming `x² - 1` near x = 1 throws
/// away half the significant bits of `x - 1`, which is the whole result there.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f64(x: __m256d) -> __m256d {
    use inv_hyperbolic_breakpoints::{BIG_F64, NEAR_F64};

    let one = _mm256_set1_pd(1.0);
    let two = _mm256_set1_pd(2.0);
    let t = _mm256_sub_pd(x, one);

    // 1 <= x < 2: acosh(1+t) = log1p(t + sqrt(2t + t²)), which never forms a
    // difference of two nearly equal quantities.
    let disc = _mm256_sqrt_pd(_mm256_fmadd_pd(t, t, _mm256_mul_pd(two, t)));
    let near = log1p_f64(_mm256_add_pd(t, disc));
    // 2 <= x <= 2^28.
    let root = _mm256_sqrt_pd(_mm256_fmsub_pd(x, x, one));
    let mid = log_f64(_mm256_fmsub_pd(
        two,
        x,
        _mm256_div_pd(one, _mm256_add_pd(x, root)),
    ));
    // x > 2^28: sqrt(x²-1) equals x in double, so acosh collapses to log(2x).
    let far = _mm256_add_pd(log_f64(x), _mm256_set1_pd(std::f64::consts::LN_2));

    let r = _mm256_blendv_pd(
        near,
        mid,
        _mm256_cmp_pd::<_CMP_GE_OQ>(x, _mm256_set1_pd(NEAR_F64)),
    );
    let r = _mm256_blendv_pd(
        r,
        far,
        _mm256_cmp_pd::<_CMP_GT_OQ>(x, _mm256_set1_pd(BIG_F64)),
    );

    // acosh is undefined below 1. NaN fails the ordered compare and keeps the
    // NaN the log1p branch already produced.
    _mm256_blendv_pd(
        r,
        _mm256_set1_pd(f64::NAN),
        _mm256_cmp_pd::<_CMP_LT_OQ>(x, one),
    )
}

/// Fast SIMD atanh for f32 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `0.5*log((1+x)/(1-x))` rounds
/// `1 + x` before the log, which at small |x| discards every bit the result is
/// made of, and it has no domain handling at all: at x = 1 the quotient is
/// infinity, whose logarithm the old kernel reported as 88.72.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f32(x: __m256) -> __m256 {
    use inv_hyperbolic_breakpoints::ATANH_SPLIT_F32;

    let sign_mask = _mm256_set1_ps(-0.0);
    let one = _mm256_set1_ps(1.0);
    let a = _mm256_andnot_ps(sign_mask, x);
    let t = _mm256_add_ps(a, a);
    let den = _mm256_sub_ps(one, a);

    // a < 0.5: t + t*a/(1-a) is 2a/(1-a) written so the leading term stays
    // exact, which is what carries atanh(x) == x through the subnormal range.
    let small = log1p_f32(_mm256_add_ps(t, _mm256_div_ps(_mm256_mul_ps(t, a), den)));
    // 0.5 <= a: at a = 1 the quotient is +inf and log1p returns +inf; past 1 it
    // is at most -2, so log1p of it is NaN.
    let large = log1p_f32(_mm256_div_ps(t, den));

    let picked = _mm256_blendv_ps(
        large,
        small,
        _mm256_cmp_ps::<_CMP_LT_OQ>(a, _mm256_set1_ps(ATANH_SPLIT_F32)),
    );
    let r = _mm256_mul_ps(_mm256_set1_ps(0.5), picked);

    _mm256_or_ps(r, _mm256_and_ps(x, sign_mask))
}

/// Fast SIMD atanh for f64 using AVX2
///
/// See `common::_HYPERBOLIC_ALGORITHM_DOC`. `0.5*log((1+x)/(1-x))` rounds
/// `1 + x` before the log, which at x = 7e-4 discards every bit the result is
/// made of; log1p keeps them.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f64(x: __m256d) -> __m256d {
    use inv_hyperbolic_breakpoints::ATANH_SPLIT_F64;

    let sign_mask = _mm256_set1_pd(-0.0);
    let one = _mm256_set1_pd(1.0);
    let a = _mm256_andnot_pd(sign_mask, x);
    let t = _mm256_add_pd(a, a);
    let den = _mm256_sub_pd(one, a);

    // a < 0.5: t + t*a/(1-a) is 2a/(1-a) written so the leading term stays
    // exact, which is what carries atanh(x) == x through the subnormal range.
    let small = log1p_f64(_mm256_add_pd(t, _mm256_div_pd(_mm256_mul_pd(t, a), den)));
    // 0.5 <= a: at a = 1 the quotient is +inf and log1p returns +inf; past 1 it
    // is at most -2, so log1p of it is NaN.
    let large = log1p_f64(_mm256_div_pd(t, den));

    let picked = _mm256_blendv_pd(
        large,
        small,
        _mm256_cmp_pd::<_CMP_LT_OQ>(a, _mm256_set1_pd(ATANH_SPLIT_F64)),
    );
    let r = _mm256_mul_pd(_mm256_set1_pd(0.5), picked);

    _mm256_or_pd(r, _mm256_and_pd(x, sign_mask))
}

/// Shared polynomial correction `t * R(t)` for f32 asin/acos.
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC`. Valid for t in [0, 0.5]; the caller
/// multiplies by its own argument to form `y * t * R(t)`.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn asin_r_f32(t: __m256) -> __m256 {
    use asin_coefficients::*;

    let mut p = _mm256_set1_ps(PS0_F32);
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(PS1_F32));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(PS2_F32));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(PS3_F32));
    p = _mm256_fmadd_ps(p, t, _mm256_set1_ps(PS4_F32));

    _mm256_mul_ps(t, p)
}

/// Fast SIMD asin for f32 using AVX2
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asin_f32(x: __m256) -> __m256 {
    use asin_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(HALF_F32);
    let sign_mask = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_mask);
    let ax = _mm256_andnot_ps(sign_mask, x);

    // |x| > 1 leaves the reflection argument negative, so sqrt yields NaN.
    // NaN input fails the comparison and takes the same reflection path.
    let small = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, half);
    let t_refl = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    let t = _mm256_blendv_ps(t_refl, _mm256_mul_ps(ax, ax), small);
    let s = _mm256_sqrt_ps(t);
    let v = _mm256_blendv_ps(s, ax, small);

    // w is asin(|x|) on the direct branch and asin(sqrt(t)) on the reflection.
    let w = _mm256_fmadd_ps(v, asin_r_f32(t), v);

    // π/2 - 2*asin(sqrt(t)), with the low half of π/2 folded into the
    // subtracted term so the cancellation keeps the trailing bits.
    let two_w = _mm256_add_ps(w, w);
    let res_refl = _mm256_sub_ps(
        _mm256_set1_ps(PIO2_HI_F32),
        _mm256_sub_ps(two_w, _mm256_set1_ps(PIO2_LO_F32)),
    );

    let result = _mm256_blendv_ps(res_refl, w, small);
    _mm256_or_ps(result, sign)
}

/// Shared rational correction `R(t) = p(t)/q(t)` for f64 asin/acos.
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC`. Valid for t in [0, 0.5].
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn asin_r_f64(t: __m256d) -> __m256d {
    use asin_coefficients::*;

    let mut p = _mm256_set1_pd(PS5_F64);
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(PS4_F64));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(PS3_F64));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(PS2_F64));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(PS1_F64));
    p = _mm256_fmadd_pd(p, t, _mm256_set1_pd(PS0_F64));
    p = _mm256_mul_pd(p, t);

    let mut q = _mm256_set1_pd(QS4_F64);
    q = _mm256_fmadd_pd(q, t, _mm256_set1_pd(QS3_F64));
    q = _mm256_fmadd_pd(q, t, _mm256_set1_pd(QS2_F64));
    q = _mm256_fmadd_pd(q, t, _mm256_set1_pd(QS1_F64));
    q = _mm256_fmadd_pd(q, t, _mm256_set1_pd(1.0));

    _mm256_div_pd(p, q)
}

/// Fast SIMD asin for f64 using AVX2
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asin_f64(x: __m256d) -> __m256d {
    use asin_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(HALF_F64);
    let sign_mask = _mm256_set1_pd(-0.0);
    let sign = _mm256_and_pd(x, sign_mask);
    let ax = _mm256_andnot_pd(sign_mask, x);

    // |x| > 1 leaves the reflection argument negative, so sqrt yields NaN.
    // NaN input fails the comparison and takes the same reflection path.
    let small = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, half);
    let t_refl = _mm256_mul_pd(_mm256_sub_pd(one, ax), half);
    let t = _mm256_blendv_pd(t_refl, _mm256_mul_pd(ax, ax), small);
    let r = asin_r_f64(t);
    let s = _mm256_sqrt_pd(t);

    let res_small = _mm256_fmadd_pd(ax, r, ax);

    // π/2 - 2*asin(sqrt(t)), with the low half of π/2 folded into the
    // subtracted term so the cancellation keeps the trailing bits.
    let s_sr = _mm256_fmadd_pd(s, r, s);
    let two_s = _mm256_add_pd(s_sr, s_sr);
    let res_refl = _mm256_sub_pd(
        _mm256_set1_pd(PIO2_HI_F64),
        _mm256_sub_pd(two_s, _mm256_set1_pd(PIO2_LO_F64)),
    );

    let result = _mm256_blendv_pd(res_refl, res_small, small);
    _mm256_or_pd(result, sign)
}

/// Fast SIMD acos for f32 using AVX2
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details. Built from the
/// reflection directly, not as π/2 - asin(x): that subtraction cancels away the
/// whole result as x approaches 1.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f32(x: __m256) -> __m256 {
    use asin_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(HALF_F32);
    let pio2_lo = _mm256_set1_ps(PIO2_LO_F32);
    let sign_mask = _mm256_set1_ps(-0.0);
    let ax = _mm256_andnot_ps(sign_mask, x);

    let small = _mm256_cmp_ps::<_CMP_LT_OQ>(ax, half);
    let t_refl = _mm256_mul_ps(_mm256_sub_ps(one, ax), half);
    let t = _mm256_blendv_ps(t_refl, _mm256_mul_ps(x, x), small);
    let q = asin_r_f32(t);
    let s = _mm256_sqrt_ps(t);

    // |x| <= 0.5: π/2 - asin(x), evaluated without forming asin(x) first.
    let px = _mm256_mul_ps(x, q);
    let res_small = _mm256_sub_ps(
        _mm256_set1_ps(PIO2_HI_F32),
        _mm256_add_ps(x, _mm256_sub_ps(px, pio2_lo)),
    );

    // x >= 0.5: 2*asin(sqrt(t)), which is small and free of cancellation.
    let ps = _mm256_mul_ps(s, q);
    let s_sr = _mm256_add_ps(s, ps);
    let res_pos = _mm256_add_ps(s_sr, s_sr);

    // x <= -0.5: π - 2*asin(sqrt(t)).
    let w = _mm256_add_ps(s, _mm256_sub_ps(ps, pio2_lo));
    let res_neg = _mm256_sub_ps(_mm256_set1_ps(PI_HI_F32), _mm256_add_ps(w, w));

    // NaN fails both comparisons and lands on the negative branch, where the
    // NaN sqrt argument propagates.
    let positive = _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_setzero_ps());
    let res_refl = _mm256_blendv_ps(res_neg, res_pos, positive);
    _mm256_blendv_ps(res_refl, res_small, small)
}

/// Fast SIMD acos for f64 using AVX2
///
/// See `common::_ASIN_ACOS_ALGORITHM_DOC` for algorithm details. Built from the
/// reflection directly, not as π/2 - asin(x): that subtraction cancels away the
/// whole result as x approaches 1.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f64(x: __m256d) -> __m256d {
    use asin_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(HALF_F64);
    let pio2_lo = _mm256_set1_pd(PIO2_LO_F64);
    let sign_mask = _mm256_set1_pd(-0.0);
    let ax = _mm256_andnot_pd(sign_mask, x);

    let small = _mm256_cmp_pd::<_CMP_LT_OQ>(ax, half);
    let t_refl = _mm256_mul_pd(_mm256_sub_pd(one, ax), half);
    let t = _mm256_blendv_pd(t_refl, _mm256_mul_pd(ax, ax), small);
    let r = asin_r_f64(t);
    let s = _mm256_sqrt_pd(t);

    // |x| <= 0.5: π/2 - asin(x), evaluated without forming asin(x) first.
    let res_small = _mm256_sub_pd(
        _mm256_set1_pd(PIO2_HI_F64),
        _mm256_add_pd(x, _mm256_sub_pd(_mm256_mul_pd(x, r), pio2_lo)),
    );

    // x >= 0.5: 2*asin(sqrt(t)), which is small and free of cancellation.
    let s_sr = _mm256_fmadd_pd(s, r, s);
    let res_pos = _mm256_add_pd(s_sr, s_sr);

    // x <= -0.5: π - 2*asin(sqrt(t)).
    let w = _mm256_add_pd(s, _mm256_sub_pd(_mm256_mul_pd(s, r), pio2_lo));
    let res_neg = _mm256_sub_pd(_mm256_set1_pd(PI_HI_F64), _mm256_add_pd(w, w));

    // NaN fails both comparisons and lands on the negative branch, where the
    // NaN sqrt argument propagates.
    let positive = _mm256_cmp_pd::<_CMP_GT_OQ>(x, _mm256_setzero_pd());
    let res_refl = _mm256_blendv_pd(res_neg, res_pos, positive);
    _mm256_blendv_pd(res_refl, res_small, small)
}

/// Fast SIMD cbrt (cube root) for f32 using AVX2
///
/// See `common::cbrt_constants`. The seed divides the whole bit pattern of
/// |x| by three, mantissa included, which lands within 3.3%; two Halley steps
/// then cube that error twice over. Seeding from the exponent alone is off by
/// up to 37%, and the same two steps only bring that to 5e-2.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cbrt_f32(x: __m256) -> __m256 {
    use cbrt_constants::*;

    let sign_mask = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);
    let one = _mm256_set1_ps(1.0);

    // Move the two ends of the range into the middle: the iteration forms
    // x + 2*t³, which overflows near f32::MAX, and the seed assumes an
    // implicit leading 1, which a subnormal does not have. Both shifts are
    // 2^96, a power of two whose exponent is a multiple of three, so undoing
    // them on the result is exact.
    let big = _mm256_cmp_ps::<_CMP_GE_OQ>(abs_x, _mm256_set1_ps(BIG_F32));
    let small = _mm256_cmp_ps::<_CMP_LT_OQ>(abs_x, _mm256_set1_ps(SMALL_F32));
    let s_in = _mm256_blendv_ps(one, _mm256_set1_ps(SCALE_UP_F32), small);
    let s_in = _mm256_blendv_ps(s_in, _mm256_set1_ps(SCALE_DOWN_F32), big);
    let s_out = _mm256_blendv_ps(one, _mm256_set1_ps(UNSCALE_DOWN_F32), small);
    let s_out = _mm256_blendv_ps(s_out, _mm256_set1_ps(UNSCALE_UP_F32), big);
    let a = _mm256_mul_ps(abs_x, s_in);

    // Seed: t = bits(a)/3 + B1, read back as a float. The integer-to-float
    // conversion drops the low bits of the bit pattern, which is worth 1e-5
    // of relative error against a seed that is already only good to 3.3%.
    let a_bits = _mm256_cvtepi32_ps(_mm256_castps_si256(a));
    let q = _mm256_fmadd_ps(a_bits, _mm256_set1_ps(1.0 / 3.0), _mm256_set1_ps(B1_F32));
    let t = _mm256_castsi256_ps(_mm256_cvtps_epi32(q));

    // Halley: t*(2a + t³)/(a + 2t³), whose error cubes each step.
    let r = _mm256_mul_ps(_mm256_mul_ps(t, t), t);
    let num = _mm256_add_ps(_mm256_add_ps(a, a), r);
    let den = _mm256_add_ps(_mm256_add_ps(a, r), r);
    let t = _mm256_mul_ps(t, _mm256_div_ps(num, den));

    let r = _mm256_mul_ps(_mm256_mul_ps(t, t), t);
    let num = _mm256_add_ps(_mm256_add_ps(a, a), r);
    let den = _mm256_add_ps(_mm256_add_ps(a, r), r);
    let t = _mm256_mul_ps(t, _mm256_div_ps(num, den));

    let out = _mm256_or_ps(_mm256_mul_ps(t, s_out), sign);

    // ±0, ±inf and NaN are their own cube roots, and none of them survives the
    // iteration: zero divides zero and infinity subtracts infinity.
    let ordinary = _mm256_and_ps(
        _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, _mm256_setzero_ps()),
        _mm256_cmp_ps::<_CMP_LT_OQ>(abs_x, _mm256_set1_ps(f32::INFINITY)),
    );
    _mm256_blendv_ps(x, out, ordinary)
}

/// Fast SIMD cbrt (cube root) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cbrt_f64(x: __m256d) -> __m256d {
    let sign_mask = _mm256_set1_pd(-0.0);
    let sign = _mm256_and_pd(x, sign_mask);
    let abs_x = _mm256_andnot_pd(sign_mask, x);

    let one_third = _mm256_set1_pd(1.0 / 3.0);

    // Initial guess: cbrt(x) ≈ exp(log(x) / 3)
    let log_x = log_f64(abs_x);
    let guess = exp_f64(_mm256_mul_pd(log_x, one_third));

    let two = _mm256_set1_pd(2.0);
    let three = _mm256_set1_pd(3.0);

    let y = guess;
    let y2 = _mm256_mul_pd(y, y);
    let y_new = _mm256_div_pd(_mm256_fmadd_pd(two, y, _mm256_div_pd(abs_x, y2)), three);

    let y2 = _mm256_mul_pd(y_new, y_new);
    let result = _mm256_div_pd(_mm256_fmadd_pd(two, y_new, _mm256_div_pd(abs_x, y2)), three);

    _mm256_or_pd(result, sign)
}
