//! AVX2 mathematical function implementations
//!
//! Provides vectorized transcendental functions using 256-bit registers.
//! All algorithms and coefficients are documented in `common.rs`.
//!
//! # Supported Functions
//!
//! | Function | f32 | f64 | Relative Error |
//! |----------|-----|-----|----------------|
//! | exp      | ✓   | ✓   | < 1e-6 / 1e-12 |
//! | tanh     | ✓   | ✓   | < 1e-6 / 1e-12 |
//! | log      | ✓   | ✓   | < 1e-6 / 2 ulp |
//! | sin      | ✓   | ✓   | < 1e-6 / 1e-10 |
//! | cos      | ✓   | ✓   | < 1e-6 / 1e-10 |
//! | tan      | ✓   | ✓   | < 2e-4 / 1e-4  |
//! | atan     | ✓   | ✓   | see note / 2 ulp |
//! | asin     | ✓   | ✓   | see note / 2 ulp |
//! | acos     | ✓   | ✓   | see note / 2 ulp |
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
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::common::{
    asin_coefficients, atan_coefficients, exp_coefficients, log_coefficients, tan_coefficients,
    trig_coefficients,
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

    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);

    let c0 = _mm256_set1_ps(C0_F32);
    let c1 = _mm256_set1_ps(C1_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c3 = _mm256_set1_ps(C3_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c5 = _mm256_set1_ps(C5_F32);
    let c6 = _mm256_set1_ps(C6_F32);

    // Clamp input to avoid overflow/underflow
    let x = _mm256_max_ps(x, _mm256_set1_ps(MIN_F32));
    let x = _mm256_min_ps(x, _mm256_set1_ps(MAX_F32));

    // y = x * log2(e)
    let y = _mm256_mul_ps(x, log2e);

    // n = round(y) - integer part
    let n = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // f = y - n - fractional part in [-0.5, 0.5]
    let f = _mm256_sub_ps(y, n);

    // r = f * ln(2) - convert back to natural log scale
    let r = _mm256_mul_ps(f, ln2);

    // Polynomial approximation using Horner's method
    let r2 = _mm256_mul_ps(r, r);
    let r3 = _mm256_mul_ps(r2, r);
    let r4 = _mm256_mul_ps(r2, r2);
    let r5 = _mm256_mul_ps(r4, r);
    let r6 = _mm256_mul_ps(r4, r2);

    let mut poly = c0;
    poly = _mm256_fmadd_ps(c1, r, poly);
    poly = _mm256_fmadd_ps(c2, r2, poly);
    poly = _mm256_fmadd_ps(c3, r3, poly);
    poly = _mm256_fmadd_ps(c4, r4, poly);
    poly = _mm256_fmadd_ps(c5, r5, poly);
    poly = _mm256_fmadd_ps(c6, r6, poly);

    // Compute 2^n using IEEE 754 bit manipulation
    // 2^n = reinterpret((n + 127) << 23) for f32
    let n_i32 = _mm256_cvtps_epi32(n);
    let bias = _mm256_set1_epi32(127);
    let exp_bits = _mm256_slli_epi32::<23>(_mm256_add_epi32(n_i32, bias));
    let pow2n = _mm256_castsi256_ps(exp_bits);

    // Result = 2^n * exp(r)
    _mm256_mul_ps(pow2n, poly)
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

    // Clamp input
    let x = _mm256_max_pd(x, _mm256_set1_pd(MIN_F64));
    let x = _mm256_min_pd(x, _mm256_set1_pd(MAX_F64));

    let y = _mm256_mul_pd(x, log2e);
    let n = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(y);

    // Cody-Waite reduction: r = x - n*ln2, split so that n*LN2_HI_F64 is exact
    let r = _mm256_fnmadd_pd(n, ln2_hi, x);
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

    // AVX2 lacks _mm256_cvtpd_epi64, use scalar conversion for 2^n
    // This is a known AVX2 limitation - polynomial eval is still SIMD
    let mut result = [0.0f64; 4];
    let mut n_arr = [0.0f64; 4];
    let mut poly_arr = [0.0f64; 4];

    _mm256_storeu_pd(n_arr.as_mut_ptr(), n);
    _mm256_storeu_pd(poly_arr.as_mut_ptr(), poly);

    for i in 0..4 {
        let n_i = n_arr[i] as i64;
        let exp_bits = ((n_i + 1023) as u64) << 52;
        let pow2n = f64::from_bits(exp_bits);
        result[i] = pow2n * poly_arr[i];
    }

    _mm256_loadu_pd(result.as_ptr())
}

// ============================================================================
// Hyperbolic tangent: tanh(x)
// ============================================================================

/// Fast SIMD tanh approximation for f32 using AVX2+FMA
///
/// Algorithm: tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f32(x: __m256) -> __m256 {
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);

    let exp2x = exp_f32(_mm256_mul_ps(two, x));
    let num = _mm256_sub_ps(exp2x, one);
    let den = _mm256_add_ps(exp2x, one);

    _mm256_div_ps(num, den)
}

/// Fast SIMD tanh approximation for f64 using AVX2+FMA
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tanh_f64(x: __m256d) -> __m256d {
    let two = _mm256_set1_pd(2.0);
    let one = _mm256_set1_pd(1.0);

    let exp2x = exp_f64(_mm256_mul_pd(two, x));
    let num = _mm256_sub_pd(exp2x, one);
    let den = _mm256_add_pd(exp2x, one);

    _mm256_div_pd(num, den)
}

// ============================================================================
// Natural logarithm: log(x)
// ============================================================================

/// Fast SIMD log approximation for f32 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f32(x: __m256) -> __m256 {
    use log_coefficients::*;

    let one = _mm256_set1_ps(1.0);
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    let sqrt2 = _mm256_set1_ps(std::f32::consts::SQRT_2);
    let half = _mm256_set1_ps(0.5);

    let c1 = _mm256_set1_ps(C1_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c3 = _mm256_set1_ps(C3_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c5 = _mm256_set1_ps(C5_F32);
    let c6 = _mm256_set1_ps(C6_F32);
    let c7 = _mm256_set1_ps(C7_F32);

    // Extract exponent: reinterpret as int, shift right by 23, subtract bias
    let x_bits = _mm256_castps_si256(x);
    let exp_raw = _mm256_srli_epi32::<23>(x_bits);
    let exp_unbiased = _mm256_sub_epi32(exp_raw, _mm256_set1_epi32(EXP_BIAS_F32));
    let mut n = _mm256_cvtepi32_ps(exp_unbiased);

    // Extract mantissa and set exponent to 0 (so mantissa is in [1, 2))
    let mantissa_mask = _mm256_set1_epi32(MANTISSA_MASK_F32);
    let exp_zero = _mm256_set1_epi32(EXP_ZERO_F32);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let mut m = _mm256_castsi256_ps(m_bits);

    // Normalize: if m > sqrt(2), divide by 2 and increment exponent
    // This keeps f in [-0.2929, 0.4142] for better polynomial accuracy
    let need_adjust = _mm256_cmp_ps::<_CMP_GT_OQ>(m, sqrt2);
    m = _mm256_blendv_ps(m, _mm256_mul_ps(m, half), need_adjust);
    n = _mm256_blendv_ps(n, _mm256_add_ps(n, one), need_adjust);

    // f = m - 1, so log(m) = log(1 + f), f is now in [-0.2929, 0.4142]
    let f = _mm256_sub_ps(m, one);

    // Horner's method: ((((((c7*f + c6)*f + c5)*f + c4)*f + c3)*f + c2)*f + c1)*f
    let mut poly = c7;
    poly = _mm256_fmadd_ps(poly, f, c6);
    poly = _mm256_fmadd_ps(poly, f, c5);
    poly = _mm256_fmadd_ps(poly, f, c4);
    poly = _mm256_fmadd_ps(poly, f, c3);
    poly = _mm256_fmadd_ps(poly, f, c2);
    poly = _mm256_fmadd_ps(poly, f, c1);
    poly = _mm256_mul_ps(poly, f);

    // Result = n * ln(2) + log(m)
    _mm256_fmadd_ps(n, ln2, poly)
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

/// Fast SIMD sin approximation for f32 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f32(x: __m256) -> __m256 {
    use trig_coefficients::*;

    let two_over_pi = _mm256_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    let s1 = _mm256_set1_ps(S1_F32);
    let s3 = _mm256_set1_ps(S3_F32);
    let s5 = _mm256_set1_ps(S5_F32);
    let s7 = _mm256_set1_ps(S7_F32);

    let c0 = _mm256_set1_ps(C0_F32);
    let c2 = _mm256_set1_ps(C2_F32);
    let c4 = _mm256_set1_ps(C4_F32);
    let c6 = _mm256_set1_ps(C6_F32);

    // Range reduction: j = round(x * 2/π), y = x - j * π/2
    let j = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        two_over_pi,
    ));
    let j_int = _mm256_cvtps_epi32(j);

    let y = _mm256_fnmadd_ps(j, pi_over_2, x);

    let y2 = _mm256_mul_ps(y, y);
    let y3 = _mm256_mul_ps(y2, y);
    let y4 = _mm256_mul_ps(y2, y2);
    let y5 = _mm256_mul_ps(y4, y);
    let y6 = _mm256_mul_ps(y4, y2);
    let y7 = _mm256_mul_ps(y4, y3);

    // sin(y) polynomial
    let sin_y = _mm256_fmadd_ps(
        s7,
        y7,
        _mm256_fmadd_ps(s5, y5, _mm256_fmadd_ps(s3, y3, _mm256_mul_ps(s1, y))),
    );

    // cos(y) polynomial
    let cos_y = _mm256_fmadd_ps(c6, y6, _mm256_fmadd_ps(c4, y4, _mm256_fmadd_ps(c2, y2, c0)));

    // Select sin or cos based on j mod 4
    // j mod 4 = 0: sin(y), 1: cos(y), 2: -sin(y), 3: -cos(y)
    let j_mod_4 = _mm256_and_si256(j_int, _mm256_set1_epi32(3));

    // Use cos when j mod 4 is 1 or 3
    let use_cos_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    );
    let use_cos_mask = _mm256_castsi256_ps(use_cos_mask);

    // Negate when j mod 4 is 2 or 3
    let negate_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_mod_4, _mm256_set1_epi32(2)),
        _mm256_set1_epi32(2),
    );
    let negate_mask = _mm256_castsi256_ps(negate_mask);
    let sign_bit = _mm256_set1_ps(-0.0); // Just the sign bit

    let result = _mm256_blendv_ps(sin_y, cos_y, use_cos_mask);
    let negated = _mm256_xor_ps(result, sign_bit);
    _mm256_blendv_ps(result, negated, negate_mask)
}

/// Fast SIMD sin approximation for f64 using AVX2+FMA
///
/// See `common::_TRIG_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sin_f64(x: __m256d) -> __m256d {
    use trig_coefficients::*;

    let two_over_pi = _mm256_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    let s1 = _mm256_set1_pd(S1_F64);
    let s3 = _mm256_set1_pd(S3_F64);
    let s5 = _mm256_set1_pd(S5_F64);
    let s7 = _mm256_set1_pd(S7_F64);
    let s9 = _mm256_set1_pd(S9_F64);

    let c0 = _mm256_set1_pd(C0_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c6 = _mm256_set1_pd(C6_F64);
    let c8 = _mm256_set1_pd(C8_F64);

    let j = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        two_over_pi,
    ));

    // Get j as integers for quadrant selection (AVX2 lacks 64-bit int conversion)
    let mut j_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    let j_int: [i32; 4] = [
        j_arr[0] as i32,
        j_arr[1] as i32,
        j_arr[2] as i32,
        j_arr[3] as i32,
    ];

    let y = _mm256_fnmadd_pd(j, pi_over_2, x);

    let y2 = _mm256_mul_pd(y, y);
    let y3 = _mm256_mul_pd(y2, y);
    let y4 = _mm256_mul_pd(y2, y2);
    let y5 = _mm256_mul_pd(y4, y);
    let y6 = _mm256_mul_pd(y4, y2);
    let y7 = _mm256_mul_pd(y4, y3);
    let y8 = _mm256_mul_pd(y4, y4);
    let y9 = _mm256_mul_pd(y8, y);

    // sin(y) and cos(y) polynomials
    let mut sin_y = _mm256_mul_pd(s1, y);
    sin_y = _mm256_fmadd_pd(s3, y3, sin_y);
    sin_y = _mm256_fmadd_pd(s5, y5, sin_y);
    sin_y = _mm256_fmadd_pd(s7, y7, sin_y);
    sin_y = _mm256_fmadd_pd(s9, y9, sin_y);

    let mut cos_y = c0;
    cos_y = _mm256_fmadd_pd(c2, y2, cos_y);
    cos_y = _mm256_fmadd_pd(c4, y4, cos_y);
    cos_y = _mm256_fmadd_pd(c6, y6, cos_y);
    cos_y = _mm256_fmadd_pd(c8, y8, cos_y);

    // Compute result per-element based on quadrant
    let mut sin_arr = [0.0f64; 4];
    let mut cos_arr = [0.0f64; 4];
    _mm256_storeu_pd(sin_arr.as_mut_ptr(), sin_y);
    _mm256_storeu_pd(cos_arr.as_mut_ptr(), cos_y);

    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let quadrant = j_int[i] & 3;
        result[i] = match quadrant {
            0 => sin_arr[i],
            1 => cos_arr[i],
            2 => -sin_arr[i],
            3 => -cos_arr[i],
            _ => unreachable!(),
        };
    }

    _mm256_loadu_pd(result.as_ptr())
}

/// Fast SIMD cos approximation for f32 using AVX2+FMA
///
/// Implemented as: cos(x) = sin(x + π/2)
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f32(x: __m256) -> __m256 {
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    sin_f32(_mm256_add_ps(x, pi_over_2))
}

/// Fast SIMD cos approximation for f64 using AVX2+FMA
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cos_f64(x: __m256d) -> __m256d {
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);
    sin_f64(_mm256_add_pd(x, pi_over_2))
}

/// Fast SIMD tan approximation for f32 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f32(x: __m256) -> __m256 {
    use tan_coefficients::*;

    let two_over_pi = _mm256_set1_ps(std::f32::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    // Range reduction
    let j = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_ps(
        x,
        two_over_pi,
    ));
    let y = _mm256_fnmadd_ps(j, pi_over_2, x);

    let t1 = _mm256_set1_ps(T1_F32);
    let t3 = _mm256_set1_ps(T3_F32);
    let t5 = _mm256_set1_ps(T5_F32);
    let t7 = _mm256_set1_ps(T7_F32);
    let t9 = _mm256_set1_ps(T9_F32);
    let t11 = _mm256_set1_ps(T11_F32);

    let y2 = _mm256_mul_ps(y, y);

    // Horner's method: tan(y) ≈ y * (1 + y²*(t3 + y²*(t5 + y²*(t7 + y²*(t9 + y²*t11)))))
    let mut poly = t11;
    poly = _mm256_fmadd_ps(poly, y2, t9);
    poly = _mm256_fmadd_ps(poly, y2, t7);
    poly = _mm256_fmadd_ps(poly, y2, t5);
    poly = _mm256_fmadd_ps(poly, y2, t3);
    poly = _mm256_fmadd_ps(poly, y2, t1);
    let tan_y = _mm256_mul_ps(y, poly);

    // For quadrants 1 and 3, tan(y + π/2) = -1/tan(y) = -cot(y)
    let j_int = _mm256_cvtps_epi32(j);
    let use_cot_mask = _mm256_cmpeq_epi32(
        _mm256_and_si256(j_int, _mm256_set1_epi32(1)),
        _mm256_set1_epi32(1),
    );
    let use_cot_mask = _mm256_castsi256_ps(use_cot_mask);

    let neg_one = _mm256_set1_ps(-1.0);
    let cot_y = _mm256_div_ps(neg_one, tan_y);

    _mm256_blendv_ps(tan_y, cot_y, use_cot_mask)
}

/// Fast SIMD tan approximation for f64 using AVX2+FMA
///
/// See `common::_TAN_ALGORITHM_DOC` for algorithm details.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn tan_f64(x: __m256d) -> __m256d {
    use tan_coefficients::*;

    let two_over_pi = _mm256_set1_pd(std::f64::consts::FRAC_2_PI);
    let pi_over_2 = _mm256_set1_pd(std::f64::consts::FRAC_PI_2);

    let j = _mm256_round_pd::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(_mm256_mul_pd(
        x,
        two_over_pi,
    ));
    let y = _mm256_fnmadd_pd(j, pi_over_2, x);

    let t1 = _mm256_set1_pd(T1_F64);
    let t3 = _mm256_set1_pd(T3_F64);
    let t5 = _mm256_set1_pd(T5_F64);
    let t7 = _mm256_set1_pd(T7_F64);
    let t9 = _mm256_set1_pd(T9_F64);
    let t11 = _mm256_set1_pd(T11_F64);
    let t13 = _mm256_set1_pd(T13_F64);

    let y2 = _mm256_mul_pd(y, y);

    // Horner's method
    let mut poly = t13;
    poly = _mm256_fmadd_pd(poly, y2, t11);
    poly = _mm256_fmadd_pd(poly, y2, t9);
    poly = _mm256_fmadd_pd(poly, y2, t7);
    poly = _mm256_fmadd_pd(poly, y2, t5);
    poly = _mm256_fmadd_pd(poly, y2, t3);
    poly = _mm256_fmadd_pd(poly, y2, t1);
    let tan_y = _mm256_mul_pd(y, poly);

    // Handle quadrant for cotangent (AVX2 lacks 64-bit int comparison)
    let mut j_arr = [0.0f64; 4];
    let mut tan_arr = [0.0f64; 4];
    _mm256_storeu_pd(j_arr.as_mut_ptr(), j);
    _mm256_storeu_pd(tan_arr.as_mut_ptr(), tan_y);

    let mut result = [0.0f64; 4];
    for i in 0..4 {
        let j_int = j_arr[i] as i32;
        result[i] = if (j_int & 1) == 1 {
            -1.0 / tan_arr[i]
        } else {
            tan_arr[i]
        };
    }

    _mm256_loadu_pd(result.as_ptr())
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
    let pi_over_2 = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);

    // Save sign and work with absolute value
    let sign_mask = _mm256_set1_ps(-0.0); // 0x80000000
    let sign = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);

    // Range reduction: for |x| > 1, compute atan(1/x) then adjust
    let need_recip = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, one);
    let recip_x = _mm256_div_ps(one, abs_x);
    let y = _mm256_blendv_ps(abs_x, recip_x, need_recip);

    // Polynomial approximation for atan(y) where y in [0, 1]
    let a0 = _mm256_set1_ps(A0_F32);
    let a2 = _mm256_set1_ps(A2_F32);
    let a4 = _mm256_set1_ps(A4_F32);
    let a6 = _mm256_set1_ps(A6_F32);
    let a8 = _mm256_set1_ps(A8_F32);
    let a10 = _mm256_set1_ps(A10_F32);
    let a12 = _mm256_set1_ps(A12_F32);

    let y2 = _mm256_mul_ps(y, y);

    // Horner's method: a0 + y²*(a2 + y²*(a4 + y²*(a6 + y²*(a8 + y²*(a10 + y²*a12)))))
    let mut poly = a12;
    poly = _mm256_fmadd_ps(poly, y2, a10);
    poly = _mm256_fmadd_ps(poly, y2, a8);
    poly = _mm256_fmadd_ps(poly, y2, a6);
    poly = _mm256_fmadd_ps(poly, y2, a4);
    poly = _mm256_fmadd_ps(poly, y2, a2);
    poly = _mm256_fmadd_ps(poly, y2, a0);
    let atan_y = _mm256_mul_ps(y, poly);

    // Apply range reduction inverse: if |x| > 1, result = π/2 - atan(1/x)
    let adjusted = _mm256_sub_ps(pi_over_2, atan_y);
    let result = _mm256_blendv_ps(atan_y, adjusted, need_recip);

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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f32(x: __m256) -> __m256 {
    // 2^x = e^(x * ln(2))
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    exp_f32(_mm256_mul_ps(x, ln2))
}

/// Fast SIMD exp2 (2^x) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f64(x: __m256d) -> __m256d {
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    exp_f64(_mm256_mul_pd(x, ln2))
}

/// Fast SIMD expm1 (e^x - 1) for f32 using AVX2
/// Uses direct computation for |x| > 0.5, Taylor series for small x
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // For small |x|, use Taylor series: x + x^2/2 + x^3/6 + x^4/24
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x2, x2);
    let c2 = _mm256_set1_ps(0.5);
    let c3 = _mm256_set1_ps(1.0 / 6.0);
    let c4 = _mm256_set1_ps(1.0 / 24.0);
    let taylor = _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c3, x3, _mm256_fmadd_ps(c2, x2, x)));

    // For large |x|, use exp(x) - 1
    let exp_result = _mm256_sub_ps(exp_f32(x), one);

    // Blend based on |x| > 0.5
    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_ps(taylor, exp_result, mask)
}

/// Fast SIMD expm1 (e^x - 1) for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn expm1_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(0.5);
    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let c2 = _mm256_set1_pd(0.5);
    let c3 = _mm256_set1_pd(1.0 / 6.0);
    let c4 = _mm256_set1_pd(1.0 / 24.0);
    let taylor = _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)));

    let exp_result = _mm256_sub_pd(exp_f64(x), one);
    let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_pd(taylor, exp_result, mask)
}

/// Fast SIMD log2 for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f32(x: __m256) -> __m256 {
    // log2(x) = log(x) * log2(e)
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    _mm256_mul_ps(log_f32(x), log2e)
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f32(x: __m256) -> __m256 {
    // log10(x) = log(x) * log10(e)
    let log10e = _mm256_set1_ps(std::f32::consts::LOG10_E);
    _mm256_mul_ps(log_f32(x), log10e)
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0), x);

    // For small |x|, use Taylor series: x - x^2/2 + x^3/3 - x^4/4
    let x2 = _mm256_mul_ps(x, x);
    let x3 = _mm256_mul_ps(x2, x);
    let x4 = _mm256_mul_ps(x2, x2);
    let c2 = _mm256_set1_ps(-0.5);
    let c3 = _mm256_set1_ps(1.0 / 3.0);
    let c4 = _mm256_set1_ps(-0.25);
    let taylor = _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c3, x3, _mm256_fmadd_ps(c2, x2, x)));

    // For large |x|, use log(1 + x)
    let log_result = log_f32(_mm256_add_ps(one, x));

    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_ps(taylor, log_result, mask)
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
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f32(x: __m256) -> __m256 {
    // sinh(x) = (exp(x) - exp(-x)) / 2
    let half = _mm256_set1_ps(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_mul_ps(half, _mm256_sub_ps(exp_x, exp_neg_x))
}

/// Fast SIMD sinh for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn sinh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_sub_pd(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f32 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f32(x: __m256) -> __m256 {
    // cosh(x) = (exp(x) + exp(-x)) / 2
    let half = _mm256_set1_ps(0.5);
    let exp_x = exp_f32(x);
    let exp_neg_x = exp_f32(_mm256_sub_ps(_mm256_setzero_ps(), x));
    _mm256_mul_ps(half, _mm256_add_ps(exp_x, exp_neg_x))
}

/// Fast SIMD cosh for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let exp_x = exp_f64(x);
    let exp_neg_x = exp_f64(_mm256_sub_pd(_mm256_setzero_pd(), x));
    _mm256_mul_pd(half, _mm256_add_pd(exp_x, exp_neg_x))
}

/// Fast SIMD asinh for f32 using AVX2
/// asinh(x) = log(x + sqrt(x^2 + 1))
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_add_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD asinh for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asinh_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let sqrt_term = _mm256_sqrt_pd(_mm256_add_pd(x2, one));
    log_f64(_mm256_add_pd(x, sqrt_term))
}

/// Fast SIMD acosh for f32 using AVX2
/// acosh(x) = log(x + sqrt(x^2 - 1)) for x >= 1
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f32(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(x2, one));
    log_f32(_mm256_add_ps(x, sqrt_term))
}

/// Fast SIMD acosh for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acosh_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let x2 = _mm256_mul_pd(x, x);
    let sqrt_term = _mm256_sqrt_pd(_mm256_sub_pd(x2, one));
    log_f64(_mm256_add_pd(x, sqrt_term))
}

/// Fast SIMD atanh for f32 using AVX2
/// atanh(x) = 0.5 * log((1 + x) / (1 - x)) for |x| < 1
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f32(x: __m256) -> __m256 {
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let one_plus_x = _mm256_add_ps(one, x);
    let one_minus_x = _mm256_sub_ps(one, x);
    let ratio = _mm256_div_ps(one_plus_x, one_minus_x);
    _mm256_mul_ps(half, log_f32(ratio))
}

/// Fast SIMD atanh for f64 using AVX2
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn atanh_f64(x: __m256d) -> __m256d {
    let half = _mm256_set1_pd(0.5);
    let one = _mm256_set1_pd(1.0);
    let one_plus_x = _mm256_add_pd(one, x);
    let one_minus_x = _mm256_sub_pd(one, x);
    let ratio = _mm256_div_pd(one_plus_x, one_minus_x);
    _mm256_mul_pd(half, log_f64(ratio))
}

/// Fast SIMD asin for f32 using AVX2
/// Uses polynomial approximation with range reduction
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn asin_f32(x: __m256) -> __m256 {
    // asin(x) = atan(x / sqrt(1 - x^2))
    let one = _mm256_set1_ps(1.0);
    let x2 = _mm256_mul_ps(x, x);
    let sqrt_term = _mm256_sqrt_ps(_mm256_sub_ps(one, x2));
    let ratio = _mm256_div_ps(x, sqrt_term);
    atan_f32(ratio)
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
/// acos(x) = pi/2 - asin(x)
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn acos_f32(x: __m256) -> __m256 {
    let pi_half = _mm256_set1_ps(std::f32::consts::FRAC_PI_2);
    _mm256_sub_ps(pi_half, asin_f32(x))
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
/// Uses Halley's method for refinement
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cbrt_f32(x: __m256) -> __m256 {
    // Handle sign separately
    let sign_mask = _mm256_set1_ps(-0.0);
    let sign = _mm256_and_ps(x, sign_mask);
    let abs_x = _mm256_andnot_ps(sign_mask, x);

    // Initial approximation using bit manipulation
    // cbrt(x) ≈ 2^(log2(x)/3) via IEEE 754
    let one_third = _mm256_set1_ps(1.0 / 3.0);
    let bias = _mm256_set1_ps(127.0);

    // Extract exponent: e = floor(log2(|x|))
    let xi = _mm256_castps_si256(abs_x);
    let exp_bits = _mm256_srli_epi32::<23>(xi);
    let exp_f = _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

    // Initial guess: 2^(e/3)
    let new_exp = _mm256_mul_ps(exp_f, one_third);
    let new_exp_i = _mm256_cvtps_epi32(_mm256_add_ps(new_exp, bias));
    let guess = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(new_exp_i));

    // Newton-Raphson iteration: y = y * (2*y^3 + x) / (2*x + y^3)
    // Simplified: y = (2*y + x/y^2) / 3
    let two = _mm256_set1_ps(2.0);
    let three = _mm256_set1_ps(3.0);

    let y = guess;
    let y2 = _mm256_mul_ps(y, y);
    let y_new = _mm256_div_ps(_mm256_fmadd_ps(two, y, _mm256_div_ps(abs_x, y2)), three);

    // One more iteration
    let y2 = _mm256_mul_ps(y_new, y_new);
    let result = _mm256_div_ps(_mm256_fmadd_ps(two, y_new, _mm256_div_ps(abs_x, y2)), three);

    // Restore sign
    _mm256_or_ps(result, sign)
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
