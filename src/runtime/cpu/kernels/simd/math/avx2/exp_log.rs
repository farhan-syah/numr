//! AVX2 exponential and logarithm implementations (exp, log, and derived functions)
//!
//! # Safety
//!
//! All functions require AVX2 and FMA CPU features.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::common::{exp_coefficients, log_coefficients};

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

/// Fast SIMD log approximation for f64 using AVX2+FMA
///
/// See `common::_LOG_ALGORITHM_DOC` for algorithm details.
///
/// # Implementation Note
/// Unlike the naive scalar-loop approach, this implementation uses native AVX2
/// 64-bit SIMD operations for exponent extraction. The only scalar operations
/// are for the normalization conditional and final reconstruction, which cannot
/// be avoided due to AVX2's lack of 64-bit comparison and conversion intrinsics.
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log_f64(x: __m256d) -> __m256d {
    use log_coefficients::*;

    let one = _mm256_set1_pd(1.0);
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    let sqrt2_val = std::f64::consts::SQRT_2;

    let c1 = _mm256_set1_pd(C1_F64);
    let c2 = _mm256_set1_pd(C2_F64);
    let c3 = _mm256_set1_pd(C3_F64);
    let c4 = _mm256_set1_pd(C4_F64);
    let c5 = _mm256_set1_pd(C5_F64);
    let c6 = _mm256_set1_pd(C6_F64);
    let c7 = _mm256_set1_pd(C7_F64);
    let c8 = _mm256_set1_pd(C8_F64);
    let c9 = _mm256_set1_pd(C9_F64);

    // Use SIMD for bit manipulation - AVX2 has 64-bit shifts
    let x_bits = _mm256_castpd_si256(x);

    // Extract exponent using 64-bit SIMD shift
    let exp_raw = _mm256_srli_epi64::<52>(x_bits);

    // Extract mantissa and set exponent to bias (so mantissa is in [1, 2))
    let mantissa_mask = _mm256_set1_epi64x(MANTISSA_MASK_F64 as i64);
    let exp_zero = _mm256_set1_epi64x(EXP_ZERO_F64 as i64);
    let m_bits = _mm256_or_si256(_mm256_and_si256(x_bits, mantissa_mask), exp_zero);
    let m_initial = _mm256_castsi256_pd(m_bits);

    // AVX2 lacks 64-bit int comparison and conversion, so we extract for
    // normalization and exponent calculation. The heavy lifting (polynomial
    // evaluation) remains fully vectorized.
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

    let n = _mm256_loadu_pd(n_arr.as_ptr());
    let m = _mm256_loadu_pd(m_arr.as_ptr());

    // f = m - 1 (fully SIMD from here)
    let f = _mm256_sub_pd(m, one);

    // Horner's method for polynomial (fully vectorized)
    let mut poly = c9;
    poly = _mm256_fmadd_pd(poly, f, c8);
    poly = _mm256_fmadd_pd(poly, f, c7);
    poly = _mm256_fmadd_pd(poly, f, c6);
    poly = _mm256_fmadd_pd(poly, f, c5);
    poly = _mm256_fmadd_pd(poly, f, c4);
    poly = _mm256_fmadd_pd(poly, f, c3);
    poly = _mm256_fmadd_pd(poly, f, c2);
    poly = _mm256_fmadd_pd(poly, f, c1);
    poly = _mm256_mul_pd(poly, f);

    // Result = n * ln(2) + log(m) (fully SIMD)
    _mm256_fmadd_pd(n, ln2, poly)
}

// ============================================================================
// Derived exponential/logarithm functions
// ============================================================================

/// Fast SIMD exp2 (2^x) for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f32(x: __m256) -> __m256 {
    // 2^x = e^(x * ln(2))
    let ln2 = _mm256_set1_ps(std::f32::consts::LN_2);
    exp_f32(_mm256_mul_ps(x, ln2))
}

/// Fast SIMD exp2 (2^x) for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn exp2_f64(x: __m256d) -> __m256d {
    let ln2 = _mm256_set1_pd(std::f64::consts::LN_2);
    exp_f64(_mm256_mul_pd(x, ln2))
}

/// Fast SIMD expm1 (e^x - 1) for f32 using AVX2
/// Uses direct computation for |x| > 0.5, Taylor series for small x
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
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
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
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
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f32(x: __m256) -> __m256 {
    // log2(x) = log(x) * log2(e)
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    _mm256_mul_ps(log_f32(x), log2e)
}

/// Fast SIMD log2 for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log2_f64(x: __m256d) -> __m256d {
    let log2e = _mm256_set1_pd(std::f64::consts::LOG2_E);
    _mm256_mul_pd(log_f64(x), log2e)
}

/// Fast SIMD log10 for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f32(x: __m256) -> __m256 {
    // log10(x) = log(x) * log10(e)
    let log10e = _mm256_set1_ps(std::f32::consts::LOG10_E);
    _mm256_mul_ps(log_f32(x), log10e)
}

/// Fast SIMD log10 for f64 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log10_f64(x: __m256d) -> __m256d {
    let log10e = _mm256_set1_pd(std::f64::consts::LOG10_E);
    _mm256_mul_pd(log_f64(x), log10e)
}

/// Fast SIMD log1p (log(1+x)) for f32 using AVX2
///
/// # Safety
/// Requires AVX2 and FMA CPU features.
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
/// # Safety
/// Requires AVX2 and FMA CPU features.
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn log1p_f64(x: __m256d) -> __m256d {
    let one = _mm256_set1_pd(1.0);
    let half = _mm256_set1_pd(0.5);
    let abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);

    let x2 = _mm256_mul_pd(x, x);
    let x3 = _mm256_mul_pd(x2, x);
    let x4 = _mm256_mul_pd(x2, x2);
    let c2 = _mm256_set1_pd(-0.5);
    let c3 = _mm256_set1_pd(1.0 / 3.0);
    let c4 = _mm256_set1_pd(-0.25);
    let taylor = _mm256_fmadd_pd(c4, x4, _mm256_fmadd_pd(c3, x3, _mm256_fmadd_pd(c2, x2, x)));

    let log_result = log_f64(_mm256_add_pd(one, x));
    let mask = _mm256_cmp_pd::<_CMP_GT_OQ>(abs_x, half);
    _mm256_blendv_pd(taylor, log_result, mask)
}
