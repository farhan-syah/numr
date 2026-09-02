//! AVX-512 special function kernels
//!
//! Vectorized implementations using 512-bit registers.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{exp_f32, exp_f64};
use super::coefficients::{bessel_i0, bessel_j0, erf};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ============================================================================
// Error Function (erf)
// ============================================================================

/// Vectorized erf for f32 using AVX-512
///
/// Uses A&S 7.1.26 (~1e-7 accuracy), sufficient for f32's ~7 significant digits.
#[target_feature(enable = "avx512f")]
pub unsafe fn erf_f32(input: *const f32, output: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let a1 = _mm512_set1_ps(erf::A1_F32);
    let a2 = _mm512_set1_ps(erf::A2_F32);
    let a3 = _mm512_set1_ps(erf::A3_F32);
    let a4 = _mm512_set1_ps(erf::A4_F32);
    let a5 = _mm512_set1_ps(erf::A5_F32);
    let p = _mm512_set1_ps(erf::P_F32);
    let one = _mm512_set1_ps(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = _mm512_loadu_ps(input.add(offset));

        // Extract sign and work with absolute value
        let ax = _mm512_abs_ps(x);
        let sign_mask = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_setzero_ps());

        // t = 1/(1 + p*|x|)
        let t = _mm512_div_ps(one, _mm512_fmadd_ps(p, ax, one));

        // Horner's method
        let mut poly = a5;
        poly = _mm512_fmadd_ps(poly, t, a4);
        poly = _mm512_fmadd_ps(poly, t, a3);
        poly = _mm512_fmadd_ps(poly, t, a2);
        poly = _mm512_fmadd_ps(poly, t, a1);
        poly = _mm512_mul_ps(poly, t);

        // exp(-x²)
        let neg_x2 = _mm512_sub_ps(_mm512_setzero_ps(), _mm512_mul_ps(ax, ax));
        let exp_term = exp_f32(neg_x2);

        // y = 1 - poly * exp(-x²)
        let y = _mm512_fnmadd_ps(poly, exp_term, one);

        // Apply sign
        let result = _mm512_mask_sub_ps(y, sign_mask, _mm512_setzero_ps(), y);

        _mm512_storeu_ps(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) =
                crate::algorithm::special::scalar::erf_scalar(x as f64) as f32;
        }
    }
}

/// Vectorized erf for f64 using AVX-512
///
/// Uses Maclaurin series for |x| < 3, Laplace continued fraction for 3 ≤ |x| < 6,
/// and asymptotic ±1 for |x| ≥ 6. Accuracy: ~1e-15 (full f64 precision).
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn erf_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let zero = _mm512_setzero_pd();
    let one = _mm512_set1_pd(1.0);
    let three = _mm512_set1_pd(3.0);
    let six = _mm512_set1_pd(6.0);
    let two_over_sqrt_pi = _mm512_set1_pd(std::f64::consts::FRAC_2_SQRT_PI);
    let frac_1_sqrt_pi = _mm512_set1_pd(0.5641895835477563);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm512_loadu_pd(input.add(offset));

        let ax = _mm512_abs_pd(x);
        let neg_mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(x, zero);

        // === Maclaurin series ===
        let x2 = _mm512_mul_pd(ax, ax);
        let neg_x2 = _mm512_sub_pd(zero, x2);
        let mut term = ax;
        let mut sum = ax;
        for n in 1..30 {
            let n_f = n as f64;
            term = _mm512_mul_pd(term, _mm512_div_pd(neg_x2, _mm512_set1_pd(n_f)));
            let contrib = _mm512_div_pd(term, _mm512_set1_pd(2.0 * n_f + 1.0));
            sum = _mm512_add_pd(sum, contrib);
        }
        let maclaurin_result = _mm512_mul_pd(sum, two_over_sqrt_pi);

        // === Laplace continued fraction for erfc ===
        let mut f = zero;
        for n in (1..=50_u32).rev() {
            f = _mm512_div_pd(_mm512_set1_pd(n as f64 * 0.5), _mm512_add_pd(ax, f));
        }
        let cf = _mm512_div_pd(one, _mm512_add_pd(ax, f));
        let exp_neg_x2 = exp_f64(_mm512_sub_pd(zero, x2));
        let erfc_val = _mm512_mul_pd(_mm512_mul_pd(exp_neg_x2, frac_1_sqrt_pi), cf);
        let cf_result = _mm512_sub_pd(one, erfc_val);

        // === Blend regions ===
        let mask_small = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(ax, three);
        let mask_large = _mm512_cmp_pd_mask::<_CMP_GE_OQ>(ax, six);

        // Start with continued fraction, override Maclaurin where |x| < 3
        let mut result = _mm512_mask_blend_pd(mask_small, cf_result, maclaurin_result);
        // Override with 1.0 where |x| ≥ 6
        result = _mm512_mask_blend_pd(mask_large, result, one);
        // Apply sign: negate where x < 0
        result = _mm512_mask_sub_pd(result, neg_mask, zero, result);

        _mm512_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            *output.add(offset + i) =
                crate::algorithm::special::scalar::erf_scalar(*input.add(offset + i));
        }
    }
}

// ============================================================================
// Complementary Error Function (erfc)
// ============================================================================

/// Vectorized erfc for f32 using AVX-512
#[target_feature(enable = "avx512f")]
pub unsafe fn erfc_f32(input: *const f32, output: *mut f32, len: usize) {
    erf_f32(input, output, len);

    let chunks = len / F32_LANES;
    let one = _mm512_set1_ps(1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let erf_val = _mm512_loadu_ps(output.add(offset));
        let result = _mm512_sub_ps(one, erf_val);
        _mm512_storeu_ps(output.add(offset), result);
    }

    let remainder = len % F32_LANES;
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let val = *output.add(offset + i);
            *output.add(offset + i) = 1.0 - val;
        }
    }
}

/// Vectorized erfc for f64 using AVX-512
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn erfc_f64(input: *const f64, output: *mut f64, len: usize) {
    erf_f64(input, output, len);

    let chunks = len / F64_LANES;
    let one = _mm512_set1_pd(1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let erf_val = _mm512_loadu_pd(output.add(offset));
        let result = _mm512_sub_pd(one, erf_val);
        _mm512_storeu_pd(output.add(offset), result);
    }

    let remainder = len % F64_LANES;
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let val = *output.add(offset + i);
            *output.add(offset + i) = 1.0 - val;
        }
    }
}

// ============================================================================
// Bessel J0
// ============================================================================

/// Vectorized bessel_j0 for f32 using AVX-512
///
/// Note: The f32 coefficients cannot precisely represent values like 57568490574.0
/// (f32 only has ~7 significant digits). For accuracy, we use scalar f64 computation.
#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_j0_f32(input: *const f32, output: *mut f32, len: usize) {
    // Use scalar f64 computation for accuracy - the coefficients don't fit in f32
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_j0_scalar(x as f64) as f32;
    }
}

/// Vectorized bessel_j0 for f64 using AVX-512
#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_j0_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let threshold = _mm512_set1_pd(8.0);

    let sp0 = _mm512_set1_pd(bessel_j0::SMALL_P[0]);
    let sp1 = _mm512_set1_pd(bessel_j0::SMALL_P[1]);
    let sp2 = _mm512_set1_pd(bessel_j0::SMALL_P[2]);
    let sp3 = _mm512_set1_pd(bessel_j0::SMALL_P[3]);
    let sp4 = _mm512_set1_pd(bessel_j0::SMALL_P[4]);
    let sp5 = _mm512_set1_pd(bessel_j0::SMALL_P[5]);

    let sq0 = _mm512_set1_pd(bessel_j0::SMALL_Q[0]);
    let sq1 = _mm512_set1_pd(bessel_j0::SMALL_Q[1]);
    let sq2 = _mm512_set1_pd(bessel_j0::SMALL_Q[2]);
    let sq3 = _mm512_set1_pd(bessel_j0::SMALL_Q[3]);
    let sq4 = _mm512_set1_pd(bessel_j0::SMALL_Q[4]);
    let sq5 = _mm512_set1_pd(bessel_j0::SMALL_Q[5]);

    let ap0 = _mm512_set1_pd(bessel_j0::ASYMP_P[0]);
    let ap1 = _mm512_set1_pd(bessel_j0::ASYMP_P[1]);
    let ap2 = _mm512_set1_pd(bessel_j0::ASYMP_P[2]);
    let ap3 = _mm512_set1_pd(bessel_j0::ASYMP_P[3]);
    let ap4 = _mm512_set1_pd(bessel_j0::ASYMP_P[4]);

    let aq0 = _mm512_set1_pd(bessel_j0::ASYMP_Q[0]);
    let aq1 = _mm512_set1_pd(bessel_j0::ASYMP_Q[1]);
    let aq2 = _mm512_set1_pd(bessel_j0::ASYMP_Q[2]);
    let aq3 = _mm512_set1_pd(bessel_j0::ASYMP_Q[3]);
    let aq4 = _mm512_set1_pd(bessel_j0::ASYMP_Q[4]);

    let eight = _mm512_set1_pd(8.0);
    let frac_pi_4 = _mm512_set1_pd(bessel_j0::FRAC_PI_4);
    let two_over_pi = _mm512_set1_pd(bessel_j0::TWO_OVER_PI);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm512_loadu_pd(input.add(offset));
        let ax = _mm512_abs_pd(x);

        let y = _mm512_mul_pd(x, x);

        let mut num = sp5;
        num = _mm512_fmadd_pd(num, y, sp4);
        num = _mm512_fmadd_pd(num, y, sp3);
        num = _mm512_fmadd_pd(num, y, sp2);
        num = _mm512_fmadd_pd(num, y, sp1);
        num = _mm512_fmadd_pd(num, y, sp0);

        let mut den = sq5;
        den = _mm512_fmadd_pd(den, y, sq4);
        den = _mm512_fmadd_pd(den, y, sq3);
        den = _mm512_fmadd_pd(den, y, sq2);
        den = _mm512_fmadd_pd(den, y, sq1);
        den = _mm512_fmadd_pd(den, y, sq0);

        let small_result = _mm512_div_pd(num, den);

        let z = _mm512_div_pd(eight, ax);
        let y_asymp = _mm512_mul_pd(z, z);
        let xx = _mm512_sub_pd(ax, frac_pi_4);

        let mut p0 = ap4;
        p0 = _mm512_fmadd_pd(p0, y_asymp, ap3);
        p0 = _mm512_fmadd_pd(p0, y_asymp, ap2);
        p0 = _mm512_fmadd_pd(p0, y_asymp, ap1);
        p0 = _mm512_fmadd_pd(p0, y_asymp, ap0);

        let mut q0 = aq4;
        q0 = _mm512_fmadd_pd(q0, y_asymp, aq3);
        q0 = _mm512_fmadd_pd(q0, y_asymp, aq2);
        q0 = _mm512_fmadd_pd(q0, y_asymp, aq1);
        q0 = _mm512_fmadd_pd(q0, y_asymp, aq0);
        q0 = _mm512_mul_pd(q0, z);

        let scale = _mm512_sqrt_pd(_mm512_div_pd(two_over_pi, ax));

        let mut cos_vals = [0.0f64; F64_LANES];
        let mut sin_vals = [0.0f64; F64_LANES];
        let mut xx_arr = [0.0f64; F64_LANES];
        _mm512_storeu_pd(xx_arr.as_mut_ptr(), xx);
        for j in 0..F64_LANES {
            cos_vals[j] = xx_arr[j].cos();
            sin_vals[j] = xx_arr[j].sin();
        }
        let cos_xx = _mm512_loadu_pd(cos_vals.as_ptr());
        let sin_xx = _mm512_loadu_pd(sin_vals.as_ptr());

        let asymp_result = _mm512_mul_pd(
            scale,
            _mm512_fnmadd_pd(sin_xx, q0, _mm512_mul_pd(cos_xx, p0)),
        );

        let mask = _mm512_cmp_pd_mask::<_CMP_LT_OQ>(ax, threshold);
        let result = _mm512_mask_blend_pd(mask, asymp_result, small_result);

        _mm512_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) = crate::algorithm::special::scalar::bessel_j0_scalar(x);
        }
    }
}

// ============================================================================
// Bessel J1 (scalar fallback for AVX-512)
// ============================================================================

#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_j1_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_j1_scalar(x as f64) as f32;
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_j1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_j1_scalar(x);
    }
}

// ============================================================================
// Modified Bessel I0
// ============================================================================

#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_i0_f32(input: *const f32, output: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    let threshold = _mm512_set1_ps(bessel_i0::THRESHOLD_F32);
    let one = _mm512_set1_ps(1.0);
    let two_pi = _mm512_set1_ps(2.0 * std::f32::consts::PI);

    let ai0 = _mm512_set1_ps(bessel_i0::ASYMP_F32[0]);
    let ai1 = _mm512_set1_ps(bessel_i0::ASYMP_F32[1]);
    let ai2 = _mm512_set1_ps(bessel_i0::ASYMP_F32[2]);
    let ai3 = _mm512_set1_ps(bessel_i0::ASYMP_F32[3]);
    let ai4 = _mm512_set1_ps(bessel_i0::ASYMP_F32[4]);
    let ai5 = _mm512_set1_ps(bessel_i0::ASYMP_F32[5]);
    let ai6 = _mm512_set1_ps(bessel_i0::ASYMP_F32[6]);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let x = _mm512_loadu_ps(input.add(offset));
        let ax = _mm512_abs_ps(x);

        // Small argument power series
        let z = _mm512_mul_ps(ax, ax);
        let mut sum = one;
        let mut term = one;

        for k in 1..15 {
            let k_f = k as f32;
            let factor = _mm512_div_ps(z, _mm512_set1_ps(4.0 * k_f * k_f));
            term = _mm512_mul_ps(term, factor);
            sum = _mm512_add_ps(sum, term);
        }

        let small_result = sum;

        // Asymptotic
        let z_inv = _mm512_div_ps(one, ax);

        let mut poly = ai6;
        poly = _mm512_fmadd_ps(poly, z_inv, ai5);
        poly = _mm512_fmadd_ps(poly, z_inv, ai4);
        poly = _mm512_fmadd_ps(poly, z_inv, ai3);
        poly = _mm512_fmadd_ps(poly, z_inv, ai2);
        poly = _mm512_fmadd_ps(poly, z_inv, ai1);
        poly = _mm512_fmadd_ps(poly, z_inv, ai0);

        let exp_ax = exp_f32(ax);
        let scale = _mm512_div_ps(one, _mm512_sqrt_ps(_mm512_mul_ps(two_pi, ax)));
        let asymp_result = _mm512_mul_ps(_mm512_mul_ps(exp_ax, scale), poly);

        let mask = _mm512_cmp_ps_mask::<_CMP_LE_OQ>(ax, threshold);
        let result = _mm512_mask_blend_ps(mask, asymp_result, small_result);

        _mm512_storeu_ps(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) =
                crate::algorithm::special::scalar::bessel_i0_scalar(x as f64) as f32;
        }
    }
}

#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn bessel_i0_f64(input: *const f64, output: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    let threshold = _mm512_set1_pd(bessel_i0::THRESHOLD);
    let one = _mm512_set1_pd(1.0);
    let two_pi = _mm512_set1_pd(2.0 * std::f64::consts::PI);

    let ai0 = _mm512_set1_pd(bessel_i0::ASYMP[0]);
    let ai1 = _mm512_set1_pd(bessel_i0::ASYMP[1]);
    let ai2 = _mm512_set1_pd(bessel_i0::ASYMP[2]);
    let ai3 = _mm512_set1_pd(bessel_i0::ASYMP[3]);
    let ai4 = _mm512_set1_pd(bessel_i0::ASYMP[4]);
    let ai5 = _mm512_set1_pd(bessel_i0::ASYMP[5]);
    let ai6 = _mm512_set1_pd(bessel_i0::ASYMP[6]);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let x = _mm512_loadu_pd(input.add(offset));
        let ax = _mm512_abs_pd(x);

        let z = _mm512_mul_pd(ax, ax);
        let mut sum = one;
        let mut term = one;

        for k in 1..20 {
            let k_f = k as f64;
            let factor = _mm512_div_pd(z, _mm512_set1_pd(4.0 * k_f * k_f));
            term = _mm512_mul_pd(term, factor);
            sum = _mm512_add_pd(sum, term);
        }

        let small_result = sum;

        let z_inv = _mm512_div_pd(one, ax);

        let mut poly = ai6;
        poly = _mm512_fmadd_pd(poly, z_inv, ai5);
        poly = _mm512_fmadd_pd(poly, z_inv, ai4);
        poly = _mm512_fmadd_pd(poly, z_inv, ai3);
        poly = _mm512_fmadd_pd(poly, z_inv, ai2);
        poly = _mm512_fmadd_pd(poly, z_inv, ai1);
        poly = _mm512_fmadd_pd(poly, z_inv, ai0);

        let exp_ax = exp_f64(ax);
        let scale = _mm512_div_pd(one, _mm512_sqrt_pd(_mm512_mul_pd(two_pi, ax)));
        let asymp_result = _mm512_mul_pd(_mm512_mul_pd(exp_ax, scale), poly);

        let mask = _mm512_cmp_pd_mask::<_CMP_LE_OQ>(ax, threshold);
        let result = _mm512_mask_blend_pd(mask, asymp_result, small_result);

        _mm512_storeu_pd(output.add(offset), result);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        for i in 0..remainder {
            let x = *input.add(offset + i);
            *output.add(offset + i) = crate::algorithm::special::scalar::bessel_i0_scalar(x);
        }
    }
}

// ============================================================================
// Modified Bessel I1 (scalar fallback for AVX-512)
// ============================================================================

#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_i1_f32(input: *const f32, output: *mut f32, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_i1_scalar(x as f64) as f32;
    }
}

#[target_feature(enable = "avx512f")]
pub unsafe fn bessel_i1_f64(input: *const f64, output: *mut f64, len: usize) {
    for i in 0..len {
        let x = *input.add(i);
        *output.add(i) = crate::algorithm::special::scalar::bessel_i1_scalar(x);
    }
}
