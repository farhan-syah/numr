//! AVX-512 fused activation-mul kernels
//!
//! Vectorized implementations of fused activation * multiplication using 512-bit registers.
//! Functions take two inputs (a, b) and compute activation(a) * b in a single pass.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{exp_f32, exp_f64, tanh_f32, tanh_f64};
use super::{
    gelu_mul_scalar_f32, gelu_mul_scalar_f64, relu_mul_scalar_f32, relu_mul_scalar_f64,
    sigmoid_mul_scalar_f32, sigmoid_mul_scalar_f64, silu_mul_scalar_f32, silu_mul_scalar_f64,
};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 silu_mul for f32
///
/// Computes: (a / (1 + exp(-a))) * b
#[target_feature(enable = "avx512f")]
pub unsafe fn silu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let one = _mm512_set1_ps(1.0);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm512_loadu_ps(a.add(offset));
        let y = _mm512_loadu_ps(b.add(offset));
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        let exp_neg_x = exp_f32(neg_x);
        let activation = _mm512_div_ps(x, _mm512_add_ps(one, exp_neg_x));
        let result = _mm512_mul_ps(activation, y);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        silu_mul_scalar_f32(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 silu_mul for f64
///
/// Computes: (a / (1 + exp(-a))) * b
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn silu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let one = _mm512_set1_pd(1.0);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm512_loadu_pd(a.add(offset));
        let y = _mm512_loadu_pd(b.add(offset));
        let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);
        let exp_neg_x = exp_f64(neg_x);
        let activation = _mm512_div_pd(x, _mm512_add_pd(one, exp_neg_x));
        let result = _mm512_mul_pd(activation, y);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        silu_mul_scalar_f64(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 gelu_mul for f32
///
/// Computes: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b
#[target_feature(enable = "avx512f")]
pub unsafe fn gelu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let half = _mm512_set1_ps(0.5);
    let one = _mm512_set1_ps(1.0);
    let sqrt_2_over_pi = _mm512_set1_ps(0.7978845608);
    let tanh_coef = _mm512_set1_ps(0.044715);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm512_loadu_ps(a.add(offset));
        let y = _mm512_loadu_ps(b.add(offset));

        let x_cubed = _mm512_mul_ps(_mm512_mul_ps(x, x), x);
        let inner = _mm512_mul_ps(sqrt_2_over_pi, _mm512_fmadd_ps(tanh_coef, x_cubed, x));

        let tanh_inner = tanh_f32(inner);
        let activation = _mm512_mul_ps(half, _mm512_mul_ps(x, _mm512_add_ps(one, tanh_inner)));

        let result = _mm512_mul_ps(activation, y);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        gelu_mul_scalar_f32(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 gelu_mul for f64
///
/// Computes: 0.5 * a * (1 + tanh(sqrt(2/pi) * (a + 0.044715 * a^3))) * b
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn gelu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let half = _mm512_set1_pd(0.5);
    let one = _mm512_set1_pd(1.0);
    let sqrt_2_over_pi = _mm512_set1_pd(0.7978845608028654);
    let tanh_coef = _mm512_set1_pd(0.044715);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm512_loadu_pd(a.add(offset));
        let y = _mm512_loadu_pd(b.add(offset));

        let x_cubed = _mm512_mul_pd(_mm512_mul_pd(x, x), x);
        let inner = _mm512_mul_pd(sqrt_2_over_pi, _mm512_fmadd_pd(tanh_coef, x_cubed, x));

        let tanh_inner = tanh_f64(inner);
        let activation = _mm512_mul_pd(half, _mm512_mul_pd(x, _mm512_add_pd(one, tanh_inner)));

        let result = _mm512_mul_pd(activation, y);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        gelu_mul_scalar_f64(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 relu_mul for f32
///
/// Computes: max(0, a) * b
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let zero = _mm512_setzero_ps();

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm512_loadu_ps(a.add(offset));
        let y = _mm512_loadu_ps(b.add(offset));
        let activation = _mm512_max_ps(zero, x);
        let result = _mm512_mul_ps(activation, y);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        relu_mul_scalar_f32(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 relu_mul for f64
///
/// Computes: max(0, a) * b
#[target_feature(enable = "avx512f")]
pub unsafe fn relu_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let zero = _mm512_setzero_pd();

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm512_loadu_pd(a.add(offset));
        let y = _mm512_loadu_pd(b.add(offset));
        let activation = _mm512_max_pd(zero, x);
        let result = _mm512_mul_pd(activation, y);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        relu_mul_scalar_f64(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 sigmoid_mul for f32
///
/// Computes: (1 / (1 + exp(-a))) * b
#[target_feature(enable = "avx512f")]
pub unsafe fn sigmoid_mul_f32(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let one = _mm512_set1_ps(1.0);

    for c in 0..chunks {
        let offset = c * F32_LANES;
        let x = _mm512_loadu_ps(a.add(offset));
        let y = _mm512_loadu_ps(b.add(offset));
        let neg_x = _mm512_sub_ps(_mm512_setzero_ps(), x);
        let exp_neg_x = exp_f32(neg_x);
        let activation = _mm512_div_ps(one, _mm512_add_ps(one, exp_neg_x));
        let result = _mm512_mul_ps(activation, y);
        _mm512_storeu_ps(out.add(offset), result);
    }

    let processed = chunks * F32_LANES;
    if processed < len {
        sigmoid_mul_scalar_f32(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 sigmoid_mul for f64
///
/// Computes: (1 / (1 + exp(-a))) * b
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn sigmoid_mul_f64(a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let one = _mm512_set1_pd(1.0);

    for c in 0..chunks {
        let offset = c * F64_LANES;
        let x = _mm512_loadu_pd(a.add(offset));
        let y = _mm512_loadu_pd(b.add(offset));
        let neg_x = _mm512_sub_pd(_mm512_setzero_pd(), x);
        let exp_neg_x = exp_f64(neg_x);
        let activation = _mm512_div_pd(one, _mm512_add_pd(one, exp_neg_x));
        let result = _mm512_mul_pd(activation, y);
        _mm512_storeu_pd(out.add(offset), result);
    }

    let processed = chunks * F64_LANES;
    if processed < len {
        sigmoid_mul_scalar_f64(
            a.add(processed),
            b.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}
