//! AVX2 unary operation kernels
//!
//! Processes 8 f32s or 4 f64s per iteration using 256-bit vectors.
//!
//! # Streaming Stores
//!
//! For large arrays (> 1MB), ReLU uses non-temporal stores (`_mm256_stream_ps`)
//! to bypass the cache.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::super::math::avx2::{
    acos_f32 as acos_vec_f32, acos_f64 as acos_vec_f64, acosh_f32 as acosh_vec_f32,
    acosh_f64 as acosh_vec_f64, asin_f32 as asin_vec_f32, asin_f64 as asin_vec_f64,
    asinh_f32 as asinh_vec_f32, asinh_f64 as asinh_vec_f64, atan_f32 as atan_vec_f32,
    atan_f64 as atan_vec_f64, atanh_f32 as atanh_vec_f32, atanh_f64 as atanh_vec_f64,
    cbrt_f32 as cbrt_vec_f32, cbrt_f64 as cbrt_vec_f64, cos_f32 as cos_vec_f32,
    cos_f64 as cos_vec_f64, cosh_f32 as cosh_vec_f32, cosh_f64 as cosh_vec_f64,
    exp_f32 as exp_vec_f32, exp_f64 as exp_vec_f64, exp2_f32 as exp2_vec_f32,
    exp2_f64 as exp2_vec_f64, expm1_f32 as expm1_vec_f32, expm1_f64 as expm1_vec_f64,
    log_f32 as log_vec_f32, log_f64 as log_vec_f64, log1p_f32 as log1p_vec_f32,
    log1p_f64 as log1p_vec_f64, log2_f32 as log2_vec_f32, log2_f64 as log2_vec_f64,
    log10_f32 as log10_vec_f32, log10_f64 as log10_vec_f64, rsqrt_f32 as rsqrt_vec_f32,
    rsqrt_f64 as rsqrt_vec_f64, sin_f32 as sin_vec_f32, sin_f64 as sin_vec_f64,
    sinh_f32 as sinh_vec_f32, sinh_f64 as sinh_vec_f64, tan_f32 as tan_vec_f32,
    tan_f64 as tan_vec_f64, tanh_f32 as tanh_vec_f32, tanh_f64 as tanh_vec_f64,
};
use super::super::super::streaming::{is_aligned_avx2, should_stream_f32, should_stream_f64};
use super::super::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};
use crate::ops::UnaryOp;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 unary operation for f32
#[target_feature(enable = "avx2")]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    match op {
        // Sign and Absolute
        UnaryOp::Neg => unary_neg_f32(a, out, chunks),
        UnaryOp::Abs => unary_abs_f32(a, out, chunks),
        UnaryOp::Sign => unary_sign_f32(a, out, chunks),

        // Power and Root
        UnaryOp::Sqrt => unary_sqrt_f32(a, out, chunks),
        UnaryOp::Rsqrt => unary_rsqrt_f32(a, out, chunks),
        UnaryOp::Cbrt => unary_cbrt_f32(a, out, chunks),
        UnaryOp::Square => unary_square_f32(a, out, chunks),
        UnaryOp::Recip => unary_recip_f32(a, out, chunks),

        // Exponential and Logarithmic
        UnaryOp::Exp => unary_exp_f32(a, out, chunks),
        UnaryOp::Exp2 => unary_exp2_f32(a, out, chunks),
        UnaryOp::Expm1 => unary_expm1_f32(a, out, chunks),
        UnaryOp::Log => unary_log_f32(a, out, chunks),
        UnaryOp::Log2 => unary_log2_f32(a, out, chunks),
        UnaryOp::Log10 => unary_log10_f32(a, out, chunks),
        UnaryOp::Log1p => unary_log1p_f32(a, out, chunks),

        // Trigonometric
        UnaryOp::Sin => unary_sin_f32(a, out, chunks),
        UnaryOp::Cos => unary_cos_f32(a, out, chunks),
        UnaryOp::Tan => unary_tan_f32(a, out, chunks),
        UnaryOp::Asin => unary_asin_f32(a, out, chunks),
        UnaryOp::Acos => unary_acos_f32(a, out, chunks),
        UnaryOp::Atan => unary_atan_f32(a, out, chunks),

        // Hyperbolic
        UnaryOp::Sinh => unary_sinh_f32(a, out, chunks),
        UnaryOp::Cosh => unary_cosh_f32(a, out, chunks),
        UnaryOp::Tanh => unary_tanh_f32(a, out, chunks),
        UnaryOp::Asinh => unary_asinh_f32(a, out, chunks),
        UnaryOp::Acosh => unary_acosh_f32(a, out, chunks),
        UnaryOp::Atanh => unary_atanh_f32(a, out, chunks),

        // Rounding
        UnaryOp::Floor => unary_floor_f32(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f32(a, out, chunks),
        UnaryOp::Round => unary_round_f32(a, out, chunks),
        UnaryOp::RoundTiesEven => unary_round_ties_even_f32(a, out, chunks),
        UnaryOp::Trunc => unary_trunc_f32(a, out, chunks),
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        unary_scalar_f32(op, a.add(offset), out.add(offset), remainder);
    }
}

/// AVX2 unary operation for f64
#[target_feature(enable = "avx2")]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    match op {
        // Sign and Absolute
        UnaryOp::Neg => unary_neg_f64(a, out, chunks),
        UnaryOp::Abs => unary_abs_f64(a, out, chunks),
        UnaryOp::Sign => unary_sign_f64(a, out, chunks),

        // Power and Root
        UnaryOp::Sqrt => unary_sqrt_f64(a, out, chunks),
        UnaryOp::Rsqrt => unary_rsqrt_f64(a, out, chunks),
        UnaryOp::Cbrt => unary_cbrt_f64(a, out, chunks),
        UnaryOp::Square => unary_square_f64(a, out, chunks),
        UnaryOp::Recip => unary_recip_f64(a, out, chunks),

        // Exponential and Logarithmic
        UnaryOp::Exp => unary_exp_f64(a, out, chunks),
        UnaryOp::Exp2 => unary_exp2_f64(a, out, chunks),
        UnaryOp::Expm1 => unary_expm1_f64(a, out, chunks),
        UnaryOp::Log => unary_log_f64(a, out, chunks),
        UnaryOp::Log2 => unary_log2_f64(a, out, chunks),
        UnaryOp::Log10 => unary_log10_f64(a, out, chunks),
        UnaryOp::Log1p => unary_log1p_f64(a, out, chunks),

        // Trigonometric
        UnaryOp::Sin => unary_sin_f64(a, out, chunks),
        UnaryOp::Cos => unary_cos_f64(a, out, chunks),
        UnaryOp::Tan => unary_tan_f64(a, out, chunks),
        UnaryOp::Asin => unary_asin_f64(a, out, chunks),
        UnaryOp::Acos => unary_acos_f64(a, out, chunks),
        UnaryOp::Atan => unary_atan_f64(a, out, chunks),

        // Hyperbolic
        UnaryOp::Sinh => unary_sinh_f64(a, out, chunks),
        UnaryOp::Cosh => unary_cosh_f64(a, out, chunks),
        UnaryOp::Tanh => unary_tanh_f64(a, out, chunks),
        UnaryOp::Asinh => unary_asinh_f64(a, out, chunks),
        UnaryOp::Acosh => unary_acosh_f64(a, out, chunks),
        UnaryOp::Atanh => unary_atanh_f64(a, out, chunks),

        // Rounding
        UnaryOp::Floor => unary_floor_f64(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f64(a, out, chunks),
        UnaryOp::Round => unary_round_f64(a, out, chunks),
        UnaryOp::RoundTiesEven => unary_round_ties_even_f64(a, out, chunks),
        UnaryOp::Trunc => unary_trunc_f64(a, out, chunks),
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        unary_scalar_f64(op, a.add(offset), out.add(offset), remainder);
    }
}

/// AVX2 ReLU for f32
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is aligned.
#[target_feature(enable = "avx2")]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let zero = _mm256_setzero_ps();

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f32(len) && is_aligned_avx2(out);

    if use_streaming {
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let va = _mm256_loadu_ps(a.add(offset));
            let vr = _mm256_max_ps(va, zero);
            _mm256_stream_ps(out.add(offset), vr);
        }
        _mm_sfence();
    } else {
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let va = _mm256_loadu_ps(a.add(offset));
            let vr = _mm256_max_ps(va, zero);
            _mm256_storeu_ps(out.add(offset), vr);
        }
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        relu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// AVX2 ReLU for f64
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is aligned.
#[target_feature(enable = "avx2")]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let zero = _mm256_setzero_pd();

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f64(len) && is_aligned_avx2(out);

    if use_streaming {
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let va = _mm256_loadu_pd(a.add(offset));
            let vr = _mm256_max_pd(va, zero);
            _mm256_stream_pd(out.add(offset), vr);
        }
        _mm_sfence();
    } else {
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let va = _mm256_loadu_pd(a.add(offset));
            let vr = _mm256_max_pd(va, zero);
            _mm256_storeu_pd(out.add(offset), vr);
        }
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        relu_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn unary_neg_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let zero = _mm256_setzero_ps();
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_sub_ps(zero, va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_abs_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_and_ps(va, mask);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_sqrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_sqrt_ps(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_square_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_mul_ps(va, va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_recip_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let one = _mm256_set1_ps(1.0);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_div_ps(one, va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_floor_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_floor_ps(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_ceil_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = _mm256_ceil_ps(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

/// Round to nearest with ties away from zero, matching `f32::round`.
///
/// AVX2 has no ties-away rounding mode, so this rounds the magnitude to nearest
/// (ties to even) and then pushes the exact-tie lanes one step further out.
/// `n - mag` is exact for every finite input, so the `<= -0.5` test fires only
/// on a true tie that rounded toward zero.
#[target_feature(enable = "avx2")]
unsafe fn unary_round_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let sign_mask = _mm256_set1_ps(-0.0);
    let neg_half = _mm256_set1_ps(-0.5);
    let one = _mm256_set1_ps(1.0);
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let sign = _mm256_and_ps(va, sign_mask);
        let mag = _mm256_andnot_ps(sign_mask, va);
        // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x08 (ties to even)
        let n = _mm256_round_ps::<0x08>(mag);
        let y = _mm256_sub_ps(n, mag);
        // NaN and infinity compare false here, so they pass through as `n`.
        let tie = _mm256_cmp_ps::<_CMP_LE_OQ>(y, neg_half);
        let vr = _mm256_or_ps(_mm256_add_ps(n, _mm256_and_ps(tie, one)), sign);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

/// Round to nearest with ties to even, matching `f32::round_ties_even`.
#[target_feature(enable = "avx2")]
unsafe fn unary_round_ties_even_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        // _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC = 0x08
        let vr = _mm256_round_ps::<0x08>(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_exp_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = exp_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_tanh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = tanh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = log_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_sin_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = sin_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cos_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = cos_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_tan_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = tan_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_atan_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = atan_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_sign_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
        let pos_mask = _mm256_cmp_ps::<_CMP_GT_OQ>(va, zero);
        let neg_mask = _mm256_cmp_ps::<_CMP_LT_OQ>(va, zero);
        let pos_part = _mm256_and_ps(pos_mask, one);
        let neg_part = _mm256_and_ps(neg_mask, neg_one);
        let vr = _mm256_or_ps(pos_part, neg_part);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_rsqrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = rsqrt_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cbrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = cbrt_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_exp2_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = exp2_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_expm1_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = expm1_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log2_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = log2_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log10_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = log10_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log1p_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = log1p_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_asin_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = asin_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_acos_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = acos_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_sinh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = sinh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cosh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = cosh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_asinh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = asinh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_acosh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = acosh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_atanh_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        let vr = atanh_vec_f32(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_trunc_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = _mm256_loadu_ps(a.add(offset));
        // _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC = 0x0B
        let vr = _mm256_round_ps::<0x0B>(va);
        _mm256_storeu_ps(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[target_feature(enable = "avx2")]
unsafe fn unary_neg_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let zero = _mm256_setzero_pd();
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_sub_pd(zero, va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_abs_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let mask = _mm256_set1_pd(f64::from_bits(0x7FFF_FFFF_FFFF_FFFF));
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_and_pd(va, mask);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_sqrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_sqrt_pd(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_square_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_mul_pd(va, va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_recip_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let one = _mm256_set1_pd(1.0);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_div_pd(one, va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_floor_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_floor_pd(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_ceil_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_ceil_pd(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

/// Round to nearest with ties away from zero, matching `f64::round`.
///
/// Same magnitude-then-correct construction as `unary_round_f32`.
#[target_feature(enable = "avx2")]
unsafe fn unary_round_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let sign_mask = _mm256_set1_pd(-0.0);
    let neg_half = _mm256_set1_pd(-0.5);
    let one = _mm256_set1_pd(1.0);
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let sign = _mm256_and_pd(va, sign_mask);
        let mag = _mm256_andnot_pd(sign_mask, va);
        let n = _mm256_round_pd::<0x08>(mag);
        let y = _mm256_sub_pd(n, mag);
        let tie = _mm256_cmp_pd::<_CMP_LE_OQ>(y, neg_half);
        let vr = _mm256_or_pd(_mm256_add_pd(n, _mm256_and_pd(tie, one)), sign);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

/// Round to nearest with ties to even, matching `f64::round_ties_even`.
#[target_feature(enable = "avx2")]
unsafe fn unary_round_ties_even_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = _mm256_round_pd::<0x08>(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_exp_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = exp_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_tanh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = tanh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = log_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_sin_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = sin_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cos_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = cos_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_tan_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = tan_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_atan_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = atan_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_sign_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let zero = _mm256_setzero_pd();
    let one = _mm256_set1_pd(1.0);
    let neg_one = _mm256_set1_pd(-1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
        let pos_mask = _mm256_cmp_pd::<_CMP_GT_OQ>(va, zero);
        let neg_mask = _mm256_cmp_pd::<_CMP_LT_OQ>(va, zero);
        let pos_part = _mm256_and_pd(pos_mask, one);
        let neg_part = _mm256_and_pd(neg_mask, neg_one);
        let vr = _mm256_or_pd(pos_part, neg_part);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_rsqrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = rsqrt_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cbrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = cbrt_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_exp2_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = exp2_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_expm1_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = expm1_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log2_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = log2_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log10_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = log10_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_log1p_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = log1p_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_asin_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = asin_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_acos_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = acos_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_sinh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = sinh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_cosh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = cosh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_asinh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = asinh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_acosh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = acosh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn unary_atanh_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        let vr = atanh_vec_f64(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}

#[target_feature(enable = "avx2")]
unsafe fn unary_trunc_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = _mm256_loadu_pd(a.add(offset));
        // _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC = 0x0B
        let vr = _mm256_round_pd::<0x0B>(va);
        _mm256_storeu_pd(out.add(offset), vr);
    }
}
