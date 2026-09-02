//! AVX-512 binary operation kernels
//!
//! Processes 16 f32s or 8 f64s per iteration using 512-bit vectors.
//!
//! # Streaming Stores
//!
//! For large arrays (> 1MB), this module uses streaming (non-temporal) stores
//! (`_mm512_stream_ps`) to bypass the cache. This improves performance by
//! avoiding cache pollution when the output won't be read immediately.
//!
//! Streaming stores require 64-byte aligned output pointers. If alignment
//! requirements are not met, the code falls back to regular cached stores.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::super::math::avx512::{
    exp_f32 as exp_vec_f32, exp_f64 as exp_vec_f64, log_f32 as log_vec_f32, log_f64 as log_vec_f64,
};
use super::super::super::streaming::{is_aligned_avx512, should_stream_f32, should_stream_f64};
use super::super::{binary_scalar_f32, binary_scalar_f64};
use crate::ops::BinaryOp;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ============================================================================
// Macro to generate binary kernels (eliminates duplication between regular/streaming)
// ============================================================================

/// Generates a binary kernel for f32 with the specified store instruction.
macro_rules! impl_binary_f32_avx512 {
    ($name:ident, $vec_op:ident, $store:ident) => {
        #[target_feature(enable = "avx512f")]
        unsafe fn $name(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = _mm512_loadu_ps(a.add(offset));
                let vb = _mm512_loadu_ps(b.add(offset));
                let vr = $vec_op(va, vb);
                $store(out.add(offset), vr);
            }
        }
    };
}

/// Generates a binary kernel for f64 with the specified store instruction.
macro_rules! impl_binary_f64_avx512 {
    ($name:ident, $vec_op:ident, $store:ident) => {
        #[target_feature(enable = "avx512f")]
        unsafe fn $name(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = _mm512_loadu_pd(a.add(offset));
                let vb = _mm512_loadu_pd(b.add(offset));
                let vr = $vec_op(va, vb);
                $store(out.add(offset), vr);
            }
        }
    };
}

// Generate regular (cached) f32 kernels
impl_binary_f32_avx512!(binary_add_f32, _mm512_add_ps, _mm512_storeu_ps);
impl_binary_f32_avx512!(binary_sub_f32, _mm512_sub_ps, _mm512_storeu_ps);
impl_binary_f32_avx512!(binary_mul_f32, _mm512_mul_ps, _mm512_storeu_ps);
impl_binary_f32_avx512!(binary_div_f32, _mm512_div_ps, _mm512_storeu_ps);
impl_binary_f32_avx512!(binary_max_f32, _mm512_max_ps, _mm512_storeu_ps);
impl_binary_f32_avx512!(binary_min_f32, _mm512_min_ps, _mm512_storeu_ps);

// Generate streaming (non-temporal) f32 kernels
impl_binary_f32_avx512!(binary_add_f32_stream, _mm512_add_ps, _mm512_stream_ps);
impl_binary_f32_avx512!(binary_sub_f32_stream, _mm512_sub_ps, _mm512_stream_ps);
impl_binary_f32_avx512!(binary_mul_f32_stream, _mm512_mul_ps, _mm512_stream_ps);
impl_binary_f32_avx512!(binary_div_f32_stream, _mm512_div_ps, _mm512_stream_ps);
impl_binary_f32_avx512!(binary_max_f32_stream, _mm512_max_ps, _mm512_stream_ps);
impl_binary_f32_avx512!(binary_min_f32_stream, _mm512_min_ps, _mm512_stream_ps);

// Generate regular (cached) f64 kernels
impl_binary_f64_avx512!(binary_add_f64, _mm512_add_pd, _mm512_storeu_pd);
impl_binary_f64_avx512!(binary_sub_f64, _mm512_sub_pd, _mm512_storeu_pd);
impl_binary_f64_avx512!(binary_mul_f64, _mm512_mul_pd, _mm512_storeu_pd);
impl_binary_f64_avx512!(binary_div_f64, _mm512_div_pd, _mm512_storeu_pd);
impl_binary_f64_avx512!(binary_max_f64, _mm512_max_pd, _mm512_storeu_pd);
impl_binary_f64_avx512!(binary_min_f64, _mm512_min_pd, _mm512_storeu_pd);

// Generate streaming (non-temporal) f64 kernels
impl_binary_f64_avx512!(binary_add_f64_stream, _mm512_add_pd, _mm512_stream_pd);
impl_binary_f64_avx512!(binary_sub_f64_stream, _mm512_sub_pd, _mm512_stream_pd);
impl_binary_f64_avx512!(binary_mul_f64_stream, _mm512_mul_pd, _mm512_stream_pd);
impl_binary_f64_avx512!(binary_div_f64_stream, _mm512_div_pd, _mm512_stream_pd);
impl_binary_f64_avx512!(binary_max_f64_stream, _mm512_max_pd, _mm512_stream_pd);
impl_binary_f64_avx512!(binary_min_f64_stream, _mm512_min_pd, _mm512_stream_pd);

// ============================================================================
// Pow kernels (more complex, not macro-generated)
// ============================================================================

/// Generates pow kernel with specified store instruction.
macro_rules! impl_pow_f32_avx512 {
    ($name:ident, $store:ident) => {
        #[target_feature(enable = "avx512f")]
        unsafe fn $name(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
            let inf = _mm512_set1_ps(f32::INFINITY);
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = _mm512_loadu_ps(a.add(offset));
                let vb = _mm512_loadu_ps(b.add(offset));
                // pow(a, b) = exp(b * log(a))
                let log_a = log_vec_f32(va);
                let b_log_a = _mm512_mul_ps(vb, log_a);
                let vr = exp_vec_f32(b_log_a);

                // The identity only holds for a > 0. Elsewhere log(a) is -inf
                // or NaN, and results like pow(0, 0) = 1 or pow(-2, 3) = -8
                // have no form in it at all. Unordered compare, so NaN bases
                // land here too.
                let undefined = _mm512_cmp_ps_mask::<_CMP_NGT_UQ>(va, _mm512_setzero_ps());
                // A non-finite product is the other escape: `b * log(a)` is
                // 0 * inf for pow(+inf, 0) and pow(1, ±inf), which IEEE 754
                // fixes at exactly 1, and the exponential of NaN cannot
                // recover that.
                let wild = _mm512_cmp_ps_mask::<_CMP_NLT_UQ>(_mm512_abs_ps(b_log_a), inf);
                let undefined = undefined | wild;

                if undefined == 0 {
                    $store(out.add(offset), vr);
                } else {
                    let mut lanes = [0.0f32; F32_LANES];
                    let mut products = [0.0f32; F32_LANES];
                    _mm512_storeu_ps(lanes.as_mut_ptr(), vr);
                    _mm512_storeu_ps(products.as_mut_ptr(), b_log_a);
                    for (lane, slot) in lanes.iter_mut().enumerate() {
                        let base = *a.add(offset + lane);
                        if base.is_nan() || base <= 0.0 || !products[lane].is_finite() {
                            *slot = base.powf(*b.add(offset + lane));
                        }
                    }
                    $store(out.add(offset), _mm512_loadu_ps(lanes.as_ptr()));
                }
            }
        }
    };
}

macro_rules! impl_pow_f64_avx512 {
    ($name:ident, $store:ident) => {
        #[target_feature(enable = "avx512f", enable = "avx512dq")]
        unsafe fn $name(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
            let inf = _mm512_set1_pd(f64::INFINITY);
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = _mm512_loadu_pd(a.add(offset));
                let vb = _mm512_loadu_pd(b.add(offset));
                // pow(a, b) = exp(b * log(a))
                let log_a = log_vec_f64(va);
                let b_log_a = _mm512_mul_pd(vb, log_a);
                let vr = exp_vec_f64(b_log_a);

                // The identity only holds for a > 0. Elsewhere log(a) is -inf
                // or NaN, and results like pow(0, 0) = 1 or pow(-2, 3) = -8
                // have no form in it at all. Unordered compare, so NaN bases
                // land here too.
                let undefined = _mm512_cmp_pd_mask::<_CMP_NGT_UQ>(va, _mm512_setzero_pd());
                // A non-finite product is the other escape: `b * log(a)` is
                // 0 * inf for pow(+inf, 0) and pow(1, ±inf), which IEEE 754
                // fixes at exactly 1, and the exponential of NaN cannot
                // recover that.
                let wild = _mm512_cmp_pd_mask::<_CMP_NLT_UQ>(_mm512_abs_pd(b_log_a), inf);
                let undefined = undefined | wild;

                if undefined == 0 {
                    $store(out.add(offset), vr);
                } else {
                    let mut lanes = [0.0f64; F64_LANES];
                    let mut products = [0.0f64; F64_LANES];
                    _mm512_storeu_pd(lanes.as_mut_ptr(), vr);
                    _mm512_storeu_pd(products.as_mut_ptr(), b_log_a);
                    for (lane, slot) in lanes.iter_mut().enumerate() {
                        let base = *a.add(offset + lane);
                        if base.is_nan() || base <= 0.0 || !products[lane].is_finite() {
                            *slot = base.powf(*b.add(offset + lane));
                        }
                    }
                    $store(out.add(offset), _mm512_loadu_pd(lanes.as_ptr()));
                }
            }
        }
    };
}

impl_pow_f32_avx512!(binary_pow_f32, _mm512_storeu_ps);
impl_pow_f32_avx512!(binary_pow_f32_stream, _mm512_stream_ps);
impl_pow_f64_avx512!(binary_pow_f64, _mm512_storeu_pd);
impl_pow_f64_avx512!(binary_pow_f64_stream, _mm512_stream_pd);

// ============================================================================
// Public dispatch functions
// ============================================================================

/// AVX-512 binary operation for f32
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is 64-byte aligned.
///
/// # Safety
/// - CPU must support AVX-512F
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx512f")]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f32(len) && is_aligned_avx512(out);

    // Atan2 has no SIMD implementation - use scalar fallback
    if op == BinaryOp::Atan2 {
        binary_scalar_f32(op, a, b, out, len);
        return;
    }

    if use_streaming {
        match op {
            BinaryOp::Add => binary_add_f32_stream(a, b, out, chunks),
            BinaryOp::Sub => binary_sub_f32_stream(a, b, out, chunks),
            BinaryOp::Mul => binary_mul_f32_stream(a, b, out, chunks),
            BinaryOp::Div => binary_div_f32_stream(a, b, out, chunks),
            BinaryOp::Max => binary_max_f32_stream(a, b, out, chunks),
            BinaryOp::Min => binary_min_f32_stream(a, b, out, chunks),
            BinaryOp::Pow => binary_pow_f32_stream(a, b, out, chunks),
            BinaryOp::Atan2 => unreachable!(), // Handled above
        }
        // Memory fence ensures streaming stores are globally visible
        _mm_sfence();
    } else {
        match op {
            BinaryOp::Add => binary_add_f32(a, b, out, chunks),
            BinaryOp::Sub => binary_sub_f32(a, b, out, chunks),
            BinaryOp::Mul => binary_mul_f32(a, b, out, chunks),
            BinaryOp::Div => binary_div_f32(a, b, out, chunks),
            BinaryOp::Max => binary_max_f32(a, b, out, chunks),
            BinaryOp::Min => binary_min_f32(a, b, out, chunks),
            BinaryOp::Pow => binary_pow_f32(a, b, out, chunks),
            BinaryOp::Atan2 => unreachable!(), // Handled above
        }
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F32_LANES;
        binary_scalar_f32(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}

/// AVX-512 binary operation for f64
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is 64-byte aligned.
///
/// # Safety
/// - CPU must support AVX-512F and AVX-512DQ
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f64(len) && is_aligned_avx512(out);

    // Atan2 has no SIMD implementation - use scalar fallback
    if op == BinaryOp::Atan2 {
        binary_scalar_f64(op, a, b, out, len);
        return;
    }

    if use_streaming {
        match op {
            BinaryOp::Add => binary_add_f64_stream(a, b, out, chunks),
            BinaryOp::Sub => binary_sub_f64_stream(a, b, out, chunks),
            BinaryOp::Mul => binary_mul_f64_stream(a, b, out, chunks),
            BinaryOp::Div => binary_div_f64_stream(a, b, out, chunks),
            BinaryOp::Max => binary_max_f64_stream(a, b, out, chunks),
            BinaryOp::Min => binary_min_f64_stream(a, b, out, chunks),
            BinaryOp::Pow => binary_pow_f64_stream(a, b, out, chunks),
            BinaryOp::Atan2 => unreachable!(), // Handled above
        }
        // Memory fence ensures streaming stores are globally visible
        _mm_sfence();
    } else {
        match op {
            BinaryOp::Add => binary_add_f64(a, b, out, chunks),
            BinaryOp::Sub => binary_sub_f64(a, b, out, chunks),
            BinaryOp::Mul => binary_mul_f64(a, b, out, chunks),
            BinaryOp::Div => binary_div_f64(a, b, out, chunks),
            BinaryOp::Max => binary_max_f64(a, b, out, chunks),
            BinaryOp::Min => binary_min_f64(a, b, out, chunks),
            BinaryOp::Pow => binary_pow_f64(a, b, out, chunks),
            BinaryOp::Atan2 => unreachable!(), // Handled above
        }
    }

    // Handle tail with scalar
    if remainder > 0 {
        let offset = chunks * F64_LANES;
        binary_scalar_f64(op, a.add(offset), b.add(offset), out.add(offset), remainder);
    }
}
