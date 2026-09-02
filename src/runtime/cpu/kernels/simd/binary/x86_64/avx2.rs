//! AVX2 binary operation kernels
//!
//! Processes 8 f32s or 4 f64s per iteration using 256-bit vectors.
//!
//! # Streaming Stores
//!
//! For large arrays (> 1MB), this module uses streaming (non-temporal) stores
//! (`_mm256_stream_ps`) to bypass the cache. This improves performance by
//! avoiding cache pollution when the output won't be read immediately.
//!
//! Streaming stores require 32-byte aligned output pointers. If alignment
//! requirements are not met, the code falls back to regular cached stores.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::super::math::avx2::{
    exp_f32 as exp_vec_f32, exp_f64 as exp_vec_f64, log_f32 as log_vec_f32, log_f64 as log_vec_f64,
};
use super::super::super::streaming::{is_aligned_avx2, should_stream_f32, should_stream_f64};
use super::super::{binary_scalar_f32, binary_scalar_f64};
use crate::ops::BinaryOp;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

// ============================================================================
// Macro to generate binary kernels (eliminates duplication between regular/streaming)
// ============================================================================

/// Generates a binary kernel for f32 with the specified store instruction.
macro_rules! impl_binary_f32_avx2 {
    ($name:ident, $vec_op:ident, $store:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = _mm256_loadu_ps(a.add(offset));
                let vb = _mm256_loadu_ps(b.add(offset));
                let vr = $vec_op(va, vb);
                $store(out.add(offset), vr);
            }
        }
    };
}

/// Generates a binary kernel for f64 with the specified store instruction.
macro_rules! impl_binary_f64_avx2 {
    ($name:ident, $vec_op:ident, $store:ident) => {
        #[target_feature(enable = "avx2")]
        unsafe fn $name(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = _mm256_loadu_pd(a.add(offset));
                let vb = _mm256_loadu_pd(b.add(offset));
                let vr = $vec_op(va, vb);
                $store(out.add(offset), vr);
            }
        }
    };
}

// Generate regular (cached) f32 kernels
impl_binary_f32_avx2!(binary_add_f32, _mm256_add_ps, _mm256_storeu_ps);
impl_binary_f32_avx2!(binary_sub_f32, _mm256_sub_ps, _mm256_storeu_ps);
impl_binary_f32_avx2!(binary_mul_f32, _mm256_mul_ps, _mm256_storeu_ps);
impl_binary_f32_avx2!(binary_div_f32, _mm256_div_ps, _mm256_storeu_ps);
impl_binary_f32_avx2!(binary_max_f32, _mm256_max_ps, _mm256_storeu_ps);
impl_binary_f32_avx2!(binary_min_f32, _mm256_min_ps, _mm256_storeu_ps);

// Generate streaming (non-temporal) f32 kernels
impl_binary_f32_avx2!(binary_add_f32_stream, _mm256_add_ps, _mm256_stream_ps);
impl_binary_f32_avx2!(binary_sub_f32_stream, _mm256_sub_ps, _mm256_stream_ps);
impl_binary_f32_avx2!(binary_mul_f32_stream, _mm256_mul_ps, _mm256_stream_ps);
impl_binary_f32_avx2!(binary_div_f32_stream, _mm256_div_ps, _mm256_stream_ps);
impl_binary_f32_avx2!(binary_max_f32_stream, _mm256_max_ps, _mm256_stream_ps);
impl_binary_f32_avx2!(binary_min_f32_stream, _mm256_min_ps, _mm256_stream_ps);

// Generate regular (cached) f64 kernels
impl_binary_f64_avx2!(binary_add_f64, _mm256_add_pd, _mm256_storeu_pd);
impl_binary_f64_avx2!(binary_sub_f64, _mm256_sub_pd, _mm256_storeu_pd);
impl_binary_f64_avx2!(binary_mul_f64, _mm256_mul_pd, _mm256_storeu_pd);
impl_binary_f64_avx2!(binary_div_f64, _mm256_div_pd, _mm256_storeu_pd);
impl_binary_f64_avx2!(binary_max_f64, _mm256_max_pd, _mm256_storeu_pd);
impl_binary_f64_avx2!(binary_min_f64, _mm256_min_pd, _mm256_storeu_pd);

// Generate streaming (non-temporal) f64 kernels
impl_binary_f64_avx2!(binary_add_f64_stream, _mm256_add_pd, _mm256_stream_pd);
impl_binary_f64_avx2!(binary_sub_f64_stream, _mm256_sub_pd, _mm256_stream_pd);
impl_binary_f64_avx2!(binary_mul_f64_stream, _mm256_mul_pd, _mm256_stream_pd);
impl_binary_f64_avx2!(binary_div_f64_stream, _mm256_div_pd, _mm256_stream_pd);
impl_binary_f64_avx2!(binary_max_f64_stream, _mm256_max_pd, _mm256_stream_pd);
impl_binary_f64_avx2!(binary_min_f64_stream, _mm256_min_pd, _mm256_stream_pd);

// ============================================================================
// Pow kernels (more complex, uses FMA)
// ============================================================================

/// Generates pow kernel with specified store instruction.
macro_rules! impl_pow_f32_avx2 {
    ($name:ident, $store:ident) => {
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn $name(a: *const f32, b: *const f32, out: *mut f32, chunks: usize) {
            let inf = _mm256_set1_ps(f32::INFINITY);
            let abs_mask = _mm256_set1_ps(-0.0);
            for i in 0..chunks {
                let offset = i * F32_LANES;
                let va = _mm256_loadu_ps(a.add(offset));
                let vb = _mm256_loadu_ps(b.add(offset));
                // pow(a, b) = exp(b * log(a))
                let log_a = log_vec_f32(va);
                let b_log_a = _mm256_mul_ps(vb, log_a);
                let vr = exp_vec_f32(b_log_a);

                // The identity only holds for a > 0. Elsewhere log(a) is -inf
                // or NaN, and results like pow(0, 0) = 1 or pow(-2, 3) = -8
                // have no form in it at all. Unordered compare, so NaN bases
                // land here too.
                let undefined = _mm256_cmp_ps::<_CMP_NGT_UQ>(va, _mm256_setzero_ps());
                // A non-finite product is the other escape: `b * log(a)` is
                // 0 * inf for pow(+inf, 0) and pow(1, ±inf), which IEEE 754
                // fixes at exactly 1, and the exponential of NaN cannot
                // recover that.
                let wild = _mm256_cmp_ps::<_CMP_NLT_UQ>(_mm256_andnot_ps(abs_mask, b_log_a), inf);
                let undefined = _mm256_or_ps(undefined, wild);

                if _mm256_movemask_ps(undefined) == 0 {
                    $store(out.add(offset), vr);
                } else {
                    let mut lanes = [0.0f32; F32_LANES];
                    let mut products = [0.0f32; F32_LANES];
                    _mm256_storeu_ps(lanes.as_mut_ptr(), vr);
                    _mm256_storeu_ps(products.as_mut_ptr(), b_log_a);
                    for (lane, slot) in lanes.iter_mut().enumerate() {
                        let base = *a.add(offset + lane);
                        if base.is_nan() || base <= 0.0 || !products[lane].is_finite() {
                            *slot = base.powf(*b.add(offset + lane));
                        }
                    }
                    $store(out.add(offset), _mm256_loadu_ps(lanes.as_ptr()));
                }
            }
        }
    };
}

macro_rules! impl_pow_f64_avx2 {
    ($name:ident, $store:ident) => {
        #[target_feature(enable = "avx2", enable = "fma")]
        unsafe fn $name(a: *const f64, b: *const f64, out: *mut f64, chunks: usize) {
            let inf = _mm256_set1_pd(f64::INFINITY);
            let abs_mask = _mm256_set1_pd(-0.0);
            for i in 0..chunks {
                let offset = i * F64_LANES;
                let va = _mm256_loadu_pd(a.add(offset));
                let vb = _mm256_loadu_pd(b.add(offset));
                // pow(a, b) = exp(b * log(a))
                let log_a = log_vec_f64(va);
                let b_log_a = _mm256_mul_pd(vb, log_a);
                let vr = exp_vec_f64(b_log_a);

                // The identity only holds for a > 0. Elsewhere log(a) is -inf
                // or NaN, and results like pow(0, 0) = 1 or pow(-2, 3) = -8
                // have no form in it at all. Unordered compare, so NaN bases
                // land here too.
                let undefined = _mm256_cmp_pd::<_CMP_NGT_UQ>(va, _mm256_setzero_pd());
                // A non-finite product is the other escape: `b * log(a)` is
                // 0 * inf for pow(+inf, 0) and pow(1, ±inf), which IEEE 754
                // fixes at exactly 1, and the exponential of NaN cannot
                // recover that.
                let wild = _mm256_cmp_pd::<_CMP_NLT_UQ>(_mm256_andnot_pd(abs_mask, b_log_a), inf);
                let undefined = _mm256_or_pd(undefined, wild);

                if _mm256_movemask_pd(undefined) == 0 {
                    $store(out.add(offset), vr);
                } else {
                    let mut lanes = [0.0f64; F64_LANES];
                    let mut products = [0.0f64; F64_LANES];
                    _mm256_storeu_pd(lanes.as_mut_ptr(), vr);
                    _mm256_storeu_pd(products.as_mut_ptr(), b_log_a);
                    for (lane, slot) in lanes.iter_mut().enumerate() {
                        let base = *a.add(offset + lane);
                        if base.is_nan() || base <= 0.0 || !products[lane].is_finite() {
                            *slot = base.powf(*b.add(offset + lane));
                        }
                    }
                    $store(out.add(offset), _mm256_loadu_pd(lanes.as_ptr()));
                }
            }
        }
    };
}

impl_pow_f32_avx2!(binary_pow_f32, _mm256_storeu_ps);
impl_pow_f32_avx2!(binary_pow_f32_stream, _mm256_stream_ps);
impl_pow_f64_avx2!(binary_pow_f64, _mm256_storeu_pd);
impl_pow_f64_avx2!(binary_pow_f64_stream, _mm256_stream_pd);

// ============================================================================
// Public dispatch functions
// ============================================================================

/// AVX2 binary operation for f32
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is 32-byte aligned.
///
/// # Safety
/// - CPU must support AVX2
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f32(len) && is_aligned_avx2(out);

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

/// AVX2 binary operation for f64
///
/// Uses streaming (non-temporal) stores for arrays > 1MB when output is 32-byte aligned.
///
/// # Safety
/// - CPU must support AVX2
/// - All pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    // Use streaming stores for large aligned arrays
    let use_streaming = should_stream_f64(len) && is_aligned_avx2(out);

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
