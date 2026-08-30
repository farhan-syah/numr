//! NEON unary operation kernels for ARM64
//!
//! Processes 4 f32s or 2 f64s per iteration using 128-bit vectors.
//!
//! # Operations
//!
//! Direct SIMD operations (native NEON instructions):
//! - Neg, Abs, Sqrt, Recip
//! - Floor, Ceil, Round, RoundTiesEven, Trunc (AArch64 NEON has rounding modes)
//! - Sign (comparison-based)
//! - Square (mul x * x)
//!
//! Vectorized transcendentals (using math module):
//! - Exp, Exp2, Expm1, Log, Log2, Log10, Log1p, Cbrt, Rsqrt
//! - Sin, Cos, Tan, Asin, Acos, Atan
//! - Sinh, Cosh, Tanh, Asinh, Acosh, Atanh

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::{relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64};
use crate::ops::UnaryOp;
use crate::runtime::cpu::kernels::simd::math::aarch64::neon as math;

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// Check if operation has NEON SIMD support (native or math module)
#[inline]
const fn has_neon_support(op: UnaryOp) -> bool {
    matches!(
        op,
        // Native NEON instructions
        UnaryOp::Neg
            | UnaryOp::Abs
            | UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Recip
            | UnaryOp::Square
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::RoundTiesEven
            | UnaryOp::Trunc
            | UnaryOp::Sign
            // Transcendentals from math module
            | UnaryOp::Exp
            | UnaryOp::Exp2
            | UnaryOp::Expm1
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Log10
            | UnaryOp::Log1p
            | UnaryOp::Cbrt
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Asin
            | UnaryOp::Acos
            | UnaryOp::Atan
            | UnaryOp::Sinh
            | UnaryOp::Cosh
            | UnaryOp::Tanh
            | UnaryOp::Asinh
            | UnaryOp::Acosh
            | UnaryOp::Atanh
    )
}

/// NEON unary operation for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    // Fall back to scalar for unsupported ops
    if !has_neon_support(op) {
        unary_scalar_f32(op, a, out, len);
        return;
    }

    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;

    match op {
        // Native NEON instructions
        UnaryOp::Neg => unary_neg_f32(a, out, chunks),
        UnaryOp::Abs => unary_abs_f32(a, out, chunks),
        UnaryOp::Sqrt => unary_sqrt_f32(a, out, chunks),
        UnaryOp::Rsqrt => unary_rsqrt_f32(a, out, chunks),
        UnaryOp::Recip => unary_recip_f32(a, out, chunks),
        UnaryOp::Square => unary_square_f32(a, out, chunks),
        UnaryOp::Floor => unary_floor_f32(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f32(a, out, chunks),
        UnaryOp::Round => unary_round_f32(a, out, chunks),
        UnaryOp::RoundTiesEven => unary_round_ties_even_f32(a, out, chunks),
        UnaryOp::Trunc => unary_trunc_f32(a, out, chunks),
        UnaryOp::Sign => unary_sign_f32(a, out, chunks),
        // Transcendentals from math module
        UnaryOp::Exp => unary_transcendental_f32(a, out, chunks, math::exp_f32),
        UnaryOp::Exp2 => unary_transcendental_f32(a, out, chunks, math::exp2_f32),
        UnaryOp::Expm1 => unary_transcendental_f32(a, out, chunks, math::expm1_f32),
        UnaryOp::Log => unary_transcendental_f32(a, out, chunks, math::log_f32),
        UnaryOp::Log2 => unary_transcendental_f32(a, out, chunks, math::log2_f32),
        UnaryOp::Log10 => unary_transcendental_f32(a, out, chunks, math::log10_f32),
        UnaryOp::Log1p => unary_transcendental_f32(a, out, chunks, math::log1p_f32),
        UnaryOp::Cbrt => unary_transcendental_f32(a, out, chunks, math::cbrt_f32),
        UnaryOp::Sin => unary_transcendental_f32(a, out, chunks, math::sin_f32),
        UnaryOp::Cos => unary_transcendental_f32(a, out, chunks, math::cos_f32),
        UnaryOp::Tan => unary_transcendental_f32(a, out, chunks, math::tan_f32),
        UnaryOp::Asin => unary_transcendental_f32(a, out, chunks, math::asin_f32),
        UnaryOp::Acos => unary_transcendental_f32(a, out, chunks, math::acos_f32),
        UnaryOp::Atan => unary_transcendental_f32(a, out, chunks, math::atan_f32),
        UnaryOp::Sinh => unary_transcendental_f32(a, out, chunks, math::sinh_f32),
        UnaryOp::Cosh => unary_transcendental_f32(a, out, chunks, math::cosh_f32),
        UnaryOp::Tanh => unary_transcendental_f32(a, out, chunks, math::tanh_f32),
        UnaryOp::Asinh => unary_transcendental_f32(a, out, chunks, math::asinh_f32),
        UnaryOp::Acosh => unary_transcendental_f32(a, out, chunks, math::acosh_f32),
        UnaryOp::Atanh => unary_transcendental_f32(a, out, chunks, math::atanh_f32),
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        unary_scalar_f32(op, a.add(offset), out.add(offset), remainder);
    }
}

/// NEON unary operation for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - All pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    // Fall back to scalar for unsupported ops
    if !has_neon_support(op) {
        unary_scalar_f64(op, a, out, len);
        return;
    }

    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;

    match op {
        // Native NEON instructions
        UnaryOp::Neg => unary_neg_f64(a, out, chunks),
        UnaryOp::Abs => unary_abs_f64(a, out, chunks),
        UnaryOp::Sqrt => unary_sqrt_f64(a, out, chunks),
        UnaryOp::Rsqrt => unary_rsqrt_f64(a, out, chunks),
        UnaryOp::Recip => unary_recip_f64(a, out, chunks),
        UnaryOp::Square => unary_square_f64(a, out, chunks),
        UnaryOp::Floor => unary_floor_f64(a, out, chunks),
        UnaryOp::Ceil => unary_ceil_f64(a, out, chunks),
        UnaryOp::Round => unary_round_f64(a, out, chunks),
        UnaryOp::RoundTiesEven => unary_round_ties_even_f64(a, out, chunks),
        UnaryOp::Trunc => unary_trunc_f64(a, out, chunks),
        UnaryOp::Sign => unary_sign_f64(a, out, chunks),
        // Transcendentals from math module
        UnaryOp::Exp => unary_transcendental_f64(a, out, chunks, math::exp_f64),
        UnaryOp::Exp2 => unary_transcendental_f64(a, out, chunks, math::exp2_f64),
        UnaryOp::Expm1 => unary_transcendental_f64(a, out, chunks, math::expm1_f64),
        UnaryOp::Log => unary_transcendental_f64(a, out, chunks, math::log_f64),
        UnaryOp::Log2 => unary_transcendental_f64(a, out, chunks, math::log2_f64),
        UnaryOp::Log10 => unary_transcendental_f64(a, out, chunks, math::log10_f64),
        UnaryOp::Log1p => unary_transcendental_f64(a, out, chunks, math::log1p_f64),
        UnaryOp::Cbrt => unary_transcendental_f64(a, out, chunks, math::cbrt_f64),
        UnaryOp::Sin => unary_transcendental_f64(a, out, chunks, math::sin_f64),
        UnaryOp::Cos => unary_transcendental_f64(a, out, chunks, math::cos_f64),
        UnaryOp::Tan => unary_transcendental_f64(a, out, chunks, math::tan_f64),
        UnaryOp::Asin => unary_transcendental_f64(a, out, chunks, math::asin_f64),
        UnaryOp::Acos => unary_transcendental_f64(a, out, chunks, math::acos_f64),
        UnaryOp::Atan => unary_transcendental_f64(a, out, chunks, math::atan_f64),
        UnaryOp::Sinh => unary_transcendental_f64(a, out, chunks, math::sinh_f64),
        UnaryOp::Cosh => unary_transcendental_f64(a, out, chunks, math::cosh_f64),
        UnaryOp::Tanh => unary_transcendental_f64(a, out, chunks, math::tanh_f64),
        UnaryOp::Asinh => unary_transcendental_f64(a, out, chunks, math::asinh_f64),
        UnaryOp::Acosh => unary_transcendental_f64(a, out, chunks, math::acosh_f64),
        UnaryOp::Atanh => unary_transcendental_f64(a, out, chunks, math::atanh_f64),
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        unary_scalar_f64(op, a.add(offset), out.add(offset), remainder);
    }
}

/// NEON ReLU for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let remainder = len % F32_LANES;
    let zero = vdupq_n_f32(0.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vmaxq_f32(va, zero);
        vst1q_f32(out.add(offset), vr);
    }

    if remainder > 0 {
        let offset = chunks * F32_LANES;
        relu_scalar_f32(a.add(offset), out.add(offset), remainder);
    }
}

/// NEON ReLU for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let remainder = len % F64_LANES;
    let zero = vdupq_n_f64(0.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vmaxq_f64(va, zero);
        vst1q_f64(out.add(offset), vr);
    }

    if remainder > 0 {
        let offset = chunks * F64_LANES;
        relu_scalar_f64(a.add(offset), out.add(offset), remainder);
    }
}

// ============================================================================
// f32 kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_neg_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vnegq_f32(va);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_abs_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vabsq_f32(va);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_sqrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vsqrtq_f32(va);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_rsqrt_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = math::rsqrt_f32(va);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_recip_f32(a: *const f32, out: *mut f32, chunks: usize) {
    // NEON provides vrecpeq_f32 (approximate reciprocal)
    // followed by Newton-Raphson refinement for full precision
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));

        // Initial estimate
        let est = vrecpeq_f32(va);

        // Newton-Raphson refinement: est = est * (2 - va * est)
        // NEON provides vrecpsq_f32 which computes (2 - a*b)
        let step1 = vrecpsq_f32(va, est);
        let refined = vmulq_f32(est, step1);

        // Second refinement for better accuracy
        let step2 = vrecpsq_f32(va, refined);
        let vr = vmulq_f32(refined, step2);

        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_square_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vmulq_f32(va, va);
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_floor_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vrndmq_f32(va); // Round toward minus infinity
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_ceil_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vrndpq_f32(va); // Round toward plus infinity
        vst1q_f32(out.add(offset), vr);
    }
}

/// Round to nearest with ties away from zero, matching `f32::round`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_round_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vrndaq_f32(va); // FRINTA: round to nearest, ties away from zero
        vst1q_f32(out.add(offset), vr);
    }
}

/// Round to nearest with ties to even, matching `f32::round_ties_even`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_round_ties_even_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vrndnq_f32(va); // FRINTN: round to nearest, ties to even
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_trunc_f32(a: *const f32, out: *mut f32, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = vrndq_f32(va); // Round toward zero
        vst1q_f32(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_sign_f32(a: *const f32, out: *mut f32, chunks: usize) {
    let zero = vdupq_n_f32(0.0);
    let one = vdupq_n_f32(1.0);
    let neg_one = vdupq_n_f32(-1.0);

    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));

        // sign(x) = (x > 0) ? 1 : ((x < 0) ? -1 : 0)
        let pos_mask = vcgtq_f32(va, zero);
        let neg_mask = vcltq_f32(va, zero);

        // Select 1.0 where x > 0, -1.0 where x < 0, 0.0 otherwise
        let pos_part = vbslq_f32(pos_mask, one, zero);
        let vr = vbslq_f32(neg_mask, neg_one, pos_part);

        vst1q_f32(out.add(offset), vr);
    }
}

// ============================================================================
// f64 kernels
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_neg_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vnegq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_abs_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vabsq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_sqrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vsqrtq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_rsqrt_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = math::rsqrt_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_recip_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));

        // Initial estimate
        let est = vrecpeq_f64(va);

        // Newton-Raphson refinement
        let step1 = vrecpsq_f64(va, est);
        let refined = vmulq_f64(est, step1);

        // Second refinement
        let step2 = vrecpsq_f64(va, refined);
        let vr = vmulq_f64(refined, step2);

        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_square_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vmulq_f64(va, va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_floor_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vrndmq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_ceil_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vrndpq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

/// Round to nearest with ties away from zero, matching `f64::round`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_round_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vrndaq_f64(va); // FRINTA: ties away from zero
        vst1q_f64(out.add(offset), vr);
    }
}

/// Round to nearest with ties to even, matching `f64::round_ties_even`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_round_ties_even_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vrndnq_f64(va); // FRINTN: ties to even
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_trunc_f64(a: *const f64, out: *mut f64, chunks: usize) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = vrndq_f64(va);
        vst1q_f64(out.add(offset), vr);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_sign_f64(a: *const f64, out: *mut f64, chunks: usize) {
    let zero = vdupq_n_f64(0.0);
    let one = vdupq_n_f64(1.0);
    let neg_one = vdupq_n_f64(-1.0);

    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));

        let pos_mask = vcgtq_f64(va, zero);
        let neg_mask = vcltq_f64(va, zero);

        let pos_part = vbslq_f64(pos_mask, one, zero);
        let vr = vbslq_f64(neg_mask, neg_one, pos_part);

        vst1q_f64(out.add(offset), vr);
    }
}

// ============================================================================
// Transcendental helper functions
// ============================================================================

/// Apply a transcendental function from the math module to f32 chunks
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_transcendental_f32(
    a: *const f32,
    out: *mut f32,
    chunks: usize,
    func: unsafe fn(float32x4_t) -> float32x4_t,
) {
    for i in 0..chunks {
        let offset = i * F32_LANES;
        let va = vld1q_f32(a.add(offset));
        let vr = func(va);
        vst1q_f32(out.add(offset), vr);
    }
}

/// Apply a transcendental function from the math module to f64 chunks
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn unary_transcendental_f64(
    a: *const f64,
    out: *mut f64,
    chunks: usize,
    func: unsafe fn(float64x2_t) -> float64x2_t,
) {
    for i in 0..chunks {
        let offset = i * F64_LANES;
        let va = vld1q_f64(a.add(offset));
        let vr = func(va);
        vst1q_f64(out.add(offset), vr);
    }
}
