//! NEON matmul microkernels for ARM64
//!
//! Provides vectorized matrix multiplication microkernels using 128-bit NEON registers.
//!
//! # Microkernel Dimensions
//!
//! - f32: 6×4 (6 rows × 4 columns = 24 elements per microkernel invocation)
//! - f64: 6×2 (6 rows × 2 columns = 12 elements per microkernel invocation)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Matmul microkernel 6x4 for f32: C[0:6, 0:4] += A[0:6, 0:K] @ B[0:K, 0:4]
///
/// When `first_k` is true, accumulators start from zero (beta=0).
/// When false, they load from C and accumulate (beta=1).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x4_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5);

    if first_k {
        c0 = vdupq_n_f32(0.0);
        c1 = vdupq_n_f32(0.0);
        c2 = vdupq_n_f32(0.0);
        c3 = vdupq_n_f32(0.0);
        c4 = vdupq_n_f32(0.0);
        c5 = vdupq_n_f32(0.0);
    } else {
        c0 = vld1q_f32(c);
        c1 = vld1q_f32(c.add(ldc));
        c2 = vld1q_f32(c.add(ldc * 2));
        c3 = vld1q_f32(c.add(ldc * 3));
        c4 = vld1q_f32(c.add(ldc * 4));
        c5 = vld1q_f32(c.add(ldc * 5));
    }

    for kk in 0..k {
        let b_row = vld1q_f32(b.add(kk * 4));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f32(a_base);
        c0 = vfmaq_f32(c0, a0, b_row);

        let a1 = vld1q_dup_f32(a_base.add(1));
        c1 = vfmaq_f32(c1, a1, b_row);

        let a2 = vld1q_dup_f32(a_base.add(2));
        c2 = vfmaq_f32(c2, a2, b_row);

        let a3 = vld1q_dup_f32(a_base.add(3));
        c3 = vfmaq_f32(c3, a3, b_row);

        let a4 = vld1q_dup_f32(a_base.add(4));
        c4 = vfmaq_f32(c4, a4, b_row);

        let a5 = vld1q_dup_f32(a_base.add(5));
        c5 = vfmaq_f32(c5, a5, b_row);
    }

    vst1q_f32(c, c0);
    vst1q_f32(c.add(ldc), c1);
    vst1q_f32(c.add(ldc * 2), c2);
    vst1q_f32(c.add(ldc * 3), c3);
    vst1q_f32(c.add(ldc * 4), c4);
    vst1q_f32(c.add(ldc * 5), c5);
}

/// Matmul microkernel 6x2 for f64: C[0:6, 0:2] += A[0:6, 0:K] @ B[0:K, 0:2]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x2_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0, mut c1, mut c2, mut c3, mut c4, mut c5);

    if first_k {
        c0 = vdupq_n_f64(0.0);
        c1 = vdupq_n_f64(0.0);
        c2 = vdupq_n_f64(0.0);
        c3 = vdupq_n_f64(0.0);
        c4 = vdupq_n_f64(0.0);
        c5 = vdupq_n_f64(0.0);
    } else {
        c0 = vld1q_f64(c);
        c1 = vld1q_f64(c.add(ldc));
        c2 = vld1q_f64(c.add(ldc * 2));
        c3 = vld1q_f64(c.add(ldc * 3));
        c4 = vld1q_f64(c.add(ldc * 4));
        c5 = vld1q_f64(c.add(ldc * 5));
    }

    for kk in 0..k {
        let b_row = vld1q_f64(b.add(kk * 2));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f64(a_base);
        c0 = vfmaq_f64(c0, a0, b_row);

        let a1 = vld1q_dup_f64(a_base.add(1));
        c1 = vfmaq_f64(c1, a1, b_row);

        let a2 = vld1q_dup_f64(a_base.add(2));
        c2 = vfmaq_f64(c2, a2, b_row);

        let a3 = vld1q_dup_f64(a_base.add(3));
        c3 = vfmaq_f64(c3, a3, b_row);

        let a4 = vld1q_dup_f64(a_base.add(4));
        c4 = vfmaq_f64(c4, a4, b_row);

        let a5 = vld1q_dup_f64(a_base.add(5));
        c5 = vfmaq_f64(c5, a5, b_row);
    }

    vst1q_f64(c, c0);
    vst1q_f64(c.add(ldc), c1);
    vst1q_f64(c.add(ldc * 2), c2);
    vst1q_f64(c.add(ldc * 3), c3);
    vst1q_f64(c.add(ldc * 4), c4);
    vst1q_f64(c.add(ldc * 5), c5);
}

/// Matmul microkernel 6x8 for f32: C[0:6, 0:8] += A[0:6, 0:K] @ B[0:K, 0:8]
///
/// The double-width kernel for a FULL `NR = 8` block. It is not two
/// [`microkernel_6x4_f32`] calls: `pack_b` writes a full block INTERLEAVED, so
/// element `(kk, j)` sits at `kk * 8 + j`. Two half-width calls would read
/// `kk * 4` and `4 * k + kk * 4`, which is a different buffer entirely. Both
/// halves here belong to the SAME `kk`.
///
/// 6 rows x 2 vectors = 12 accumulators, matching the 12 FMA chains the AVX2
/// and AVX-512 double-width kernels keep in flight.
///
/// When `first_k` is true, accumulators start from zero (beta=0).
/// When false, they load from C and accumulate (beta=1).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x8_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0lo, mut c1lo, mut c2lo, mut c3lo, mut c4lo, mut c5lo);
    let (mut c0hi, mut c1hi, mut c2hi, mut c3hi, mut c4hi, mut c5hi);

    if first_k {
        c0lo = vdupq_n_f32(0.0);
        c1lo = vdupq_n_f32(0.0);
        c2lo = vdupq_n_f32(0.0);
        c3lo = vdupq_n_f32(0.0);
        c4lo = vdupq_n_f32(0.0);
        c5lo = vdupq_n_f32(0.0);
        c0hi = vdupq_n_f32(0.0);
        c1hi = vdupq_n_f32(0.0);
        c2hi = vdupq_n_f32(0.0);
        c3hi = vdupq_n_f32(0.0);
        c4hi = vdupq_n_f32(0.0);
        c5hi = vdupq_n_f32(0.0);
    } else {
        c0lo = vld1q_f32(c);
        c1lo = vld1q_f32(c.add(ldc));
        c2lo = vld1q_f32(c.add(ldc * 2));
        c3lo = vld1q_f32(c.add(ldc * 3));
        c4lo = vld1q_f32(c.add(ldc * 4));
        c5lo = vld1q_f32(c.add(ldc * 5));
        c0hi = vld1q_f32(c.add(4));
        c1hi = vld1q_f32(c.add(ldc + 4));
        c2hi = vld1q_f32(c.add(ldc * 2 + 4));
        c3hi = vld1q_f32(c.add(ldc * 3 + 4));
        c4hi = vld1q_f32(c.add(ldc * 4 + 4));
        c5hi = vld1q_f32(c.add(ldc * 5 + 4));
    }

    for kk in 0..k {
        let b_lo = vld1q_f32(b.add(kk * 8));
        let b_hi = vld1q_f32(b.add(kk * 8 + 4));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f32(a_base);
        c0lo = vfmaq_f32(c0lo, a0, b_lo);
        c0hi = vfmaq_f32(c0hi, a0, b_hi);

        let a1 = vld1q_dup_f32(a_base.add(1));
        c1lo = vfmaq_f32(c1lo, a1, b_lo);
        c1hi = vfmaq_f32(c1hi, a1, b_hi);

        let a2 = vld1q_dup_f32(a_base.add(2));
        c2lo = vfmaq_f32(c2lo, a2, b_lo);
        c2hi = vfmaq_f32(c2hi, a2, b_hi);

        let a3 = vld1q_dup_f32(a_base.add(3));
        c3lo = vfmaq_f32(c3lo, a3, b_lo);
        c3hi = vfmaq_f32(c3hi, a3, b_hi);

        let a4 = vld1q_dup_f32(a_base.add(4));
        c4lo = vfmaq_f32(c4lo, a4, b_lo);
        c4hi = vfmaq_f32(c4hi, a4, b_hi);

        let a5 = vld1q_dup_f32(a_base.add(5));
        c5lo = vfmaq_f32(c5lo, a5, b_lo);
        c5hi = vfmaq_f32(c5hi, a5, b_hi);
    }

    vst1q_f32(c, c0lo);
    vst1q_f32(c.add(ldc), c1lo);
    vst1q_f32(c.add(ldc * 2), c2lo);
    vst1q_f32(c.add(ldc * 3), c3lo);
    vst1q_f32(c.add(ldc * 4), c4lo);
    vst1q_f32(c.add(ldc * 5), c5lo);
    vst1q_f32(c.add(4), c0hi);
    vst1q_f32(c.add(ldc + 4), c1hi);
    vst1q_f32(c.add(ldc * 2 + 4), c2hi);
    vst1q_f32(c.add(ldc * 3 + 4), c3hi);
    vst1q_f32(c.add(ldc * 4 + 4), c4hi);
    vst1q_f32(c.add(ldc * 5 + 4), c5hi);
}

/// Matmul microkernel 6x4 for f64: C[0:6, 0:4] += A[0:6, 0:K] @ B[0:K, 0:4]
///
/// The f64 twin of [`microkernel_6x8_f32`], for a full `NR = 4` block packed
/// interleaved at `kk * 4 + j`. See that kernel for why this is not two
/// [`microkernel_6x2_f64`] calls.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn microkernel_6x4_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
    first_k: bool,
) {
    let (mut c0lo, mut c1lo, mut c2lo, mut c3lo, mut c4lo, mut c5lo);
    let (mut c0hi, mut c1hi, mut c2hi, mut c3hi, mut c4hi, mut c5hi);

    if first_k {
        c0lo = vdupq_n_f64(0.0);
        c1lo = vdupq_n_f64(0.0);
        c2lo = vdupq_n_f64(0.0);
        c3lo = vdupq_n_f64(0.0);
        c4lo = vdupq_n_f64(0.0);
        c5lo = vdupq_n_f64(0.0);
        c0hi = vdupq_n_f64(0.0);
        c1hi = vdupq_n_f64(0.0);
        c2hi = vdupq_n_f64(0.0);
        c3hi = vdupq_n_f64(0.0);
        c4hi = vdupq_n_f64(0.0);
        c5hi = vdupq_n_f64(0.0);
    } else {
        c0lo = vld1q_f64(c);
        c1lo = vld1q_f64(c.add(ldc));
        c2lo = vld1q_f64(c.add(ldc * 2));
        c3lo = vld1q_f64(c.add(ldc * 3));
        c4lo = vld1q_f64(c.add(ldc * 4));
        c5lo = vld1q_f64(c.add(ldc * 5));
        c0hi = vld1q_f64(c.add(2));
        c1hi = vld1q_f64(c.add(ldc + 2));
        c2hi = vld1q_f64(c.add(ldc * 2 + 2));
        c3hi = vld1q_f64(c.add(ldc * 3 + 2));
        c4hi = vld1q_f64(c.add(ldc * 4 + 2));
        c5hi = vld1q_f64(c.add(ldc * 5 + 2));
    }

    for kk in 0..k {
        let b_lo = vld1q_f64(b.add(kk * 4));
        let b_hi = vld1q_f64(b.add(kk * 4 + 2));
        let a_base = a.add(kk * 6);

        let a0 = vld1q_dup_f64(a_base);
        c0lo = vfmaq_f64(c0lo, a0, b_lo);
        c0hi = vfmaq_f64(c0hi, a0, b_hi);

        let a1 = vld1q_dup_f64(a_base.add(1));
        c1lo = vfmaq_f64(c1lo, a1, b_lo);
        c1hi = vfmaq_f64(c1hi, a1, b_hi);

        let a2 = vld1q_dup_f64(a_base.add(2));
        c2lo = vfmaq_f64(c2lo, a2, b_lo);
        c2hi = vfmaq_f64(c2hi, a2, b_hi);

        let a3 = vld1q_dup_f64(a_base.add(3));
        c3lo = vfmaq_f64(c3lo, a3, b_lo);
        c3hi = vfmaq_f64(c3hi, a3, b_hi);

        let a4 = vld1q_dup_f64(a_base.add(4));
        c4lo = vfmaq_f64(c4lo, a4, b_lo);
        c4hi = vfmaq_f64(c4hi, a4, b_hi);

        let a5 = vld1q_dup_f64(a_base.add(5));
        c5lo = vfmaq_f64(c5lo, a5, b_lo);
        c5hi = vfmaq_f64(c5hi, a5, b_hi);
    }

    vst1q_f64(c, c0lo);
    vst1q_f64(c.add(ldc), c1lo);
    vst1q_f64(c.add(ldc * 2), c2lo);
    vst1q_f64(c.add(ldc * 3), c3lo);
    vst1q_f64(c.add(ldc * 4), c4lo);
    vst1q_f64(c.add(ldc * 5), c5lo);
    vst1q_f64(c.add(2), c0hi);
    vst1q_f64(c.add(ldc + 2), c1hi);
    vst1q_f64(c.add(ldc * 2 + 2), c2hi);
    vst1q_f64(c.add(ldc * 3 + 2), c3hi);
    vst1q_f64(c.add(ldc * 4 + 2), c4hi);
    vst1q_f64(c.add(ldc * 5 + 2), c5hi);
}
