//! SIMD-accelerated f32 dot products for the half-precision GEMV-BT path.
//!
//! Split out of the matmul kernel module: these dispatch by ISA and carry no
//! matmul-specific logic. Only `gemv_bt_via_f32` uses them.

/// SIMD-accelerated f32 dot product for use in half-precision GEMV-BT.
///
/// Dispatches to AVX-512 or AVX2+FMA based on detected SIMD level.
/// Each backend is a separate function with `#[target_feature]` so the compiler
/// can optimize the entire function body for that ISA.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` f32 elements
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
pub(super) unsafe fn simd_dot_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
    level: super::super::simd::SimdLevel,
) -> f32 {
    use super::super::simd::SimdLevel;

    match level {
        SimdLevel::Avx512 => simd_dot_f32_avx512(a, b, len),
        SimdLevel::Avx2Fma => simd_dot_f32_avx2(a, b, len),
        _ => {
            let mut sum = 0.0f32;
            for i in 0..len {
                sum += *a.add(i) * *b.add(i);
            }
            sum
        }
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn simd_dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut offset = 0;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    while offset + 32 <= len {
        let av0 = _mm512_loadu_ps(a.add(offset));
        let bv0 = _mm512_loadu_ps(b.add(offset));
        acc0 = _mm512_fmadd_ps(av0, bv0, acc0);
        let av1 = _mm512_loadu_ps(a.add(offset + 16));
        let bv1 = _mm512_loadu_ps(b.add(offset + 16));
        acc1 = _mm512_fmadd_ps(av1, bv1, acc1);
        offset += 32;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    while offset + 16 <= len {
        let av = _mm512_loadu_ps(a.add(offset));
        let bv = _mm512_loadu_ps(b.add(offset));
        acc0 = _mm512_fmadd_ps(av, bv, acc0);
        offset += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc0);
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}

#[cfg(all(feature = "f16", target_arch = "aarch64"))]
#[inline]
pub(super) unsafe fn simd_dot_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
    _level: super::super::simd::SimdLevel,
) -> f32 {
    simd_dot_f32_neon(a, b, len)
}

#[cfg(all(feature = "f16", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn simd_dot_f32_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut offset = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    // Process 8 floats per iteration with dual accumulators
    while offset + 8 <= len {
        let av0 = vld1q_f32(a.add(offset));
        let bv0 = vld1q_f32(b.add(offset));
        acc0 = vfmaq_f32(acc0, av0, bv0);
        let av1 = vld1q_f32(a.add(offset + 4));
        let bv1 = vld1q_f32(b.add(offset + 4));
        acc1 = vfmaq_f32(acc1, av1, bv1);
        offset += 8;
    }
    acc0 = vaddq_f32(acc0, acc1);
    // Handle remaining 4-float chunk
    while offset + 4 <= len {
        let av = vld1q_f32(a.add(offset));
        let bv = vld1q_f32(b.add(offset));
        acc0 = vfmaq_f32(acc0, av, bv);
        offset += 4;
    }
    let mut sum = vaddvq_f32(acc0);
    // Scalar tail
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut offset = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    // Process 16 floats per iteration with 2 independent accumulators
    // to hide FMA latency (4-5 cycles on modern x86)
    while offset + 16 <= len {
        let av0 = _mm256_loadu_ps(a.add(offset));
        let bv0 = _mm256_loadu_ps(b.add(offset));
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        let av1 = _mm256_loadu_ps(a.add(offset + 8));
        let bv1 = _mm256_loadu_ps(b.add(offset + 8));
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        offset += 16;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Handle remaining 8-float chunk
    while offset + 8 <= len {
        let av = _mm256_loadu_ps(a.add(offset));
        let bv = _mm256_loadu_ps(b.add(offset));
        acc0 = _mm256_fmadd_ps(av, bv, acc0);
        offset += 8;
    }
    // Horizontal sum of 256-bit accumulator
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(sums2);
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}
