//! AVX2 i8 dot product kernels
//!
//! Uses i8 → i16 widening + _mm256_madd_epi16 for correct signed i8 x i8 → i32 accumulation.
//! Processes 32 elements per iteration (two 16-element halves widened to i16).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::runtime::cpu::kernels::simd::dot::{DOT_SPILL_ITERS, saturate_i64_to_i32};

const I8_LANES: usize = 32; // Process 32 i8s per iteration

/// Horizontal sum of 8 i32 lanes into an i64.
///
/// Widened deliberately: each lane can legitimately hold up to `2^30` between
/// spills, so summing eight of them in i32 would overflow even when no
/// individual lane has. Storing and adding in i64 costs one store plus eight
/// adds, and it runs once per `DOT_SPILL_ITERS` iterations.
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_wide(v: __m256i) -> i64 {
    let mut lanes = [0i32; 8];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, v);
    lanes.iter().map(|&x| x as i64).sum()
}

/// Dot product of signed i8 vectors, accumulated exactly and clamped to i32.
///
/// Strategy: Load 32 bytes, split into low/high 16 bytes, sign-extend to i16,
/// use _mm256_madd_epi16 (signed i16 pairs → i32) to accumulate.
///
/// The i32 lanes are spilled into an i64 total every [`DOT_SPILL_ITERS`]
/// iterations. Without that, the lanes wrap after about a million elements and
/// the result comes back with the wrong sign.
///
/// # Safety
/// - CPU must support AVX2
/// - Pointers must be valid for `len` elements
#[target_feature(enable = "avx2")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut total = 0i64;
    let mut acc = _mm256_setzero_si256();

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = _mm256_loadu_si256(a.add(offset) as *const __m256i);
        let vb = _mm256_loadu_si256(b.add(offset) as *const __m256i);

        // Process low 16 bytes: sign-extend i8 → i16
        let va_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(va));
        let vb_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vb));
        // madd: multiply pairs of i16, sum adjacent → i32
        let prod_lo = _mm256_madd_epi16(va_lo, vb_lo);
        acc = _mm256_add_epi32(acc, prod_lo);

        // Process high 16 bytes
        let va_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1));
        let vb_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1));
        let prod_hi = _mm256_madd_epi16(va_hi, vb_hi);
        acc = _mm256_add_epi32(acc, prod_hi);

        if (i + 1) % DOT_SPILL_ITERS == 0 {
            total += hsum_epi32_wide(acc);
            acc = _mm256_setzero_si256();
        }
    }

    total += hsum_epi32_wide(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        total += (*a.add(offset) as i64) * (*b.add(offset) as i64);
    }

    saturate_i64_to_i32(total)
}
