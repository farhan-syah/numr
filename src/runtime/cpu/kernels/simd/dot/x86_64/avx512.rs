//! AVX-512 i8 dot product kernels
//!
//! Uses i8 → i16 widening + _mm512_madd_epi16 for correct signed i8 x i8 → i32 accumulation.
//! Processes 64 elements per iteration (two 32-element halves widened to i16).

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::runtime::cpu::kernels::simd::dot::{DOT_SPILL_ITERS, saturate_i64_to_i32};

const I8_LANES: usize = 64; // Process 64 i8s per iteration

/// Dot product of signed i8 vectors using AVX-512BW, accumulated exactly and
/// clamped to i32.
///
/// The i32 lanes are spilled into an i64 total every [`DOT_SPILL_ITERS`]
/// iterations; see the AVX2 kernel for the bound.
///
/// Strategy: Load 64 bytes, split into low/high 32 bytes, sign-extend to i16,
/// use _mm512_madd_epi16 (signed i16 pairs → i32) to accumulate.
///
/// # Safety
/// - CPU must support AVX-512F + AVX-512BW
/// - Pointers must be valid for `len` elements
#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut total = 0i64;
    let mut acc = _mm512_setzero_si512();

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = _mm512_loadu_si512(a.add(offset) as *const __m512i);
        let vb = _mm512_loadu_si512(b.add(offset) as *const __m512i);

        // Process low 32 bytes: sign-extend i8 → i16 in 512-bit
        let va_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(va));
        let vb_lo = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(vb));
        let prod_lo = _mm512_madd_epi16(va_lo, vb_lo);
        acc = _mm512_add_epi32(acc, prod_lo);

        // Process high 32 bytes
        let va_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(va, 1));
        let vb_hi = _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(vb, 1));
        let prod_hi = _mm512_madd_epi16(va_hi, vb_hi);
        acc = _mm512_add_epi32(acc, prod_hi);

        if (i + 1) % DOT_SPILL_ITERS == 0 {
            total += hsum_epi32_512_wide(acc);
            acc = _mm512_setzero_si512();
        }
    }

    total += hsum_epi32_512_wide(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        total += (*a.add(offset) as i64) * (*b.add(offset) as i64);
    }

    saturate_i64_to_i32(total)
}

/// Horizontal sum of 16 i32 lanes into an i64.
///
/// Widened for the same reason as the AVX2 version: a lane can hold `2^30`
/// between spills, so summing sixteen of them in i32 would overflow even when
/// no lane has.
#[target_feature(enable = "avx512f")]
unsafe fn hsum_epi32_512_wide(v: __m512i) -> i64 {
    let mut lanes = [0i32; 16];
    _mm512_storeu_si512(lanes.as_mut_ptr() as *mut __m512i, v);
    lanes.iter().map(|&x| x as i64).sum()
}
