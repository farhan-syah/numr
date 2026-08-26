//! AVX-512 conditional select (where) kernels
//!
//! Uses mask-based blend operations.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{where_scalar_f32, where_scalar_f64};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 where for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn where_f32(cond: *const u8, x: *const f32, y: *const f32, out: *mut f32, len: usize) {
    let chunks = len / F32_LANES;
    let zeros = _mm_setzero_si128();

    for c in 0..chunks {
        let offset = c * F32_LANES;

        // Load 16 condition bytes
        let cond_bytes = _mm_loadu_si128(cond.add(offset) as *const __m128i);

        // Compare bytes to zero: result is 0xFF where cond != 0, 0x00 where cond == 0
        let cmp_result = _mm_cmpgt_epi8(cond_bytes, zeros);

        // Convert to 16-bit mask
        let mask = _mm_movemask_epi8(cmp_result) as u16;

        // Load x and y values
        let vx = _mm512_loadu_ps(x.add(offset));
        let vy = _mm512_loadu_ps(y.add(offset));

        // Blend: where mask bit is 1, select x; where 0, select y
        let result = _mm512_mask_blend_ps(mask, vy, vx);
        _mm512_storeu_ps(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F32_LANES;
    if processed < len {
        where_scalar_f32(
            cond.add(processed),
            x.add(processed),
            y.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

/// AVX-512 where for f64
#[target_feature(enable = "avx512f")]
pub unsafe fn where_f64(cond: *const u8, x: *const f64, y: *const f64, out: *mut f64, len: usize) {
    let chunks = len / F64_LANES;
    let zeros = _mm_setzero_si128();

    for c in 0..chunks {
        let offset = c * F64_LANES;

        // Load 8 condition bytes (we only need lower 8 bytes for f64 lanes)
        let cond_bytes = _mm_loadl_epi64(cond.add(offset) as *const __m128i);

        // Compare bytes to zero
        let cmp_result = _mm_cmpgt_epi8(cond_bytes, zeros);

        // Convert to 8-bit mask (only lower 8 bits are valid)
        let mask = (_mm_movemask_epi8(cmp_result) & 0xFF) as u8;

        // Load x and y values
        let vx = _mm512_loadu_pd(x.add(offset));
        let vy = _mm512_loadu_pd(y.add(offset));

        // Blend
        let result = _mm512_mask_blend_pd(mask, vy, vx);
        _mm512_storeu_pd(out.add(offset), result);
    }

    // Scalar tail
    let processed = chunks * F64_LANES;
    if processed < len {
        where_scalar_f64(
            cond.add(processed),
            x.add(processed),
            y.add(processed),
            out.add(processed),
            len - processed,
        );
    }
}

// Suppress unused warnings
const _: () = {
    let _ = where_scalar_f32 as unsafe fn(*const u8, *const f32, *const f32, *mut f32, usize);
    let _ = where_scalar_f64 as unsafe fn(*const u8, *const f64, *const f64, *mut f64, usize);
};
