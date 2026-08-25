//! NEON i8 dot product kernels for ARM64
//!
//! Uses vmull_s8 + vpadalq_s16 for i8 x i8 → i32 accumulation.

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use crate::runtime::cpu::kernels::simd::dot::{DOT_SPILL_ITERS, saturate_i64_to_i32};

const I8_LANES: usize = 16; // 128-bit / 8-bit (process 8 at a time via vmull)

/// Dot product of signed i8 vectors, accumulated exactly and clamped to i32.
///
/// Processes 16 i8 elements per iteration using two vmull_s8 (low/high halves).
///
/// The i32 lanes are spilled into an i64 total every [`DOT_SPILL_ITERS`]
/// iterations. Without that they wrap after about a million elements and the
/// result comes back with the wrong sign.
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - Pointers must be valid for `len` elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let chunks = len / I8_LANES;
    let remainder = len % I8_LANES;

    let mut total = 0i64;
    let mut acc = vdupq_n_s32(0);

    for i in 0..chunks {
        let offset = i * I8_LANES;
        let va = vld1q_s8(a.add(offset));
        let vb = vld1q_s8(b.add(offset));

        // Multiply low 8 elements: i8 x i8 → 8x i16
        let prod_lo = vmull_s8(vget_low_s8(va), vget_low_s8(vb));
        // Multiply high 8 elements: i8 x i8 → 8x i16
        let prod_hi = vmull_s8(vget_high_s8(va), vget_high_s8(vb));

        // Pairwise add and accumulate i16 → i32
        acc = vpadalq_s16(acc, prod_lo);
        acc = vpadalq_s16(acc, prod_hi);

        if (i + 1) % DOT_SPILL_ITERS == 0 {
            total += hsum_s32_wide(acc);
            acc = vdupq_n_s32(0);
        }
    }

    total += hsum_s32_wide(acc);

    // Scalar tail
    for i in 0..remainder {
        let offset = chunks * I8_LANES + i;
        total += (*a.add(offset) as i64) * (*b.add(offset) as i64);
    }

    saturate_i64_to_i32(total)
}

/// Horizontal sum of 4 i32 lanes into an i64.
///
/// `vaddvq_s32` returns i32 and would overflow on its own: a lane can hold up
/// to `2^30` between spills, and four of those exceed i32.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hsum_s32_wide(v: int32x4_t) -> i64 {
    let mut lanes = [0i32; 4];
    vst1q_s32(lanes.as_mut_ptr(), v);
    lanes.iter().map(|&x| x as i64).sum()
}
