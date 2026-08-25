//! SIMD-accelerated integer dot product operations
//!
//! Provides high-throughput i8 x i8 → i32 dot products for quantized inference.
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set  | Elements/cycle | Key Intrinsic          |
//! |--------------|------------------|----------------|------------------------|
//! | x86-64       | AVX-512BW        | 64             | maddubs + madd         |
//! | x86-64       | AVX2             | 32             | maddubs + madd         |
//! | ARM64        | NEON             | 16             | vmull_s8 + vpadalq_s16 |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use super::{SimdLevel, detect_simd};

/// Minimum elements to justify SIMD overhead for dot products
const DOT_SIMD_THRESHOLD: usize = 32;

/// SIMD iterations between spills of the i32 lane accumulator into an i64 total.
///
/// Every backend here accumulates products in i32 SIMD lanes, and every one of
/// them adds at most `2^16` to a single lane per iteration: a product of two i8
/// is bounded by `128 * 128 = 2^14`, and each lane receives at most four of them
/// per iteration. `2^14 * 2^16 = 2^30` stays inside i32, so spilling this often
/// is provably safe with a full bit of headroom.
///
/// Without a spill the lanes wrap after roughly a million elements and the dot
/// product returns a value with the wrong sign — silently, in release.
pub(super) const DOT_SPILL_ITERS: usize = 16_384;

/// Narrow an exact i64 total to the i32 this op returns, clamping on overflow.
///
/// Saturating rather than wrapping, matching
/// [`crate::runtime::cpu::kernels::wide_acc`]: a wrapped total reports the
/// wrong sign and magnitude, while a clamped one is at least ordered correctly
/// and stays a total function.
#[inline]
pub(super) fn saturate_i64_to_i32(acc: i64) -> i32 {
    acc.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

/// Dot product of signed i8 vectors, accumulated in i32.
///
/// Automatically dispatches to the best SIMD implementation available:
/// - x86-64/AVX-512BW: 64 elements per iteration via `_mm512_maddubs_epi16` + `_mm512_madd_epi16`
/// - x86-64/AVX2: 32 elements per iteration via `_mm256_maddubs_epi16` + `_mm256_madd_epi16`
/// - ARM64/NEON: 16 elements per iteration via `vmull_s8` + `vpadalq_s16`
/// - Scalar fallback for small arrays (<32 elements) or unsupported platforms
///
/// Computes sum(a[i] * b[i]) for i in 0..len.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` elements
#[inline]
pub unsafe fn i8xi8_dot_i32(a: *const i8, b: *const i8, len: usize) -> i32 {
    let level = detect_simd();

    if len < DOT_SIMD_THRESHOLD || level == SimdLevel::Scalar {
        return i8xi8_dot_scalar(a, b, len);
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => return x86_64::avx512::i8xi8_dot_i32(a, b, len),
        SimdLevel::Avx2Fma => return x86_64::avx2::i8xi8_dot_i32(a, b, len),
        _ => return i8xi8_dot_scalar(a, b, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => return aarch64::neon::i8xi8_dot_i32(a, b, len),
        _ => return i8xi8_dot_scalar(a, b, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    i8xi8_dot_scalar(a, b, len)
}

/// Scaled dot product of signed i8 vectors, returning f32.
///
/// Computes scale * sum(a[i] * b[i]) for i in 0..len.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` elements
#[inline]
#[allow(dead_code)] // Public API for downstream crates (e.g., boostr quantized ops)
pub unsafe fn i8xi8_dot_f32(a: *const i8, b: *const i8, scale: f32, len: usize) -> f32 {
    (i8xi8_dot_i32(a, b, len) as f32) * scale
}

/// Scalar fallback for i8 dot product.
///
/// Accumulates in i64 and clamps once at the end. An i32 accumulator wraps
/// after about 131k terms (`i32::MAX / 128^2`), which is well inside the sizes
/// a quantized matmul reaches along K. The i64 accumulator cannot overflow for
/// any reachable length: it would take `2^63 / 2^14` terms.
#[inline]
unsafe fn i8xi8_dot_scalar(a: *const i8, b: *const i8, len: usize) -> i32 {
    let mut acc = 0i64;
    for i in 0..len {
        acc += (*a.add(i) as i64) * (*b.add(i) as i64);
    }
    saturate_i64_to_i32(acc)
}

#[cfg(test)]
mod tests;
