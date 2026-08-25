//! AVX2 i32 matmul with a 64-bit accumulator, guarded by a magnitude prescan.
//!
//! # Why this is guarded rather than unconditional
//!
//! `i32 * i32` reaches `2^62`, so two products can already overflow i64. An
//! unconditional i64 SIMD path would therefore be exact for the small integers
//! most callers use and silently wrong for the rest — the same class of defect
//! as the `_mm256_add_epi32` accumulator this replaces, just with a higher
//! threshold.
//!
//! Instead [`matmul_i32_fits_i64`] measures the operands first. The bound
//! `max|a| * max|b| * k <= i64::MAX` is sufficient for every partial sum, and
//! the scan is `O(mk + kn)` against the matmul's `O(mnk)` — free at any size
//! where SIMD would matter. When the bound does not hold, the caller keeps the
//! exact i128 scalar path.
//!
//! Results are bit-identical to that scalar path within the guarded range: both
//! compute the exact integer sum and clamp once at write-out.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Largest magnitude in a strided i32 matrix, as i64.
///
/// Returns i64 because `-i32::MIN` does not fit i32.
unsafe fn max_abs(p: *const i32, rows: usize, cols: usize, stride: usize) -> i64 {
    let mut m = 0i64;
    for r in 0..rows {
        for c in 0..cols {
            let v = (*p.add(r * stride + c) as i64).abs();
            if v > m {
                m = v;
            }
        }
    }
    m
}

/// Whether every partial sum of this matmul fits i64.
///
/// `max|a| * max|b| * k` bounds the largest dot product in absolute value.
/// Checked with i128 arithmetic so the test itself cannot overflow.
///
/// # Safety
/// - `a` must be valid for `m * lda` i32 elements
/// - `b` must be valid for `k * ldb` i32 elements
pub unsafe fn matmul_i32_fits_i64(
    a: *const i32,
    b: *const i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
) -> bool {
    if k == 0 || m == 0 || n == 0 {
        return true;
    }
    let bound = (max_abs(a, m, k, lda) as i128) * (max_abs(b, k, n, ldb) as i128) * (k as i128);
    bound <= i64::MAX as i128
}

/// Number of i64 accumulator lanes held in one AVX2 register.
const LANES: usize = 4;

/// `C = A @ B` for i32, accumulating in i64 and clamping to i32 on write-out.
///
/// Caller MUST have checked [`matmul_i32_fits_i64`]; this does not re-check.
///
/// One row of i64 accumulators is held across the `k` loop, matching the scalar
/// wide-accumulator kernel's `ikj` order so B is walked contiguously.
///
/// # Safety
/// - CPU must support AVX2
/// - `a` valid for `m * lda`, `b` for `k * ldb`, `out` for `m * ldc` elements
/// - `out` must not alias `a` or `b`
/// - every partial sum must fit i64 (see [`matmul_i32_fits_i64`])
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i32_avx2(
    a: *const i32,
    b: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let mut row_acc = vec![0i64; n];
    let chunks = n / LANES;
    let tail = n % LANES;

    for i in 0..m {
        row_acc.iter_mut().for_each(|s| *s = 0);

        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            if a_val == 0 {
                continue;
            }
            // Broadcast into 64-bit lanes. `_mm256_mul_epi32` reads bits 31:0 of
            // each lane and sign-extends them, so the low half of each lane
            // carrying `a_val` is exactly what it needs.
            let va = _mm256_set1_epi64x(a_val as i64);
            let b_row = b.add(kk * ldb);

            for c in 0..chunks {
                let j = c * LANES;
                // Sign-extend 4 i32 to 4 i64 so both operands present their
                // value in the low half of a 64-bit lane.
                let vb = _mm256_cvtepi32_epi64(_mm_loadu_si128(b_row.add(j) as *const __m128i));
                let prod = _mm256_mul_epi32(va, vb);
                let slot = row_acc.as_mut_ptr().add(j) as *mut __m256i;
                _mm256_storeu_si256(slot, _mm256_add_epi64(_mm256_loadu_si256(slot), prod));
            }

            for j in (chunks * LANES)..(chunks * LANES + tail) {
                row_acc[j] += a_val as i64 * *b_row.add(j) as i64;
            }
        }

        for (j, &slot) in row_acc.iter().enumerate() {
            *out.add(i * ldc + j) = slot.clamp(i32::MIN as i64, i32::MAX as i64) as i32;
        }
    }
}

#[cfg(test)]
mod tests;
