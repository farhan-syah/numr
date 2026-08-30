//! SIMD-optimized matrix multiplication with cache-aware tiling
//!
//! This module provides the tiled matmul algorithm that dispatches to
//! SIMD microkernels based on runtime CPU feature detection.
//!
//! # Algorithm Overview (BLIS-style)
//!
//! ```text
//! for jc in (0..N).step_by(NC):           # L3 cache blocking
//!   for pc in (0..K).step_by(KC):         # L2 cache blocking
//!     pack B[pc:pc+KC, jc:jc+NC] → B̃       # Pack B panel
//!     for ic in (0..M).step_by(MC):       # L2 cache blocking
//!       pack A[ic:ic+MC, pc:pc+KC] → Ã    # Pack A panel
//!       for jr in (0..NC).step_by(NR):    # Microkernel loop
//!         for ir in (0..MC).step_by(MR):
//!           microkernel(Ã[ir], B̃[jr], C[ic+ir, jc+jr])
//! ```
//!
//! # Microkernel Dimensions
//!
//! | SIMD Level | f32 (MR×NR) | f64 (MR×NR) |
//! |------------|-------------|-------------|
//! | AVX-512    | 6×16        | 6×8         |
//! | AVX2+FMA   | 6×8         | 6×4         |
//! | Scalar     | 6×4         | 6×4         |

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;
use super::gemv_bt;
use super::scalar::{matmul_bias_scalar_f32, matmul_bias_scalar_f64};
use super::scalar::{matmul_scalar_f32, matmul_scalar_f64};
use super::scalar::{microkernel_edge_f32, microkernel_edge_f64};
use super::small;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::tiling::{matmul_bias_tiled_f32, matmul_bias_tiled_f64};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::tiling::{matmul_bt_tiled_f32, matmul_bt_tiled_f64};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use super::tiling::{matmul_tiled_f32, matmul_tiled_f64};
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

// ============================================================================
// Constants
// ============================================================================

/// Micro-kernel row dimension (Mr)
pub const MR: usize = 6;

/// L3 cache blocking: M dimension (Mc)
/// Must be a multiple of MR to avoid buffer overflow in packing.
pub const MC: usize = 126; // 21 * MR(6)

/// L2 cache blocking: K dimension (Kc)
/// Sized so packed_A (MC×KC×4) fits in L2 cache (~256KB):
/// 126 × 256 × 4 = 129KB
pub const KC: usize = 256;

/// L3 cache blocking: N dimension (Nc)
pub const NC: usize = 512;

/// Small matrix threshold - below this, register-blocked SIMD is faster than tiled
const SMALL_MATRIX_THRESHOLD: usize = 128 * 128 * 128 + 1;

// ============================================================================
// Public API
// ============================================================================

/// SIMD-optimized matrix multiplication: C = A @ B
///
/// Dispatches to the best available SIMD implementation based on CPU features.
/// Falls back to scalar for unsupported CPUs or small matrices.
///
/// # Safety
/// - All pointers must be valid for the specified dimensions
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        small::small_matmul_f32(a, b, out, m, n, k, lda, ldb, ldc, level);
        return;
    }

    // Use double-width NR for 12 FMA chains (2×NR columns per microkernel)
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => matmul_tiled_f32::<32>(a, b, out, m, n, k, lda, ldb, ldc, level),
        SimdLevel::Avx2Fma => matmul_tiled_f32::<16>(a, b, out, m, n, k, lda, ldb, ldc, level),
        _ => matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_tiled_f32::<8>(a, b, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    matmul_scalar_f32(a, b, out, m, n, k, lda, ldb, ldc);
}

/// SIMD-optimized matrix multiplication for f64
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        small::small_matmul_f64(a, b, out, m, n, k, lda, ldb, ldc, level);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => matmul_tiled_f64::<16>(a, b, out, m, n, k, lda, ldb, ldc, level),
        SimdLevel::Avx2Fma => matmul_tiled_f64::<8>(a, b, out, m, n, k, lda, ldb, ldc, level),
        _ => matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_tiled_f64::<4>(a, b, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    matmul_scalar_f64(a, b, out, m, n, k, lda, ldb, ldc);
}

/// Do the transposed-B entry points take the tiled path for this shape?
///
/// The tiled path is the one that packs B panels straight out of the `[N, K]`
/// buffer, producing panels byte-identical to a materialized `[K, N]` operand —
/// so on that path, and only on that path, [`matmul_bt_f32`] and [`matmul_f32`]
/// agree bit for bit. Anywhere else the transposed operand runs a different
/// kernel (the dot-product GEMV-BT one), which agrees within tolerance but
/// accumulates in its own order. Callers that require identity gate on this.
pub fn matmul_bt_is_tiled(m: usize, n: usize, k: usize) -> bool {
    m.saturating_mul(n).saturating_mul(k) >= SMALL_MATRIX_THRESHOLD
        && tiled_level_available(detect_simd())
}

/// Fewest output columns that keep an `m × n × k` product on the tiled path.
///
/// A caller splitting N across threads uses this as its chunk floor. A chunk
/// narrower than this drops below `SMALL_MATRIX_THRESHOLD` and runs the
/// small-matrix kernel instead, which both accumulates in a different order
/// and — for a transposed B — is a *different* kernel from the contiguous
/// case, breaking the bit-for-bit agreement [`matmul_bt_is_tiled`] promises.
///
/// Returns `usize::MAX` when `m * k` is zero, so no split can clear the floor.
pub fn min_tiled_columns(m: usize, k: usize) -> usize {
    let rows = m.saturating_mul(k);
    if rows == 0 {
        return usize::MAX;
    }
    SMALL_MATRIX_THRESHOLD.div_ceil(rows)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(target_arch = "x86_64")]
fn tiled_level_available(level: SimdLevel) -> bool {
    matches!(level, SimdLevel::Avx512 | SimdLevel::Avx2Fma)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(target_arch = "aarch64")]
fn tiled_level_available(level: SimdLevel) -> bool {
    matches!(level, SimdLevel::Neon | SimdLevel::NeonFp16)
}

/// Does this SIMD level have a tiled microkernel, rather than the scalar path?
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn tiled_level_available(_level: SimdLevel) -> bool {
    false
}

/// SIMD matrix multiplication with a transposed B operand: C = A @ B
///
/// `b` holds the logical `[K, N]` operand as a contiguous `[N, K]` buffer — the
/// layout a transposed weight matrix already has. Nothing is materialized: the
/// tiled kernel packs its B panels straight out of that buffer, and the packed
/// panels are byte-identical to the ones a materialized `[K, N]` operand would
/// produce, so the result matches [`matmul_f32`] exactly.
///
/// Below the small-matrix threshold the dot-product GEMV-BT kernel wins, which
/// is the same kernel the decode path uses.
///
/// # Safety
/// - `a` must be valid for `m * k` contiguous elements (row stride `k`)
/// - `b` must be valid for `n * k` contiguous elements (row stride `k`)
/// - `out` must be valid for `m * ldc` writable elements
/// - `out` must not alias `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bt_f32(
    a: *const f32,
    b: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        gemv_bt::gemv_bt_f32(a, b, out, m, n, k, ldc, level);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => matmul_bt_tiled_f32::<32>(a, b, out, m, n, k, k, k, ldc, level),
        SimdLevel::Avx2Fma => matmul_bt_tiled_f32::<16>(a, b, out, m, n, k, k, k, ldc, level),
        _ => gemv_bt::gemv_bt_f32(a, b, out, m, n, k, ldc, level),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_bt_tiled_f32::<8>(a, b, out, m, n, k, k, k, ldc, level)
        }
        _ => gemv_bt::gemv_bt_f32(a, b, out, m, n, k, ldc, level),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gemv_bt::gemv_bt_f32(a, b, out, m, n, k, ldc, level);
}

/// SIMD matrix multiplication with a transposed B operand for f64
///
/// See [`matmul_bt_f32`].
///
/// # Safety
/// Same as [`matmul_bt_f32`].
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bt_f64(
    a: *const f64,
    b: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        gemv_bt::gemv_bt_f64(a, b, out, m, n, k, ldc, level);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => matmul_bt_tiled_f64::<16>(a, b, out, m, n, k, k, k, ldc, level),
        SimdLevel::Avx2Fma => matmul_bt_tiled_f64::<8>(a, b, out, m, n, k, k, k, ldc, level),
        _ => gemv_bt::gemv_bt_f64(a, b, out, m, n, k, ldc, level),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_bt_tiled_f64::<4>(a, b, out, m, n, k, k, k, ldc, level)
        }
        _ => gemv_bt::gemv_bt_f64(a, b, out, m, n, k, ldc, level),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    gemv_bt::gemv_bt_f64(a, b, out, m, n, k, ldc, level);
}

/// Fused matmul with bias: C = A @ B + bias (single-pass, cache-efficient)
///
/// Initializes C with bias, then accumulates the matmul result.
/// This is more cache-efficient than separate matmul + bias addition.
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_f32(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    out: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        small::small_matmul_bias_f32(a, b, bias, out, m, n, k, lda, ldb, ldc, level);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => {
            matmul_bias_tiled_f32::<32>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        SimdLevel::Avx2Fma => {
            matmul_bias_tiled_f32::<16>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_bias_tiled_f32::<8>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    matmul_bias_scalar_f32(a, b, bias, out, m, n, k, lda, ldb, ldc);
}

/// Fused matmul with bias for f64
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_f64(
    a: *const f64,
    b: *const f64,
    bias: *const f64,
    out: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let level = detect_simd();

    if m * n * k < SMALL_MATRIX_THRESHOLD {
        small::small_matmul_bias_f64(a, b, bias, out, m, n, k, lda, ldb, ldc, level);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => {
            matmul_bias_tiled_f64::<16>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        SimdLevel::Avx2Fma => {
            matmul_bias_tiled_f64::<8>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            matmul_bias_tiled_f64::<4>(a, b, bias, out, m, n, k, lda, ldb, ldc, level)
        }
        _ => matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    matmul_bias_scalar_f64(a, b, bias, out, m, n, k, lda, ldb, ldc);
}

// ============================================================================
// Microkernel dispatch
// ============================================================================

/// Dispatch to the appropriate SIMD microkernel for f32 (single-width NR)
///
/// `first_k`: when true, accumulators start from zero (beta=0, no load from C).
#[inline]
pub unsafe fn call_microkernel_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::microkernel_6x16_f32(a, b, c, k, ldc, first_k),
        SimdLevel::Avx2Fma => avx2::microkernel_6x8_f32(a, b, c, k, ldc, first_k),
        _ => microkernel_edge_f32(a, b, c, MR, 4, k, ldc, first_k),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::microkernel_6x4_f32(a, b, c, k, ldc, first_k)
        }
        _ => microkernel_edge_f32(a, b, c, MR, 4, k, ldc, first_k),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    microkernel_edge_f32(a, b, c, MR, 4, k, ldc, first_k);
}

/// Dispatch to the double-width SIMD microkernel for f32 (2×NR columns)
///
/// Processes 6 rows × 2*NR columns = 12 independent FMA chains.
#[inline]
pub unsafe fn call_microkernel_2x_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    k: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::microkernel_6x32_f32(a, b, c, k, ldc, first_k),
        SimdLevel::Avx2Fma => avx2::microkernel_6x16_f32(a, b, c, k, ldc, first_k),
        // One call over the whole block: `pack_b` interleaves a full block at
        // `kk * nr + j`, so two half-width calls would read the wrong buffer.
        _ => microkernel_edge_f32(a, b, c, MR, 8, k, ldc, first_k),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::microkernel_6x8_f32(a, b, c, k, ldc, first_k)
        }
        _ => microkernel_edge_f32(a, b, c, MR, 8, k, ldc, first_k),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    microkernel_edge_f32(a, b, c, MR, 8, k, ldc, first_k);
}

/// Dispatch to the appropriate SIMD microkernel for f64 (single-width NR)
#[inline]
pub unsafe fn call_microkernel_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::microkernel_6x8_f64(a, b, c, k, ldc, first_k),
        SimdLevel::Avx2Fma => avx2::microkernel_6x4_f64(a, b, c, k, ldc, first_k),
        _ => microkernel_edge_f64(a, b, c, MR, 4, k, ldc, first_k),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::microkernel_6x2_f64(a, b, c, k, ldc, first_k)
        }
        _ => microkernel_edge_f64(a, b, c, MR, 2, k, ldc, first_k),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    microkernel_edge_f64(a, b, c, MR, 4, k, ldc, first_k);
}

/// Dispatch to the double-width SIMD microkernel for f64 (2×NR columns)
#[inline]
pub unsafe fn call_microkernel_2x_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    k: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::microkernel_6x16_f64(a, b, c, k, ldc, first_k),
        SimdLevel::Avx2Fma => avx2::microkernel_6x8_f64(a, b, c, k, ldc, first_k),
        _ => microkernel_edge_f64(a, b, c, MR, 8, k, ldc, first_k),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            aarch64::neon::microkernel_6x4_f64(a, b, c, k, ldc, first_k)
        }
        _ => microkernel_edge_f64(a, b, c, MR, 4, k, ldc, first_k),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    microkernel_edge_f64(a, b, c, MR, 8, k, ldc, first_k);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_matmul_f32(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn reference_matmul_f64(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    fn reference_matmul_bias_f32(
        a: &[f32],
        b: &[f32],
        bias: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let mut c = reference_matmul_f32(a, b, m, n, k);
        for i in 0..m {
            for j in 0..n {
                c[i * n + j] += bias[j];
            }
        }
        c
    }

    const F32_SMALL_TOL: f32 = 1e-4;
    const F32_LARGE_TOL: f32 = 1e-3;
    const F64_SMALL_TOL: f64 = 1e-10;
    const F64_LARGE_TOL: f64 = 1e-9;

    #[test]
    fn test_matmul_f32_small() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_matmul_f32(&a, &b, m, n, k);

        unsafe { matmul_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        for i in 0..m * n {
            assert!((c[i] - expected[i]).abs() < F32_SMALL_TOL);
        }
    }

    #[test]
    fn test_matmul_f32_large() {
        let (m, n, k) = (128, 128, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_matmul_f32(&a, &b, m, n, k);

        unsafe { matmul_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        let max_diff = (0..m * n)
            .map(|i| (c[i] - expected[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < F32_LARGE_TOL);
    }

    #[test]
    fn test_matmul_f64_small() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f64> = (0..m * k).map(|i| (i + 1) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i + 1) as f64).collect();
        let mut c = vec![0.0f64; m * n];
        let expected = reference_matmul_f64(&a, &b, m, n, k);

        unsafe { matmul_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        for i in 0..m * n {
            assert!((c[i] - expected[i]).abs() < F64_SMALL_TOL);
        }
    }

    #[test]
    fn test_matmul_f64_large() {
        let (m, n, k) = (128, 128, 128);
        let a: Vec<f64> = (0..m * k).map(|i| ((i % 17) as f64) * 0.1).collect();
        let b: Vec<f64> = (0..k * n).map(|i| ((i % 13) as f64) * 0.1).collect();
        let mut c = vec![0.0f64; m * n];
        let expected = reference_matmul_f64(&a, &b, m, n, k);

        unsafe { matmul_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        let max_diff = (0..m * n)
            .map(|i| (c[i] - expected[i]).abs())
            .fold(0.0f64, f64::max);
        assert!(max_diff < F64_LARGE_TOL);
    }

    #[test]
    fn test_matmul_non_square() {
        let (m, n, k) = (37, 53, 41);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32) * 0.5).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32) * 0.3).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_matmul_f32(&a, &b, m, n, k);

        unsafe { matmul_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        let max_diff = (0..m * n)
            .map(|i| (c[i] - expected[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < F32_LARGE_TOL);
    }

    #[test]
    fn test_matmul_bias_f32_small() {
        let (m, n, k) = (4, 4, 4);
        let a: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let bias: Vec<f32> = (0..n).map(|i| (i * 10) as f32).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_matmul_bias_f32(&a, &b, &bias, m, n, k);

        unsafe {
            matmul_bias_f32(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                k,
                n,
                n,
            )
        };

        for i in 0..m * n {
            assert!((c[i] - expected[i]).abs() < F32_SMALL_TOL);
        }
    }

    #[test]
    fn test_matmul_bias_f32_large() {
        let (m, n, k) = (128, 128, 128);
        let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32) * 0.1).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i % 13) as f32) * 0.1).collect();
        let bias: Vec<f32> = (0..n).map(|i| ((i % 7) as f32) * 0.5).collect();
        let mut c = vec![0.0f32; m * n];
        let expected = reference_matmul_bias_f32(&a, &b, &bias, m, n, k);

        unsafe {
            matmul_bias_f32(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                m,
                n,
                k,
                k,
                n,
                n,
            )
        };

        let max_diff = (0..m * n)
            .map(|i| (c[i] - expected[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < F32_LARGE_TOL);
    }

    #[test]
    fn test_simd_level_detection() {
        let level = detect_simd();
        println!("Detected SIMD level: {level:?}");
    }
}
