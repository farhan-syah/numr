//! SIMD-accelerated unary operations
//!
//! This module provides multi-architecture SIMD implementations for element-wise
//! unary operations.
//!
//! # SIMD Support
//!
//! ALL operations now have SIMD implementations:
//! - Neg, Abs, Sqrt, Square, Recip, Floor, Ceil, Round, RoundTiesEven, Trunc
//!   (direct SIMD)
//! - Exp, Log, Sin, Cos, Tan, Atan, Tanh (polynomial approximations from math module)
//! - Sign (comparison-based)
//! - ReLU (critical for ML)
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | f32 lanes | f64 lanes |
//! |--------------|-----------------|--------------|-----------|-----------|
//! | x86-64       | AVX-512         | 512 bits     | 16        | 8         |
//! | x86-64       | AVX2 + FMA      | 256 bits     | 8         | 4         |
//! | ARM64        | NEON            | 128 bits     | 4         | 2         |

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

use super::{SimdLevel, detect_simd};
use crate::ops::UnaryOp;

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::unary::{
    relu_scalar_f32, relu_scalar_f64, unary_scalar_f32, unary_scalar_f64,
};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// Check if operation has SIMD support
#[inline]
const fn is_simd_supported(op: UnaryOp) -> bool {
    matches!(
        op,
        UnaryOp::Neg
            | UnaryOp::Abs
            | UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Cbrt
            | UnaryOp::Exp
            | UnaryOp::Exp2
            | UnaryOp::Expm1
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Log10
            | UnaryOp::Log1p
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
            | UnaryOp::Square
            | UnaryOp::Recip
            | UnaryOp::Floor
            | UnaryOp::Ceil
            | UnaryOp::Round
            | UnaryOp::RoundTiesEven
            | UnaryOp::Trunc
            | UnaryOp::Sign
    )
}

/// SIMD unary operation for f32
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f32(op: UnaryOp, a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unary_scalar_f32(op, a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::unary_f32(op, a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::unary_f32(op, a, out, len),
        _ => unary_scalar_f32(op, a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::unary_f32(op, a, out, len),
        _ => unary_scalar_f32(op, a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unary_scalar_f32(op, a, out, len);
}

/// SIMD unary operation for f64
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn unary_f64(op: UnaryOp, a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar || !is_simd_supported(op) {
        unary_scalar_f64(op, a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::unary_f64(op, a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::unary_f64(op, a, out, len),
        _ => unary_scalar_f64(op, a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::unary_f64(op, a, out, len),
        _ => unary_scalar_f64(op, a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    unary_scalar_f64(op, a, out, len);
}

/// SIMD ReLU for f32
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_f32(a: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_scalar_f32(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::relu_f32(a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::relu_f32(a, out, len),
        _ => relu_scalar_f32(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_f32(a, out, len),
        _ => relu_scalar_f32(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_scalar_f32(a, out, len);
}

/// SIMD ReLU for f64
///
/// # Safety
/// - `a` and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn relu_f64(a: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        relu_scalar_f64(a, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::relu_f64(a, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::relu_f64(a, out, len),
        _ => relu_scalar_f64(a, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    relu_scalar_f64(a, out, len);
}

// ---------------------------------------------------------------------------
// f16/bf16 via f32 block-convert-compute
// ---------------------------------------------------------------------------

half_unary_op!(unary, unary_f32, UnaryOp);
half_unary!(relu, relu_f32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unary_neg_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Neg, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], -a[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_abs_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Abs, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i].abs(), "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_exp_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].exp();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5 * expected.abs().max(1.0),
                "exp mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// f64 exp must reach double precision, not merely f32 precision.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number of
    /// AVX2/AVX-512/NEON f64 lanes), so no element falls through to the exact
    /// scalar fallback. A degree-6 Taylor polynomial leaves a truncation error
    /// near 1.2e-7 here and fails this bound by nine orders of magnitude.
    #[test]
    fn test_unary_exp_f64_double_precision() {
        const LEN: usize = 2048;
        let a: Vec<f64> = (0..LEN)
            .map(|i| -700.0 + 1400.0 * (i as f64) / (LEN as f64 - 1.0))
            .collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { unary_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            let rel_err = (out[i] - expected).abs() / expected;
            assert!(
                rel_err < 1e-14,
                "exp({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// Relative error against a reference, safe when the reference is zero.
    ///
    /// asin(0) and acos(1) are exactly zero, so a plain division would report
    /// NaN for a bit-exact result.
    fn rel_err_f64(got: f64, expected: f64) -> f64 {
        // An exact match reports no error even when both sides are infinite or
        // both NaN. Subtracting two equal infinities yields NaN, which would
        // otherwise fail a comparison the kernel got exactly right.
        if got == expected || (got.is_nan() && expected.is_nan()) {
            return 0.0;
        }
        if !got.is_finite() || !expected.is_finite() {
            return f64::INFINITY;
        }
        (got - expected).abs() / expected.abs().max(f64::MIN_POSITIVE)
    }

    /// Fill `a` up to `len` with a sweep over [-limit, limit].
    fn fill_sweep_f64(a: &mut Vec<f64>, len: usize, limit: f64) {
        let start = a.len();
        let span = len - start;
        for i in 0..span {
            a.push(-limit + 2.0 * limit * (i as f64) / (span as f64 - 1.0));
        }
    }

    #[test]
    fn test_unary_asin_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // 1/sqrt(2) is where asin crosses from the direct series to the
        // reflection in every naive atan-based formulation, so a wrong branch
        // shows up here first. 0.5 is this implementation's own branch point.
        for &b in &[std::f64::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -100i32..=100 {
                let d = b + (k as f64) * 1e-12;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }
        fill_sweep_f64(&mut a, LEN, 1.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].asin();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "asin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_acos_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for &b in &[std::f64::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -100i32..=100 {
                let d = b + (k as f64) * 1e-12;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }
        fill_sweep_f64(&mut a, LEN, 1.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].acos();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "acos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_atan_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // Every reduction breakpoint, from both sides. |x| = 1 is the boundary
        // of the naive reciprocal reduction and the worst point of a truncated
        // Gregory series.
        for &b in &[0.4375f64, 0.6875, 1.0, 1.1875, 2.4375] {
            for k in -40i32..=40 {
                let d = b + (k as f64) * 1e-12;
                a.push(d);
                a.push(-d);
            }
        }
        // Magnitudes far past the last breakpoint, where t = -1/|x| is used.
        for k in 0..100 {
            let d = 10.0f64.powi(k % 25 + 2);
            a.push(d);
            a.push(-d);
        }
        fill_sweep_f64(&mut a, LEN, 8.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].atan();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "atan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_inverse_trig_f64_domain_edges() {
        const LEN: usize = 2048;

        // ±1 are exact endpoints; beyond them asin/acos are undefined.
        let asin_edges = [1.0f64, -1.0, 0.0, -0.0, 1.5, -1.5, f64::INFINITY, f64::NAN];
        let a: Vec<f64> = (0..LEN).map(|i| asin_edges[i % asin_edges.len()]).collect();

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].asin();
            if expected.is_nan() {
                assert!(out[i].is_nan(), "asin({}) = {}, expected NaN", a[i], out[i]);
            } else {
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "asin({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }

        unsafe { unary_f64(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].acos();
            if expected.is_nan() {
                assert!(out[i].is_nan(), "acos({}) = {}, expected NaN", a[i], out[i]);
            } else {
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "acos({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }

        // atan is defined everywhere; ±inf must saturate to ±pi/2.
        let atan_edges = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0f64,
            1.0,
            -1.0,
            1e308,
            -1e308,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| atan_edges[i % atan_edges.len()]).collect();
        unsafe { unary_f64(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            if a[i].is_nan() {
                assert!(out[i].is_nan(), "atan(NaN) = {}", out[i]);
            } else {
                let expected = a[i].atan();
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "atan({}) = {}, expected {}",
                    a[i],
                    out[i],
                    expected
                );
            }
        }
    }

    /// Fill `a` up to `len` with a linear sweep over [lo, hi].
    fn fill_range_f64(a: &mut Vec<f64>, len: usize, lo: f64, hi: f64) {
        let start = a.len();
        let span = len - start;
        for i in 0..span {
            a.push(lo + (hi - lo) * (i as f64) / (span as f64 - 1.0));
        }
    }

    /// Arguments that expose every weak point of a log reduction: the region
    /// around 1 where `log` cancels, one point per binade across the whole
    /// exponent range, and subnormals, which carry no implicit leading 1.
    fn log_probe_points_f64(len: usize) -> Vec<f64> {
        let mut a: Vec<f64> = Vec::with_capacity(len);

        // Near 1 the mantissa polynomial is the entire result, so a series that
        // is merely "close" over the reduction interval shows up here.
        for k in -60i32..=60 {
            a.push(1.0 + (k as f64) * 1e-12);
            a.push(1.0 + (k as f64) * 1e-3);
        }

        // sqrt(2) is the normalization breakpoint; a wrong branch lands here.
        for k in -40i32..=40 {
            a.push(std::f64::consts::SQRT_2 + (k as f64) * 1e-12);
            a.push(std::f64::consts::FRAC_1_SQRT_2 + (k as f64) * 1e-12);
        }

        // One value per binade over the full exponent range, powers of two
        // included, plus subnormals below f64::MIN_POSITIVE.
        for k in -1074i32..=1023 {
            if k % 3 == 0 {
                a.push(2.0f64.powi(k));
            }
        }
        a.push(f64::MIN_POSITIVE);
        a.push(f64::MIN_POSITIVE * 0.5);
        a.push(5e-324);
        a.push(f64::MAX);

        fill_range_f64(&mut a, len, 1e-8, 1e8);
        a
    }

    #[test]
    fn test_unary_log_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Log, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].ln();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log2_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Log2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].log2();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log10_f64_double_precision() {
        const LEN: usize = 2048;
        let a = log_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Log10, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].log10();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log10({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log1p_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // Tiny |x|, where 1 + x rounds x away entirely and log1p must fall back
        // on log1p(x) == x. This is the whole reason log1p exists separately.
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Around -0.5, far enough from 0 that a low-degree series in x diverges
        // from log(1+x) but still inside any |x| <= 0.5 fast path.
        for k in -60i32..=60 {
            a.push(-0.5 + (k as f64) * 1e-3);
        }

        // Approaching -1 from above, where log1p(x) -> -inf. 2^-53 is the last
        // offset that still rounds to something other than -1 itself.
        for k in 1i32..=53 {
            a.push(-1.0 + 2.0f64.powi(-k));
        }

        // Both sides of |x| = 1, the Fast2Sum branch point.
        for k in -20i32..=20 {
            a.push(1.0 + (k as f64) * 1e-12);
        }

        fill_range_f64(&mut a, LEN, -0.9, 10.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Log1p, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].ln_1p();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "log1p({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_log_f64_domain_edges() {
        const LEN: usize = 2048;

        // log is undefined at and below zero, and 1 must come out exactly zero.
        let log_edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| log_edges[i % log_edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Log, f64::ln as fn(f64) -> f64),
            (UnaryOp::Log2, f64::log2 as fn(f64) -> f64),
            (UnaryOp::Log10, f64::log10 as fn(f64) -> f64),
        ] {
            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i]);
                if expected.is_nan() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                } else {
                    assert_eq!(out[i], expected, "{:?}({}) mismatch", op, a[i]);
                }
            }
        }

        // log1p(-1) = -inf and log1p(x < -1) = NaN; log1p(0) keeps its sign.
        let log1p_edges = [
            -1.0f64,
            -1.5,
            0.0,
            -0.0,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN)
            .map(|i| log1p_edges[i % log1p_edges.len()])
            .collect();
        unsafe { unary_f64(UnaryOp::Log1p, a.as_ptr(), out.as_mut_ptr(), LEN) }
        for i in 0..LEN {
            let expected = a[i].ln_1p();
            if expected.is_nan() {
                assert!(
                    out[i].is_nan(),
                    "log1p({}) = {}, expected NaN",
                    a[i],
                    out[i]
                );
            } else {
                assert_eq!(out[i], expected, "log1p({}) mismatch", a[i]);
            }
        }
    }

    /// Arguments that expose every weak point of a π/2 range reduction: the
    /// multiples of π/2, where the reduced argument cancels down to almost
    /// nothing, and large |x|, where a single rounded π/2 leaves an absolute
    /// error proportional to |x|. A uniform sweep alone hits neither.
    fn trig_probe_points_f64(len: usize) -> Vec<f64> {
        let mut a: Vec<f64> = Vec::with_capacity(len);

        // Every multiple of π/2 out to |x| = 100, and its immediate
        // neighbourhood. One of sin or cos crosses zero at each of them, so the
        // reduced argument carries the whole result.
        for k in -64i32..=64 {
            let c = (k as f64) * std::f64::consts::FRAC_PI_2;
            a.push(c);
            for step in 1i32..=3 {
                a.push(c + (step as f64) * 1e-13);
                a.push(c - (step as f64) * 1e-13);
            }
        }

        // Magnitudes far past the test sweep, where reduction error dominates
        // polynomial error. The stride is prime-ish so the points do not line
        // up with any multiple of π/2.
        for k in 0i32..200 {
            let d = 1e3 + (k as f64) * 977.0;
            a.push(d);
            a.push(-d);
        }

        fill_sweep_f64(&mut a, len, 100.0);
        a
    }

    /// f64 sin must reach double precision across the reduction range.
    ///
    /// The degree-9 Taylor series this replaced left ~2.6e-8 of truncation
    /// error near the ends of [-π/4, π/4], and reducing with a single rounded
    /// π/2 left ~7e-12 at |x| = 2e5. Both fail this bound outright.
    #[test]
    fn test_unary_sin_f64_double_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].sin();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "sin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f64 cos must reach double precision across the reduction range.
    ///
    /// Building cos as `sin(x + π/2)` rounds the sum before reduction, so this
    /// also fails for any x large enough that the addition is inexact.
    #[test]
    fn test_unary_cos_f64_double_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f64(LEN);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].cos();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "cos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f64 tan must reach double precision away from its poles.
    ///
    /// The truncated Taylor series this replaced left ~5e-5 of relative error
    /// at the edge of the reduction interval, which ±π/4 targets directly.
    #[test]
    fn test_unary_tan_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // ±π/4 is the edge of the reduction interval and the worst point of any
        // fixed-degree polynomial in the reduced argument.
        for &b in &[std::f64::consts::FRAC_PI_4, 0.5, 1.0] {
            for k in -60i32..=60 {
                let d = b + (k as f64) * 1e-13;
                a.push(d);
                a.push(-d);
            }
        }

        // Multiples of π, where tan crosses zero and the reduction cancels.
        for k in -30i32..=30 {
            a.push((k as f64) * std::f64::consts::PI);
        }

        // Large |x|, where reduction error dominates. Points near a pole are
        // dropped: there the result itself is ill-conditioned, not the kernel.
        for k in 0i32..200 {
            let d = 1e3 + (k as f64) * 977.0 + 0.37;
            if d.cos().abs() > 1e-3 {
                a.push(d);
                a.push(-d);
            }
        }

        fill_range_f64(&mut a, LEN, -1.5, 1.5);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].tan();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "tan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_trig_f64_domain_edges() {
        const LEN: usize = 2048;

        // ±inf and NaN have no finite reduction, so all three are NaN there.
        // ±0 must keep its own sign, which `x - j*π/2` destroys for j = -0.
        let edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            std::f64::consts::FRAC_PI_2,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Sin, f64::sin as fn(f64) -> f64),
            (UnaryOp::Cos, f64::cos as fn(f64) -> f64),
            (UnaryOp::Tan, f64::tan as fn(f64) -> f64),
        ] {
            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                if !a[i].is_finite() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                    continue;
                }
                let expected = reference(a[i]);
                assert!(
                    rel_err_f64(out[i], expected) < 1e-14,
                    "{:?}({}) = {}, expected {}",
                    op,
                    a[i],
                    out[i],
                    expected
                );
                if expected == 0.0 {
                    assert_eq!(
                        out[i].is_sign_negative(),
                        expected.is_sign_negative(),
                        "{:?}({}) lost the sign of zero",
                        op,
                        a[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_unary_tanh_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Tanh, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].tanh();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "tanh mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_sign_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Sign, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 {
                1.0
            } else if a[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
            assert_eq!(out[i], expected, "sign mismatch at index {}", i);
        }
    }

    #[test]
    fn test_unary_log_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Log, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].ln();
            let diff = (out[i] - expected).abs();
            // Relative error tolerance of ~1e-4 is acceptable for f32 SIMD approximations
            assert!(
                diff < 5e-5 * expected.abs().max(1.0),
                "log mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_sin_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].sin();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "sin mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_cos_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.1).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].cos();
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-5,
                "cos mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_tan_f32() {
        // Avoid values near π/2 where tan approaches infinity
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.02).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { unary_f32(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].tan();
            let diff = (out[i] - expected).abs();
            // Relative error tolerance of ~2e-4 is acceptable for f32 SIMD tan approximations
            assert!(
                diff < 2e-4 * expected.abs().max(1.0),
                "tan mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    /// Inputs that stress both rounding modes: every tie between -4.5 and 4.5,
    /// the largest f32 below 0.5 (where a naive `floor(|x| + 0.5)` rounds the
    /// wrong way), values above 2^23 that are already integers, and infinities.
    fn rounding_probe_f32() -> Vec<f32> {
        let mut v = vec![
            -4.5,
            -3.5,
            -2.5,
            -1.5,
            -0.5,
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            0.0,
            -0.0,
            1.1,
            -1.1,
            3.9,
            -4.7,
            f32::from_bits(0x3EFF_FFFF),
            -f32::from_bits(0x3EFF_FFFF),
            8_388_609.0,
            -8_388_609.0,
            1e30,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ];
        // Pad past SIMD_THRESHOLD with a length that leaves a scalar tail for
        // every vector width in use (16, 8 and 4 lanes).
        while v.len() < 35 {
            v.push(v.len() as f32 * 0.25);
        }
        v
    }

    #[test]
    fn test_unary_round_f32_ties_away_from_zero() {
        let a = rounding_probe_f32();
        let len = a.len();
        let mut out = vec![0.0f32; len];

        unsafe { unary_f32(UnaryOp::Round, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_ties_even_f32() {
        let a = rounding_probe_f32();
        let len = a.len();
        let mut out = vec![0.0f32; len];

        unsafe { unary_f32(UnaryOp::RoundTiesEven, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round_ties_even();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round_ties_even mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_f64_ties_away_from_zero() {
        let a: Vec<f64> = rounding_probe_f32().iter().map(|&x| f64::from(x)).collect();
        let len = a.len();
        let mut out = vec![0.0f64; len];

        unsafe { unary_f64(UnaryOp::Round, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_unary_round_ties_even_f64() {
        let a: Vec<f64> = rounding_probe_f32().iter().map(|&x| f64::from(x)).collect();
        let len = a.len();
        let mut out = vec![0.0f64; len];

        unsafe { unary_f64(UnaryOp::RoundTiesEven, a.as_ptr(), out.as_mut_ptr(), len) }

        for i in 0..len {
            let expected = a[i].round_ties_even();
            assert_eq!(
                out[i].to_bits(),
                expected.to_bits(),
                "round_ties_even mismatch at {}: input {}, got {}, expected {}",
                i,
                a[i],
                out[i],
                expected
            );
        }
    }

    /// The SIMD kernels only run at or above `SIMD_THRESHOLD`, so a short input
    /// silently tests the scalar path alone. Sweep the lengths around and past
    /// the threshold to cover both.
    #[test]
    fn test_round_ops_across_simd_threshold_f32() {
        let probe = rounding_probe_f32();
        for len in [1usize, 7, 31, 32, 33, 35, 64, 65] {
            let a: Vec<f32> = probe.iter().copied().cycle().take(len).collect();
            let mut away = vec![0.0f32; len];
            let mut even = vec![0.0f32; len];

            unsafe {
                unary_f32(UnaryOp::Round, a.as_ptr(), away.as_mut_ptr(), len);
                unary_f32(UnaryOp::RoundTiesEven, a.as_ptr(), even.as_mut_ptr(), len);
            }

            for i in 0..len {
                assert_eq!(
                    away[i].to_bits(),
                    a[i].round().to_bits(),
                    "round mismatch at len {} index {} (input {})",
                    len,
                    i,
                    a[i]
                );
                assert_eq!(
                    even[i].to_bits(),
                    a[i].round_ties_even().to_bits(),
                    "round_ties_even mismatch at len {} index {} (input {})",
                    len,
                    i,
                    a[i]
                );
            }
        }
    }

    /// One f64 subnormal step, the whole precision available below
    /// `f64::MIN_POSITIVE`. Relative error is meaningless there, so the
    /// subnormal test below bounds the absolute error in these units instead.
    /// One step is also the ulp of every normal number in the last binade, so
    /// the bound stays meaningful across the whole tail.
    const SUBNORMAL_STEP_F64: f64 = 5e-324;

    /// f64 exp must cover its whole representable range, not the ±709 the
    /// clamp used to allow.
    ///
    /// ln(f64::MAX) = 709.7827, so every input in [709, 709.7827] has a
    /// perfectly representable result that a clamp at 709 replaces with
    /// exp(709). At 709.78 that is 8.2184074616e307 against a true
    /// 1.7928227944e308 — 54% low.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f64 lanes), so no element falls through to
    /// the exact scalar fallback.
    #[test]
    fn test_unary_exp_f64_upper_band() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        // The band the ±709 clamp used to swallow, up to ln(f64::MAX).
        for k in 0i32..1024 {
            a.push(709.0 + 0.782_712_893_383 * (k as f64) / 1023.0);
        }
        fill_range_f64(&mut a, LEN, 700.0, 709.782_712_893_383);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            assert!(
                expected.is_finite(),
                "probe {} is outside the representable range at index {}",
                a[i],
                i
            );
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "exp({}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp keeps producing subnormals down to -745. A clamp at -709 returns
    /// 1.2e-308 for every one of them, sixteen orders of magnitude high at the
    /// bottom of the range.
    ///
    /// Every result in this range has an ulp of one subnormal step or less, so
    /// the absolute bound below is a bound of two ulps throughout.
    #[test]
    fn test_unary_exp_f64_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);
        fill_range_f64(&mut a, LEN, -745.0, -708.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F64,
                "exp({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F64,
                i
            );
        }
    }

    /// Past ln(f64::MAX) the result genuinely overflows and below -745.13 it
    /// genuinely underflows, so the clamp bounds must sit outside both: a bound
    /// that is itself representable turns an infinity into a finite number and
    /// a zero into a subnormal.
    #[test]
    fn test_unary_exp_f64_range_ends() {
        const LEN: usize = 2048;
        let probes: [f64; 8] = [
            // Past ln(f64::MAX) = 709.782712893384, so exp is +inf.
            709.782_712_893_4,
            710.0,
            745.0,
            1.0e300,
            // Past ln(2^-1075) = -745.1332, so exp rounds to zero.
            -746.0,
            -750.0,
            -1.0e300,
            f64::NEG_INFINITY,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { unary_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = if a[i] > 0.0 { f64::INFINITY } else { 0.0 };
            assert_eq!(
                out[i], expected,
                "exp({}) = {:e}, expected {:e} at index {}",
                a[i], out[i], expected, i
            );
        }
    }

    /// exp(NaN) must be NaN. On x86 maxpd/minpd return their second operand
    /// when either input is NaN, so a clamp written as `max(x, MIN)` replaces
    /// NaN with the clamp bound and returns approximately zero; NEON's
    /// FMAX/FMIN propagate it, so the same source diverges across ISAs.
    #[test]
    fn test_unary_exp_f64_domain_edges() {
        const LEN: usize = 2048;
        let probes: [f64; 8] = [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -0.0,
            1.0,
            -1.0,
            709.0,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { unary_f64(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp();
            assert!(
                rel_err_f64(out[i], expected) < 1e-14,
                "exp({}) = {}, expected {} at index {}",
                a[i],
                out[i],
                expected,
                i
            );
        }
    }

    /// sinh and cosh stay finite up to 710.4758, past the 709.7827 where exp
    /// and expm1 overflow, so the band between the two has to be reached by
    /// squaring `exp(|x|/2)` rather than by scaling `exp(|x|)`.
    #[test]
    fn test_unary_hyperbolic_f64_upper_band() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in 0i32..1024 {
            a.push(709.0 + 1.475 * (k as f64) / 1023.0);
        }
        fill_range_f64(&mut a, LEN, -710.475, -709.0);

        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Cosh, f64::cosh as fn(f64) -> f64),
        ] {
            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i]);
                assert!(
                    expected.is_finite(),
                    "probe {} is outside the representable range at index {}",
                    a[i],
                    i
                );
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
                    "{:?}({}) = {:e}, expected {:e}, rel_err = {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    rel_err,
                    i
                );
            }
        }
    }

    /// Arguments that break a naive expm1: the ±0.5 boundary of the old
    /// degree-4 Taylor branch, where the dropped `x⁵/120` term is 2.6e-4, and
    /// arguments small enough that `exp(x) - 1` keeps none of the result.
    #[test]
    fn test_unary_expm1_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -60i32..=60 {
            a.push(0.5 + (k as f64) * 1e-3);
            a.push(-0.5 + (k as f64) * 1e-3);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }
        fill_range_f64(&mut a, LEN, -700.0, 700.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Expm1, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp_m1();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "expm1({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp2 must not borrow `exp(x * ln2)`. That premultiply rounds a value as
    /// large as 710 to one ulp, and the exponential turns the absolute error
    /// into a relative one, so the failure only shows at large |x|.
    #[test]
    fn test_unary_exp2_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(978.022 + (k as f64) * 1e-3);
            a.push(-978.022 + (k as f64) * 1e-3);
        }

        // Exact integers and half-integers are the ends of the reduction
        // interval, where `r = x - n` is largest.
        for k in -40i32..=40 {
            a.push(k as f64);
            a.push(k as f64 + 0.5);
        }
        fill_range_f64(&mut a, LEN, -1000.0, 1000.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].exp2();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "exp2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// Small |x| for the two hyperbolic functions. `(e^x - e^-x)/2` and
    /// `(e^2x - 1)/(e^2x + 1)` both subtract quantities that approach each
    /// other as x approaches zero, so they lose the result they are computing.
    #[test]
    fn test_unary_sinh_tanh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut probes: Vec<f64> = Vec::with_capacity(LEN);

        for k in 1i32..=300 {
            probes.push(10.0f64.powi(-k));
            probes.push(-10.0f64.powi(-k));
        }
        for k in -100i32..=100 {
            probes.push(-0.0024 + (k as f64) * 1e-5);
        }

        let mut out = vec![0.0f64; LEN];

        for (op, reference, hi) in [
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64, 700.0f64),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64, 30.0f64),
        ] {
            let mut a = probes.clone();
            fill_range_f64(&mut a, LEN, -hi, hi);

            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i]);
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
                    "{:?}({}) = {}, expected {}, rel_err = {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    rel_err,
                    i
                );
            }
        }
    }

    /// asinh at negative arguments, where `log(x + sqrt(x²+1))` cancels: at
    /// x = -49.6 the two addends agree to twelve digits and the sum keeps only
    /// the top three. Small |x| exercises the same cancellation from the other
    /// side, where asinh(x) == x.
    #[test]
    fn test_unary_asinh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(-49.6093 + (k as f64) * 1e-6);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Both sides of the 2 and 2^28 branch points, and the far tail.
        for k in -20i32..=20 {
            a.push(2.0 + (k as f64) * 1e-12);
            a.push(268_435_456.0 + (k as f64) * 1e-3);
        }
        for k in 1i32..=30 {
            a.push(10.0f64.powi(k));
            a.push(-10.0f64.powi(k));
        }
        fill_sweep_f64(&mut a, LEN, 3.0);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Asinh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].asinh();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "asinh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// acosh near 1, where `x² - 1` throws away half the significant bits of
    /// `x - 1` — and `x - 1` is the whole result there.
    #[test]
    fn test_unary_acosh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(1.01 + (k as f64) * 1e-6);
        }
        for k in 1i32..=300 {
            a.push(1.0 + 10.0f64.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(2.0 + (k as f64) * 1e-12);
            a.push(268_435_456.0 + (k as f64) * 1e-3);
        }
        fill_range_f64(&mut a, LEN, 1.0, 1e6);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Acosh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].acosh();
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "acosh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// Reference atanh, evaluated on the side where `f64::atanh` is accurate.
    ///
    /// `f64::atanh` is the only one of these seven references not delegated to
    /// libm: std computes `0.5 * ((2x)/(1-x)).ln_1p()` directly, which is not
    /// odd-symmetric. For x >= 0 the small quantity is the denominator `1 - x`,
    /// exact by Sterbenz on [0.5, 1), and the quotient is large and well
    /// conditioned. For x -> -1 the quotient instead approaches -1, where
    /// `ln_1p` amplifies its half-ulp rounding by `1/(1+q)` without bound: 107
    /// ulps at x = -(1 - 2^-13), and 1.8e6 ulps at x = -(1 - 2^-26). atanh is
    /// odd, so the negative side is referenced through the positive one.
    fn atanh_reference(x: f64) -> f64 {
        if x < 0.0 { -(-x).atanh() } else { x.atanh() }
    }

    /// atanh at small |x|, where forming `(1+x)/(1-x)` rounds the ratio to
    /// one ulp of 1 and the log of it keeps nothing: at x = 7e-4 that is a
    /// relative error of 1.4e-13, 600 ulps.
    #[test]
    fn test_unary_atanh_f64_double_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f64> = Vec::with_capacity(LEN);

        for k in -100i32..=100 {
            a.push(0.0007 + (k as f64) * 1e-9);
            a.push(-0.0007 + (k as f64) * 1e-9);
        }
        for k in 1i32..=300 {
            a.push(10.0f64.powi(-k));
            a.push(-10.0f64.powi(-k));
        }

        // Approaching ±1, where atanh -> ±inf, and both sides of the 0.5 split.
        for k in 1i32..=40 {
            a.push(1.0 - 2.0f64.powi(-k));
            a.push(-1.0 + 2.0f64.powi(-k));
        }
        for k in -20i32..=20 {
            a.push(0.5 + (k as f64) * 1e-12);
        }
        fill_sweep_f64(&mut a, LEN, 0.9999);

        let mut out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Atanh, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = atanh_reference(a[i]);
            let rel_err = rel_err_f64(out[i], expected);
            assert!(
                rel_err < 1e-14,
                "atanh({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }

        // The reference only exercises std on x >= 0, so the sign is pinned
        // separately: atanh is odd, and this kernel is odd bit for bit because
        // it works on |x| and restores the sign bit.
        let neg: Vec<f64> = a.iter().map(|v| -v).collect();
        let mut neg_out = vec![0.0f64; LEN];
        unsafe { unary_f64(UnaryOp::Atanh, neg.as_ptr(), neg_out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            if out[i].is_nan() {
                assert!(neg_out[i].is_nan(), "atanh({}) lost NaN", neg[i]);
            } else {
                assert_eq!(
                    neg_out[i].to_bits(),
                    (-out[i]).to_bits(),
                    "atanh({}) and atanh({}) are not exact negatives",
                    neg[i],
                    a[i]
                );
            }
        }
    }

    #[test]
    fn test_unary_hyperbolic_f64_domain_edges() {
        const LEN: usize = 2048;

        let edges = [
            0.0f64,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            1.5,
            -1.5,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            5e-324,
            -5e-324,
            1e-310,
        ];
        let a: Vec<f64> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f64; LEN];

        for (op, reference) in [
            (UnaryOp::Expm1, f64::exp_m1 as fn(f64) -> f64),
            (UnaryOp::Exp2, f64::exp2 as fn(f64) -> f64),
            (UnaryOp::Sinh, f64::sinh as fn(f64) -> f64),
            (UnaryOp::Tanh, f64::tanh as fn(f64) -> f64),
            (UnaryOp::Asinh, f64::asinh as fn(f64) -> f64),
            (UnaryOp::Acosh, f64::acosh as fn(f64) -> f64),
            (UnaryOp::Atanh, f64::atanh as fn(f64) -> f64),
        ] {
            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i]);
                let rel_err = rel_err_f64(out[i], expected);
                assert!(
                    rel_err < 1e-14,
                    "{:?}({}) = {}, expected {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    i
                );
            }
        }
    }

    /// The odd functions carry the sign of zero, which they can only do by
    /// working on |x| and restoring the sign bit rather than negating.
    #[test]
    fn test_unary_hyperbolic_f64_signed_zero() {
        const LEN: usize = 2048;
        let a: Vec<f64> = (0..LEN)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut out = vec![0.0f64; LEN];

        for op in [
            UnaryOp::Expm1,
            UnaryOp::Sinh,
            UnaryOp::Tanh,
            UnaryOp::Asinh,
            UnaryOp::Atanh,
        ] {
            unsafe { unary_f64(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                assert_eq!(
                    out[i].to_bits(),
                    a[i].to_bits(),
                    "{:?} lost the sign of zero at index {}",
                    op,
                    i
                );
            }
        }
    }

    /// Relative error against a reference, safe when the reference is zero.
    ///
    /// The f32 counterpart of `rel_err_f64`: the reference is always the f64
    /// result rounded once to f32, so the bound below is a bound on the kernel
    /// alone and not on the reference.
    fn rel_err_f32(got: f32, expected: f32) -> f32 {
        // An exact match reports no error even when both sides are infinite or
        // both NaN. Subtracting two equal infinities yields NaN, which would
        // otherwise fail a comparison the kernel got exactly right.
        if got == expected || (got.is_nan() && expected.is_nan()) {
            return 0.0;
        }
        if !got.is_finite() || !expected.is_finite() {
            return f32::INFINITY;
        }
        (got - expected).abs() / expected.abs().max(f32::MIN_POSITIVE)
    }

    /// Fill `a` up to `len` with a linear sweep over [lo, hi].
    fn fill_range_f32(a: &mut Vec<f32>, len: usize, lo: f32, hi: f32) {
        let start = a.len();
        let span = len - start;
        for i in 0..span {
            a.push(lo + (hi - lo) * (i as f32) / (span as f32 - 1.0));
        }
    }

    /// One f32 subnormal step, the whole precision available below
    /// `f32::MIN_POSITIVE`. Relative error is meaningless there — the grid
    /// itself is only 5e-6 fine near 2.6e-40 — so the subnormal tests bound the
    /// absolute error in these units instead.
    const SUBNORMAL_STEP_F32: f32 = 1.401_298_5e-45;

    /// f32 exp must reach single precision over its whole representable range.
    ///
    /// Two defects show up here. Clamping the input to ±88 replaces every
    /// result in [88, ln(f32::MAX)] with exp(88), a factor of two out at the
    /// top. Reducing as `(x*log2(e) - n) * ln(2)` instead of Cody-Waite leaves
    /// an absolute error in r proportional to |x|, worth 3e-6 relative near
    /// |x| = 88 — thirty ulps — which the sweep below reaches everywhere.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f32 lanes), so no element falls through to
    /// the exact scalar fallback.
    #[test]
    fn test_unary_exp_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // The band the ±88 clamp used to swallow, up to ln(f32::MAX).
        for k in 0i32..400 {
            a.push(88.0 + 0.722 * (k as f32) / 399.0);
        }
        fill_range_f32(&mut a, LEN, -87.3, 88.72);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "exp({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp keeps producing subnormals down to -104. A clamp at -88 returns
    /// 6e-39 for every one of them, twenty-eight orders of magnitude high at
    /// the bottom of the range.
    #[test]
    fn test_unary_exp_f32_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN, -104.0, -87.4);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Exp, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp() as f32;
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F32,
                "exp({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F32,
                i
            );
        }
    }

    /// f32 exp2 must not borrow `exp(x * ln2)`. That premultiply rounds a value
    /// as large as 128 before the exponential, worth 5e-6 relative — forty
    /// ulps. The ±88 clamp of `exp` also truncates the top of the range, where
    /// 2^127 is still perfectly representable.
    #[test]
    fn test_unary_exp2_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // The last binade, which the exp clamp used to cut off at 2^88.
        for k in 0i32..400 {
            a.push(127.0 + (k as f32) / 400.0);
        }

        // Exact integers and half-integers are the ends of the reduction
        // interval, where `r = x - n` is largest.
        for k in -126i32..=127 {
            a.push(k as f32);
            if k < 127 {
                a.push(k as f32 + 0.5);
            }
        }
        fill_range_f32(&mut a, LEN, -126.0, 127.9);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp2() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "exp2({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// exp2 keeps producing subnormals down to -149, the whole range a clamp at
    /// -88 discards.
    #[test]
    fn test_unary_exp2_f32_subnormal_tail() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN, -149.0, -126.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Exp2, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp2() as f32;
            let abs_err = (out[i] - expected).abs();
            assert!(
                abs_err <= 2.0 * SUBNORMAL_STEP_F32,
                "exp2({}) = {:e}, expected {:e}, off by {} subnormal steps at index {}",
                a[i],
                out[i],
                expected,
                abs_err / SUBNORMAL_STEP_F32,
                i
            );
        }
    }

    /// Arguments that break a naive f32 expm1: the ±0.5 boundary of the old
    /// degree-4 Taylor branch, where the dropped `x⁵/120` term is 2.6e-4, and
    /// arguments small enough that `exp(x) - 1` keeps none of the result.
    #[test]
    fn test_unary_expm1_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        for k in -60i32..=60 {
            a.push(0.5 + (k as f32) * 1e-3);
            a.push(-0.5 + (k as f32) * 1e-3);
        }
        for k in 1i32..=30 {
            a.push(10.0f32.powi(-k));
            a.push(-10.0f32.powi(-k));
        }
        fill_range_f32(&mut a, LEN, -87.0, 88.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Expm1, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).exp_m1() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "expm1({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 cbrt over every binade, both signs. Seeding the iteration from the
    /// exponent alone leaves the mantissa unaccounted for: the seed is off by
    /// up to 37%, and two Newton steps square that to 5e-2, not to 1e-7.
    #[test]
    fn test_unary_cbrt_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // One value per binade over the full exponent range, powers of two
        // included, subnormals below 2^-126 among them. Both are built from
        // bit patterns: a subnormal power of two has no exponent field to set.
        for k in -149i32..=127 {
            let p = if k >= -126 {
                f32::from_bits(((k + 127) as u32) << 23)
            } else {
                f32::from_bits(1u32 << (k + 149))
            };
            a.push(p);
            a.push(-p);
            a.push(1.5 * p);
            a.push(-1.5 * p);
        }
        a.push(f32::MAX);
        a.push(-f32::MAX);
        a.push(-31.965_813);
        fill_range_f32(&mut a, LEN, -100.0, 100.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Cbrt, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).cbrt() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "cbrt({:e}) = {:e}, expected {:e}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// cbrt is odd, so the two halves of the line must agree exactly.
    #[test]
    fn test_unary_cbrt_f32_odd_symmetry() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);
        fill_range_f32(&mut a, LEN / 2, 1e-30, 1e30);
        let positive = a.clone();
        for &v in &positive {
            a.push(-v);
        }

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Cbrt, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN / 2 {
            assert_eq!(
                out[i],
                -out[i + LEN / 2],
                "cbrt({:e}) = {:e} but cbrt of its negation is {:e}",
                a[i],
                out[i],
                out[i + LEN / 2]
            );
        }
    }

    /// The clamp that keeps the reduction in range must not eat the domain
    /// edges: on x86 `maxps`/`minps` return their second operand for NaN, so a
    /// clamp written the wrong way round turns NaN into the clamp bound.
    #[test]
    fn test_unary_exp_family_f32_domain_edges() {
        const LEN: usize = 2048;
        let probes: [f32; 8] = [
            f32::NEG_INFINITY,
            f32::INFINITY,
            f32::NAN,
            0.0,
            -0.0,
            1.0,
            -1.0,
            120.0,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| probes[i % probes.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Exp, f64::exp as fn(f64) -> f64),
            (UnaryOp::Exp2, f64::exp2 as fn(f64) -> f64),
            (UnaryOp::Expm1, f64::exp_m1 as fn(f64) -> f64),
            (UnaryOp::Cbrt, f64::cbrt as fn(f64) -> f64),
        ] {
            unsafe { unary_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {} at index {}",
                    op,
                    a[i],
                    out[i],
                    expected,
                    i
                );
            }
        }
    }

    /// expm1 and cbrt are odd, so they carry the sign of zero, which they can
    /// only do by restoring the sign bit rather than negating.
    #[test]
    fn test_unary_exp_family_f32_signed_zero() {
        const LEN: usize = 2048;
        let a: Vec<f32> = (0..LEN)
            .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
            .collect();
        let mut out = vec![0.0f32; LEN];

        for op in [UnaryOp::Expm1, UnaryOp::Cbrt] {
            unsafe { unary_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                assert_eq!(
                    out[i].to_bits(),
                    a[i].to_bits(),
                    "{:?} lost the sign of zero at index {}",
                    op,
                    i
                );
            }
        }
    }

    /// Probe points for f32 sin/cos: every multiple of π/2 the reduction has to
    /// survive, plus magnitudes far past the sweep.
    fn trig_probe_points_f32(len: usize) -> Vec<f32> {
        let mut a: Vec<f32> = Vec::with_capacity(len);

        // Every multiple of π/2 out to |x| = 100, and its immediate
        // neighbourhood. One of sin or cos crosses zero at each of them, so the
        // reduced argument carries the whole result.
        for k in -64i32..=64 {
            let c = (k as f32) * std::f32::consts::FRAC_PI_2;
            a.push(c);
            for step in 1i32..=3 {
                a.push(c + (step as f32) * 1e-6);
                a.push(c - (step as f32) * 1e-6);
            }
        }

        // Multiples of π/2 out to |x| = 1.2e5. Reducing with a single rounded
        // π/2 leaves an absolute phase error of |j| * 4.4e-8 here — over 3e-3,
        // which is the whole answer.
        for k in 0i32..120 {
            let c = ((20_000 + k * 500) as f32) * std::f32::consts::FRAC_PI_2;
            a.push(c);
            a.push(-c);
        }

        // Magnitudes on a stride that lines up with no multiple of π/2.
        for k in 0i32..200 {
            let d = 1.0e3 + (k as f32) * 601.0;
            a.push(d);
            a.push(-d);
        }

        fill_range_f32(&mut a, len, -100.0, 100.0);
        a
    }

    /// f32 sin must reach single precision across the reduction range.
    ///
    /// Reducing with a single rounded π/2 costs |j| * 4.4e-8 of absolute phase,
    /// so this fails by more than the answer itself past |x| ~ 1e4. The
    /// degree-6 cos Taylor series it paired with is separately worth 3.6e-6 at
    /// y = π/4, thirty ulps.
    ///
    /// The length forces the SIMD path (>= SIMD_THRESHOLD, and a whole number
    /// of AVX2, AVX-512 and NEON f32 lanes), so no element falls through to the
    /// exact scalar fallback.
    #[test]
    fn test_unary_sin_f32_single_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Sin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).sin() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "sin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 cos must reach single precision across the reduction range.
    ///
    /// Building cos as `sin(x + π/2)` rounds the sum before reduction, which
    /// costs an ulp of x — already the whole answer near a zero of cos.
    #[test]
    fn test_unary_cos_f32_single_precision() {
        const LEN: usize = 2048;
        let a = trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Cos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).cos() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "cos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 tan must reach single precision away from its poles.
    ///
    /// The truncated Taylor series this replaced dropped a term worth 1.5e-4 at
    /// y = ±π/4, which the dense band there reaches on every point.
    #[test]
    fn test_unary_tan_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // ±π/4 is the edge of the reduction interval and the worst point of any
        // fixed-degree polynomial in the reduced argument.
        for &b in &[std::f32::consts::FRAC_PI_4, 0.5, 1.0] {
            for k in -60i32..=60 {
                let d = b + (k as f32) * 1e-6;
                a.push(d);
                a.push(-d);
            }
        }

        // Multiples of π, where tan crosses zero and the reduction cancels.
        for k in -30i32..=30 {
            a.push((k as f32) * std::f32::consts::PI);
        }

        // Large |x|, where reduction error dominates. Points near a pole are
        // dropped: there the result itself is ill-conditioned, not the kernel.
        for k in 0i32..200 {
            let d = 1.0e3 + (k as f32) * 601.0 + 0.37;
            if (d as f64).cos().abs() > 1e-3 {
                a.push(d);
                a.push(-d);
            }
        }

        fill_range_f32(&mut a, LEN, -1.5, 1.5);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Tan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).tan() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "tan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// Probe points for f32 asin/acos: the branch points, the endpoints, and a
    /// sweep of the whole domain.
    fn inverse_trig_probe_points_f32(len: usize) -> Vec<f32> {
        let mut a: Vec<f32> = Vec::with_capacity(len);

        // 1/sqrt(2) is where asin crosses from the direct series to the
        // reflection in any atan-based formulation, because atan's argument
        // reaches exactly 1 there. 0.5 is this implementation's own branch
        // point, and ±1 is the endpoint the reflection has to land on exactly.
        for &b in &[std::f32::consts::FRAC_1_SQRT_2, 0.5, 1.0] {
            for k in -80i32..=80 {
                let d = b + (k as f32) * 1e-7;
                if d <= 1.0 {
                    a.push(d);
                    a.push(-d);
                }
            }
        }

        fill_range_f32(&mut a, len, -1.0, 1.0);
        a
    }

    /// f32 asin must reach single precision over [-1, 1].
    ///
    /// Composing it as `atan(x / sqrt(1 - x²))` sends atan's argument to 1 at
    /// |x| = 1/sqrt(2), the slowest-converging point of the Gregory series it
    /// used to call, worth 4.5e-2 relative.
    #[test]
    fn test_unary_asin_f32_single_precision() {
        const LEN: usize = 2048;
        let a = inverse_trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Asin, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).asin() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "asin({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 acos must reach single precision over [-1, 1].
    ///
    /// `π/2 - asin(x)` inherits every defect of asin and adds cancellation as
    /// x approaches 1, where the result is the difference that vanishes.
    #[test]
    fn test_unary_acos_f32_single_precision() {
        const LEN: usize = 2048;
        let a = inverse_trig_probe_points_f32(LEN);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Acos, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).acos() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "acos({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    /// f32 atan must reach single precision for every finite input.
    ///
    /// The Gregory series it used to evaluate on [0, 1] converges like
    /// 1/(2n+3) at the boundary: seven terms leave ~4e-2 relative error, which
    /// is where the old peak sat.
    #[test]
    fn test_unary_atan_f32_single_precision() {
        const LEN: usize = 2048;
        let mut a: Vec<f32> = Vec::with_capacity(LEN);

        // |x| = 1 is the old reduction boundary; the two tan(π/8) and tan(3π/8)
        // values are the new ones.
        for &b in &[1.0f32, 0.414_213_56, 2.414_213_6, 0.989_011] {
            for k in -60i32..=60 {
                let d = b * (1.0 + (k as f32) * 1e-6);
                a.push(d);
                a.push(-d);
            }
        }

        // Many magnitudes: atan has to hold from the smallest normal up to the
        // point where the result is π/2 to the last bit.
        for k in -35i32..=35 {
            let d = 10f32.powi(k);
            a.push(d);
            a.push(-d);
        }

        fill_range_f32(&mut a, LEN, -20.0, 20.0);

        let mut out = vec![0.0f32; LEN];
        unsafe { unary_f32(UnaryOp::Atan, a.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = (a[i] as f64).atan() as f32;
            let rel_err = rel_err_f32(out[i], expected);
            assert!(
                rel_err < 1e-6,
                "atan({}) = {}, expected {}, rel_err = {} at index {}",
                a[i],
                out[i],
                expected,
                rel_err,
                i
            );
        }
    }

    #[test]
    fn test_unary_trig_f32_domain_edges() {
        const LEN: usize = 2048;

        // ±inf and NaN have no finite reduction, so all three are NaN there.
        // ±0 must keep its own sign, which `x - j*π/2` destroys for j = -0.
        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            std::f32::consts::FRAC_PI_2,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Sin, f64::sin as fn(f64) -> f64),
            (UnaryOp::Cos, f64::cos as fn(f64) -> f64),
            (UnaryOp::Tan, f64::tan as fn(f64) -> f64),
        ] {
            unsafe { unary_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                if !a[i].is_finite() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                    continue;
                }
                let expected = reference(a[i] as f64) as f32;
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {}",
                    op,
                    a[i],
                    out[i],
                    expected
                );
                if expected == 0.0 {
                    assert_eq!(
                        out[i].is_sign_negative(),
                        expected.is_sign_negative(),
                        "{:?}({}) lost the sign of zero",
                        op,
                        a[i]
                    );
                }
            }
        }
    }

    #[test]
    fn test_unary_inverse_trig_f32_domain_edges() {
        const LEN: usize = 2048;

        // asin/acos are NaN outside [-1, 1] and exact at the endpoints; atan
        // saturates to ±π/2 at infinity.
        let edges = [
            0.0f32,
            -0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            1.000_001,
            -1.000_001,
            2.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ];
        let a: Vec<f32> = (0..LEN).map(|i| edges[i % edges.len()]).collect();
        let mut out = vec![0.0f32; LEN];

        for (op, reference) in [
            (UnaryOp::Asin, f64::asin as fn(f64) -> f64),
            (UnaryOp::Acos, f64::acos as fn(f64) -> f64),
            (UnaryOp::Atan, f64::atan as fn(f64) -> f64),
        ] {
            unsafe { unary_f32(op, a.as_ptr(), out.as_mut_ptr(), LEN) }
            for i in 0..LEN {
                let expected = reference(a[i] as f64) as f32;
                if expected.is_nan() {
                    assert!(
                        out[i].is_nan(),
                        "{:?}({}) = {}, expected NaN",
                        op,
                        a[i],
                        out[i]
                    );
                    continue;
                }
                assert!(
                    rel_err_f32(out[i], expected) < 1e-6,
                    "{:?}({}) = {}, expected {}",
                    op,
                    a[i],
                    out[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_relu_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32 - 50.0).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { relu_f32(a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_relu_f64() {
        let a: Vec<f64> = (0..100).map(|x| x as f64 - 50.0).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { relu_f64(a.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = if a[i] > 0.0 { a[i] } else { 0.0 };
            assert_eq!(out[i], expected, "mismatch at index {}", i);
        }
    }
}
