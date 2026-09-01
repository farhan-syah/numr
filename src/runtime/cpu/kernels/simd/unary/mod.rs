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
