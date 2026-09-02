//! SIMD-accelerated special functions
//!
//! Provides vectorized implementations of error functions and Bessel functions.
//! Uses polynomial evaluation with FMA for accuracy and performance.
//!
//! # Supported Functions
//!
//! | Function  | Algorithm                        | Accuracy        |
//! |-----------|----------------------------------|-----------------|
//! | erf       | A&S 7.1.26 polynomial            | ~1e-7 rel err   |
//! | erfc      | 1 - erf(x)                       | ~1e-7 rel err   |
//! | bessel_j0 | Rational poly + asymptotic       | ~1e-8 rel err   |
//! | bessel_j1 | Rational poly + asymptotic       | ~1e-8 rel err   |
//! | bessel_i0 | Power series + asymptotic        | ~1e-8 rel err   |
//! | bessel_i1 | Power series + asymptotic        | ~1e-8 rel err   |

#[cfg(target_arch = "x86_64")]
mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;

#[cfg(target_arch = "aarch64")]
mod aarch64;

pub mod coefficients;

use super::{SimdLevel, detect_simd};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

// ============================================================================
// Dispatch Macros - Eliminate duplication across special functions
// ============================================================================

/// Generate SIMD dispatch function for f32 with architecture-specific backends
macro_rules! impl_simd_dispatch_f32 {
    ($fn_name:ident, $scalar_fn:ident) => {
        #[inline]
        pub unsafe fn $fn_name(input: *const f32, output: *mut f32, len: usize) {
            let level = detect_simd();

            if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
                $scalar_fn(input, output, len);
                return;
            }

            #[cfg(target_arch = "x86_64")]
            match level {
                SimdLevel::Avx512 => avx512::$fn_name(input, output, len),
                SimdLevel::Avx2Fma => avx2::$fn_name(input, output, len),
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(target_arch = "aarch64")]
            match level {
                SimdLevel::Neon | SimdLevel::NeonFp16 => {
                    aarch64::neon::$fn_name(input, output, len)
                }
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            $scalar_fn(input, output, len);
        }
    };
}

/// Generate SIMD dispatch function for f64 with architecture-specific backends
macro_rules! impl_simd_dispatch_f64 {
    ($fn_name:ident, $scalar_fn:ident) => {
        #[inline]
        pub unsafe fn $fn_name(input: *const f64, output: *mut f64, len: usize) {
            let level = detect_simd();

            if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
                $scalar_fn(input, output, len);
                return;
            }

            #[cfg(target_arch = "x86_64")]
            match level {
                SimdLevel::Avx512 => avx512::$fn_name(input, output, len),
                SimdLevel::Avx2Fma => avx2::$fn_name(input, output, len),
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(target_arch = "aarch64")]
            match level {
                SimdLevel::Neon | SimdLevel::NeonFp16 => {
                    aarch64::neon::$fn_name(input, output, len)
                }
                _ => $scalar_fn(input, output, len),
            }

            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            $scalar_fn(input, output, len);
        }
    };
}

/// Generate both f32 and f64 dispatch functions
macro_rules! impl_simd_dispatch {
    ($base_name:ident) => {
        paste::paste! {
            impl_simd_dispatch_f32!([<$base_name _f32>], [<$base_name _scalar_f32>]);
            impl_simd_dispatch_f64!([<$base_name _f64>], [<$base_name _scalar_f64>]);
        }
    };
}

/// Generate scalar fallback pair (f32 + f64) from a scalar function
macro_rules! impl_scalar_fallback {
    ($base_name:ident, $scalar_fn:path) => {
        paste::paste! {
            #[inline]
            unsafe fn [<$base_name _scalar_f32>](input: *const f32, output: *mut f32, len: usize) {
                for i in 0..len {
                    let x = *input.add(i);
                    *output.add(i) = $scalar_fn(x as f64) as f32;
                }
            }

            #[inline]
            unsafe fn [<$base_name _scalar_f64>](input: *const f64, output: *mut f64, len: usize) {
                for i in 0..len {
                    *output.add(i) = $scalar_fn(*input.add(i));
                }
            }
        }
    };
}

/// Generate scalar-only dispatch (no SIMD implementation available)
macro_rules! impl_scalar_only {
    ($base_name:ident) => {
        paste::paste! {
            #[inline]
            pub unsafe fn [<$base_name _f32>](input: *const f32, output: *mut f32, len: usize) {
                [<$base_name _scalar_f32>](input, output, len);
            }

            #[inline]
            pub unsafe fn [<$base_name _f64>](input: *const f64, output: *mut f64, len: usize) {
                [<$base_name _scalar_f64>](input, output, len);
            }
        }
    };
}

// ============================================================================
// Scalar Fallbacks - Import and generate typed wrappers
// ============================================================================

use crate::algorithm::special::scalar::{
    bessel_i0_scalar, bessel_i1_scalar, bessel_j0_scalar, bessel_j1_scalar, digamma_scalar,
    erf_scalar, erfc_scalar, gamma_scalar, lgamma_scalar,
};

impl_scalar_fallback!(erf, erf_scalar);
impl_scalar_fallback!(erfc, erfc_scalar);
impl_scalar_fallback!(bessel_j0, bessel_j0_scalar);
impl_scalar_fallback!(bessel_j1, bessel_j1_scalar);
impl_scalar_fallback!(bessel_i0, bessel_i0_scalar);
impl_scalar_fallback!(bessel_i1, bessel_i1_scalar);
impl_scalar_fallback!(gamma, gamma_scalar);
impl_scalar_fallback!(lgamma, lgamma_scalar);
impl_scalar_fallback!(digamma, digamma_scalar);

// ============================================================================
// SIMD Dispatch Functions - Error Functions
// ============================================================================

impl_simd_dispatch!(erf);
impl_simd_dispatch!(erfc);

// ============================================================================
// SIMD Dispatch Functions - Bessel Functions
// ============================================================================

impl_simd_dispatch!(bessel_j0);
impl_simd_dispatch!(bessel_j1);
impl_simd_dispatch!(bessel_i0);
impl_simd_dispatch!(bessel_i1);

// ============================================================================
// Gamma Functions - Scalar only (no SIMD implementation yet)
// ============================================================================

impl_scalar_only!(gamma);
impl_scalar_only!(lgamma);
impl_scalar_only!(digamma);

// F16/BF16 Wrappers via macros
half_unary!(erf, erf_f32);
half_unary!(erfc, erfc_f32);
half_unary!(bessel_j0, bessel_j0_f32);
half_unary!(bessel_j1, bessel_j1_f32);
half_unary!(bessel_i0, bessel_i0_f32);
half_unary!(bessel_i1, bessel_i1_f32);
half_unary!(gamma, gamma_f32);
half_unary!(lgamma, lgamma_f32);
half_unary!(digamma, digamma_f32);

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erf_f32() {
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.05 - 3.0).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            erf_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            erf_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            assert!(
                diff < 1e-5,
                "erf mismatch at {}: SIMD={} scalar={} diff={}",
                i,
                out_simd[i],
                out_scalar[i],
                diff
            );
        }
    }

    #[test]
    fn test_bessel_j0_f32() {
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.2).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            bessel_j0_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            bessel_j0_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let rel_err = if out_scalar[i].abs() > 1e-6 {
                diff / out_scalar[i].abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-4,
                "bessel_j0 mismatch at {}: SIMD={} scalar={} rel_err={}",
                i,
                out_simd[i],
                out_scalar[i],
                rel_err
            );
        }
    }

    /// Relative error against a reference, tolerant of an exact match.
    ///
    /// Two equal infinities subtract to NaN, which would otherwise fail a
    /// comparison the kernel got exactly right.
    fn rel_err_f64(got: f64, expected: f64) -> f64 {
        if got == expected {
            return 0.0;
        }
        if !got.is_finite() || !expected.is_finite() {
            return f64::INFINITY;
        }
        (got - expected).abs() / expected.abs()
    }

    fn rel_err_f32(got: f32, expected: f32) -> f32 {
        if got == expected {
            return 0.0;
        }
        if !got.is_finite() || !expected.is_finite() {
            return f32::INFINITY;
        }
        (got - expected).abs() / expected.abs()
    }

    /// Length forcing the SIMD path: past SIMD_THRESHOLD and a whole number of
    /// AVX2, AVX-512 and NEON lanes in both precisions, so no probe falls
    /// through to the scalar remainder loop.
    const SIMD_LEN: usize = 2048;

    /// Build a SIMD-length input whose leading elements are the probe points.
    /// The filler sits in the power-series branch and is never asserted on.
    fn probe_input_f64(points: &[f64]) -> Vec<f64> {
        let mut a = vec![0.5f64; SIMD_LEN];
        a[..points.len()].copy_from_slice(points);
        a
    }

    fn probe_input_f32(points: &[f32]) -> Vec<f32> {
        let mut a = vec![0.5f32; SIMD_LEN];
        a[..points.len()].copy_from_slice(points);
        a
    }

    /// I0(x) = exp(x)/sqrt(2*pi*x) * P(1/x) stays finite past the point where
    /// exp(x) alone overflows, because sqrt(2*pi*x) divides it back under
    /// f64::MAX. Forming exp(x) first returns infinity over the whole band
    /// (709.7827, 713.9869], which no later division can recover.
    ///
    /// The limit 713.9869085439683 solves x - ln(sqrt(2*pi*x)) = ln(f64::MAX);
    /// references are mpmath at 40 digits.
    #[test]
    fn test_bessel_i0_f64_overflow_band() {
        // (x, I0(x), relative tolerance)
        const PROBES: [(f64, f64, f64); 7] = [
            (5.0, 27.239_871_823_604_446, 1e-12),
            (100.0, 1.073_751_707_131_073_8e42, 1e-12),
            (700.0, 1.529_593_347_671_873_7e302, 1e-12),
            (710.0, 3.345_334_558_619_656e306, 1e-12),
            (712.0, 2.468_411_057_762_752_3e307, 1e-12),
            (713.5, 1.105_101_208_117_827_9e308, 1e-12),
            (713.9, 1.648_155_186_695_175_2e308, 1e-12),
        ];

        let points: Vec<f64> = PROBES.iter().map(|p| p.0).collect();
        let a = probe_input_f64(&points);
        let mut out = vec![0.0f64; SIMD_LEN];
        unsafe { bessel_i0_f64(a.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }

        for (i, (x, expected, tol)) in PROBES.iter().enumerate() {
            let err = rel_err_f64(out[i], *expected);
            assert!(
                err < *tol,
                "bessel_i0({}) = {}, expected {}, rel_err = {}",
                x,
                out[i],
                expected,
                err
            );
        }

        // At the limit the true value equals f64::MAX to eighteen digits, so
        // either the largest finite double or infinity is correct.
        let edge = probe_input_f64(&[713.986_908_543_968_3, 714.0, 720.0]);
        let mut out = vec![0.0f64; SIMD_LEN];
        unsafe { bessel_i0_f64(edge.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }
        assert!(
            out[0] >= 1.797e308,
            "bessel_i0 at the overflow limit = {}, expected f64::MAX or infinity",
            out[0]
        );
        assert!(out[1].is_infinite(), "bessel_i0(714.0) = {}", out[1]);
        assert!(out[2].is_infinite(), "bessel_i0(720.0) = {}", out[2]);
    }

    /// I1 has the same overflow band as I0, ending at 713.9876098185423 — its
    /// asymptotic polynomial is slightly below one, so it stays finite a little
    /// further out. See `test_bessel_i0_f64_overflow_band`.
    #[test]
    fn test_bessel_i1_f64_overflow_band() {
        const PROBES: [(f64, f64, f64); 7] = [
            (5.0, 24.335_642_142_450_528, 1e-12),
            (100.0, 1.068_369_390_338_162_5e42, 1e-12),
            (700.0, 1.528_500_390_233_900_6e302, 1e-12),
            (710.0, 3.342_977_858_509_762_6e306, 1e-12),
            (712.0, 2.466_677_013_524_615_3e307, 1e-12),
            (713.5, 1.104_326_513_679_595_2e308, 1e-12),
            (713.9, 1.647_000_449_923_271_8e308, 1e-12),
        ];

        let points: Vec<f64> = PROBES.iter().map(|p| p.0).collect();
        let a = probe_input_f64(&points);
        let mut out = vec![0.0f64; SIMD_LEN];
        unsafe { bessel_i1_f64(a.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }

        for (i, (x, expected, tol)) in PROBES.iter().enumerate() {
            let err = rel_err_f64(out[i], *expected);
            assert!(
                err < *tol,
                "bessel_i1({}) = {}, expected {}, rel_err = {}",
                x,
                out[i],
                expected,
                err
            );
        }

        let edge = probe_input_f64(&[713.987_609_818_542_3, 714.0, 720.0]);
        let mut out = vec![0.0f64; SIMD_LEN];
        unsafe { bessel_i1_f64(edge.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }
        assert!(
            out[0] >= 1.796e308,
            "bessel_i1 at the overflow limit = {}, expected f64::MAX or infinity",
            out[0]
        );
        assert!(out[1].is_infinite(), "bessel_i1(714.0) = {}", out[1]);
        assert!(out[2].is_infinite(), "bessel_i1(720.0) = {}", out[2]);

        // Odd function: the sign follows the argument even where it overflows.
        let neg = probe_input_f64(&[-710.0, -714.0]);
        let mut out = vec![0.0f64; SIMD_LEN];
        unsafe { bessel_i1_f64(neg.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }
        let err = rel_err_f64(out[0], -3.342_977_858_509_762_6e306);
        assert!(
            err < 1e-12,
            "bessel_i1(-710.0) = {}, rel_err = {}",
            out[0],
            err
        );
        assert!(
            out[1] == f64::NEG_INFINITY,
            "bessel_i1(-714.0) = {}",
            out[1]
        );
    }

    /// The f32 analogue: exp overflows past ln(f32::MAX) = 88.7228 while I0
    /// stays finite up to 91.9008, so the band (88.7228, 91.9008] returned
    /// infinity for a representable result.
    #[test]
    fn test_bessel_i0_f32_overflow_band() {
        // Tolerance matches the f32 asymptotic accuracy; the defect it guards
        // against is infinity against a finite value, not a last-digit shift.
        const PROBES: [(f32, f32); 7] = [
            (5.0, 27.239_872),
            (20.0, 4.355_828_4e7),
            (88.0, 7.034_019_5e36),
            (89.0, 1.901_241_9e37),
            (90.0, 5.139_238_6e37),
            (91.0, 1.389_271_4e38),
            (91.5, 2.284_237_2e38),
        ];

        let points: Vec<f32> = PROBES.iter().map(|p| p.0).collect();
        let a = probe_input_f32(&points);
        let mut out = vec![0.0f32; SIMD_LEN];
        unsafe { bessel_i0_f32(a.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }

        for (i, (x, expected)) in PROBES.iter().enumerate() {
            let err = rel_err_f32(out[i], *expected);
            assert!(
                err < 1e-4,
                "bessel_i0({}) = {}, expected {}, rel_err = {}",
                x,
                out[i],
                expected,
                err
            );
        }

        // 91.90076 solves x - ln(sqrt(2*pi*x)) = ln(f32::MAX).
        let edge = probe_input_f32(&[91.900_764, 92.0, 93.0]);
        let mut out = vec![0.0f32; SIMD_LEN];
        unsafe { bessel_i0_f32(edge.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }
        assert!(
            out[0] >= 3.402e38,
            "bessel_i0 at the overflow limit = {}, expected f32::MAX or infinity",
            out[0]
        );
        assert!(out[1].is_infinite(), "bessel_i0(92.0) = {}", out[1]);
        assert!(out[2].is_infinite(), "bessel_i0(93.0) = {}", out[2]);
    }

    /// I1 in f32 stays finite up to 91.9063. See
    /// `test_bessel_i0_f32_overflow_band`.
    #[test]
    fn test_bessel_i1_f32_overflow_band() {
        const PROBES: [(f32, f32); 7] = [
            (5.0, 24.335_642),
            (20.0, 4.245_497_2e7),
            (88.0, 6.993_939e36),
            (89.0, 1.890_530_5e37),
            (90.0, 5.110_607e37),
            (91.0, 1.381_616_9e38),
            (91.5, 2.271_720_5e38),
        ];

        let points: Vec<f32> = PROBES.iter().map(|p| p.0).collect();
        let a = probe_input_f32(&points);
        let mut out = vec![0.0f32; SIMD_LEN];
        unsafe { bessel_i1_f32(a.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }

        for (i, (x, expected)) in PROBES.iter().enumerate() {
            let err = rel_err_f32(out[i], *expected);
            assert!(
                err < 1e-4,
                "bessel_i1({}) = {}, expected {}, rel_err = {}",
                x,
                out[i],
                expected,
                err
            );
        }

        let edge = probe_input_f32(&[91.906_265, 92.0, 93.0]);
        let mut out = vec![0.0f32; SIMD_LEN];
        unsafe { bessel_i1_f32(edge.as_ptr(), out.as_mut_ptr(), SIMD_LEN) }
        assert!(
            out[0] >= 3.402e38,
            "bessel_i1 at the overflow limit = {}, expected f32::MAX or infinity",
            out[0]
        );
        assert!(out[1].is_infinite(), "bessel_i1(92.0) = {}", out[1]);
        assert!(out[2].is_infinite(), "bessel_i1(93.0) = {}", out[2]);
    }

    #[test]
    fn test_bessel_i0_f32() {
        // Test for modest arguments to avoid overflow
        let input: Vec<f32> = (0..128).map(|x| (x as f32) * 0.1).collect();
        let mut out_simd = vec![0.0f32; 128];
        let mut out_scalar = vec![0.0f32; 128];

        unsafe {
            bessel_i0_f32(input.as_ptr(), out_simd.as_mut_ptr(), 128);
            bessel_i0_scalar_f32(input.as_ptr(), out_scalar.as_mut_ptr(), 128);
        }

        for i in 0..128 {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            let rel_err = if out_scalar[i].abs() > 1e-6 {
                diff / out_scalar[i].abs()
            } else {
                diff
            };
            assert!(
                rel_err < 1e-4,
                "bessel_i0 mismatch at {}: SIMD={} scalar={} rel_err={}",
                i,
                out_simd[i],
                out_scalar[i],
                rel_err
            );
        }
    }
}
