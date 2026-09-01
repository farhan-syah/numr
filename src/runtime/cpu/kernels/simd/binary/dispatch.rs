//! SIMD-accelerated binary operation dispatch.
//!
//! This module provides multi-architecture SIMD implementations for element-wise
//! binary operations (add, sub, mul, div, max, min, pow).
//!
//! # Architecture Support
//!
//! | Architecture | Instruction Set | Vector Width | f32 lanes | f64 lanes |
//! |--------------|-----------------|--------------|-----------|-----------|
//! | x86-64       | AVX-512         | 512 bits     | 16        | 8         |
//! | x86-64       | AVX2 + FMA      | 256 bits     | 8         | 4         |
//! | ARM64        | NEON            | 128 bits     | 4         | 2         |

#[cfg(target_arch = "aarch64")]
use super::aarch64;
#[cfg(target_arch = "x86_64")]
use super::x86_64;

use crate::ops::BinaryOp;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

// Import scalar fallbacks from kernels module (single source of truth)
pub use crate::runtime::cpu::kernels::binary::{
    binary_scalar_f32, binary_scalar_f64, binary_scalar_i32,
};

/// Minimum elements to justify SIMD overhead
const SIMD_THRESHOLD: usize = 32;

/// SIMD binary operation for f32
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f32(op: BinaryOp, a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f32(op, a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::binary_f32(op, a, b, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::binary_f32(op, a, b, out, len),
        _ => binary_scalar_f32(op, a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::binary_f32(op, a, b, out, len),
        _ => binary_scalar_f32(op, a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    binary_scalar_f32(op, a, b, out, len);
}

/// SIMD binary operation for f64
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_f64(op: BinaryOp, a: *const f64, b: *const f64, out: *mut f64, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_f64(op, a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512::binary_f64(op, a, b, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2::binary_f64(op, a, b, out, len),
        _ => binary_scalar_f64(op, a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::binary_f64(op, a, b, out, len),
        _ => binary_scalar_f64(op, a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    binary_scalar_f64(op, a, b, out, len);
}

/// SIMD binary operation for i32
///
/// # Safety
/// - `a`, `b`, and `out` must be valid pointers to `len` elements
#[inline]
pub unsafe fn binary_i32(op: BinaryOp, a: *const i32, b: *const i32, out: *mut i32, len: usize) {
    let level = detect_simd();

    if len < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        binary_scalar_i32(op, a, b, out, len);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => x86_64::avx512_int::binary_i32(op, a, b, out, len),
        SimdLevel::Avx2Fma => x86_64::avx2_int::binary_i32(op, a, b, out, len),
        _ => binary_scalar_i32(op, a, b, out, len),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon_int::binary_i32(op, a, b, out, len),
        _ => binary_scalar_i32(op, a, b, out, len),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    binary_scalar_i32(op, a, b, out, len);
}

half_binary_op!(binary, binary_f32, BinaryOp);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_add_f32() {
        let a: Vec<f32> = (0..100).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..100).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_mul_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..101).map(|x| (x * 2) as f64).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { binary_f64(BinaryOp::Mul, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] * b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_small_array_uses_scalar() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let b = [5.0f32, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 4];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) }

        assert_eq!(out, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_non_aligned_length() {
        let a: Vec<f32> = (0..67).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..67).map(|x| (x * 2) as f32).collect();
        let mut out = vec![0.0f32; 67];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 67) }

        for i in 0..67 {
            assert_eq!(out[i], a[i] + b[i], "mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_pow_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32 * 0.1).collect();
        let b: Vec<f32> = (1..101).map(|x| (x % 5) as f32 + 0.5).collect();
        let mut out = vec![0.0f32; 100];

        unsafe { binary_f32(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].powf(b[i]);
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-3 * expected.abs().max(1.0),
                "pow mismatch at {}: got {}, expected {} (a={}, b={})",
                i,
                out[i],
                expected,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_binary_pow_f64() {
        let a: Vec<f64> = (1..101).map(|x| x as f64 * 0.1).collect();
        let b: Vec<f64> = (1..101).map(|x| (x % 5) as f64 + 0.5).collect();
        let mut out = vec![0.0f64; 100];

        unsafe { binary_f64(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            let expected = a[i].powf(b[i]);
            let diff = (out[i] - expected).abs();
            assert!(
                diff < 1e-4 * expected.abs().max(1.0),
                "pow mismatch at {}: got {}, expected {} (a={}, b={})",
                i,
                out[i],
                expected,
                a[i],
                b[i]
            );
        }
    }

    #[test]
    fn test_binary_max_min_f32() {
        let a: Vec<f32> = (0..100).map(|x| (x as f32 - 50.0) * 0.5).collect();
        let b: Vec<f32> = (0..100).map(|x| ((x + 25) as f32 - 50.0) * 0.5).collect();
        let mut out_max = vec![0.0f32; 100];
        let mut out_min = vec![0.0f32; 100];

        unsafe {
            binary_f32(
                BinaryOp::Max,
                a.as_ptr(),
                b.as_ptr(),
                out_max.as_mut_ptr(),
                100,
            );
            binary_f32(
                BinaryOp::Min,
                a.as_ptr(),
                b.as_ptr(),
                out_min.as_mut_ptr(),
                100,
            );
        }

        for i in 0..100 {
            assert_eq!(out_max[i], a[i].max(b[i]), "max mismatch at {}", i);
            assert_eq!(out_min[i], a[i].min(b[i]), "min mismatch at {}", i);
        }
    }

    #[test]
    fn test_binary_sub_div_f32() {
        let a: Vec<f32> = (1..101).map(|x| x as f32 * 2.0).collect();
        let b: Vec<f32> = (1..101).map(|x| x as f32).collect();
        let mut out_sub = vec![0.0f32; 100];
        let mut out_div = vec![0.0f32; 100];

        unsafe {
            binary_f32(
                BinaryOp::Sub,
                a.as_ptr(),
                b.as_ptr(),
                out_sub.as_mut_ptr(),
                100,
            );
            binary_f32(
                BinaryOp::Div,
                a.as_ptr(),
                b.as_ptr(),
                out_div.as_mut_ptr(),
                100,
            );
        }

        for i in 0..100 {
            assert_eq!(out_sub[i], a[i] - b[i], "sub mismatch at {}", i);
            assert_eq!(out_div[i], a[i] / b[i], "div mismatch at {}", i);
        }
    }

    // ============================================================================
    // Streaming store tests (x86-64 only)
    // ============================================================================

    #[cfg(target_arch = "x86_64")]
    mod streaming_tests {
        use super::super::super::super::streaming::{
            STREAMING_THRESHOLD_F32, STREAMING_THRESHOLD_F64,
        };

        /// Test streaming threshold constant is correctly defined
        #[test]
        fn test_streaming_threshold_defined() {
            // 1MB = 262144 f32s, 131072 f64s
            assert_eq!(STREAMING_THRESHOLD_F32, 262144);
            assert_eq!(STREAMING_THRESHOLD_F64, 131072);
        }
    }

    /// Test that large arrays produce correct results (exercises streaming path if aligned)
    #[test]
    fn test_large_array_correctness_f32() {
        const LEN: usize = 1024;
        let a: Vec<f32> = (0..LEN).map(|x| (x as f32) * 0.1).collect();
        let b: Vec<f32> = (0..LEN).map(|x| (x as f32) * 0.2 + 1.0).collect();
        let mut out = vec![0.0f32; LEN];

        unsafe { binary_f32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i] + b[i];
            assert!(
                (out[i] - expected).abs() < 1e-6,
                "large array mismatch at {}: got {}, expected {}",
                i,
                out[i],
                expected
            );
        }
    }

    #[test]
    fn test_binary_pow_f64_nonpositive_base() {
        // exp(b*log(a)) has no answer for a <= 0: log(0) is -inf and log of a
        // negative is NaN, so these lanes must come from the scalar reference.
        const LEN: usize = 512;
        let bases = [0.0f64, -0.0, -2.0, -1.0, 2.0, 0.5, -0.5, 4.0];
        let exps = [0.0f64, 1.0, 2.0, 3.0, -1.0, 0.5, -2.0, 10.0];
        let a: Vec<f64> = (0..LEN).map(|i| bases[i % bases.len()]).collect();
        let b: Vec<f64> = (0..LEN)
            .map(|i| exps[(i / bases.len()) % exps.len()])
            .collect();
        let mut out = vec![0.0f64; LEN];

        unsafe { binary_f64(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), LEN) }

        for i in 0..LEN {
            let expected = a[i].powf(b[i]);
            if a[i] > 0.0 {
                continue; // accuracy of the positive branch is covered above
            }
            if expected.is_nan() {
                assert!(
                    out[i].is_nan(),
                    "pow({}, {}) = {}, expected NaN",
                    a[i],
                    b[i],
                    out[i]
                );
            } else {
                assert_eq!(out[i], expected, "pow({}, {}) mismatch", a[i], b[i]);
            }
        }
    }

    /// Test that large arrays produce correct results for all operations
    #[test]
    fn test_large_array_all_ops_f32() {
        const LEN: usize = 512;
        let a: Vec<f32> = (1..=LEN as i32).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=LEN as i32).map(|x| (x as f32) * 0.5 + 0.5).collect();

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let mut out = vec![0.0f32; LEN];
            unsafe { binary_f32(op, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), LEN) }

            for i in 0..LEN {
                let expected = match op {
                    BinaryOp::Add => a[i] + b[i],
                    BinaryOp::Sub => a[i] - b[i],
                    BinaryOp::Mul => a[i] * b[i],
                    BinaryOp::Div => a[i] / b[i],
                    BinaryOp::Max => a[i].max(b[i]),
                    BinaryOp::Min => a[i].min(b[i]),
                    BinaryOp::Pow => a[i].powf(b[i]),
                    BinaryOp::Atan2 => a[i].atan2(b[i]),
                };
                assert!(
                    (out[i] - expected).abs() < 1e-5 * expected.abs().max(1.0),
                    "{:?} mismatch at {}: got {}, expected {}",
                    op,
                    i,
                    out[i],
                    expected
                );
            }
        }
    }

    #[test]
    fn test_binary_add_i32() {
        let a: Vec<i32> = (0..100).collect();
        let b: Vec<i32> = (0..100).map(|x| x * 2).collect();
        let mut out = vec![0i32; 100];

        unsafe { binary_i32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] + b[i], "i32 add mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_all_ops_i32() {
        let a: Vec<i32> = (1..101).collect();
        let b: Vec<i32> = (1..101).map(|x| x * 2 + 1).collect();

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Max,
            BinaryOp::Min,
        ] {
            let mut out = vec![0i32; 100];
            unsafe { binary_i32(op, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

            for i in 0..100 {
                let expected = match op {
                    BinaryOp::Add => a[i] + b[i],
                    BinaryOp::Sub => a[i] - b[i],
                    BinaryOp::Mul => a[i] * b[i],
                    BinaryOp::Max => a[i].max(b[i]),
                    BinaryOp::Min => a[i].min(b[i]),
                    _ => unreachable!(),
                };
                assert_eq!(out[i], expected, "{:?} i32 mismatch at {}", op, i);
            }
        }
    }

    #[test]
    fn test_binary_i32_non_aligned_length() {
        let a: Vec<i32> = (0..67).collect();
        let b: Vec<i32> = (0..67).map(|x| x * 3).collect();
        let mut out = vec![0i32; 67];

        unsafe { binary_i32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 67) }

        for i in 0..67 {
            assert_eq!(out[i], a[i] + b[i], "i32 add tail mismatch at index {}", i);
        }
    }

    #[test]
    fn test_binary_i32_small_array() {
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut out = [0i32; 4];

        unsafe { binary_i32(BinaryOp::Add, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) }

        assert_eq!(out, [6, 8, 10, 12]);
    }

    #[test]
    fn test_binary_div_i32() {
        let a: Vec<i32> = (1..101).collect();
        let b: Vec<i32> = (1..101).map(|x| x * 2 + 1).collect();
        let mut out = vec![0i32; 100];

        unsafe { binary_i32(BinaryOp::Div, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 100) }

        for i in 0..100 {
            assert_eq!(out[i], a[i] / b[i], "div mismatch at {}", i);
        }
    }

    #[test]
    fn test_binary_div_i32_by_zero() {
        let a = [10i32, 20, 0, 30, -5, 100, i32::MAX, i32::MIN];
        let b = [0i32, 2, 5, 0, 0, -3, 0, 0];
        let mut out = [0i32; 8];

        unsafe { binary_i32(BinaryOp::Div, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 8) }

        assert_eq!(out[0], 0, "10 / 0 should be 0");
        assert_eq!(out[1], 10, "20 / 2 should be 10");
        assert_eq!(out[2], 0, "0 / 5 should be 0");
        assert_eq!(out[3], 0, "30 / 0 should be 0");
        assert_eq!(out[4], 0, "-5 / 0 should be 0");
        assert_eq!(out[5], -33, "100 / -3 should be -33");
        assert_eq!(out[6], 0, "i32::MAX / 0 should be 0");
        assert_eq!(out[7], 0, "i32::MIN / 0 should be 0");
    }

    #[test]
    fn test_binary_pow_i32() {
        let a = [2i32, 3, 10, 0, -2, 1, 5, 100];
        let b = [10i32, 5, 3, 5, 3, 100, 0, 1];
        let mut out = [0i32; 8];

        unsafe { binary_i32(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 8) }

        assert_eq!(out[0], 1024, "2^10");
        assert_eq!(out[1], 243, "3^5");
        assert_eq!(out[2], 1000, "10^3");
        assert_eq!(out[3], 0, "0^5");
        assert_eq!(out[4], -8, "(-2)^3");
        assert_eq!(out[5], 1, "1^100");
        assert_eq!(out[6], 1, "5^0");
        assert_eq!(out[7], 100, "100^1");
    }

    #[test]
    fn test_binary_pow_i32_saturates_on_overflow() {
        let a = [2i32, -2, 46341, -46341, 3];
        let b = [40i32, 41, 2, 3, 30];
        let mut out = [0i32; 5];

        unsafe { binary_i32(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 5) }

        assert_eq!(out[0], i32::MAX, "2^40 saturates");
        assert_eq!(out[1], i32::MIN, "(-2)^41 saturates negative");
        assert_eq!(out[2], i32::MAX, "46341^2 saturates");
        assert_eq!(out[3], i32::MIN, "(-46341)^3 saturates negative");
        assert_eq!(out[4], i32::MAX, "3^30 saturates");
    }

    #[test]
    fn test_binary_pow_i32_negative_exponent_truncates() {
        let a = [2i32, 1, -1, 5];
        let b = [-1i32, -5, -3, -2];
        let mut out = [0i32; 4];

        unsafe { binary_i32(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 4) }

        assert_eq!(out[0], 0, "2^-1 truncates to 0");
        assert_eq!(out[1], 1, "1^-5 is 1");
        assert_eq!(out[2], -1, "(-1)^-3 is -1");
        assert_eq!(out[3], 0, "5^-2 truncates to 0");
    }

    #[test]
    fn test_binary_pow_i64_is_exact_past_the_f64_mantissa() {
        use crate::runtime::cpu::kernels::binary_op_kernel;

        let a = [7i64, 3, 2, -2, 10];
        let b = [20i64, 39, 62, 63, 18];
        let mut out = [0i64; 5];

        unsafe { binary_op_kernel(BinaryOp::Pow, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 5) }

        assert_eq!(out[0], 79792266297612001i64, "7^20");
        assert_eq!(out[1], 4052555153018976267i64, "3^39");
        assert_eq!(out[2], 4611686018427387904i64, "2^62");
        assert_eq!(out[3], i64::MIN, "(-2)^63 is exactly i64::MIN");
        assert_eq!(out[4], 1000000000000000000i64, "10^18");
    }

    #[test]
    fn test_binary_atan2_i32() {
        let a = [0i32, 1, -1, 10, 0, 100];
        let b = [1i32, 0, 0, 10, 0, 1];
        let mut out = [0i32; 6];

        unsafe { binary_i32(BinaryOp::Atan2, a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), 6) }

        assert_eq!(out[0], 0, "atan2(0,1) = 0");
        assert_eq!(out[1], 1, "atan2(1,0) truncates to 1");
        assert_eq!(out[2], -1, "atan2(-1,0) truncates to -1");
        assert_eq!(out[3], 0, "atan2(10,10) truncates to 0");
    }

    /// Test alignment check functions (x86-64 only)
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_alignment_checks() {
        use crate::runtime::cpu::kernels::simd::streaming::{is_aligned_avx2, is_aligned_avx512};

        assert!(is_aligned_avx2(32 as *const f32));
        assert!(is_aligned_avx2(64 as *const f32));
        assert!(!is_aligned_avx2(16 as *const f32));

        assert!(is_aligned_avx512(64 as *const f32));
        assert!(is_aligned_avx512(128 as *const f32));
        assert!(!is_aligned_avx512(32 as *const f32));
    }
}
