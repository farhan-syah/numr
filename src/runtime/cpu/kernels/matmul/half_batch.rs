//! Batched half-precision to f32 conversion.
//!
//! Pure format conversion: whole rows widened before the f32 kernels can be
//! used on them. Kept apart from the matmul kernels for that reason.

#[cfg(feature = "f16")]
use crate::dtype::Element;

/// Batch convert half-precision (f16/bf16) elements to f32 using SIMD when available.
#[cfg(feature = "f16")]
#[inline]
pub(super) unsafe fn batch_half_to_f32<T: Element>(src: *const T, dst: *mut f32, len: usize) {
    use crate::dtype::DType;
    match T::DTYPE {
        #[cfg(target_arch = "x86_64")]
        DType::BF16 => {
            // BF16 → f32: shift left by 16 bits (bf16 is upper 16 bits of f32)
            batch_bf16_to_f32(src as *const u16, dst, len);
        }
        #[cfg(target_arch = "x86_64")]
        DType::F16 => {
            // F16 → f32: use F16C instruction if available
            batch_f16_to_f32(src as *const u16, dst, len);
        }
        _ => {
            for i in 0..len {
                *dst.add(i) = (*src.add(i)).to_f32();
            }
        }
    }
}

/// BF16 → f32 conversion using SIMD bit-shift (bf16 is just f32 with lower 16 bits zeroed)
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn batch_bf16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    if is_x86_feature_detected!("avx2") {
        batch_bf16_to_f32_avx2(src, dst, len);
    } else {
        batch_bf16_to_f32_scalar(src, dst, len);
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn batch_bf16_to_f32_avx2(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    while i + 8 <= len {
        let bf16_vals = _mm_loadu_si128(src.add(i) as *const __m128i);
        let i32_vals = _mm256_cvtepu16_epi32(bf16_vals);
        let f32_bits = _mm256_slli_epi32(i32_vals, 16);
        _mm256_storeu_ps(dst.add(i), _mm256_castsi256_ps(f32_bits));
        i += 8;
    }
    // Scalar tail
    while i < len {
        let bits = (*src.add(i) as u32) << 16;
        *dst.add(i) = f32::from_bits(bits);
        i += 1;
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
unsafe fn batch_bf16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        let bits = (*src.add(i) as u32) << 16;
        *dst.add(i) = f32::from_bits(bits);
    }
}

/// F16 → f32 conversion using F16C instructions (vcvtph2ps)
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn batch_f16_to_f32(src: *const u16, dst: *mut f32, len: usize) {
    if is_x86_feature_detected!("f16c") {
        batch_f16_to_f32_f16c(src, dst, len);
    } else {
        batch_f16_to_f32_scalar(src, dst, len);
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "f16c", enable = "avx")]
unsafe fn batch_f16_to_f32_f16c(src: *const u16, dst: *mut f32, len: usize) {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    while i + 8 <= len {
        let f16_vals = _mm_loadu_si128(src.add(i) as *const __m128i);
        let f32_vals = _mm256_cvtph_ps(f16_vals);
        _mm256_storeu_ps(dst.add(i), f32_vals);
        i += 8;
    }
    // Scalar tail
    while i < len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
        i += 1;
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
unsafe fn batch_f16_to_f32_scalar(src: *const u16, dst: *mut f32, len: usize) {
    for i in 0..len {
        *dst.add(i) = half::f16::from_bits(*src.add(i)).to_f32();
    }
}
