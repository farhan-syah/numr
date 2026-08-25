//! Matrix multiplication kernels
//!
//! This module provides matrix multiplication with automatic SIMD dispatch.
//! On x86-64, f32 and f64 matmuls use AVX-512 or AVX2+FMA when available.
//!
//! # Accumulator width
//!
//! A dot product of length `k` outgrows the element type for every float
//! narrower than F32 and for every integer dtype, so those accumulate in a
//! wider type and narrow once per output element. See
//! [`crate::runtime::cpu::kernels::wide_acc`] for why the accumulators are f32
//! and i128, and why integer narrowing saturates. Output dtypes are unchanged.

use super::wide_acc::WideAcc;
use crate::dtype::Element;

/// SIMD-accelerated f32 dot product for use in half-precision GEMV-BT.
///
/// Dispatches to AVX-512 or AVX2+FMA based on detected SIMD level.
/// Each backend is a separate function with `#[target_feature]` so the compiler
/// can optimize the entire function body for that ISA.
///
/// # Safety
/// - `a` and `b` must be valid pointers to `len` f32 elements
#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[inline]
unsafe fn simd_dot_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
    level: super::simd::SimdLevel,
) -> f32 {
    use super::simd::SimdLevel;

    match level {
        SimdLevel::Avx512 => simd_dot_f32_avx512(a, b, len),
        SimdLevel::Avx2Fma => simd_dot_f32_avx2(a, b, len),
        _ => {
            let mut sum = 0.0f32;
            for i in 0..len {
                sum += *a.add(i) * *b.add(i);
            }
            sum
        }
    }
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn simd_dot_f32_avx512(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut offset = 0;
    let mut acc0 = _mm512_setzero_ps();
    let mut acc1 = _mm512_setzero_ps();
    while offset + 32 <= len {
        let av0 = _mm512_loadu_ps(a.add(offset));
        let bv0 = _mm512_loadu_ps(b.add(offset));
        acc0 = _mm512_fmadd_ps(av0, bv0, acc0);
        let av1 = _mm512_loadu_ps(a.add(offset + 16));
        let bv1 = _mm512_loadu_ps(b.add(offset + 16));
        acc1 = _mm512_fmadd_ps(av1, bv1, acc1);
        offset += 32;
    }
    acc0 = _mm512_add_ps(acc0, acc1);
    while offset + 16 <= len {
        let av = _mm512_loadu_ps(a.add(offset));
        let bv = _mm512_loadu_ps(b.add(offset));
        acc0 = _mm512_fmadd_ps(av, bv, acc0);
        offset += 16;
    }
    let mut sum = _mm512_reduce_add_ps(acc0);
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}

#[cfg(all(feature = "f16", target_arch = "aarch64"))]
#[inline]
unsafe fn simd_dot_f32(
    a: *const f32,
    b: *const f32,
    len: usize,
    _level: super::simd::SimdLevel,
) -> f32 {
    simd_dot_f32_neon(a, b, len)
}

#[cfg(all(feature = "f16", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
unsafe fn simd_dot_f32_neon(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::aarch64::*;
    let mut offset = 0;
    let mut acc0 = vdupq_n_f32(0.0);
    let mut acc1 = vdupq_n_f32(0.0);
    // Process 8 floats per iteration with dual accumulators
    while offset + 8 <= len {
        let av0 = vld1q_f32(a.add(offset));
        let bv0 = vld1q_f32(b.add(offset));
        acc0 = vfmaq_f32(acc0, av0, bv0);
        let av1 = vld1q_f32(a.add(offset + 4));
        let bv1 = vld1q_f32(b.add(offset + 4));
        acc1 = vfmaq_f32(acc1, av1, bv1);
        offset += 8;
    }
    acc0 = vaddq_f32(acc0, acc1);
    // Handle remaining 4-float chunk
    while offset + 4 <= len {
        let av = vld1q_f32(a.add(offset));
        let bv = vld1q_f32(b.add(offset));
        acc0 = vfmaq_f32(acc0, av, bv);
        offset += 4;
    }
    let mut sum = vaddvq_f32(acc0);
    // Scalar tail
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}

#[cfg(all(feature = "f16", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_dot_f32_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut offset = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    // Process 16 floats per iteration with 2 independent accumulators
    // to hide FMA latency (4-5 cycles on modern x86)
    while offset + 16 <= len {
        let av0 = _mm256_loadu_ps(a.add(offset));
        let bv0 = _mm256_loadu_ps(b.add(offset));
        acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
        let av1 = _mm256_loadu_ps(a.add(offset + 8));
        let bv1 = _mm256_loadu_ps(b.add(offset + 8));
        acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
        offset += 16;
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    // Handle remaining 8-float chunk
    while offset + 8 <= len {
        let av = _mm256_loadu_ps(a.add(offset));
        let bv = _mm256_loadu_ps(b.add(offset));
        acc0 = _mm256_fmadd_ps(av, bv, acc0);
        offset += 8;
    }
    // Horizontal sum of 256-bit accumulator
    let hi = _mm256_extractf128_ps(acc0, 1);
    let lo = _mm256_castps256_ps128(acc0);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    let mut sum = _mm_cvtss_f32(sums2);
    while offset < len {
        sum += *a.add(offset) * *b.add(offset);
        offset += 1;
    }
    sum
}

/// GEMV-BT kernel: C[M,N] = A[M,K] @ B^T where B is stored as contiguous [N,K]
///
/// This avoids the costly contiguous copy of transposed weight matrices during
/// decode (M=1). Both A rows and B rows are contiguous, making this ideal for
/// SIMD dot products.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), contiguous row-major
/// * `b_nk` - Pointer to B in [N,K] layout (NOT the transposed view)
/// * `out` - Pointer to output C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `ldc` - Leading dimension of output
///
/// # Safety
/// - `a` must be valid for m*k contiguous reads
/// - `b_nk` must be valid for n*k contiguous reads
/// - `out` must be valid for m*ldc writes
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn gemv_bt_kernel<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::detect_simd;
        use super::simd::matmul::gemv_bt;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                let level = detect_simd();
                gemv_bt::gemv_bt_f32(
                    a as *const f32,
                    b_nk as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    ldc,
                    level,
                );
                return;
            }
            DType::F64 => {
                let level = detect_simd();
                gemv_bt::gemv_bt_f64(
                    a as *const f64,
                    b_nk as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    ldc,
                    level,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                gemv_bt_via_f32(a, b_nk, out, m, n, k, ldc);
                return;
            }
            _ => {}
        }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        #[allow(unused_imports)]
        use crate::dtype::DType;
        match T::DTYPE {
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                gemv_bt_via_f32(a, b_nk, out, m, n, k, ldc);
                return;
            }
            _ => {}
        }
    }

    // Narrow floats and integers cannot hold their own dot product.
    if T::DTYPE.is_narrow_float() {
        gemv_bt_scalar_acc::<T, f32>(a, b_nk, out, m, n, k, ldc);
        return;
    }
    if T::DTYPE.is_int() {
        gemv_bt_scalar_acc::<T, i128>(a, b_nk, out, m, n, k, ldc);
        return;
    }

    // Scalar fallback
    gemv_bt_scalar(a, b_nk, out, m, n, k, ldc);
}

/// GEMV-BT with a wide accumulator, for element types that cannot hold the
/// running dot product.
///
/// # Safety
/// Same as [`gemv_bt_kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_scalar_acc<T: Element, A: WideAcc>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);
        for col in 0..n {
            let b_row = b_nk.add(col * k);
            let mut sum = A::ZERO;
            for i in 0..k {
                let prod = A::from_elem(*a_row.add(i)).wide_mul(A::from_elem(*b_row.add(i)));
                sum = sum.wide_add(prod);
            }
            *out_row.add(col) = sum.to_elem::<T>();
        }
    }
}

/// Scalar GEMV-BT fallback
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_scalar<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    for row in 0..m {
        let a_row = a.add(row * k);
        let out_row = out.add(row * ldc);
        for col in 0..n {
            let b_row = b_nk.add(col * k);
            let mut sum = T::zero();
            for i in 0..k {
                sum = sum + *a_row.add(i) * *b_row.add(i);
            }
            *out_row.add(col) = sum;
        }
    }
}

/// GEMV-BT for f16/bf16 via f32 conversion
///
/// Converts A row to f32 (batch SIMD conversion), then converts each B row
/// to f32 in SIMD chunks and uses the f32 AVX2/AVX-512 dot product.
#[cfg(feature = "f16")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn gemv_bt_via_f32<T: Element>(
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
) {
    // Convert A row to f32 once (small buffer, reused per row)
    let mut a_f32 = vec![0.0f32; k];
    let mut b_f32 = vec![0.0f32; k];

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    let level = super::simd::detect_simd();

    for row in 0..m {
        let a_row = a.add(row * k);
        // Batch convert A row to f32
        batch_half_to_f32::<T>(a_row, a_f32.as_mut_ptr(), k);

        let out_row = out.add(row * ldc);

        for col in 0..n {
            let b_row = b_nk.add(col * k);
            // Batch convert B row to f32
            batch_half_to_f32::<T>(b_row, b_f32.as_mut_ptr(), k);

            // Use SIMD f32 dot product
            #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
            {
                let dot = simd_dot_f32(a_f32.as_ptr(), b_f32.as_ptr(), k, level);
                *out_row.add(col) = T::from_f32(dot);
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            {
                let mut sum = 0.0f32;
                for i in 0..k {
                    sum += a_f32[i] * b_f32[i];
                }
                *out_row.add(col) = T::from_f32(sum);
            }
        }
    }
}

/// Batch convert half-precision (f16/bf16) elements to f32 using SIMD when available.
#[cfg(feature = "f16")]
#[inline]
unsafe fn batch_half_to_f32<T: Element>(src: *const T, dst: *mut f32, len: usize) {
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

/// Matrix multiplication with automatic SIMD dispatch: C = A @ B
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 6×16 f32 microkernel, 6×8 f64 microkernel
/// - AVX2+FMA: 6×8 f32 microkernel, 6×4 f64 microkernel
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a` or `b`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_kernel<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Integers accumulate in i128 on every architecture. The AVX2 i32 kernel
    // this replaces accumulated in i32 with `_mm256_add_epi32`, so a dot product
    // whose partial sums left i32's range wrapped and reported a value with the
    // wrong sign even when the final result was representable.
    if T::DTYPE.is_int() {
        matmul_scalar_acc::<T, i128>(a, b, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Dispatch to SIMD for f32/f64 on x86-64, f16/bf16 via f32 conversion
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::matmul;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                matmul::matmul_f32(
                    a as *const f32,
                    b as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            DType::F64 => {
                matmul::matmul_f64(
                    a as *const f64,
                    b as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                matmul::half_convert::matmul_via_f32(a, b, out, m, n, k, lda, ldb, ldc);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // FP8 has no SIMD path on any architecture, and F16/BF16 reach here on
    // architectures without the block above. All of them saturate long before a
    // dot product ends if they accumulate in themselves.
    if T::DTYPE.is_narrow_float() {
        matmul_scalar_acc::<T, f32>(a, b, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Scalar fallback for non-SIMD types or non-x86 platforms
    matmul_scalar(a, b, out, m, n, k, lda, ldb, ldc);
}

/// Matmul with a wide accumulator, for element types that cannot hold the
/// running dot product.
///
/// Keeps the `ikj` loop order of [`matmul_scalar`] for cache locality by
/// holding one output row of accumulators, then narrowing that row once.
///
/// # Safety
/// Same as [`matmul_kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_scalar_acc<T: Element, A: WideAcc>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let mut row_acc = vec![A::ZERO; n];

    for i in 0..m {
        for slot in row_acc.iter_mut() {
            *slot = A::ZERO;
        }

        for kk in 0..k {
            let a_val = A::from_elem(*a.add(i * lda + kk));
            for (j, slot) in row_acc.iter_mut().enumerate() {
                let prod = a_val.wide_mul(A::from_elem(*b.add(kk * ldb + j)));
                *slot = slot.wide_add(prod);
            }
        }

        for (j, slot) in row_acc.iter().enumerate() {
            *out.add(i * ldc + j) = slot.to_elem::<T>();
        }
    }
}

/// Scalar matmul implementation for all Element types
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_scalar<T: Element>(
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Zero output first
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = T::zero();
        }
    }

    // ikj order: better cache locality for B
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let b_val = *b.add(kk * ldb + j);
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * b_val;
            }
        }
    }
}

/// Fused matrix multiplication with bias addition: C = A @ B + bias
///
/// Single-pass implementation that initializes C with bias, then accumulates
/// the matmul result. This is more cache-efficient than separate matmul + bias
/// because it avoids an extra memory round-trip through the output matrix.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), row-major with leading dimension lda
/// * `b` - Pointer to matrix B (k × n), row-major with leading dimension ldb
/// * `bias` - Pointer to bias vector (n elements, broadcast across rows)
/// * `out` - Pointer to output matrix C (m × n), row-major with leading dimension ldc
/// * `m`, `n`, `k` - Matrix dimensions
/// * `lda`, `ldb`, `ldc` - Leading dimensions (row stride in elements)
///
/// # Safety
/// - All pointers must be valid for the specified dimensions and strides
/// - `out` must not alias with `a`, `b`, or `bias`
/// - `bias` must have at least `n` elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_kernel<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Same accumulator-width rule as `matmul_kernel`: the bias is only the
    // starting value of a dot product that still has to be accumulated wide.
    if T::DTYPE.is_int() {
        matmul_bias_scalar_acc::<T, i128>(a, b, bias, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Dispatch to fused SIMD for f32/f64 on x86-64, f16/bf16 via f32 conversion
    #[cfg(target_arch = "x86_64")]
    {
        use super::simd::matmul;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                matmul::matmul_bias_f32(
                    a as *const f32,
                    b as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            DType::F64 => {
                matmul::matmul_bias_f64(
                    a as *const f64,
                    b as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    lda,
                    ldb,
                    ldc,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 | DType::BF16 => {
                matmul::half_convert::matmul_bias_via_f32(a, b, bias, out, m, n, k, lda, ldb, ldc);
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    if T::DTYPE.is_narrow_float() {
        matmul_bias_scalar_acc::<T, f32>(a, b, bias, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Scalar fallback with fused bias
    matmul_bias_scalar(a, b, bias, out, m, n, k, lda, ldb, ldc);
}

/// Fused matmul + bias with a wide accumulator.
///
/// # Safety
/// Same as [`matmul_bias_kernel`].
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_scalar_acc<T: Element, A: WideAcc>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    let mut row_acc = vec![A::ZERO; n];

    for i in 0..m {
        for (j, slot) in row_acc.iter_mut().enumerate() {
            *slot = A::from_elem(*bias.add(j));
        }

        for kk in 0..k {
            let a_val = A::from_elem(*a.add(i * lda + kk));
            for (j, slot) in row_acc.iter_mut().enumerate() {
                let prod = a_val.wide_mul(A::from_elem(*b.add(kk * ldb + j)));
                *slot = slot.wide_add(prod);
            }
        }

        for (j, slot) in row_acc.iter().enumerate() {
            *out.add(i * ldc + j) = slot.to_elem::<T>();
        }
    }
}

/// Scalar matmul with fused bias for all Element types
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn matmul_bias_scalar<T: Element>(
    a: *const T,
    b: *const T,
    bias: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    // Initialize output with bias (single write pass)
    for i in 0..m {
        for j in 0..n {
            *out.add(i * ldc + j) = *bias.add(j);
        }
    }

    // Accumulate matmul result (ikj order for cache locality)
    for i in 0..m {
        for kk in 0..k {
            let a_val = *a.add(i * lda + kk);
            for j in 0..n {
                let b_val = *b.add(kk * ldb + j);
                let out_ptr = out.add(i * ldc + j);
                *out_ptr = *out_ptr + a_val * b_val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_i32_basic() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = [[19, 22], [43, 50]]
        let a = [1i32, 2, 3, 4];
        let b = [5i32, 6, 7, 8];
        let mut c = [0i32; 4];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 2, 2, 2, 2, 2) };
        assert_eq!(c, [19, 22, 43, 50]);
    }

    #[test]
    fn test_matmul_i32_non_square() {
        // A(3x2) @ B(2x4) = C(3x4)
        let a = [1i32, 2, 3, 4, 5, 6];
        let b = [1i32, 2, 3, 4, 5, 6, 7, 8];
        let mut c = [0i32; 12];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 4, 2, 2, 4, 4) };
        assert_eq!(c, [11, 14, 17, 20, 23, 30, 37, 44, 35, 46, 57, 68]);
    }

    #[test]
    fn test_matmul_i32_wide() {
        // n > 8: the width that used to select the AVX2 i32 microkernel.
        let (m, n, k) = (2, 16, 3);
        let a: Vec<i32> = (0..m * k).map(|i| (i + 1) as i32).collect();
        let b: Vec<i32> = (0..k * n).map(|i| (i + 1) as i32).collect();
        let mut c = vec![0i32; m * n];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m, n, k, k, n, n) };

        let mut expected = vec![0i32; m * n];
        for i in 0..m {
            for j in 0..n {
                for kk in 0..k {
                    expected[i * n + j] += a[i * k + kk] * b[kk * n + j];
                }
            }
        }
        assert_eq!(c, expected);
    }

    /// Catches an i32 matmul accumulator.
    ///
    /// Column 0's dot product is 4_000_000_000, which i32 cannot hold. An i32
    /// accumulator panics on the overflow in a debug build, and in a release
    /// build wraps to -294_967_296 where the documented answer is the saturated
    /// `i32::MAX`. Column 1 stays in range and pins that ordinary results are
    /// untouched.
    #[test]
    fn test_matmul_i32_saturates_instead_of_wrapping() {
        let a = [2_000_000_000i32, 2_000_000_000];
        let b = [1i32, 1, 1, -1];
        let mut c = [0i32; 2];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 2, 2, 2, 2, 2) };
        assert_eq!(c, [i32::MAX, 0]);
    }

    /// Catches an FP8 matmul accumulator.
    ///
    /// A length-32 dot product of ones is 32. Accumulated in FP8E4M3 the
    /// running sum stalls at 16, because above 16 the format's spacing is 2 and
    /// `16 + 1` rounds back to 16.
    #[test]
    fn test_matmul_fp8_accumulates_in_f32() {
        use crate::dtype::FP8E4M3;

        let a = [FP8E4M3::from_f32(1.0); 32];
        let b = [FP8E4M3::from_f32(1.0); 32];
        let mut c = [FP8E4M3::from_f32(0.0); 1];

        unsafe { matmul_kernel(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 1, 32, 32, 1, 1) };
        assert_eq!(c[0].to_f32(), 32.0);
    }

    /// Same accumulator defect through the fused bias kernel.
    ///
    /// The bias is only the starting value of a dot product that still has to
    /// be accumulated wide, so an i32 accumulator fails here for exactly the
    /// reason it fails in `matmul_kernel`.
    #[test]
    fn test_matmul_bias_i32_saturates_instead_of_wrapping() {
        let a = [2_000_000_000i32, 2_000_000_000];
        let b = [1i32, 1, 1, -1];
        let bias = [7i32, 7];
        let mut c = [0i32; 2];

        unsafe {
            matmul_bias_kernel(
                a.as_ptr(),
                b.as_ptr(),
                bias.as_ptr(),
                c.as_mut_ptr(),
                1,
                2,
                2,
                2,
                2,
                2,
            )
        };
        assert_eq!(c, [i32::MAX, 7]);
    }

    /// Same accumulator defect through the GEMV-BT decode fast path, which has
    /// its own dot-product loop and its own accumulator.
    #[test]
    fn test_gemv_bt_i32_saturates_instead_of_wrapping() {
        // B is stored as [N, K] = [[1, 1], [1, -1]].
        let a = [2_000_000_000i32, 2_000_000_000];
        let b_nk = [1i32, 1, 1, -1];
        let mut c = [0i32; 2];

        unsafe { gemv_bt_kernel(a.as_ptr(), b_nk.as_ptr(), c.as_mut_ptr(), 1, 2, 2, 2) };
        assert_eq!(c, [i32::MAX, 0]);
    }

    /// Catches an FP8 accumulator in the GEMV-BT dot product.
    #[test]
    fn test_gemv_bt_fp8_accumulates_in_f32() {
        use crate::dtype::FP8E4M3;

        let a = [FP8E4M3::from_f32(1.0); 32];
        let b_nk = [FP8E4M3::from_f32(1.0); 32];
        let mut c = [FP8E4M3::from_f32(0.0); 1];

        unsafe { gemv_bt_kernel(a.as_ptr(), b_nk.as_ptr(), c.as_mut_ptr(), 1, 1, 32, 1) };
        assert_eq!(c[0].to_f32(), 32.0);
    }
}
