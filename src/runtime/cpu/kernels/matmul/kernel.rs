//! The general matmul kernels: `matmul_kernel` and `matmul_bias_kernel`.

use super::super::wide_acc::WideAcc;
use crate::dtype::Element;

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
    // Integers accumulate exactly, never in the element type. An older AVX2 i32
    // kernel used `_mm256_add_epi32` and wrapped mid-dot-product, reporting the
    // wrong sign even when the final result was representable.
    //
    // i32 on AVX2 gets a 64-bit accumulator when a magnitude prescan proves
    // every partial sum fits; the scan is O(mk + kn) against the matmul's
    // O(mnk). Everything else — every other integer dtype, non-AVX2 hardware,
    // and operands the scan rejects — takes the exact i128 scalar path. Both
    // produce the same clamped result, so which one ran is invisible.
    if T::DTYPE.is_int() {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::dtype::DType;
            if T::DTYPE == DType::I32 && std::arch::is_x86_feature_detected!("avx2") {
                let (ai, bi) = (a as *const i32, b as *const i32);
                if super::super::simd::matmul::int32::matmul_i32_fits_i64(ai, bi, m, n, k, lda, ldb)
                {
                    super::super::simd::matmul::int32::matmul_i32_avx2(
                        ai,
                        bi,
                        out as *mut i32,
                        m,
                        n,
                        k,
                        lda,
                        ldb,
                        ldc,
                    );
                    return;
                }
            }
        }
        matmul_scalar_acc::<T, i128>(a, b, out, m, n, k, lda, ldb, ldc);
        return;
    }

    // Dispatch to SIMD for f32/f64, f16/bf16 via f32 conversion.
    //
    // aarch64 MUST be here, not just x86_64. `matmul_bt_kernel` already gates on
    // both, and `matmul_bt_matches_contiguous` promises the two agree bit for
    // bit wherever the tiled path runs. Leaving ARM out sent contiguous down the
    // scalar path while transposed ran the NEON tiled kernel: same maths, a
    // different summation order, and a 1-ULP disagreement that broke that
    // promise. `matmul_f32` already dispatches per architecture internally.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::matmul;
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

    // Dispatch to fused SIMD for f32/f64, f16/bf16 via f32 conversion.
    // Gated on both architectures for the reason in `matmul_kernel`.
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::super::simd::matmul;
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
