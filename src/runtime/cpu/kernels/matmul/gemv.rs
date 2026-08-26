//! GEMV with a transposed B operand.
//!
//! Split out of the matmul kernel module: the matrix-vector case has its own
//! dispatch, wide-accumulator, and half-precision paths, sharing nothing with
//! the general matmul beyond the accumulator convention.

use super::super::wide_acc::WideAcc;
#[cfg(feature = "f16")]
use super::dot::simd_dot_f32;
#[cfg(feature = "f16")]
use super::half_batch::batch_half_to_f32;
use crate::dtype::Element;

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
        use super::super::simd::detect_simd;
        use super::super::simd::matmul::gemv_bt;
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
    let level = super::super::simd::detect_simd();

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
