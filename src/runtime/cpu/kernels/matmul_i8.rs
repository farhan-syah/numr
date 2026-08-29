//! i8 × i8 → i32 matrix multiplication kernel
//!
//! Entry point for i8 matmul that dispatches to SIMD dot-product-based implementation.

/// i8 × i8 → i32 matmul: C[m×n] = A[m×k] @ B[k×n]
///
/// Input matrices are i8, output is i32 (standard quantized matmul accumulation).
///
/// # Safety
/// - `a` must point to m×lda i8 elements
/// - `b` must point to k×ldb i8 elements
/// - `out` must point to m×ldc i32 elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_to_i32_kernel(
    a: *const i8,
    b: *const i8,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    super::simd::matmul::int8::matmul_i8_to_i32(a, b, out, m, n, k, lda, ldb, ldc);
}

/// i8 × i8 → i32 matmul with a fused i32 bias: C[m×n] = A[m×k] @ B[k×n] + bias
///
/// The bias is i32 because it seeds the i32 accumulator - see
/// `ops/matmul_dtype.rs` for the dtype rule both backends apply.
///
/// # Safety
/// - `a` must point to m×lda i8 elements
/// - `b` must point to k×ldb i8 elements
/// - `bias` must point to n i32 elements
/// - `out` must point to m×ldc i32 elements
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_i8_to_i32_bias_kernel(
    a: *const i8,
    b: *const i8,
    bias: *const i32,
    out: *mut i32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    super::simd::matmul::int8::matmul_i8_to_i32_bias(a, b, bias, out, m, n, k, lda, ldb, ldc);
}
