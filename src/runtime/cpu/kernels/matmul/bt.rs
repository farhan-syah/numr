//! Matmul against a transposed B operand: `matmul_bt_kernel`.
//!
//! Split out of the general matmul kernel: B arrives as the contiguous `[N, K]`
//! buffer a transposed weight matrix already owns, so the index arithmetic and
//! the dispatch differ from [`super::matmul_kernel`] even though the
//! result is the same.

use crate::dtype::{DType, Element};

/// Does [`matmul_bt_kernel`] agree bit for bit with [`super::matmul_kernel`] on
/// a materialized B, for this dtype and shape?
///
/// True only where both run the tiled kernel over identical packed panels: f32
/// and f64, at a shape the tiled path takes. Every other case reaches the
/// transposed layout through a different kernel — half floats convert B
/// wholesale, integers and narrow floats dot-product with a wide accumulator —
/// which agrees within tolerance but accumulates in its own order. A caller
/// that must not change results materializes B for those.
pub fn matmul_bt_matches_contiguous(dtype: DType, m: usize, n: usize, k: usize) -> bool {
    matches!(dtype, DType::F32 | DType::F64)
        && super::super::simd::matmul::matmul_bt_is_tiled(m, n, k)
}

/// Matrix multiplication with a transposed B operand: C = A @ B
///
/// `b_nk` holds the logical `[K, N]` operand as a contiguous `[N, K]` buffer,
/// which is what a `[K, N]` view with strides `[1, K]` points at. The matrix is
/// never materialized: the tiled kernel packs its B panels directly out of that
/// buffer, producing panels byte-identical to the ones a materialized operand
/// would give. Shapes and dtypes the tiled path does not cover fall back to the
/// dot-product kernel, which agrees within tolerance but not bit for bit — see
/// [`matmul_bt_matches_contiguous`].
///
/// # Why this exists
///
/// Every `Linear::forward` multiplies against a transposed weight view. Making
/// that view contiguous copied the whole weight matrix per call: a profiled
/// VoxCPM2 decode moved ~50 GB through `copy_strided` over four generated
/// patches, and 41% of all program instructions landed under
/// `Tensor::contiguous` on that path.
///
/// # Arguments
/// * `a` - Pointer to matrix A (m × k), contiguous row-major (row stride `k`)
/// * `b_nk` - Pointer to B in `[N, K]` layout, contiguous (row stride `k`)
/// * `out` - Pointer to output C (m × n), row-major with leading dimension `ldc`
/// * `m`, `n`, `k` - Matrix dimensions of the logical product
/// * `ldc` - Leading dimension of the output
///
/// # Safety
/// - `a` must be valid for `m * k` contiguous reads
/// - `b_nk` must be valid for `n * k` contiguous reads
/// - `out` must be valid for `m * ldc` writes
/// - `out` must not alias `a` or `b_nk`
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bt_kernel<T: Element>(
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
        use super::super::simd::matmul;

        match T::DTYPE {
            DType::F32 => {
                matmul::matmul_bt_f32(
                    a as *const f32,
                    b_nk as *const f32,
                    out as *mut f32,
                    m,
                    n,
                    k,
                    ldc,
                );
                return;
            }
            DType::F64 => {
                matmul::matmul_bt_f64(
                    a as *const f64,
                    b_nk as *const f64,
                    out as *mut f64,
                    m,
                    n,
                    k,
                    ldc,
                );
                return;
            }
            _ => {}
        }
    }

    // Every remaining dtype — half floats, FP8, integers — reaches the
    // dot-product kernel, which carries their wide-accumulator and conversion
    // paths and reads the same `[N, K]` layout.
    super::gemv_bt_kernel::<T>(a, b_nk, out, m, n, k, ldc);
}
