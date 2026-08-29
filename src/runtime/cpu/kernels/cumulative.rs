//! Cumulative operation kernels (cumsum, cumprod, logsumexp)

use super::wide_acc::WideAcc;
use crate::dtype::Element;

/// Cumulative sum along a contiguous dimension
///
/// # Arguments
/// * `a` - Input pointer (scan_size * outer_size elements, contiguous)
/// * `out` - Output pointer (scan_size * outer_size elements)
/// * `scan_size` - Number of elements to scan over per segment
/// * `outer_size` - Number of independent scans
///
/// # Safety
/// - `a` must point to `scan_size * outer_size` elements
/// - `out` must point to `scan_size * outer_size` elements
#[inline]
pub unsafe fn cumsum_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
) {
    // The running total outgrows the element type long before a scan of any
    // length ends, for every float narrower than F32 and for every integer.
    // Those accumulate wide and narrow once per element written; F32, F64, and
    // the complex types already accumulate in a type as wide as their output
    // and keep the direct path.
    if T::DTYPE.is_narrow_float() {
        cumsum_kernel_acc::<T, f32>(a, out, scan_size, outer_size);
        return;
    }
    if T::DTYPE.is_int() {
        cumsum_kernel_acc::<T, i128>(a, out, scan_size, outer_size);
        return;
    }

    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = T::zero();
        for i in 0..scan_size {
            acc = acc + *a.add(base + i);
            *out.add(base + i) = acc;
        }
    }
}

/// Cumulative sum over a contiguous dimension with a wide accumulator.
///
/// # Safety
/// Same as [`cumsum_kernel`].
#[inline]
unsafe fn cumsum_kernel_acc<T: Element, A: WideAcc>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = A::ZERO;
        for i in 0..scan_size {
            acc = acc.wide_add(A::from_elem(*a.add(base + i)));
            *out.add(base + i) = acc.to_elem::<T>();
        }
    }
}

/// Cumulative sum along a strided dimension
///
/// # Arguments
/// * `a` - Input pointer
/// * `out` - Output pointer
/// * `scan_size` - Number of elements to scan over per segment
/// * `outer_size` - Number of independent scans
/// * `inner_size` - Stride between consecutive elements in scan dimension
///
/// # Safety
/// - Pointers must be valid for the given strides and sizes
#[inline]
pub unsafe fn cumsum_strided_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    // Use SIMD for f32/f64 on x86_64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::cumulative;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                cumulative::cumsum_strided_f32(
                    a as *const f32,
                    out as *mut f32,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            DType::F64 => {
                cumulative::cumsum_strided_f64(
                    a as *const f64,
                    out as *mut f64,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                cumulative::cumsum_strided_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                cumulative::cumsum_strided_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback. FP8 never reaches the SIMD block above, and neither do
    // the integer dtypes, so both need the wide accumulator here for the same
    // reason `cumsum_kernel` does.
    if T::DTYPE.is_narrow_float() {
        cumsum_strided_kernel_acc::<T, f32>(a, out, scan_size, outer_size, inner_size);
        return;
    }
    if T::DTYPE.is_int() {
        cumsum_strided_kernel_acc::<T, i128>(a, out, scan_size, outer_size, inner_size);
        return;
    }

    // For strided access: element [o, s, i] is at offset o * scan_size * inner_size + s * inner_size + i
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = T::zero();
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc = acc + *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// Cumulative sum over a strided dimension with a wide accumulator.
///
/// # Safety
/// Same as [`cumsum_strided_kernel`].
#[inline]
unsafe fn cumsum_strided_kernel_acc<T: Element, A: WideAcc>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = A::ZERO;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc = acc.wide_add(A::from_elem(*a.add(idx)));
                *out.add(idx) = acc.to_elem::<T>();
            }
        }
    }
}

/// Cumulative product along a contiguous dimension
///
/// # Arguments
/// * `a` - Input pointer (scan_size * outer_size elements, contiguous)
/// * `out` - Output pointer (scan_size * outer_size elements)
/// * `scan_size` - Number of elements to scan over per segment
/// * `outer_size` - Number of independent scans
///
/// # Safety
/// - `a` must point to `scan_size * outer_size` elements
/// - `out` must point to `scan_size * outer_size` elements
#[inline]
pub unsafe fn cumprod_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
) {
    // An integer running product leaves the element type's range far faster
    // than a running sum does: it wraps in release and panics in debug. It
    // accumulates in i128 and clamps once per element written, the same
    // convention `cumsum_kernel` follows.
    if T::DTYPE.is_int() {
        cumprod_kernel_acc::<T, i128>(a, out, scan_size, outer_size);
        return;
    }

    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = T::one();
        for i in 0..scan_size {
            acc = acc * *a.add(base + i);
            *out.add(base + i) = acc;
        }
    }
}

/// Cumulative product over a contiguous dimension with a wide accumulator.
///
/// The accumulator saturates, so the output is the true product clamped to the
/// element type's range. i128 is exact for any product that stays inside it,
/// and once it saturates a later factor still moves the sign correctly:
/// `i128::MAX * -1` saturates to `i128::MIN`, which narrows to `T::MIN`.
///
/// # Safety
/// Same as [`cumprod_kernel`].
#[inline]
unsafe fn cumprod_kernel_acc<T: Element, A: WideAcc>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = A::ONE;
        for i in 0..scan_size {
            acc = acc.wide_mul(A::from_elem(*a.add(base + i)));
            *out.add(base + i) = acc.to_elem::<T>();
        }
    }
}

/// Cumulative product along a strided dimension
///
/// # Arguments
/// * `a` - Input pointer
/// * `out` - Output pointer
/// * `scan_size` - Number of elements to scan over per segment
/// * `outer_size` - Number of independent scans
/// * `inner_size` - Stride between consecutive elements in scan dimension
///
/// # Safety
/// - Pointers must be valid for the given strides and sizes
#[inline]
pub unsafe fn cumprod_strided_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    // Use SIMD for f32/f64 on x86_64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::cumulative;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                cumulative::cumprod_strided_f32(
                    a as *const f32,
                    out as *mut f32,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            DType::F64 => {
                cumulative::cumprod_strided_f64(
                    a as *const f64,
                    out as *mut f64,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                cumulative::cumprod_strided_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                cumulative::cumprod_strided_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    scan_size,
                    outer_size,
                    inner_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback. Integers never reach the SIMD block above, so they need
    // the wide accumulator here for the same reason `cumprod_kernel` does.
    if T::DTYPE.is_int() {
        cumprod_strided_kernel_acc::<T, i128>(a, out, scan_size, outer_size, inner_size);
        return;
    }

    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = T::one();
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc = acc * *a.add(idx);
                *out.add(idx) = acc;
            }
        }
    }
}

/// Cumulative product over a strided dimension with a wide accumulator.
///
/// # Safety
/// Same as [`cumprod_strided_kernel`].
#[inline]
unsafe fn cumprod_strided_kernel_acc<T: Element, A: WideAcc>(
    a: *const T,
    out: *mut T,
    scan_size: usize,
    outer_size: usize,
    inner_size: usize,
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let mut acc = A::ONE;
            for s in 0..scan_size {
                let idx = o * scan_size * inner_size + s * inner_size + i;
                acc = acc.wide_mul(A::from_elem(*a.add(idx)));
                *out.add(idx) = acc.to_elem::<T>();
            }
        }
    }
}

/// Log-sum-exp along a contiguous dimension (numerically stable)
///
/// Computes log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
///
/// On x86-64, dispatches to optimized SIMD implementations for f32/f64:
/// - AVX-512: 16 f32s or 8 f64s per iteration with vectorized exp
/// - AVX2: 8 f32s or 4 f64s per iteration with vectorized exp
/// - Scalar fallback for other types or non-x86 platforms
///
/// # Arguments
/// * `a` - Input pointer (reduce_size * outer_size elements, contiguous)
/// * `out` - Output pointer (outer_size elements)
/// * `reduce_size` - Number of elements to reduce per segment
/// * `outer_size` - Number of independent reductions
///
/// # Safety
/// - `a` must point to `reduce_size * outer_size` elements
/// - `out` must point to `outer_size` elements
#[inline]
pub unsafe fn logsumexp_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    // Dispatch to SIMD for f32/f64 on x86-64 and aarch64
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::logsumexp;
        use crate::dtype::DType;

        match T::DTYPE {
            DType::F32 => {
                logsumexp::logsumexp_f32(a as *const f32, out as *mut f32, reduce_size, outer_size);
                return;
            }
            DType::F64 => {
                logsumexp::logsumexp_f64(a as *const f64, out as *mut f64, reduce_size, outer_size);
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                logsumexp::logsumexp_f16(
                    a as *const half::f16,
                    out as *mut half::f16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                logsumexp::logsumexp_bf16(
                    a as *const half::bf16,
                    out as *mut half::bf16,
                    reduce_size,
                    outer_size,
                );
                return;
            }
            _ => {} // Fall through to scalar
        }
    }

    // Scalar fallback
    logsumexp_kernel_scalar(a, out, reduce_size, outer_size);
}

/// Scalar logsumexp for all Element types
#[inline]
unsafe fn logsumexp_kernel_scalar<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
) {
    for o in 0..outer_size {
        let base = o * reduce_size;

        // Step 1: Find max
        let mut max_val = *a.add(base);
        for i in 1..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: Compute sum(exp(x - max))
        //
        // A float narrower than F32 must not hold this accumulator. Every term
        // is at most 1.0, so the running sum saturates quickly and stalls, and
        // `log(sum)` then comes back short — which makes every log_softmax
        // built on it wrong. Widen those to f64 (what `logsumexp_strided_kernel`
        // already does) and narrow only the final result. Every other dtype
        // keeps its own accumulator and its existing result.
        if T::DTYPE.is_narrow_float() {
            let mut sum = 0.0f64;
            for i in 0..reduce_size {
                let val = (*a.add(base + i)).to_f64();
                sum += (val - max_val.to_f64()).exp();
            }
            *out.add(o) = T::from_f64(max_val.to_f64() + sum.ln());
            continue;
        }

        let mut sum = T::zero();
        for i in 0..reduce_size {
            let val = *a.add(base + i);
            // Compute exp(val - max_val) using f64 for precision
            let exp_val = T::from_f64((val.to_f64() - max_val.to_f64()).exp());
            sum = sum + exp_val;
        }

        // Step 3: Result = max + log(sum)
        *out.add(o) = T::from_f64(max_val.to_f64() + sum.to_f64().ln());
    }
}

/// Log-sum-exp along a strided dimension (numerically stable)
///
/// # Safety
/// - Pointers must be valid for the given strides and sizes
#[inline]
pub unsafe fn logsumexp_strided_kernel<T: Element>(
    a: *const T,
    out: *mut T,
    reduce_size: usize,
    outer_size: usize,
    inner_size: usize,
    _in_stride: usize, // stride along the reduce dimension in input (unused, kept for API parity)
    out_stride: usize, // stride in output
) {
    for o in 0..outer_size {
        for i in 0..inner_size {
            let out_idx = o * out_stride + i;

            // Step 1: Find max along reduce dimension
            let first_idx = o * reduce_size * inner_size + i;
            let mut max_val = *a.add(first_idx);
            for r in 1..reduce_size {
                let idx = o * reduce_size * inner_size + r * inner_size + i;
                let val = *a.add(idx);
                if val > max_val {
                    max_val = val;
                }
            }

            // Step 2: Compute sum(exp(x - max))
            let mut sum = 0.0f64;
            for r in 0..reduce_size {
                let idx = o * reduce_size * inner_size + r * inner_size + i;
                let val = (*a.add(idx)).to_f64();
                sum += (val - max_val.to_f64()).exp();
            }

            // Step 3: Result = max + log(sum)
            *out.add(out_idx) = T::from_f64(max_val.to_f64() + sum.ln());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cumsum_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];

        unsafe {
            cumsum_kernel(a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out, [1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_cumsum_multiple_segments() {
        // Two segments of 3 elements each
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 6];

        unsafe {
            cumsum_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [1.0, 3.0, 6.0, 10.0, 30.0, 60.0]);
    }

    #[test]
    fn test_cumprod_basic() {
        let a = [1.0f32, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 4];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out, [1.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_cumprod_multiple_segments() {
        let a = [1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0];
        let mut out = [0.0f32; 6];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        assert_eq!(out, [1.0, 2.0, 6.0, 2.0, 6.0, 24.0]);
    }

    #[test]
    fn test_logsumexp_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let mut out = [0.0f32; 1];

        unsafe {
            logsumexp_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        // log(exp(1) + exp(2) + exp(3)) ≈ 3.4076
        let expected = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        assert!((out[0] as f64 - expected).abs() < 1e-5);
    }

    #[test]
    fn test_logsumexp_multiple_segments() {
        let a = [1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0];
        let mut out = [0.0f32; 2];

        unsafe {
            logsumexp_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 2);
        }

        let expected0 = (1.0f64.exp() + 2.0f64.exp() + 3.0f64.exp()).ln();
        let expected1 = (10.0f64.exp() + 20.0f64.exp() + 30.0f64.exp()).ln();
        assert!((out[0] as f64 - expected0).abs() < 1e-5);
        assert!((out[1] as f64 - expected1).abs() < 1e-5);
    }

    #[test]
    fn test_logsumexp_numerical_stability() {
        // Test with large values that would overflow naive exp
        let a = [1000.0f32, 1000.0, 1000.0];
        let mut out = [0.0f32; 1];

        unsafe {
            logsumexp_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        // Should be log(3) + 1000 ≈ 1001.0986
        let expected = 1000.0 + (3.0f64).ln();
        assert!((out[0] as f64 - expected).abs() < 1e-3);
    }

    /// Catches a `cumsum` accumulator held in the element type for FP8.
    ///
    /// FP8E4M3 has three mantissa bits, so above 16 its spacing is 2 and
    /// `16 + 1` rounds back to 16. An FP8 accumulator therefore stalls at 16
    /// and every later output reads 16 instead of the true partial sum.
    #[test]
    fn test_cumsum_fp8_accumulates_wider_than_the_element_type() {
        use crate::dtype::FP8E4M3;

        let a = [FP8E4M3::from_f32(1.0); 32];
        let mut out = [FP8E4M3::from_f32(0.0); 32];

        unsafe {
            cumsum_kernel(a.as_ptr(), out.as_mut_ptr(), 32, 1);
        }

        // 24 and 32 are both exactly representable in FP8E4M3.
        assert_eq!(out[23].to_f32(), 24.0);
        assert_eq!(out[31].to_f32(), 32.0);
    }

    /// Catches a `cumsum` accumulator held in the element type for F16.
    ///
    /// F16 has ten mantissa bits, so above 2048 its spacing is 2 and
    /// `2048 + 1` rounds back to 2048. An F16 accumulator stalls there.
    #[cfg(feature = "f16")]
    #[test]
    fn test_cumsum_f16_accumulates_wider_than_the_element_type() {
        let a = vec![half::f16::from_f32(1.0); 3000];
        let mut out = vec![half::f16::from_f32(0.0); 3000];

        unsafe {
            cumsum_kernel(a.as_ptr(), out.as_mut_ptr(), 3000, 1);
        }

        // 2500 and 3000 are even, so both are exact in F16.
        assert_eq!(out[2499].to_f32(), 2500.0);
        assert_eq!(out[2999].to_f32(), 3000.0);
    }

    /// Catches an i32 `cumsum` accumulator.
    ///
    /// The running total leaves i32's range at element 1 and comes back at
    /// element 2. An i32 accumulator panics on that overflow in a debug build,
    /// and in a release build stores the wrapped -294_967_296 where the
    /// documented answer is the saturated `i32::MAX`.
    #[test]
    fn test_cumsum_i32_accumulates_in_a_wider_integer() {
        let a = [2_000_000_000i32, 2_000_000_000, -2_000_000_000];
        let mut out = [0i32; 3];

        unsafe {
            cumsum_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out, [2_000_000_000, i32::MAX, 2_000_000_000]);
    }

    /// Same accumulator defect on the strided path, which the SIMD block above
    /// never covers for integers.
    ///
    /// Layout is `[scan][inner]` with `inner_size = 2`: column 0 overflows i32,
    /// column 1 stays small and pins that the fix did not disturb it.
    #[test]
    fn test_cumsum_strided_i32_accumulates_in_a_wider_integer() {
        let a = [2_000_000_000i32, 1, 2_000_000_000, 2, -2_000_000_000, 3];
        let mut out = [0i32; 6];

        unsafe {
            cumsum_strided_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1, 2);
        }

        assert_eq!(out, [2_000_000_000, 1, i32::MAX, 3, 2_000_000_000, 6]);
    }

    /// Catches an FP8 accumulator on the strided path, which has no SIMD
    /// dispatch for FP8 on any architecture.
    #[test]
    fn test_cumsum_strided_fp8_accumulates_wider_than_the_element_type() {
        use crate::dtype::FP8E4M3;

        // 32 scan steps over 2 interleaved columns of 1.0.
        let a = [FP8E4M3::from_f32(1.0); 64];
        let mut out = [FP8E4M3::from_f32(0.0); 64];

        unsafe {
            cumsum_strided_kernel(a.as_ptr(), out.as_mut_ptr(), 32, 1, 2);
        }

        // Last scan step of each column is 32; an FP8 accumulator reads 16.
        assert_eq!(out[62].to_f32(), 32.0);
        assert_eq!(out[63].to_f32(), 32.0);
    }

    /// The reference case a per-step saturating multiply gets wrong.
    ///
    /// True products are 100_000, 10^10, -10^10, so the clamped answers are
    /// `i32::MAX` then `i32::MIN`. A saturating multiply in i32 clamps to
    /// `i32::MAX` first and then reports `-i32::MAX`, one off and for the wrong
    /// reason.
    #[test]
    fn test_cumprod_i32_saturates_to_the_true_product_sign() {
        let a = [100_000i32, 100_000, -1];
        let mut out = [0i32; 3];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out, [100_000, i32::MAX, i32::MIN]);
    }

    /// A zero factor after saturation still gives 0, because the true product
    /// is 0 from that element on.
    #[test]
    fn test_cumprod_i32_zero_after_saturation_is_zero() {
        let a = [100_000i32, 100_000, 0, 7];
        let mut out = [0i32; 4];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out, [100_000, i32::MAX, 0, 0]);
    }

    /// Each further negative factor flips the sign of a saturated product.
    #[test]
    fn test_cumprod_i32_sign_flips_across_a_saturated_run() {
        let a = [-100_000i32, 100_000, -1, -1];
        let mut out = [0i32; 4];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out, [-100_000, i32::MIN, i32::MAX, i32::MIN]);
    }

    /// U32 has no sign to track, so overflow pins at `u32::MAX` and stays.
    #[test]
    fn test_cumprod_u32_saturates_to_max() {
        let a = [100_000u32, 100_000, 2];
        let mut out = [0u32; 3];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out, [100_000, u32::MAX, u32::MAX]);
    }

    /// A product that never leaves i32 is untouched by the wide accumulator.
    #[test]
    fn test_cumprod_i32_without_overflow_is_exact() {
        let a = [2i32, 3, -4, 5];
        let mut out = [0i32; 4];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 4, 1);
        }

        assert_eq!(out, [2, 6, -24, -120]);
    }

    /// I64 saturates on the same rule; i128 holds every product of two i64s.
    #[test]
    fn test_cumprod_i64_saturates_to_the_true_product_sign() {
        let a = [4_000_000_000i64, 4_000_000_000, -1];
        let mut out = [0i64; 3];

        unsafe {
            cumprod_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1);
        }

        assert_eq!(out, [4_000_000_000, i64::MAX, i64::MIN]);
    }

    /// The strided path has no SIMD dispatch for integers, so it needs the same
    /// accumulator.
    ///
    /// Layout is `[scan][inner]` with `inner_size = 2`: column 0 overflows i32
    /// and then flips sign, column 1 stays small.
    #[test]
    fn test_cumprod_strided_i32_saturates_to_the_true_product_sign() {
        let a = [100_000i32, 2, 100_000, 3, -1, 4];
        let mut out = [0i32; 6];

        unsafe {
            cumprod_strided_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1, 2);
        }

        assert_eq!(out, [100_000, 2, i32::MAX, 6, i32::MIN, 24]);
    }

    /// Strided U32 overflow, which shares no code with the contiguous path.
    #[test]
    fn test_cumprod_strided_u32_saturates_to_max() {
        let a = [100_000u32, 2, 100_000, 3, 2, 4];
        let mut out = [0u32; 6];

        unsafe {
            cumprod_strided_kernel(a.as_ptr(), out.as_mut_ptr(), 3, 1, 2);
        }

        assert_eq!(out, [100_000, 2, u32::MAX, 6, u32::MAX, 24]);
    }
}
