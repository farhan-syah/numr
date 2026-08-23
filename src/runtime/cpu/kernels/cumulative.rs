//! Cumulative operation kernels (cumsum, cumprod, logsumexp)

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
    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = T::zero();
        for i in 0..scan_size {
            acc = acc + *a.add(base + i);
            *out.add(base + i) = acc;
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

    // Scalar fallback
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
    for o in 0..outer_size {
        let base = o * scan_size;
        let mut acc = T::one();
        for i in 0..scan_size {
            acc = acc * *a.add(base + i);
            *out.add(base + i) = acc;
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

    // Scalar fallback
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
}
