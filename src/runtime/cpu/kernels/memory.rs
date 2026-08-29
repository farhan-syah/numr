//! Memory operation kernels (fill, copy, cast, random)

use super::rng;
use crate::dtype::Element;

/// Fill buffer with a constant value
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn fill_kernel<T: Element>(out: *mut T, value: T, len: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, len);
    out_slice.fill(value);
}

/// Copy elements from src to dst
///
/// # Safety
/// - `src` and `dst` must be valid pointers to `len` elements
/// - `dst` must not overlap with `src`
#[inline]
pub unsafe fn copy_kernel<T: Element>(src: *const T, dst: *mut T, len: usize) {
    std::ptr::copy_nonoverlapping(src, dst, len);
}

/// Cast tensor data from one dtype to another.
///
/// Converts elements by going through f64 as an intermediate representation,
/// which works for all numeric types via the Element trait.
///
/// # Safety
/// - `src` must be valid pointer to `len` elements of `src_dtype`
/// - `dst` must be valid pointer to `len` elements of `dst_dtype`
/// - `src` and `dst` must not overlap
#[inline]
pub unsafe fn cast_kernel(
    src: *const u8,
    dst: *mut u8,
    len: usize,
    src_dtype: crate::dtype::DType,
    dst_dtype: crate::dtype::DType,
) -> crate::error::Result<()> {
    use crate::dtype::DType;
    use crate::error::Error;

    // Helper macro to cast from a known source type to any destination type
    macro_rules! cast_from {
        ($src_ty:ty, $src_ptr:expr, $dst_ptr:expr, $len:expr, $dst_dtype:expr) => {{
            let src_slice = std::slice::from_raw_parts($src_ptr as *const $src_ty, $len);
            match $dst_dtype {
                DType::F64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64();
                    }
                }
                DType::F32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut f32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as f32;
                    }
                }
                DType::F16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::f16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::f16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::F16,
                            op: "cast",
                        });
                    }
                }
                DType::BF16 => {
                    #[cfg(feature = "f16")]
                    {
                        let dst_slice =
                            std::slice::from_raw_parts_mut($dst_ptr as *mut half::bf16, $len);
                        for i in 0..$len {
                            dst_slice[i] = half::bf16::from_f64(src_slice[i].to_f64());
                        }
                    }
                    #[cfg(not(feature = "f16"))]
                    {
                        return Err(Error::UnsupportedDType {
                            dtype: DType::BF16,
                            op: "cast",
                        });
                    }
                }
                DType::FP8E4M3 => {
                    let dst_slice = std::slice::from_raw_parts_mut(
                        $dst_ptr as *mut crate::dtype::FP8E4M3,
                        $len,
                    );
                    for i in 0..$len {
                        dst_slice[i] =
                            crate::dtype::FP8E4M3::from_f32(src_slice[i].to_f64() as f32);
                    }
                }
                DType::FP8E5M2 => {
                    let dst_slice = std::slice::from_raw_parts_mut(
                        $dst_ptr as *mut crate::dtype::FP8E5M2,
                        $len,
                    );
                    for i in 0..$len {
                        dst_slice[i] =
                            crate::dtype::FP8E5M2::from_f32(src_slice[i].to_f64() as f32);
                    }
                }
                DType::I64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i64;
                    }
                }
                DType::I32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i32;
                    }
                }
                DType::I16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i16;
                    }
                }
                DType::I8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut i8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as i8;
                    }
                }
                DType::U64 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u64, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u64;
                    }
                }
                DType::U32 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u32, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u32;
                    }
                }
                DType::U16 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u16, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u16;
                    }
                }
                DType::U8 => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = src_slice[i].to_f64() as u8;
                    }
                }
                DType::Bool => {
                    let dst_slice = std::slice::from_raw_parts_mut($dst_ptr as *mut u8, $len);
                    for i in 0..$len {
                        dst_slice[i] = if src_slice[i].to_f64() != 0.0 { 1 } else { 0 };
                    }
                }
                DType::Complex64 => {
                    let dst_slice = std::slice::from_raw_parts_mut(
                        $dst_ptr as *mut crate::dtype::Complex64,
                        $len,
                    );
                    for i in 0..$len {
                        dst_slice[i] =
                            crate::dtype::Complex64::new(src_slice[i].to_f64() as f32, 0.0);
                    }
                }
                DType::Complex128 => {
                    let dst_slice = std::slice::from_raw_parts_mut(
                        $dst_ptr as *mut crate::dtype::Complex128,
                        $len,
                    );
                    for i in 0..$len {
                        dst_slice[i] = crate::dtype::Complex128::new(src_slice[i].to_f64(), 0.0);
                    }
                }
            }
        }};
    }

    // Dispatch based on source dtype
    match src_dtype {
        DType::F64 => cast_from!(f64, src, dst, len, dst_dtype),
        DType::F32 => cast_from!(f32, src, dst, len, dst_dtype),
        DType::F16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::f16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::F16,
                    op: "cast",
                });
            }
        }
        DType::BF16 => {
            #[cfg(feature = "f16")]
            {
                cast_from!(half::bf16, src, dst, len, dst_dtype)
            }
            #[cfg(not(feature = "f16"))]
            {
                return Err(Error::UnsupportedDType {
                    dtype: DType::BF16,
                    op: "cast",
                });
            }
        }
        DType::FP8E4M3 => {
            cast_from!(crate::dtype::FP8E4M3, src, dst, len, dst_dtype)
        }
        DType::FP8E5M2 => {
            cast_from!(crate::dtype::FP8E5M2, src, dst, len, dst_dtype)
        }
        DType::I64 => cast_from!(i64, src, dst, len, dst_dtype),
        DType::I32 => cast_from!(i32, src, dst, len, dst_dtype),
        DType::I16 => cast_from!(i16, src, dst, len, dst_dtype),
        DType::I8 => cast_from!(i8, src, dst, len, dst_dtype),
        DType::U64 => cast_from!(u64, src, dst, len, dst_dtype),
        DType::U32 => cast_from!(u32, src, dst, len, dst_dtype),
        DType::U16 => cast_from!(u16, src, dst, len, dst_dtype),
        DType::U8 => cast_from!(u8, src, dst, len, dst_dtype),
        DType::Bool => {
            // Bool is stored as u8 (0 or 1)
            cast_from!(u8, src, dst, len, dst_dtype)
        }
        DType::Complex64 => {
            // Complex64 source: extract real part for non-complex destinations
            let src_slice = std::slice::from_raw_parts(src as *const crate::dtype::Complex64, len);
            match dst_dtype {
                DType::Complex64 => {
                    let dst_slice =
                        std::slice::from_raw_parts_mut(dst as *mut crate::dtype::Complex64, len);
                    dst_slice.copy_from_slice(src_slice);
                }
                DType::Complex128 => {
                    let dst_slice =
                        std::slice::from_raw_parts_mut(dst as *mut crate::dtype::Complex128, len);
                    for i in 0..len {
                        dst_slice[i] = crate::dtype::Complex128::new(
                            src_slice[i].re as f64,
                            src_slice[i].im as f64,
                        );
                    }
                }
                _ => {
                    // For non-complex destinations, take the real part
                    let dst_slice = std::slice::from_raw_parts_mut(dst as *mut f32, len);
                    for i in 0..len {
                        dst_slice[i] = src_slice[i].re;
                    }
                    // Re-interpret as the target type via Element trait
                    return Err(Error::UnsupportedDType {
                        dtype: dst_dtype,
                        op: "cast from Complex64",
                    });
                }
            }
        }
        DType::Complex128 => {
            // Complex128 source: extract real part for non-complex destinations
            let src_slice = std::slice::from_raw_parts(src as *const crate::dtype::Complex128, len);
            match dst_dtype {
                DType::Complex128 => {
                    let dst_slice =
                        std::slice::from_raw_parts_mut(dst as *mut crate::dtype::Complex128, len);
                    dst_slice.copy_from_slice(src_slice);
                }
                DType::Complex64 => {
                    let dst_slice =
                        std::slice::from_raw_parts_mut(dst as *mut crate::dtype::Complex64, len);
                    for i in 0..len {
                        dst_slice[i] = crate::dtype::Complex64::new(
                            src_slice[i].re as f32,
                            src_slice[i].im as f32,
                        );
                    }
                }
                _ => {
                    return Err(Error::UnsupportedDType {
                        dtype: dst_dtype,
                        op: "cast from Complex128",
                    });
                }
            }
        }
    }

    Ok(())
}

/// Fill output with uniform random values in [0, 1)
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_uniform_kernel<T: Element>(out: *mut T, len: usize) {
    let mut prng = rng::thread_rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val = rng::sample_uniform(&mut prng);
        *elem = rng::clamp_uniform_sample(T::from_f64(val));
    }
}

/// Fill output with standard normal random values (mean=0, std=1)
///
/// Uses the Box-Muller transform for generating normally distributed values.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn rand_normal_kernel<T: Element>(out: *mut T, len: usize) {
    let mut prng = rng::thread_rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val = rng::sample_normal(&mut prng);
        *elem = T::from_f64(val);
    }
}

/// Fill output with random integers in [low, high)
///
/// Generates uniformly distributed random integers.
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
/// - `low < high` must be satisfied
#[inline]
pub unsafe fn randint_kernel<T: Element>(out: *mut T, low: i64, high: i64, len: usize) {
    let mut prng = rng::thread_rng();
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for elem in out_slice.iter_mut() {
        let val = rng::sample_uniform_int(&mut prng, low, high);
        *elem = T::from_f64(val as f64);
    }
}

/// Fill output with evenly spaced values in [start, stop)
///
/// Generates values: start + step * i for i in 0..len
///
/// # Safety
/// - `out` must be a valid pointer to `len` elements
#[inline]
pub unsafe fn arange_kernel<T: Element>(out: *mut T, start: f64, step: f64, len: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, len);

    for (i, elem) in out_slice.iter_mut().enumerate() {
        let val = start + step * (i as f64);
        *elem = T::from_f64(val);
    }
}

/// Fill output with evenly spaced values from start to stop (inclusive)
///
/// Generates values: start + (stop - start) * i / (steps - 1) for i in 0..steps
///
/// # Safety
/// - `out` must be a valid pointer to `steps` elements
/// - `steps` must be >= 2
#[inline]
pub unsafe fn linspace_kernel<T: Element>(out: *mut T, start: f64, stop: f64, steps: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, steps);

    if steps == 1 {
        out_slice[0] = T::from_f64(start);
        return;
    }

    let divisor = (steps - 1) as f64;
    let delta = stop - start;

    for (i, elem) in out_slice.iter_mut().enumerate() {
        let val = start + delta * (i as f64) / divisor;
        *elem = T::from_f64(val);
    }
}

/// Fill output with identity matrix (1s on diagonal, 0s elsewhere)
///
/// # Safety
/// - `out` must be a valid pointer to `n * m` elements
#[inline]
pub unsafe fn eye_kernel<T: Element>(out: *mut T, n: usize, m: usize) {
    let out_slice = std::slice::from_raw_parts_mut(out, n * m);

    // Fill with zeros first
    out_slice.fill(T::from_f64(0.0));

    // Set diagonal to 1
    let diag_len = n.min(m);
    for i in 0..diag_len {
        out_slice[i * m + i] = T::from_f64(1.0);
    }
}

/// Sample from a multinomial distribution with replacement
///
/// Uses inverse transform sampling (CDF method):
/// 1. Compute cumulative sum of normalized probabilities
/// 2. For each sample, draw uniform random u ∈ [0, 1)
/// 3. Binary search to find smallest index i where CDF[i] ≥ u
///
/// # Safety
/// - `probs` must be valid pointer to `num_distributions * num_categories` floats
/// - `out` must be valid pointer to `num_distributions * num_samples` i64 values
/// - All probabilities must be non-negative
/// - Each distribution must have at least one non-zero probability
#[inline]
pub unsafe fn multinomial_kernel_with_replacement<T: Element>(
    probs: *const T,
    out: *mut i64,
    num_distributions: usize,
    num_categories: usize,
    num_samples: usize,
) {
    let mut prng = rng::thread_rng();

    for dist in 0..num_distributions {
        let prob_row = std::slice::from_raw_parts(probs.add(dist * num_categories), num_categories);

        // Compute sum for normalization
        let mut sum = 0.0f64;
        for &p in prob_row {
            sum += p.to_f64();
        }

        // Compute CDF (normalized cumulative sum)
        let mut cdf = Vec::with_capacity(num_categories);
        let mut cumsum = 0.0f64;
        for &p in prob_row {
            cumsum += p.to_f64() / sum;
            cdf.push(cumsum);
        }
        // Ensure last element is exactly 1.0 to avoid floating point issues
        if !cdf.is_empty() {
            *cdf.last_mut().unwrap() = 1.0;
        }

        // Sample using binary search
        let out_row = std::slice::from_raw_parts_mut(out.add(dist * num_samples), num_samples);

        for sample in out_row {
            let u = rng::sample_uniform(&mut prng);
            // Binary search: find first index where cdf[i] >= u
            let idx = cdf.partition_point(|&c| c < u);
            *sample = idx.min(num_categories - 1) as i64;
        }
    }
}

/// Sample from a multinomial distribution without replacement
///
/// Uses the reservoir sampling approach:
/// For each sample, draw from remaining categories weighted by their probabilities.
///
/// # Safety
/// - `probs` must be valid pointer to `num_distributions * num_categories` floats
/// - `out` must be valid pointer to `num_distributions * num_samples` i64 values
/// - `num_samples <= num_categories`
/// - All probabilities must be non-negative
/// - Each distribution must have at least `num_samples` non-zero probabilities
#[inline]
pub unsafe fn multinomial_kernel_without_replacement<T: Element>(
    probs: *const T,
    out: *mut i64,
    num_distributions: usize,
    num_categories: usize,
    num_samples: usize,
) {
    let mut prng = rng::thread_rng();

    for dist in 0..num_distributions {
        let prob_row = std::slice::from_raw_parts(probs.add(dist * num_categories), num_categories);

        // Copy probabilities so we can modify them
        let mut remaining_probs: Vec<f64> = prob_row.iter().map(|p| p.to_f64()).collect();
        let out_row = std::slice::from_raw_parts_mut(out.add(dist * num_samples), num_samples);

        for sample in out_row {
            // Compute sum of remaining probabilities
            let sum: f64 = remaining_probs.iter().sum();

            // Compute CDF for remaining probabilities
            let mut cdf = Vec::with_capacity(num_categories);
            let mut cumsum = 0.0f64;
            for &p in &remaining_probs {
                cumsum += p / sum;
                cdf.push(cumsum);
            }
            if !cdf.is_empty() {
                *cdf.last_mut().unwrap() = 1.0;
            }

            // Sample
            let u = rng::sample_uniform(&mut prng);
            let idx = cdf.partition_point(|&c| c < u).min(num_categories - 1);
            *sample = idx as i64;

            // Zero out the selected category so it can't be selected again
            remaining_probs[idx] = 0.0;
        }
    }
}

/// Fisher-Yates shuffle to generate random permutation of [0, n)
///
/// # Safety
/// - `out` must be a valid pointer to `n` elements of i64
pub unsafe fn randperm_kernel(out: *mut i64, n: usize) {
    let mut prng = rng::thread_rng();
    let out_slice = std::slice::from_raw_parts_mut(out, n);

    // Initialize with [0, 1, 2, ..., n-1]
    for i in 0..n {
        out_slice[i] = i as i64;
    }

    // Fisher-Yates shuffle
    for i in (1..n).rev() {
        let j = (prng.next() % (i as u64 + 1)) as usize;
        out_slice.swap(i, j);
    }
}

/// One-hot encode integer indices
///
/// # Safety
/// - `indices` must be a valid pointer to `numel` elements of type T (integer)
/// - `out` must be a valid pointer to `numel * num_classes` f32 elements, zero-initialized
pub unsafe fn one_hot_kernel<T: Element>(
    indices: *const T,
    out: *mut f32,
    numel: usize,
    num_classes: usize,
) {
    let indices_slice = std::slice::from_raw_parts(indices, numel);
    let out_slice = std::slice::from_raw_parts_mut(out, numel * num_classes);

    for i in 0..numel {
        let idx = indices_slice[i].to_f64() as usize;
        if idx < num_classes {
            out_slice[i * num_classes + idx] = 1.0;
        }
    }
}

#[cfg(test)]
mod rand_uniform_range_tests {
    use super::*;
    #[cfg(any(feature = "f16", feature = "fp8"))]
    use crate::dtype::DType;

    const SAMPLE_COUNT: usize = 200_000;

    /// Draw `SAMPLE_COUNT` uniform samples of `T` and assert every value stays
    /// in `[0, 1)`, then check the maximum against `bound`.
    ///
    /// `bound` is `None` for F32/F64: their `[0, 1)` contract was already
    /// correct, so this only checks the range, not a specific ceiling.
    fn check_range_and_bound<T: Element>(bound: Option<f64>) {
        let mut out = vec![T::zero(); SAMPLE_COUNT];
        unsafe {
            rand_uniform_kernel::<T>(out.as_mut_ptr(), SAMPLE_COUNT);
        }

        let mut max_seen = f64::MIN;
        for &v in &out {
            let f = v.to_f64();
            assert!(f >= 0.0, "rand produced negative value: {f}");
            assert!(f < 1.0, "rand produced value out of range [0, 1): {f}");
            if f > max_seen {
                max_seen = f;
            }
        }

        if let Some(bound) = bound {
            assert_eq!(
                max_seen, bound,
                "expected the largest observed value to reach the dtype's bound below 1.0"
            );
        }
    }

    #[test]
    fn test_rand_uniform_f32_unchanged() {
        check_range_and_bound::<f32>(None);
    }

    #[test]
    fn test_rand_uniform_f64_unchanged() {
        check_range_and_bound::<f64>(None);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_rand_uniform_f16_stays_below_one() {
        check_range_and_bound::<half::f16>(DType::F16.largest_value_below_one());
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_rand_uniform_bf16_stays_below_one() {
        check_range_and_bound::<half::bf16>(DType::BF16.largest_value_below_one());
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_rand_uniform_fp8e4m3_stays_below_one() {
        check_range_and_bound::<crate::dtype::FP8E4M3>(DType::FP8E4M3.largest_value_below_one());
    }

    #[cfg(feature = "fp8")]
    #[test]
    fn test_rand_uniform_fp8e5m2_stays_below_one() {
        check_range_and_bound::<crate::dtype::FP8E5M2>(DType::FP8E5M2.largest_value_below_one());
    }
}
