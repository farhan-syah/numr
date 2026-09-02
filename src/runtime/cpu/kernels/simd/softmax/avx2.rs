//! AVX2 softmax kernels using online algorithm (2-pass).
//!
//! Pass 1: Online SIMD max + sum (single read of input)
//! Pass 2: Compute exp(x - max) / sum and write output (one read + one write)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx2::{exp_f32, exp_f64, hmax_f32, hmax_f64, hsum_f32, hsum_f64};

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

/// AVX2 softmax for f32 using online algorithm.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum in a single read pass
        let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut sum_vec = _mm256_setzero_ps();

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm256_loadu_ps(a.add(offset));

            // Save old max, compute new max
            let old_max = max_vec;
            max_vec = _mm256_max_ps(max_vec, v);

            // Rescale previous sum: sum *= exp(old_max - new_max)
            // Guard: when old_max == new_max == -inf, exp(-inf-(-inf)) = NaN.
            // Use a validity mask to zero out -inf lanes (their sum contribution is 0).
            let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
            let valid_old = _mm256_cmp_ps(old_max, neg_inf, _CMP_GT_OQ);
            let rescale = exp_f32(_mm256_sub_ps(old_max, max_vec));
            let rescale = _mm256_and_ps(rescale, valid_old);
            sum_vec = _mm256_mul_ps(sum_vec, rescale);

            // Add new contributions: sum += exp(v - new_max)
            let valid_new = _mm256_cmp_ps(max_vec, neg_inf, _CMP_GT_OQ);
            let exp_v = exp_f32(_mm256_sub_ps(v, max_vec));
            let exp_v = _mm256_and_ps(exp_v, valid_new);
            sum_vec = _mm256_add_ps(sum_vec, exp_v);
        }

        // Horizontal reduce: reconcile per-lane max/sum to scalar
        let max_val_simd = hmax_f32(max_vec);
        let mut max_val = max_val_simd;

        // Handle scalar tail for max (online)
        let mut tail_sum = 0.0f32;
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                let rescale = if max_val == f32::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                tail_sum = tail_sum * rescale + 1.0;
                max_val = val;
            } else if val == f32::NEG_INFINITY {
                // skip: contribution is 0
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with scalar max: each lane's sum must be rescaled
        // sum_vec[i] was computed relative to max_vec[i], but we need it relative to max_val
        // Guard: if a lane's max is -inf (all elements were -inf), its sum contribution is 0,
        // not NaN. We zero out those lanes to avoid NaN from exp(-inf - (-inf)).
        let v_max_vec = max_vec; // per-lane max values
        let v_global_max = _mm256_set1_ps(max_val);
        let neg_inf = _mm256_set1_ps(f32::NEG_INFINITY);
        let valid_mask = _mm256_cmp_ps(v_max_vec, neg_inf, _CMP_GT_OQ);
        let rescale = exp_f32(_mm256_sub_ps(v_max_vec, v_global_max));
        let rescale = _mm256_and_ps(rescale, valid_mask);
        let rescaled_sum = _mm256_mul_ps(sum_vec, rescale);
        let sum = hsum_f32(rescaled_sum) + tail_sum;

        // Pass 2: Compute exp(x - max) / sum in a single write pass
        let v_max = _mm256_set1_ps(max_val);
        let v_inv_sum = _mm256_set1_ps(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm256_loadu_ps(a.add(offset));
            let diff = _mm256_sub_ps(v, v_max);
            let normalized = _mm256_mul_ps(exp_f32(diff), v_inv_sum);
            _mm256_storeu_ps(out.add(offset), normalized);
        }

        // Scalar tail
        let inv_sum = 1.0 / sum;
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = (val - max_val).exp() * inv_sum;
        }
    }
}

/// AVX2 softmax for f64 using online algorithm.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum
        let mut max_vec = _mm256_set1_pd(f64::NEG_INFINITY);
        let mut sum_vec = _mm256_setzero_pd();

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm256_loadu_pd(a.add(offset));

            let old_max = max_vec;
            max_vec = _mm256_max_pd(max_vec, v);

            // Rescale previous sum: sum *= exp(old_max - new_max)
            // Guard: when old_max == new_max == -inf, exp(-inf-(-inf)) = NaN.
            // Use a validity mask to zero out -inf lanes (their sum contribution is 0).
            let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
            let valid_old = _mm256_cmp_pd(old_max, neg_inf, _CMP_GT_OQ);
            let rescale = exp_f64(_mm256_sub_pd(old_max, max_vec));
            let rescale = _mm256_and_pd(rescale, valid_old);
            sum_vec = _mm256_mul_pd(sum_vec, rescale);

            // Add new contributions: sum += exp(v - new_max)
            let valid_new = _mm256_cmp_pd(max_vec, neg_inf, _CMP_GT_OQ);
            let exp_v = exp_f64(_mm256_sub_pd(v, max_vec));
            let exp_v = _mm256_and_pd(exp_v, valid_new);
            sum_vec = _mm256_add_pd(sum_vec, exp_v);
        }

        let max_val_simd = hmax_f64(max_vec);
        let mut max_val = max_val_simd;

        // Scalar tail (online)
        let mut tail_sum = 0.0f64;
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            if val > max_val {
                let rescale = if max_val == f64::NEG_INFINITY {
                    0.0
                } else {
                    (max_val - val).exp()
                };
                tail_sum = tail_sum * rescale + 1.0;
                max_val = val;
            } else if val == f64::NEG_INFINITY {
                // skip
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with scalar max: each lane's sum must be rescaled
        // sum_vec[i] was computed relative to max_vec[i], but we need it relative to max_val
        // Guard: if a lane's max is -inf (all elements were -inf), its sum contribution is 0,
        // not NaN. We zero out those lanes to avoid NaN from exp(-inf - (-inf)).
        let v_max_vec = max_vec;
        let v_global_max = _mm256_set1_pd(max_val);
        let neg_inf = _mm256_set1_pd(f64::NEG_INFINITY);
        let valid_mask = _mm256_cmp_pd(v_max_vec, neg_inf, _CMP_GT_OQ);
        let rescale = exp_f64(_mm256_sub_pd(v_max_vec, v_global_max));
        let rescale = _mm256_and_pd(rescale, valid_mask);
        let rescaled_sum = _mm256_mul_pd(sum_vec, rescale);
        let sum = hsum_f64(rescaled_sum) + tail_sum;

        // Pass 2: exp(x - max) / sum
        let v_max = _mm256_set1_pd(max_val);
        let v_inv_sum = _mm256_set1_pd(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm256_loadu_pd(a.add(offset));
            let diff = _mm256_sub_pd(v, v_max);
            let normalized = _mm256_mul_pd(exp_f64(diff), v_inv_sum);
            _mm256_storeu_pd(out.add(offset), normalized);
        }

        let inv_sum = 1.0 / sum;
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = (val - max_val).exp() * inv_sum;
        }
    }
}
