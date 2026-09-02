//! AVX-512 softmax kernels using online algorithm (2-pass).
//!
//! Pass 1: Online SIMD max + sum (single read of input)
//! Pass 2: Compute exp(x - max) / sum and write output (one read + one write)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{exp_f32, exp_f64};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 softmax for f32 using online algorithm.
#[target_feature(enable = "avx512f")]
pub unsafe fn softmax_f32(a: *const f32, out: *mut f32, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum
        let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
        let mut sum_vec = _mm512_setzero_ps();

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));

            let old_max = max_vec;
            max_vec = _mm512_max_ps(max_vec, v);

            // Rescale previous sum and add new contributions.
            // Guard: when old_max == max_vec == -inf, exp(-inf-(-inf)) = NaN.
            // Use mask to zero out -inf lanes (their sum contribution is 0).
            let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);
            let valid_old = _mm512_cmp_ps_mask(old_max, neg_inf, _CMP_GT_OQ);
            let rescale = exp_f32(_mm512_sub_ps(old_max, max_vec));
            let rescale = _mm512_maskz_mov_ps(valid_old, rescale);
            sum_vec = _mm512_mul_ps(sum_vec, rescale);

            let valid_new = _mm512_cmp_ps_mask(max_vec, neg_inf, _CMP_GT_OQ);
            let exp_v = exp_f32(_mm512_sub_ps(v, max_vec));
            let exp_v = _mm512_maskz_mov_ps(valid_new, exp_v);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }

        let mut max_val = _mm512_reduce_max_ps(max_vec);

        // Scalar tail (online)
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
                // skip
            } else {
                tail_sum += (val - max_val).exp();
            }
        }

        // Reconcile SIMD sum with global max
        // Guard -inf lanes to avoid NaN from exp(-inf - (-inf))
        let v_global_max = _mm512_set1_ps(max_val);
        let neg_inf = _mm512_set1_ps(f32::NEG_INFINITY);
        let valid_mask = _mm512_cmp_ps_mask(max_vec, neg_inf, _CMP_GT_OQ);
        let rescale = exp_f32(_mm512_sub_ps(max_vec, v_global_max));
        let rescale = _mm512_maskz_mov_ps(valid_mask, rescale);
        let rescaled_sum = _mm512_mul_ps(sum_vec, rescale);
        let sum = _mm512_reduce_add_ps(rescaled_sum) + tail_sum;

        // Pass 2: exp(x - max) / sum
        let v_max = _mm512_set1_ps(max_val);
        let v_inv_sum = _mm512_set1_ps(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));
            let diff = _mm512_sub_ps(v, v_max);
            let normalized = _mm512_mul_ps(exp_f32(diff), v_inv_sum);
            _mm512_storeu_ps(out.add(offset), normalized);
        }

        let inv_sum = 1.0 / sum;
        for d in (chunks * F32_LANES)..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = (val - max_val).exp() * inv_sum;
        }
    }
}

/// AVX-512 softmax for f64 using online algorithm.
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn softmax_f64(a: *const f64, out: *mut f64, outer_size: usize, dim_size: usize) {
    let chunks = dim_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * dim_size;

        // Pass 1: Online max + sum
        let mut max_vec = _mm512_set1_pd(f64::NEG_INFINITY);
        let mut sum_vec = _mm512_setzero_pd();

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));

            let old_max = max_vec;
            max_vec = _mm512_max_pd(max_vec, v);

            // Guard: when old_max == max_vec == -inf, exp(-inf-(-inf)) = NaN.
            // Use mask to zero out -inf lanes (their sum contribution is 0).
            let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);
            let valid_old = _mm512_cmp_pd_mask(old_max, neg_inf, _CMP_GT_OQ);
            let rescale = exp_f64(_mm512_sub_pd(old_max, max_vec));
            let rescale = _mm512_maskz_mov_pd(valid_old, rescale);
            sum_vec = _mm512_mul_pd(sum_vec, rescale);

            let valid_new = _mm512_cmp_pd_mask(max_vec, neg_inf, _CMP_GT_OQ);
            let exp_v = exp_f64(_mm512_sub_pd(v, max_vec));
            let exp_v = _mm512_maskz_mov_pd(valid_new, exp_v);
            sum_vec = _mm512_add_pd(sum_vec, exp_v);
        }

        let mut max_val = _mm512_reduce_max_pd(max_vec);

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

        // Reconcile SIMD sum with global max
        // Guard -inf lanes to avoid NaN from exp(-inf - (-inf))
        let v_global_max = _mm512_set1_pd(max_val);
        let neg_inf = _mm512_set1_pd(f64::NEG_INFINITY);
        let valid_mask = _mm512_cmp_pd_mask(max_vec, neg_inf, _CMP_GT_OQ);
        let rescale = exp_f64(_mm512_sub_pd(max_vec, v_global_max));
        let rescale = _mm512_maskz_mov_pd(valid_mask, rescale);
        let rescaled_sum = _mm512_mul_pd(sum_vec, rescale);
        let sum = _mm512_reduce_add_pd(rescaled_sum) + tail_sum;

        // Pass 2: exp(x - max) / sum
        let v_max = _mm512_set1_pd(max_val);
        let v_inv_sum = _mm512_set1_pd(1.0 / sum);

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));
            let diff = _mm512_sub_pd(v, v_max);
            let normalized = _mm512_mul_pd(exp_f64(diff), v_inv_sum);
            _mm512_storeu_pd(out.add(offset), normalized);
        }

        let inv_sum = 1.0 / sum;
        for d in (chunks * F64_LANES)..dim_size {
            let val = *a.add(base + d);
            *out.add(base + d) = (val - max_val).exp() * inv_sum;
        }
    }
}
