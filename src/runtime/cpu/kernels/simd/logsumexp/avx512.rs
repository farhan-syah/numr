//! AVX-512 logsumexp kernels
//!
//! Uses SIMD for max-reduce, exp computation, and sum-reduce.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::super::math::avx512::{exp_f32, exp_f64};

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

/// AVX-512 logsumexp for f32
#[target_feature(enable = "avx512f")]
pub unsafe fn logsumexp_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm512_set1_ps(f32::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));
            max_vec = _mm512_max_ps(max_vec, v);
        }
        let mut max_val = _mm512_reduce_max_ps(max_vec);

        // Scalar tail for max
        for i in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Step 2: SIMD exp(x - max) and sum
        let v_max = _mm512_set1_ps(max_val);
        let mut sum_vec = _mm512_setzero_ps();

        for c in 0..chunks {
            let offset = base + c * F32_LANES;
            let v = _mm512_loadu_ps(a.add(offset));
            let diff = _mm512_sub_ps(v, v_max);
            let exp_v = exp_f32(diff);
            sum_vec = _mm512_add_ps(sum_vec, exp_v);
        }

        let mut sum = _mm512_reduce_add_ps(sum_vec);

        // Scalar tail for exp and sum
        for i in (chunks * F32_LANES)..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Step 3: Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}

/// AVX-512 logsumexp for f64
#[target_feature(enable = "avx512f", enable = "avx512dq")]
pub unsafe fn logsumexp_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;

    for o in 0..outer_size {
        let base = o * reduce_size;

        // Step 1: SIMD max-reduce
        let mut max_vec = _mm512_set1_pd(f64::NEG_INFINITY);
        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));
            max_vec = _mm512_max_pd(max_vec, v);
        }
        let mut max_val = _mm512_reduce_max_pd(max_vec);

        // Scalar tail for max
        for i in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + i);
            if val > max_val {
                max_val = val;
            }
        }

        // Every element is -inf, so exp(x - max) would be exp(-inf - (-inf)) = NaN.
        // logsumexp of an all-masked row is -inf.
        if max_val == f64::NEG_INFINITY {
            *out.add(o) = f64::NEG_INFINITY;
            continue;
        }

        // Step 2: SIMD exp(x - max) and sum
        let v_max = _mm512_set1_pd(max_val);
        let mut sum_vec = _mm512_setzero_pd();

        for c in 0..chunks {
            let offset = base + c * F64_LANES;
            let v = _mm512_loadu_pd(a.add(offset));
            let diff = _mm512_sub_pd(v, v_max);
            let exp_v = exp_f64(diff);
            sum_vec = _mm512_add_pd(sum_vec, exp_v);
        }

        let mut sum = _mm512_reduce_add_pd(sum_vec);

        // Scalar tail for exp and sum
        for i in (chunks * F64_LANES)..reduce_size {
            let val = *a.add(base + i);
            sum += (val - max_val).exp();
        }

        // Step 3: Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}
