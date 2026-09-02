//! NEON logsumexp kernels for ARM64
//!
//! Provides vectorized logsumexp operation using 128-bit NEON registers.
//!
//! logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
//!
//! # SIMD Strategy
//!
//! 1. SIMD max-reduce to find maximum for numerical stability
//! 2. SIMD exp computation with shifted values
//! 3. SIMD sum-reduce for accumulation
//! 4. Add max + log(sum) for final result

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::math::aarch64::neon::{
    exp_f32, exp_f64, hmax_f32, hmax_f64, hsum_f32, hsum_f64,
};

const F32_LANES: usize = 4;
const F64_LANES: usize = 2;

/// NEON logsumexp for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `reduce_size * outer_size` valid f32 elements
/// - `out` must point to `outer_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn logsumexp_f32(a: *const f32, out: *mut f32, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F32_LANES;
    let remainder = reduce_size % F32_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);

        // Phase 1: Find max
        let mut max_acc = vdupq_n_f32(f32::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            max_acc = vmaxq_f32(max_acc, v);
        }
        let mut max_val = hmax_f32(max_acc);

        // Scalar tail for max
        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            if val > max_val {
                max_val = val;
            }
        }

        let v_max = vdupq_n_f32(max_val);

        // Phase 2: Sum of exp(x - max)
        let mut sum_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vld1q_f32(base.add(i * F32_LANES));
            let shifted = vsubq_f32(v, v_max);
            let exp_v = exp_f32(shifted);
            sum_acc = vaddq_f32(sum_acc, exp_v);
        }
        let mut sum = hsum_f32(sum_acc);

        // Scalar tail for exp and sum
        for i in 0..remainder {
            let val = *base.add(chunks * F32_LANES + i);
            sum += (val - max_val).exp();
        }

        // Phase 3: Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}

/// NEON logsumexp for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `a` must point to `reduce_size * outer_size` valid f64 elements
/// - `out` must point to `outer_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn logsumexp_f64(a: *const f64, out: *mut f64, reduce_size: usize, outer_size: usize) {
    let chunks = reduce_size / F64_LANES;
    let remainder = reduce_size % F64_LANES;

    for o in 0..outer_size {
        let base = a.add(o * reduce_size);

        // Phase 1: Find max
        let mut max_acc = vdupq_n_f64(f64::NEG_INFINITY);
        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            max_acc = vmaxq_f64(max_acc, v);
        }
        let mut max_val = hmax_f64(max_acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
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

        let v_max = vdupq_n_f64(max_val);

        // Phase 2: Sum of exp(x - max)
        let mut sum_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let v = vld1q_f64(base.add(i * F64_LANES));
            let shifted = vsubq_f64(v, v_max);
            let exp_v = exp_f64(shifted);
            sum_acc = vaddq_f64(sum_acc, exp_v);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in 0..remainder {
            let val = *base.add(chunks * F64_LANES + i);
            sum += (val - max_val).exp();
        }

        // Phase 3: Result = max + log(sum)
        *out.add(o) = max_val + sum.ln();
    }
}
