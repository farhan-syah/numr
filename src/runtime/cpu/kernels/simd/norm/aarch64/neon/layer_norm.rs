//! NEON layer normalization kernels

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::super::math::aarch64::neon::{hsum_f32, hsum_f64};
use super::{F32_LANES, F64_LANES};

/// NEON Layer normalization for f32
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `input` and `out` must point to `batch_size * hidden_size` valid f32 elements
/// - `weight` and `bias` must point to `hidden_size` valid f32 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn layer_norm_f32(
    input: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        // Every element is shifted by the row's first value before it is
        // accumulated. `x - mean` cancels catastrophically when a row sits far
        // from zero relative to its own spread, which is where the accumulator's
        // precision goes. The two subtractions must stay separate: folding them
        // into a single `x - (reference + shifted_mean)` is the cancelling form
        // again. Mean, variance and the normalized value are all invariant under
        // the shift in exact arithmetic.
        let reference = *base;
        let v_ref = vdupq_n_f32(reference);

        // Phase 1: Compute the mean of the shifted values
        let mut sum_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vsubq_f32(vld1q_f32(base.add(i * F32_LANES)), v_ref);
            sum_acc = vaddq_f32(sum_acc, v);
        }
        let mut sum = hsum_f32(sum_acc);

        for i in 0..remainder {
            sum += *base.add(chunks * F32_LANES + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f32;
        let v_shifted_mean = vdupq_n_f32(shifted_mean);

        // Phase 2: Compute variance
        let mut var_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let v = vsubq_f32(vld1q_f32(base.add(i * F32_LANES)), v_ref);
            let diff = vsubq_f32(v, v_shifted_mean);
            var_acc = vfmaq_f32(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f32(var_acc);

        for i in 0..remainder {
            let diff = (*base.add(chunks * F32_LANES + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();
        let v_inv_std = vdupq_n_f32(inv_std);

        // Phase 3: Apply normalization, weight, and bias
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v_in = vld1q_f32(base.add(offset));
            let v_w = vld1q_f32(weight.add(offset));
            let v_b = vld1q_f32(bias.add(offset));

            let normalized =
                vmulq_f32(vsubq_f32(vsubq_f32(v_in, v_ref), v_shifted_mean), v_inv_std);
            let result = vfmaq_f32(v_b, normalized, v_w); // b + normalized * w
            vst1q_f32(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let x = *base.add(offset);
            let w = *weight.add(offset);
            let b = *bias.add(offset);
            *out_base.add(offset) = ((x - reference) - shifted_mean) * inv_std * w + b;
        }
    }
}

/// NEON Layer normalization for f64
///
/// # Safety
/// - CPU must support NEON (always true on AArch64)
/// - `input` and `out` must point to `batch_size * hidden_size` valid f64 elements
/// - `weight` and `bias` must point to `hidden_size` valid f64 elements
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn layer_norm_f64(
    input: *const f64,
    weight: *const f64,
    bias: *const f64,
    out: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);

        // Every element is shifted by the row's first value before it is
        // accumulated. `x - mean` cancels catastrophically when a row sits far
        // from zero relative to its own spread, which is where the accumulator's
        // precision goes. The two subtractions must stay separate: folding them
        // into a single `x - (reference + shifted_mean)` is the cancelling form
        // again. Mean, variance and the normalized value are all invariant under
        // the shift in exact arithmetic.
        let reference = *base;
        let v_ref = vdupq_n_f64(reference);

        // Phase 1: Compute the mean of the shifted values
        let mut sum_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let v = vsubq_f64(vld1q_f64(base.add(i * F64_LANES)), v_ref);
            sum_acc = vaddq_f64(sum_acc, v);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in 0..remainder {
            sum += *base.add(chunks * F64_LANES + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f64;
        let v_shifted_mean = vdupq_n_f64(shifted_mean);

        // Phase 2: Compute variance
        let mut var_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let v = vsubq_f64(vld1q_f64(base.add(i * F64_LANES)), v_ref);
            let diff = vsubq_f64(v, v_shifted_mean);
            var_acc = vfmaq_f64(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f64(var_acc);

        for i in 0..remainder {
            let diff = (*base.add(chunks * F64_LANES + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();
        let v_inv_std = vdupq_n_f64(inv_std);

        // Phase 3: Apply normalization, weight, and bias
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v_in = vld1q_f64(base.add(offset));
            let v_w = vld1q_f64(weight.add(offset));
            let v_b = vld1q_f64(bias.add(offset));

            let normalized =
                vmulq_f64(vsubq_f64(vsubq_f64(v_in, v_ref), v_shifted_mean), v_inv_std);
            let result = vfmaq_f64(v_b, normalized, v_w);
            vst1q_f64(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let x = *base.add(offset);
            let w = *weight.add(offset);
            let b = *bias.add(offset);
            *out_base.add(offset) = ((x - reference) - shifted_mean) * inv_std * w + b;
        }
    }
}
