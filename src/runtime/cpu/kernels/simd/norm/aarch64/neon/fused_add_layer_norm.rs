//! NEON fused add + layer normalization kernels (forward and backward)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::super::super::super::math::aarch64::neon::{hsum_f32, hsum_f64};
use super::{F32_LANES, F64_LANES};

/// NEON Fused Add + Layer Normalization for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_f32(
    input: *const f32,
    residual: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    pre_norm: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let res_base = residual.add(b * hidden_size);
        let pn_base = pre_norm.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *base + *res_base;
        let v_ref = vdupq_n_f32(reference);

        // Phase 1: Compute mean
        let mut sum_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let v_in = vld1q_f32(base.add(offset));
            let v_res = vld1q_f32(res_base.add(offset));
            let pn = vaddq_f32(v_in, v_res);
            vst1q_f32(pn_base.add(offset), pn);
            sum_acc = vaddq_f32(sum_acc, vsubq_f32(pn, v_ref));
        }
        let mut sum = hsum_f32(sum_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let pn = *base.add(offset) + *res_base.add(offset);
            *pn_base.add(offset) = pn;
            sum += pn - reference;
        }

        let shifted_mean = sum / hidden_size as f32;
        let v_shifted_mean = vdupq_n_f32(shifted_mean);

        // Phase 2: Compute variance
        let mut var_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);
            let diff = vsubq_f32(pn, v_shifted_mean);
            var_acc = vfmaq_f32(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f32(var_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let diff = (*pn_base.add(offset) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();
        let v_inv_std = vdupq_n_f32(inv_std);

        // Phase 3: Apply normalization, weight, and bias
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);
            let v_w = vld1q_f32(weight.add(offset));
            let v_b = vld1q_f32(bias.add(offset));

            let normalized = vmulq_f32(vsubq_f32(pn, v_shifted_mean), v_inv_std);
            let result = vfmaq_f32(v_b, normalized, v_w);
            vst1q_f32(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let x = *pn_base.add(offset) - reference;
            let w = *weight.add(offset);
            let b = *bias.add(offset);
            *out_base.add(offset) = (x - shifted_mean) * inv_std * w + b;
        }
    }
}

/// NEON Fused Add + Layer Normalization for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_f64(
    input: *const f64,
    residual: *const f64,
    weight: *const f64,
    bias: *const f64,
    out: *mut f64,
    pre_norm: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let base = input.add(b * hidden_size);
        let res_base = residual.add(b * hidden_size);
        let pn_base = pre_norm.add(b * hidden_size);
        let out_base = out.add(b * hidden_size);
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *base + *res_base;
        let v_ref = vdupq_n_f64(reference);

        let mut sum_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let v_in = vld1q_f64(base.add(offset));
            let v_res = vld1q_f64(res_base.add(offset));
            let pn = vaddq_f64(v_in, v_res);
            vst1q_f64(pn_base.add(offset), pn);
            sum_acc = vaddq_f64(sum_acc, vsubq_f64(pn, v_ref));
        }
        let mut sum = hsum_f64(sum_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let pn = *base.add(offset) + *res_base.add(offset);
            *pn_base.add(offset) = pn;
            sum += pn - reference;
        }

        let shifted_mean = sum / hidden_size as f64;
        let v_shifted_mean = vdupq_n_f64(shifted_mean);

        let mut var_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);
            let diff = vsubq_f64(pn, v_shifted_mean);
            var_acc = vfmaq_f64(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f64(var_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let diff = (*pn_base.add(offset) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();
        let v_inv_std = vdupq_n_f64(inv_std);

        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);
            let v_w = vld1q_f64(weight.add(offset));
            let v_b = vld1q_f64(bias.add(offset));

            let normalized = vmulq_f64(vsubq_f64(pn, v_shifted_mean), v_inv_std);
            let result = vfmaq_f64(v_b, normalized, v_w);
            vst1q_f64(out_base.add(offset), result);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let x = *pn_base.add(offset) - reference;
            let w = *weight.add(offset);
            let b = *bias.add(offset);
            *out_base.add(offset) = (x - shifted_mean) * inv_std * w + b;
        }
    }
}

/// NEON Fused Add + Layer Norm Backward for f32
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bwd_f32(
    grad: *const f32,
    pre_norm: *const f32,
    weight: *const f32,
    d_input_residual: *mut f32,
    d_weight: *mut f32,
    d_bias: *mut f32,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let chunks = hidden_size / F32_LANES;
    let remainder = hidden_size % F32_LANES;

    for b in 0..batch_size {
        let pn_base = pre_norm.add(b * hidden_size);
        let grad_base = grad.add(b * hidden_size);
        let d_ir_base = d_input_residual.add(b * hidden_size);
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pn_base;
        let v_ref = vdupq_n_f32(reference);

        // Recompute mean from pre_norm
        let mut sum_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);
            sum_acc = vaddq_f32(sum_acc, pn);
        }
        let mut sum = hsum_f32(sum_acc);

        for i in 0..remainder {
            sum += *pn_base.add(chunks * F32_LANES + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f32;
        let v_shifted_mean = vdupq_n_f32(shifted_mean);

        // Recompute variance
        let mut var_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);
            let diff = vsubq_f32(pn, v_shifted_mean);
            var_acc = vfmaq_f32(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f32(var_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let diff = (*pn_base.add(offset) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        // Compute mean_gs = mean(grad * weight) and mean_gs_n = mean(grad * weight * normalized)
        let mut gs_acc = vdupq_n_f32(0.0);
        let mut gsn_acc = vdupq_n_f32(0.0);
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(grad_base.add(offset));
            let w = vld1q_f32(weight.add(offset));
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);

            let gs = vmulq_f32(g, w);
            gs_acc = vaddq_f32(gs_acc, gs);

            let diff = vsubq_f32(pn, v_shifted_mean);
            let normalized = vmulq_f32(diff, vdupq_n_f32(inv_std));
            let gsn = vmulq_f32(gs, normalized);
            gsn_acc = vaddq_f32(gsn_acc, gsn);
        }
        let mut mean_gs_simd = hsum_f32(gs_acc);
        let mut mean_gsn_simd = hsum_f32(gsn_acc);

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset) - reference;

            let gs = g * w;
            mean_gs_simd += gs;

            let normalized = (pn - shifted_mean) * inv_std;
            mean_gsn_simd += gs * normalized;
        }

        let mean_gs = mean_gs_simd / hidden_size as f32;
        let mean_gs_n = mean_gsn_simd / hidden_size as f32;
        let v_inv_std = vdupq_n_f32(inv_std);
        let v_shifted_mean_gs = vdupq_n_f32(mean_gs);
        let v_shifted_mean_gs_n = vdupq_n_f32(mean_gs_n);

        // Apply and accumulate
        for i in 0..chunks {
            let offset = i * F32_LANES;
            let g = vld1q_f32(grad_base.add(offset));
            let w = vld1q_f32(weight.add(offset));
            let pn = vsubq_f32(vld1q_f32(pn_base.add(offset)), v_ref);

            let normalized = vmulq_f32(vsubq_f32(pn, v_shifted_mean), v_inv_std);
            let gs = vmulq_f32(g, w);
            let d_ir = vmulq_f32(
                v_inv_std,
                vsubq_f32(
                    gs,
                    vaddq_f32(
                        v_shifted_mean_gs,
                        vmulq_f32(normalized, v_shifted_mean_gs_n),
                    ),
                ),
            );
            vst1q_f32(d_ir_base.add(offset), d_ir);

            let dw_old = vld1q_f32(d_weight.add(offset));
            let dw_add = vmulq_f32(g, normalized);
            let dw_new = vaddq_f32(dw_old, dw_add);
            vst1q_f32(d_weight.add(offset), dw_new);

            let db_old = vld1q_f32(d_bias.add(offset));
            let db_new = vaddq_f32(db_old, g);
            vst1q_f32(d_bias.add(offset), db_new);
        }

        for i in 0..remainder {
            let offset = chunks * F32_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset) - reference;

            let normalized = (pn - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_ir_base.add(offset) = d_ir;

            *d_weight.add(offset) += g * normalized;
            *d_bias.add(offset) += g;
        }
    }
}

/// NEON Fused Add + Layer Norm Backward for f64
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bwd_f64(
    grad: *const f64,
    pre_norm: *const f64,
    weight: *const f64,
    d_input_residual: *mut f64,
    d_weight: *mut f64,
    d_bias: *mut f64,
    batch_size: usize,
    hidden_size: usize,
    eps: f64,
) {
    let chunks = hidden_size / F64_LANES;
    let remainder = hidden_size % F64_LANES;

    for b in 0..batch_size {
        let pn_base = pre_norm.add(b * hidden_size);
        let grad_base = grad.add(b * hidden_size);
        let d_ir_base = d_input_residual.add(b * hidden_size);
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pn_base;
        let v_ref = vdupq_n_f64(reference);

        let mut sum_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);
            sum_acc = vaddq_f64(sum_acc, pn);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in 0..remainder {
            sum += *pn_base.add(chunks * F64_LANES + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f64;
        let v_shifted_mean = vdupq_n_f64(shifted_mean);

        let mut var_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);
            let diff = vsubq_f64(pn, v_shifted_mean);
            var_acc = vfmaq_f64(var_acc, diff, diff);
        }
        let mut var_sum = hsum_f64(var_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let diff = (*pn_base.add(offset) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        let mut gs_acc = vdupq_n_f64(0.0);
        let mut gsn_acc = vdupq_n_f64(0.0);
        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(grad_base.add(offset));
            let w = vld1q_f64(weight.add(offset));
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);

            let gs = vmulq_f64(g, w);
            gs_acc = vaddq_f64(gs_acc, gs);

            let diff = vsubq_f64(pn, v_shifted_mean);
            let normalized = vmulq_f64(diff, vdupq_n_f64(inv_std));
            let gsn = vmulq_f64(gs, normalized);
            gsn_acc = vaddq_f64(gsn_acc, gsn);
        }
        let mut mean_gs_simd = hsum_f64(gs_acc);
        let mut mean_gsn_simd = hsum_f64(gsn_acc);

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset) - reference;

            let gs = g * w;
            mean_gs_simd += gs;

            let normalized = (pn - shifted_mean) * inv_std;
            mean_gsn_simd += gs * normalized;
        }

        let mean_gs = mean_gs_simd / hidden_size as f64;
        let mean_gs_n = mean_gsn_simd / hidden_size as f64;
        let v_inv_std = vdupq_n_f64(inv_std);
        let v_shifted_mean_gs = vdupq_n_f64(mean_gs);
        let v_shifted_mean_gs_n = vdupq_n_f64(mean_gs_n);

        for i in 0..chunks {
            let offset = i * F64_LANES;
            let g = vld1q_f64(grad_base.add(offset));
            let w = vld1q_f64(weight.add(offset));
            let pn = vsubq_f64(vld1q_f64(pn_base.add(offset)), v_ref);

            let normalized = vmulq_f64(vsubq_f64(pn, v_shifted_mean), v_inv_std);
            let gs = vmulq_f64(g, w);
            let d_ir = vmulq_f64(
                v_inv_std,
                vsubq_f64(
                    gs,
                    vaddq_f64(
                        v_shifted_mean_gs,
                        vmulq_f64(normalized, v_shifted_mean_gs_n),
                    ),
                ),
            );
            vst1q_f64(d_ir_base.add(offset), d_ir);

            let dw_old = vld1q_f64(d_weight.add(offset));
            let dw_add = vmulq_f64(g, normalized);
            let dw_new = vaddq_f64(dw_old, dw_add);
            vst1q_f64(d_weight.add(offset), dw_new);

            let db_old = vld1q_f64(d_bias.add(offset));
            let db_new = vaddq_f64(db_old, g);
            vst1q_f64(d_bias.add(offset), db_new);
        }

        for i in 0..remainder {
            let offset = chunks * F64_LANES + i;
            let g = *grad_base.add(offset);
            let w = *weight.add(offset);
            let pn = *pn_base.add(offset) - reference;

            let normalized = (pn - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_ir_base.add(offset) = d_ir;

            *d_weight.add(offset) += g * normalized;
            *d_bias.add(offset) += g;
        }
    }
}
