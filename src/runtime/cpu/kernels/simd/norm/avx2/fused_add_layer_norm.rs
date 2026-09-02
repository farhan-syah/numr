//! AVX2 fused add + layer normalization kernels (forward and backward)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::{F32_LANES, F64_LANES, hsum_f32, hsum_f64};

/// AVX2 Fused Add + Layer Normalization for f32
///
/// Computes: output = (input + residual - mean) / sqrt(var + eps) * weight + bias
/// Stores intermediate (input + residual) in pre_norm for backward pass.
#[target_feature(enable = "avx2", enable = "fma")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *input.add(row_start) + *residual.add(row_start);
        let v_ref = _mm256_set1_ps(reference);

        // Phase 1: Add and store in pre_norm, compute mean
        let mut sum_acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let v_in = _mm256_loadu_ps(input.add(offset));
            let v_res = _mm256_loadu_ps(residual.add(offset));
            let pn = _mm256_add_ps(v_in, v_res);
            _mm256_storeu_ps(pre_norm.add(offset), pn);
            sum_acc = _mm256_add_ps(sum_acc, _mm256_sub_ps(pn, v_ref));
        }
        let mut sum = hsum_f32(sum_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum += pn - reference;
        }

        let shifted_mean = sum / hidden_size as f32;
        let v_shifted_mean = _mm256_set1_ps(shifted_mean);

        // Phase 2: Compute variance (dual accumulators)
        let mut var_acc0 = _mm256_setzero_ps();
        let mut var_acc1 = _mm256_setzero_ps();
        let mut c = 0;
        let chunk_pairs = chunks / 2 * 2;
        while c < chunk_pairs {
            let diff0 = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + c * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_ps(diff0, diff0, var_acc0);
            let diff1 = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + (c + 1) * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc1 = _mm256_fmadd_ps(diff1, diff1, var_acc1);
            c += 2;
        }
        while c < chunks {
            let diff = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + c * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_ps(diff, diff, var_acc0);
            c += 1;
        }
        let mut var_sum = hsum_f32(_mm256_add_ps(var_acc0, var_acc1));

        for i in (chunks * F32_LANES)..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();
        let v_inv_std = _mm256_set1_ps(inv_std);

        // Phase 3: Normalize, apply weight and bias
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let pn = _mm256_sub_ps(_mm256_loadu_ps(pre_norm.add(offset)), v_ref);
            let v_weight = _mm256_loadu_ps(weight.add(w_offset));
            let v_bias = _mm256_loadu_ps(bias.add(w_offset));

            let diff = _mm256_sub_ps(pn, v_shifted_mean);
            let normalized = _mm256_mul_ps(diff, v_inv_std);
            let scaled = _mm256_mul_ps(normalized, v_weight);
            let result = _mm256_add_ps(scaled, v_bias);

            _mm256_storeu_ps(out.add(offset), result);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i) - reference;
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (pn - shifted_mean) * inv_std * w + b;
        }
    }
}

/// AVX2 Fused Add + Layer Normalization for f64
#[target_feature(enable = "avx2", enable = "fma")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *input.add(row_start) + *residual.add(row_start);
        let v_ref = _mm256_set1_pd(reference);

        let mut sum_acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let v_in = _mm256_loadu_pd(input.add(offset));
            let v_res = _mm256_loadu_pd(residual.add(offset));
            let pn = _mm256_add_pd(v_in, v_res);
            _mm256_storeu_pd(pre_norm.add(offset), pn);
            sum_acc = _mm256_add_pd(sum_acc, _mm256_sub_pd(pn, v_ref));
        }
        let mut sum = hsum_f64(sum_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum += pn - reference;
        }

        let shifted_mean = sum / hidden_size as f64;
        let v_shifted_mean = _mm256_set1_pd(shifted_mean);

        let mut var_acc0 = _mm256_setzero_pd();
        let mut var_acc1 = _mm256_setzero_pd();
        let mut c = 0;
        let chunk_pairs_v = chunks / 2 * 2;
        while c < chunk_pairs_v {
            let diff0 = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + c * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_pd(diff0, diff0, var_acc0);
            let diff1 = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + (c + 1) * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc1 = _mm256_fmadd_pd(diff1, diff1, var_acc1);
            c += 2;
        }
        while c < chunks {
            let diff = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + c * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_pd(diff, diff, var_acc0);
            c += 1;
        }
        let mut var_sum = hsum_f64(_mm256_add_pd(var_acc0, var_acc1));

        for i in (chunks * F64_LANES)..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();
        let v_inv_std = _mm256_set1_pd(inv_std);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let pn = _mm256_sub_pd(_mm256_loadu_pd(pre_norm.add(offset)), v_ref);
            let v_weight = _mm256_loadu_pd(weight.add(w_offset));
            let v_bias = _mm256_loadu_pd(bias.add(w_offset));

            let diff = _mm256_sub_pd(pn, v_shifted_mean);
            let normalized = _mm256_mul_pd(diff, v_inv_std);
            let scaled = _mm256_mul_pd(normalized, v_weight);
            let result = _mm256_add_pd(scaled, v_bias);

            _mm256_storeu_pd(out.add(offset), result);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let pn = *pre_norm.add(row_start + i) - reference;
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = (pn - shifted_mean) * inv_std * w + b;
        }
    }
}

/// AVX2 Fused Add + Layer Norm Backward for f32
///
/// Computes gradients for backward pass of layer norm
#[target_feature(enable = "avx2", enable = "fma")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pre_norm.add(row_start);
        let v_ref = _mm256_set1_ps(reference);

        // Recompute mean from pre_norm
        let mut sum_acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let pn = _mm256_sub_ps(_mm256_loadu_ps(pre_norm.add(offset)), v_ref);
            sum_acc = _mm256_add_ps(sum_acc, pn);
        }
        let mut sum = hsum_f32(sum_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            sum += *pre_norm.add(row_start + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f32;
        let v_shifted_mean = _mm256_set1_ps(shifted_mean);

        // Recompute variance (dual accumulators)
        let mut var_acc0 = _mm256_setzero_ps();
        let mut var_acc1 = _mm256_setzero_ps();
        let mut c = 0;
        let chunk_pairs_v = chunks / 2 * 2;
        while c < chunk_pairs_v {
            let diff0 = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + c * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_ps(diff0, diff0, var_acc0);
            let diff1 = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + (c + 1) * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc1 = _mm256_fmadd_ps(diff1, diff1, var_acc1);
            c += 2;
        }
        while c < chunks {
            let diff = _mm256_sub_ps(
                _mm256_sub_ps(
                    _mm256_loadu_ps(pre_norm.add(row_start + c * F32_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_ps(diff, diff, var_acc0);
            c += 1;
        }
        let mut var_sum = hsum_f32(_mm256_add_ps(var_acc0, var_acc1));

        for i in (chunks * F32_LANES)..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        // Compute mean_gs = mean(grad * weight) and mean_gs_n = mean(grad * weight * normalized)
        let mut gs_acc = _mm256_setzero_ps();
        let mut gsn_acc = _mm256_setzero_ps();
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let g = _mm256_loadu_ps(grad.add(offset));
            let w = _mm256_loadu_ps(weight.add(w_offset));
            let pn = _mm256_sub_ps(_mm256_loadu_ps(pre_norm.add(offset)), v_ref);

            let gs = _mm256_mul_ps(g, w);
            gs_acc = _mm256_add_ps(gs_acc, gs);

            let diff = _mm256_sub_ps(pn, v_shifted_mean);
            let normalized = _mm256_mul_ps(diff, _mm256_set1_ps(inv_std));
            let gsn = _mm256_mul_ps(gs, normalized);
            gsn_acc = _mm256_add_ps(gsn_acc, gsn);
        }
        let mut mean_gs_simd = hsum_f32(gs_acc);
        let mut mean_gsn_simd = hsum_f32(gsn_acc);

        for i in (chunks * F32_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i) - reference;

            let gs = g * w;
            mean_gs_simd += gs;

            let normalized = (pn - shifted_mean) * inv_std;
            mean_gsn_simd += gs * normalized;
        }

        let mean_gs = mean_gs_simd / hidden_size as f32;
        let mean_gs_n = mean_gsn_simd / hidden_size as f32;
        let v_inv_std = _mm256_set1_ps(inv_std);
        let v_shifted_mean_gs = _mm256_set1_ps(mean_gs);
        let v_shifted_mean_gs_n = _mm256_set1_ps(mean_gs_n);

        // Apply and accumulate
        for c in 0..chunks {
            let offset = row_start + c * F32_LANES;
            let w_offset = c * F32_LANES;
            let g = _mm256_loadu_ps(grad.add(offset));
            let w = _mm256_loadu_ps(weight.add(w_offset));
            let pn = _mm256_sub_ps(_mm256_loadu_ps(pre_norm.add(offset)), v_ref);

            let normalized = _mm256_mul_ps(_mm256_sub_ps(pn, v_shifted_mean), v_inv_std);
            let gs = _mm256_mul_ps(g, w);
            let d_ir = _mm256_mul_ps(
                v_inv_std,
                _mm256_sub_ps(
                    gs,
                    _mm256_add_ps(
                        v_shifted_mean_gs,
                        _mm256_mul_ps(normalized, v_shifted_mean_gs_n),
                    ),
                ),
            );
            _mm256_storeu_ps(d_input_residual.add(offset), d_ir);

            // d_weight += g * normalized
            let dw_old = _mm256_loadu_ps(d_weight.add(w_offset));
            let dw_add = _mm256_mul_ps(g, normalized);
            let dw_new = _mm256_add_ps(dw_old, dw_add);
            _mm256_storeu_ps(d_weight.add(w_offset), dw_new);

            // d_bias += g
            let db_old = _mm256_loadu_ps(d_bias.add(w_offset));
            let db_new = _mm256_add_ps(db_old, g);
            _mm256_storeu_ps(d_bias.add(w_offset), db_new);
        }

        for i in (chunks * F32_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i) - reference;

            let normalized = (pn - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_input_residual.add(row_start + i) = d_ir;

            *d_weight.add(i) += g * normalized;
            *d_bias.add(i) += g;
        }
    }
}

/// AVX2 Fused Add + Layer Norm Backward for f64
#[target_feature(enable = "avx2", enable = "fma")]
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

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;
        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pre_norm.add(row_start);
        let v_ref = _mm256_set1_pd(reference);

        let mut sum_acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let pn = _mm256_sub_pd(_mm256_loadu_pd(pre_norm.add(offset)), v_ref);
            sum_acc = _mm256_add_pd(sum_acc, pn);
        }
        let mut sum = hsum_f64(sum_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            sum += *pre_norm.add(row_start + i) - reference;
        }

        let shifted_mean = sum / hidden_size as f64;
        let v_shifted_mean = _mm256_set1_pd(shifted_mean);

        let mut var_acc0 = _mm256_setzero_pd();
        let mut var_acc1 = _mm256_setzero_pd();
        let mut c = 0;
        let chunk_pairs_v = chunks / 2 * 2;
        while c < chunk_pairs_v {
            let diff0 = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + c * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_pd(diff0, diff0, var_acc0);
            let diff1 = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + (c + 1) * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc1 = _mm256_fmadd_pd(diff1, diff1, var_acc1);
            c += 2;
        }
        while c < chunks {
            let diff = _mm256_sub_pd(
                _mm256_sub_pd(
                    _mm256_loadu_pd(pre_norm.add(row_start + c * F64_LANES)),
                    v_ref,
                ),
                v_shifted_mean,
            );
            var_acc0 = _mm256_fmadd_pd(diff, diff, var_acc0);
            c += 1;
        }
        let mut var_sum = hsum_f64(_mm256_add_pd(var_acc0, var_acc1));

        for i in (chunks * F64_LANES)..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }

        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        let mut gs_acc = _mm256_setzero_pd();
        let mut gsn_acc = _mm256_setzero_pd();
        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let g = _mm256_loadu_pd(grad.add(offset));
            let w = _mm256_loadu_pd(weight.add(w_offset));
            let pn = _mm256_sub_pd(_mm256_loadu_pd(pre_norm.add(offset)), v_ref);

            let gs = _mm256_mul_pd(g, w);
            gs_acc = _mm256_add_pd(gs_acc, gs);

            let diff = _mm256_sub_pd(pn, v_shifted_mean);
            let normalized = _mm256_mul_pd(diff, _mm256_set1_pd(inv_std));
            let gsn = _mm256_mul_pd(gs, normalized);
            gsn_acc = _mm256_add_pd(gsn_acc, gsn);
        }
        let mut mean_gs_simd = hsum_f64(gs_acc);
        let mut mean_gsn_simd = hsum_f64(gsn_acc);

        for i in (chunks * F64_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i) - reference;

            let gs = g * w;
            mean_gs_simd += gs;

            let normalized = (pn - shifted_mean) * inv_std;
            mean_gsn_simd += gs * normalized;
        }

        let mean_gs = mean_gs_simd / hidden_size as f64;
        let mean_gs_n = mean_gsn_simd / hidden_size as f64;
        let v_inv_std = _mm256_set1_pd(inv_std);
        let v_shifted_mean_gs = _mm256_set1_pd(mean_gs);
        let v_shifted_mean_gs_n = _mm256_set1_pd(mean_gs_n);

        for c in 0..chunks {
            let offset = row_start + c * F64_LANES;
            let w_offset = c * F64_LANES;
            let g = _mm256_loadu_pd(grad.add(offset));
            let w = _mm256_loadu_pd(weight.add(w_offset));
            let pn = _mm256_sub_pd(_mm256_loadu_pd(pre_norm.add(offset)), v_ref);

            let normalized = _mm256_mul_pd(_mm256_sub_pd(pn, v_shifted_mean), v_inv_std);
            let gs = _mm256_mul_pd(g, w);
            let d_ir = _mm256_mul_pd(
                v_inv_std,
                _mm256_sub_pd(
                    gs,
                    _mm256_add_pd(
                        v_shifted_mean_gs,
                        _mm256_mul_pd(normalized, v_shifted_mean_gs_n),
                    ),
                ),
            );
            _mm256_storeu_pd(d_input_residual.add(offset), d_ir);

            let dw_old = _mm256_loadu_pd(d_weight.add(w_offset));
            let dw_add = _mm256_mul_pd(g, normalized);
            let dw_new = _mm256_add_pd(dw_old, dw_add);
            _mm256_storeu_pd(d_weight.add(w_offset), dw_new);

            let db_old = _mm256_loadu_pd(d_bias.add(w_offset));
            let db_new = _mm256_add_pd(db_old, g);
            _mm256_storeu_pd(d_bias.add(w_offset), db_new);
        }

        for i in (chunks * F64_LANES)..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i) - reference;

            let normalized = (pn - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_input_residual.add(row_start + i) = d_ir;

            *d_weight.add(i) += g * normalized;
            *d_bias.add(i) += g;
        }
    }
}
