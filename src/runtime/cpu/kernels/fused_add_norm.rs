//! Fused Add + Normalization kernels
//!
//! Provides fused add+norm operations with automatic SIMD dispatch.

use crate::dtype::Element;

/// Fused Add + RMS Norm kernel: pre_norm = input + residual, output = rms_norm(pre_norm)
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_kernel<T: Element>(
    input: *const T,
    residual: *const T,
    weight: *const T,
    out: *mut T,
    pre_norm: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;
        match T::DTYPE {
            DType::F32 => {
                norm::fused_add_rms_norm_f32(
                    input as *const f32,
                    residual as *const f32,
                    weight as *const f32,
                    out as *mut f32,
                    pre_norm as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::fused_add_rms_norm_f64(
                    input as *const f64,
                    residual as *const f64,
                    weight as *const f64,
                    out as *mut f64,
                    pre_norm as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::fused_add_rms_norm_f16(
                    input as *const half::f16,
                    residual as *const half::f16,
                    weight as *const half::f16,
                    out as *mut half::f16,
                    pre_norm as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::fused_add_rms_norm_bf16(
                    input as *const half::bf16,
                    residual as *const half::bf16,
                    weight as *const half::bf16,
                    out as *mut half::bf16,
                    pre_norm as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {}
        }
    }
    fused_add_rms_norm_scalar(
        input,
        residual,
        weight,
        out,
        pre_norm,
        batch_size,
        hidden_size,
        eps,
    );
}

#[inline]
unsafe fn fused_add_rms_norm_scalar<T: Element>(
    input: *const T,
    residual: *const T,
    weight: *const T,
    out: *mut T,
    pre_norm_out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let eps = eps as f64;
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    for batch in 0..batch_size {
        let row = batch * hidden_size;
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let pn = (*input.add(row + i)).to_f64() + (*residual.add(row + i)).to_f64();
            *pre_norm_out.add(row + i) = T::from_f64(pn);
            sum_sq += pn * pn;
        }
        let inv_rms = 1.0 / (sum_sq / hidden_size as f64 + eps).sqrt();
        for (i, &w) in weight_slice.iter().enumerate() {
            let pn = (*pre_norm_out.add(row + i)).to_f64();
            *out.add(row + i) = T::from_f64(pn * inv_rms * w.to_f64());
        }
    }
}

/// Backward pass for fused add + RMS norm
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_rms_norm_bwd_kernel<T: Element>(
    grad: *const T,
    pre_norm: *const T,
    weight: *const T,
    d_input_residual: *mut T,
    d_weight: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;
        match T::DTYPE {
            DType::F32 => {
                norm::fused_add_rms_norm_bwd_f32(
                    grad as *const f32,
                    pre_norm as *const f32,
                    weight as *const f32,
                    d_input_residual as *mut f32,
                    d_weight as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::fused_add_rms_norm_bwd_f64(
                    grad as *const f64,
                    pre_norm as *const f64,
                    weight as *const f64,
                    d_input_residual as *mut f64,
                    d_weight as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::fused_add_rms_norm_bwd_f16(
                    grad as *const half::f16,
                    pre_norm as *const half::f16,
                    weight as *const half::f16,
                    d_input_residual as *mut half::f16,
                    d_weight as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::fused_add_rms_norm_bwd_bf16(
                    grad as *const half::bf16,
                    pre_norm as *const half::bf16,
                    weight as *const half::bf16,
                    d_input_residual as *mut half::bf16,
                    d_weight as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {}
        }
    }
    fused_add_rms_norm_bwd_scalar(
        grad,
        pre_norm,
        weight,
        d_input_residual,
        d_weight,
        batch_size,
        hidden_size,
        eps,
    );
}

#[inline]
unsafe fn fused_add_rms_norm_bwd_scalar<T: Element>(
    grad: *const T,
    pre_norm: *const T,
    weight: *const T,
    d_input_residual: *mut T,
    d_weight: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let eps = eps as f64;
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    // d_weight is pre-zeroed by caller
    for batch in 0..batch_size {
        let row = batch * hidden_size;
        // Recompute inv_rms
        let mut sum_sq = 0.0f64;
        for i in 0..hidden_size {
            let pn = (*pre_norm.add(row + i)).to_f64();
            sum_sq += pn * pn;
        }
        let mean_sq = sum_sq / hidden_size as f64;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        // Compute dot = sum(grad * weight * pre_norm)
        let mut dot = 0.0f64;
        for i in 0..hidden_size {
            let g = (*grad.add(row + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let pn = (*pre_norm.add(row + i)).to_f64();
            dot += g * w * pn;
        }
        let coeff = dot * inv_rms / (hidden_size as f64 * (mean_sq + eps));
        for i in 0..hidden_size {
            let g = (*grad.add(row + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let pn = (*pre_norm.add(row + i)).to_f64();
            let d_ir = (g * w - pn * coeff) * inv_rms;
            *d_input_residual.add(row + i) = T::from_f64(d_ir);
            // Accumulate d_weight
            let dw_old = (*d_weight.add(i)).to_f64();
            *d_weight.add(i) = T::from_f64(dw_old + g * pn * inv_rms);
        }
    }
}

/// Fused Add + Layer Norm kernel
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_kernel<T: Element>(
    input: *const T,
    residual: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    pre_norm: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;
        match T::DTYPE {
            DType::F32 => {
                norm::fused_add_layer_norm_f32(
                    input as *const f32,
                    residual as *const f32,
                    weight as *const f32,
                    bias as *const f32,
                    out as *mut f32,
                    pre_norm as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::fused_add_layer_norm_f64(
                    input as *const f64,
                    residual as *const f64,
                    weight as *const f64,
                    bias as *const f64,
                    out as *mut f64,
                    pre_norm as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::fused_add_layer_norm_f16(
                    input as *const half::f16,
                    residual as *const half::f16,
                    weight as *const half::f16,
                    bias as *const half::f16,
                    out as *mut half::f16,
                    pre_norm as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::fused_add_layer_norm_bf16(
                    input as *const half::bf16,
                    residual as *const half::bf16,
                    weight as *const half::bf16,
                    bias as *const half::bf16,
                    out as *mut half::bf16,
                    pre_norm as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {}
        }
    }
    fused_add_layer_norm_scalar(
        input,
        residual,
        weight,
        bias,
        out,
        pre_norm,
        batch_size,
        hidden_size,
        eps,
    );
}

#[inline]
unsafe fn fused_add_layer_norm_scalar<T: Element>(
    input: *const T,
    residual: *const T,
    weight: *const T,
    bias: *const T,
    out: *mut T,
    pre_norm_out: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let eps = eps as f64;
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    let bias_slice = std::slice::from_raw_parts(bias, hidden_size);
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = (*input.add(row)).to_f64() + (*residual.add(row)).to_f64();

        // Pass 1: add + compute mean
        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            let pn = (*input.add(row + i)).to_f64() + (*residual.add(row + i)).to_f64();
            *pre_norm_out.add(row + i) = T::from_f64(pn);
            sum += pn - reference;
        }
        let shifted_mean = sum / hidden_size as f64;
        // Pass 2: variance
        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let pn = (*pre_norm_out.add(row + i)).to_f64();
            let diff = (pn - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();
        // Pass 3: normalize
        for i in 0..hidden_size {
            let pn = (*pre_norm_out.add(row + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let b = bias_slice[i].to_f64();
            *out.add(row + i) = T::from_f64(((pn - reference) - shifted_mean) * inv_std * w + b);
        }
    }
}

/// Backward pass for fused add + layer norm
#[inline]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fused_add_layer_norm_bwd_kernel<T: Element>(
    grad: *const T,
    pre_norm: *const T,
    weight: *const T,
    _bias: *const T,
    d_input_residual: *mut T,
    d_weight: *mut T,
    d_bias: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        use super::simd::norm;
        use crate::dtype::DType;
        match T::DTYPE {
            DType::F32 => {
                norm::fused_add_layer_norm_bwd_f32(
                    grad as *const f32,
                    pre_norm as *const f32,
                    weight as *const f32,
                    d_input_residual as *mut f32,
                    d_weight as *mut f32,
                    d_bias as *mut f32,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            DType::F64 => {
                norm::fused_add_layer_norm_bwd_f64(
                    grad as *const f64,
                    pre_norm as *const f64,
                    weight as *const f64,
                    d_input_residual as *mut f64,
                    d_weight as *mut f64,
                    d_bias as *mut f64,
                    batch_size,
                    hidden_size,
                    eps as f64,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                norm::fused_add_layer_norm_bwd_f16(
                    grad as *const half::f16,
                    pre_norm as *const half::f16,
                    weight as *const half::f16,
                    d_input_residual as *mut half::f16,
                    d_weight as *mut half::f16,
                    d_bias as *mut half::f16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                norm::fused_add_layer_norm_bwd_bf16(
                    grad as *const half::bf16,
                    pre_norm as *const half::bf16,
                    weight as *const half::bf16,
                    d_input_residual as *mut half::bf16,
                    d_weight as *mut half::bf16,
                    d_bias as *mut half::bf16,
                    batch_size,
                    hidden_size,
                    eps,
                );
                return;
            }
            _ => {}
        }
    }
    fused_add_layer_norm_bwd_scalar(
        grad,
        pre_norm,
        weight,
        d_input_residual,
        d_weight,
        d_bias,
        batch_size,
        hidden_size,
        eps,
    );
}

#[inline]
unsafe fn fused_add_layer_norm_bwd_scalar<T: Element>(
    grad: *const T,
    pre_norm: *const T,
    weight: *const T,
    d_input_residual: *mut T,
    d_weight: *mut T,
    d_bias: *mut T,
    batch_size: usize,
    hidden_size: usize,
    eps: f32,
) {
    let eps = eps as f64;
    let weight_slice = std::slice::from_raw_parts(weight, hidden_size);
    // d_weight and d_bias are pre-zeroed
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = (*pre_norm.add(row)).to_f64();

        // Recompute mean and inv_std from pre_norm
        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += (*pre_norm.add(row + i)).to_f64() - reference;
        }
        let shifted_mean = sum / hidden_size as f64;
        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let diff = ((*pre_norm.add(row + i)).to_f64() - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        // Compute intermediate sums for d_input_residual
        let mut mean_gs = 0.0f64;
        let mut mean_gs_n = 0.0f64;
        for i in 0..hidden_size {
            let g = (*grad.add(row + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let pn = (*pre_norm.add(row + i)).to_f64();
            let normalized = ((pn - reference) - shifted_mean) * inv_std;
            let gs = g * w;
            mean_gs += gs;
            mean_gs_n += gs * normalized;
        }
        mean_gs /= hidden_size as f64;
        mean_gs_n /= hidden_size as f64;

        for i in 0..hidden_size {
            let g = (*grad.add(row + i)).to_f64();
            let w = weight_slice[i].to_f64();
            let pn = (*pre_norm.add(row + i)).to_f64();
            let normalized = ((pn - reference) - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_input_residual.add(row + i) = T::from_f64(d_ir);
            // Accumulate d_weight and d_bias
            let dw_old = (*d_weight.add(i)).to_f64();
            *d_weight.add(i) = T::from_f64(dw_old + g * normalized);
            let db_old = (*d_bias.add(i)).to_f64();
            *d_bias.add(i) = T::from_f64(db_old + g);
        }
    }
}
