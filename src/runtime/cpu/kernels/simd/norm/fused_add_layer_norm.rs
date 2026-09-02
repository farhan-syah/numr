//! SIMD dispatch and scalar fallbacks for fused Add + Layer normalization

use super::super::{SimdLevel, detect_simd};
use super::SIMD_THRESHOLD;

#[cfg(target_arch = "x86_64")]
use super::avx2;
#[cfg(target_arch = "x86_64")]
use super::avx512;

#[cfg(target_arch = "aarch64")]
use super::aarch64;

// ============================================================================
// Fused Add + Layer Norm (forward)
// ============================================================================

/// SIMD Fused Add + Layer Normalization for f32
#[inline]
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
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_layer_norm_scalar_f32(
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
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_layer_norm_f32(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_layer_norm_f32(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_scalar_f32(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_layer_norm_f32(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_scalar_f32(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_layer_norm_scalar_f32(
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

/// SIMD Fused Add + Layer Normalization for f64
#[inline]
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
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_layer_norm_scalar_f64(
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
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_layer_norm_f64(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_layer_norm_f64(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_scalar_f64(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_layer_norm_f64(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_scalar_f64(
            input,
            residual,
            weight,
            bias,
            out,
            pre_norm,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_layer_norm_scalar_f64(
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

// ============================================================================
// Fused Add + Layer Norm (backward)
// ============================================================================

/// SIMD Fused Add + Layer Norm Backward for f32
#[inline]
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
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_layer_norm_bwd_scalar_f32(
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
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_layer_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_layer_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_bwd_scalar_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_layer_norm_bwd_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_bwd_scalar_f32(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_layer_norm_bwd_scalar_f32(
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

/// SIMD Fused Add + Layer Norm Backward for f64
#[inline]
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
    let level = detect_simd();

    if hidden_size < SIMD_THRESHOLD || level == SimdLevel::Scalar {
        fused_add_layer_norm_bwd_scalar_f64(
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
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::fused_add_layer_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        SimdLevel::Avx2Fma => avx2::fused_add_layer_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_bwd_scalar_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => aarch64::neon::fused_add_layer_norm_bwd_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
        _ => fused_add_layer_norm_bwd_scalar_f64(
            grad,
            pre_norm,
            weight,
            d_input_residual,
            d_weight,
            d_bias,
            batch_size,
            hidden_size,
            eps,
        ),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fused_add_layer_norm_bwd_scalar_f64(
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

// ============================================================================
// Scalar fallbacks for fused add + layer norm
// ============================================================================

/// Scalar fused add + layer norm for f32
#[inline]
pub unsafe fn fused_add_layer_norm_scalar_f32(
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
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *input.add(row_start) + *residual.add(row_start);

        // Add and store pre_norm, compute mean
        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum += pn - reference;
        }
        let shifted_mean = sum / hidden_size as f32;

        // Compute variance
        let mut var_sum = 0.0f32;
        for i in 0..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        // Normalize, apply weight and bias
        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = ((pn - reference) - shifted_mean) * inv_std * w + b;
        }
    }
}

/// Scalar fused add + layer norm for f64
#[inline]
pub unsafe fn fused_add_layer_norm_scalar_f64(
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
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. Subtracting the mean cancels catastrophically when a
        // row sits far from zero relative to its own spread, which is where the
        // accumulator's precision goes. Mean, variance and the normalized value
        // are all invariant under the shift in exact arithmetic, and every other
        // backend shifts by the same element.
        let reference = *input.add(row_start) + *residual.add(row_start);

        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            let pn = *input.add(row_start + i) + *residual.add(row_start + i);
            *pre_norm.add(row_start + i) = pn;
            sum += pn - reference;
        }
        let shifted_mean = sum / hidden_size as f64;

        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        for i in 0..hidden_size {
            let pn = *pre_norm.add(row_start + i);
            let w = *weight.add(i);
            let b = *bias.add(i);
            *out.add(row_start + i) = ((pn - reference) - shifted_mean) * inv_std * w + b;
        }
    }
}

/// Scalar fused add + layer norm backward for f32
#[inline]
pub unsafe fn fused_add_layer_norm_bwd_scalar_f32(
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
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pre_norm.add(row_start);

        let mut sum = 0.0f32;
        for i in 0..hidden_size {
            sum += *pre_norm.add(row_start + i) - reference;
        }
        let shifted_mean = sum / hidden_size as f32;

        let mut var_sum = 0.0f32;
        for i in 0..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f32 + eps).sqrt();

        let mut mean_gs = 0.0f32;
        let mut mean_gs_n = 0.0f32;
        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            let gs = g * w;
            mean_gs += gs;
            mean_gs_n += gs * ((pn - reference) - shifted_mean) * inv_std;
        }
        mean_gs /= hidden_size as f32;
        mean_gs_n /= hidden_size as f32;

        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            let normalized = ((pn - reference) - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_input_residual.add(row_start + i) = d_ir;

            *d_weight.add(i) += g * normalized;
            *d_bias.add(i) += g;
        }
    }
}

/// Scalar fused add + layer norm backward for f64
#[inline]
pub unsafe fn fused_add_layer_norm_bwd_scalar_f64(
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
    // An empty row normalizes nothing, and the reference load below would read
    // past an empty allocation.
    if hidden_size == 0 {
        return;
    }

    for batch in 0..batch_size {
        let row_start = batch * hidden_size;

        // Every element is shifted by the row's first pre-norm value before it
        // is accumulated. This pass RECOMPUTES the mean and variance rather than
        // consuming a saved statistic, so it carries the forward pass's
        // cancellation exactly. Mean, variance and the normalized value are all
        // invariant under the shift in exact arithmetic. `mean_gs` needs no
        // shift: it averages `g * w`, which never sees the mean.
        let reference = *pre_norm.add(row_start);

        let mut sum = 0.0f64;
        for i in 0..hidden_size {
            sum += *pre_norm.add(row_start + i) - reference;
        }
        let shifted_mean = sum / hidden_size as f64;

        let mut var_sum = 0.0f64;
        for i in 0..hidden_size {
            let diff = (*pre_norm.add(row_start + i) - reference) - shifted_mean;
            var_sum += diff * diff;
        }
        let inv_std = 1.0 / (var_sum / hidden_size as f64 + eps).sqrt();

        let mut mean_gs = 0.0f64;
        let mut mean_gs_n = 0.0f64;
        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            let gs = g * w;
            mean_gs += gs;
            mean_gs_n += gs * ((pn - reference) - shifted_mean) * inv_std;
        }
        mean_gs /= hidden_size as f64;
        mean_gs_n /= hidden_size as f64;

        for i in 0..hidden_size {
            let g = *grad.add(row_start + i);
            let w = *weight.add(i);
            let pn = *pre_norm.add(row_start + i);
            let normalized = ((pn - reference) - shifted_mean) * inv_std;
            let gs = g * w;
            let d_ir = inv_std * (gs - mean_gs - normalized * mean_gs_n);
            *d_input_residual.add(row_start + i) = d_ir;

            *d_weight.add(i) += g * normalized;
            *d_bias.add(i) += g;
        }
    }
}
