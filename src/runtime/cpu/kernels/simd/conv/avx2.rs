//! AVX2 2D convolution kernels
//!
//! `conv2d` vectorizes over input channels using FMA instructions.
//! - f32: 8 channels per iteration
//! - f64: 4 channels per iteration
//!
//! `depthwise_conv2d` has one channel per group, so it vectorizes over output
//! width instead. (`conv1d` lives in the `conv1d` submodule, which vectorizes
//! over output positions for every shape.)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ops::conv_common::Conv2dParams;

const F32_LANES: usize = 8;
const F64_LANES: usize = 4;

// ============================================================================
// Horizontal sum helpers
// ============================================================================

/// Horizontal sum of 8 f32 values in AVX2 register
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f32(v: __m256) -> f32 {
    // Extract high 128 bits and add to low 128 bits
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(low, high);
    // Horizontal add within 128 bits
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Horizontal sum of 4 f64 values in AVX2 register
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_f64(v: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(low, high);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}

// ============================================================================
// Conv2d AVX2 kernels
// ============================================================================

/// AVX2 conv2d for f32
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;
    let ic_chunks = c_in_per_group / F32_LANES;
    let ic_remainder = c_in_per_group % F32_LANES;
    let hw = height * width;

    for b in 0..batch {
        for g in 0..groups {
            let c_out_start = g * c_out_per_group;
            let c_in_start = g * c_in_per_group;

            for oc in 0..c_out_per_group {
                let c_out_idx = c_out_start + oc;

                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let mut sum_vec = _mm256_setzero_ps();
                        let mut sum_scalar = 0.0f32;

                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let iy_signed = (oy * stride_h) as isize
                                    + (ky * dilation_h) as isize
                                    - pad_top as isize;
                                let ix_signed = (ox * stride_w) as isize
                                    + (kx * dilation_w) as isize
                                    - pad_left as isize;

                                if iy_signed >= 0
                                    && (iy_signed as usize) < height
                                    && ix_signed >= 0
                                    && (ix_signed as usize) < width
                                {
                                    let iy = iy_signed as usize;
                                    let ix = ix_signed as usize;

                                    // Vectorized loop over input channels
                                    for ic_chunk in 0..ic_chunks {
                                        let ic_base = ic_chunk * F32_LANES;
                                        let c_in_idx = c_in_start + ic_base;

                                        // Input: (b, c_in_idx..+8, iy, ix)
                                        let input_base =
                                            b * c_in * hw + c_in_idx * hw + iy * width + ix;

                                        // Weight: (c_out_idx, ic_base..+8, ky, kx)
                                        let kw = kernel_h * kernel_w;
                                        let weight_base = c_out_idx * c_in_per_group * kw
                                            + ic_base * kw
                                            + ky * kernel_w
                                            + kx;

                                        // Gather 8 input values (strided by hw)
                                        let in0 = *input.add(input_base);
                                        let in1 = *input.add(input_base + hw);
                                        let in2 = *input.add(input_base + 2 * hw);
                                        let in3 = *input.add(input_base + 3 * hw);
                                        let in4 = *input.add(input_base + 4 * hw);
                                        let in5 = *input.add(input_base + 5 * hw);
                                        let in6 = *input.add(input_base + 6 * hw);
                                        let in7 = *input.add(input_base + 7 * hw);
                                        let input_vec =
                                            _mm256_set_ps(in7, in6, in5, in4, in3, in2, in1, in0);

                                        // Gather 8 weight values (strided by kw)
                                        let w0 = *weight.add(weight_base);
                                        let w1 = *weight.add(weight_base + kw);
                                        let w2 = *weight.add(weight_base + 2 * kw);
                                        let w3 = *weight.add(weight_base + 3 * kw);
                                        let w4 = *weight.add(weight_base + 4 * kw);
                                        let w5 = *weight.add(weight_base + 5 * kw);
                                        let w6 = *weight.add(weight_base + 6 * kw);
                                        let w7 = *weight.add(weight_base + 7 * kw);
                                        let weight_vec =
                                            _mm256_set_ps(w7, w6, w5, w4, w3, w2, w1, w0);

                                        sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                                    }

                                    // Scalar remainder
                                    for ic_rem in 0..ic_remainder {
                                        let ic = ic_chunks * F32_LANES + ic_rem;
                                        let c_in_idx = c_in_start + ic;
                                        let input_idx =
                                            b * c_in * hw + c_in_idx * hw + iy * width + ix;
                                        let kw = kernel_h * kernel_w;
                                        let weight_idx = c_out_idx * c_in_per_group * kw
                                            + ic * kw
                                            + ky * kernel_w
                                            + kx;

                                        sum_scalar +=
                                            *input.add(input_idx) * *weight.add(weight_idx);
                                    }
                                }
                            }
                        }

                        let mut sum = hsum_f32(sum_vec) + sum_scalar;

                        if let Some(bias_ptr) = bias {
                            sum += *bias_ptr.add(c_out_idx);
                        }

                        let output_idx = b * c_out * output_h * output_w
                            + c_out_idx * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = sum;
                    }
                }
            }
        }
    }
}

/// AVX2 conv2d for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    let c_in_per_group = c_in / groups;
    let c_out_per_group = c_out / groups;
    let ic_chunks = c_in_per_group / F64_LANES;
    let ic_remainder = c_in_per_group % F64_LANES;
    let hw = height * width;

    for b in 0..batch {
        for g in 0..groups {
            let c_out_start = g * c_out_per_group;
            let c_in_start = g * c_in_per_group;

            for oc in 0..c_out_per_group {
                let c_out_idx = c_out_start + oc;

                for oy in 0..output_h {
                    for ox in 0..output_w {
                        let mut sum_vec = _mm256_setzero_pd();
                        let mut sum_scalar = 0.0f64;

                        for ky in 0..kernel_h {
                            for kx in 0..kernel_w {
                                let iy_signed = (oy * stride_h) as isize
                                    + (ky * dilation_h) as isize
                                    - pad_top as isize;
                                let ix_signed = (ox * stride_w) as isize
                                    + (kx * dilation_w) as isize
                                    - pad_left as isize;

                                if iy_signed >= 0
                                    && (iy_signed as usize) < height
                                    && ix_signed >= 0
                                    && (ix_signed as usize) < width
                                {
                                    let iy = iy_signed as usize;
                                    let ix = ix_signed as usize;

                                    for ic_chunk in 0..ic_chunks {
                                        let ic_base = ic_chunk * F64_LANES;
                                        let c_in_idx = c_in_start + ic_base;
                                        let input_base =
                                            b * c_in * hw + c_in_idx * hw + iy * width + ix;
                                        let kw = kernel_h * kernel_w;
                                        let weight_base = c_out_idx * c_in_per_group * kw
                                            + ic_base * kw
                                            + ky * kernel_w
                                            + kx;

                                        let in0 = *input.add(input_base);
                                        let in1 = *input.add(input_base + hw);
                                        let in2 = *input.add(input_base + 2 * hw);
                                        let in3 = *input.add(input_base + 3 * hw);
                                        let input_vec = _mm256_set_pd(in3, in2, in1, in0);

                                        let w0 = *weight.add(weight_base);
                                        let w1 = *weight.add(weight_base + kw);
                                        let w2 = *weight.add(weight_base + 2 * kw);
                                        let w3 = *weight.add(weight_base + 3 * kw);
                                        let weight_vec = _mm256_set_pd(w3, w2, w1, w0);

                                        sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
                                    }

                                    for ic_rem in 0..ic_remainder {
                                        let ic = ic_chunks * F64_LANES + ic_rem;
                                        let c_in_idx = c_in_start + ic;
                                        let input_idx =
                                            b * c_in * hw + c_in_idx * hw + iy * width + ix;
                                        let kw = kernel_h * kernel_w;
                                        let weight_idx = c_out_idx * c_in_per_group * kw
                                            + ic * kw
                                            + ky * kernel_w
                                            + kx;

                                        sum_scalar +=
                                            *input.add(input_idx) * *weight.add(weight_idx);
                                    }
                                }
                            }
                        }

                        let mut sum = hsum_f64(sum_vec) + sum_scalar;

                        if let Some(bias_ptr) = bias {
                            sum += *bias_ptr.add(c_out_idx);
                        }

                        let output_idx = b * c_out * output_h * output_w
                            + c_out_idx * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = sum;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Depthwise Conv2d AVX2 kernels
// ============================================================================

/// AVX2 depthwise conv2d for f32
///
/// For depthwise convolution, we vectorize over output width positions
/// since there's only 1 input channel per group.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn depthwise_conv2d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv2dParams,
) {
    let Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out: _,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups: _,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    let ox_chunks = output_w / F32_LANES;
    let ox_remainder = output_w % F32_LANES;

    for b in 0..batch {
        for c in 0..c_in {
            for oy in 0..output_h {
                // Vectorized loop over output width
                for ox_chunk in 0..ox_chunks {
                    let ox_base = ox_chunk * F32_LANES;

                    let mut sum_vec = _mm256_setzero_ps();

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;

                            if iy_signed >= 0 && (iy_signed as usize) < height {
                                let iy = iy_signed as usize;

                                // Load weight value (same for all 8 output positions)
                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                                let w_val = *weight.add(weight_idx);
                                let weight_vec = _mm256_set1_ps(w_val);

                                // Gather 8 input values at different x positions
                                let mut in_vals = [0.0f32; F32_LANES];
                                for i in 0..F32_LANES {
                                    let ox = ox_base + i;
                                    let ix_signed = (ox * stride_w) as isize
                                        + (kx * dilation_w) as isize
                                        - pad_left as isize;

                                    if ix_signed >= 0 && (ix_signed as usize) < width {
                                        let ix = ix_signed as usize;
                                        let input_idx = b * c_in * height * width
                                            + c * height * width
                                            + iy * width
                                            + ix;
                                        in_vals[i] = *input.add(input_idx);
                                    }
                                    // Zero padding already initialized
                                }

                                let input_vec = _mm256_loadu_ps(in_vals.as_ptr());
                                sum_vec = _mm256_fmadd_ps(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }

                    // Add bias if present
                    if let Some(bias_ptr) = bias {
                        let bias_val = *bias_ptr.add(c);
                        sum_vec = _mm256_add_ps(sum_vec, _mm256_set1_ps(bias_val));
                    }

                    // Store 8 output values
                    let mut out_vals = [0.0f32; F32_LANES];
                    _mm256_storeu_ps(out_vals.as_mut_ptr(), sum_vec);

                    for i in 0..F32_LANES {
                        let ox = ox_base + i;
                        let output_idx = b * c_in * output_h * output_w
                            + c * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = out_vals[i];
                    }
                }

                // Scalar remainder for output width
                for ox_rem in 0..ox_remainder {
                    let ox = ox_chunks * F32_LANES + ox_rem;
                    let mut sum = 0.0f32;

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;
                            let ix_signed = (ox * stride_w) as isize + (kx * dilation_w) as isize
                                - pad_left as isize;

                            if iy_signed >= 0
                                && (iy_signed as usize) < height
                                && ix_signed >= 0
                                && (ix_signed as usize) < width
                            {
                                let iy = iy_signed as usize;
                                let ix = ix_signed as usize;

                                let input_idx = b * c_in * height * width
                                    + c * height * width
                                    + iy * width
                                    + ix;
                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;

                                sum += *input.add(input_idx) * *weight.add(weight_idx);
                            }
                        }
                    }

                    if let Some(bias_ptr) = bias {
                        sum += *bias_ptr.add(c);
                    }

                    let output_idx = b * c_in * output_h * output_w
                        + c * output_h * output_w
                        + oy * output_w
                        + ox;
                    *output.add(output_idx) = sum;
                }
            }
        }
    }
}

/// AVX2 depthwise conv2d for f64
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn depthwise_conv2d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv2dParams,
) {
    let Conv2dParams {
        batch,
        c_in,
        height,
        width,
        c_out: _,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        groups: _,
        pad_top,
        pad_bottom: _,
        pad_left,
        pad_right: _,
        output_h,
        output_w,
    } = params;

    let ox_chunks = output_w / F64_LANES;
    let ox_remainder = output_w % F64_LANES;

    for b in 0..batch {
        for c in 0..c_in {
            for oy in 0..output_h {
                for ox_chunk in 0..ox_chunks {
                    let ox_base = ox_chunk * F64_LANES;
                    let mut sum_vec = _mm256_setzero_pd();

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;

                            if iy_signed >= 0 && (iy_signed as usize) < height {
                                let iy = iy_signed as usize;

                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                                let w_val = *weight.add(weight_idx);
                                let weight_vec = _mm256_set1_pd(w_val);

                                let mut in_vals = [0.0f64; F64_LANES];
                                for i in 0..F64_LANES {
                                    let ox = ox_base + i;
                                    let ix_signed = (ox * stride_w) as isize
                                        + (kx * dilation_w) as isize
                                        - pad_left as isize;

                                    if ix_signed >= 0 && (ix_signed as usize) < width {
                                        let ix = ix_signed as usize;
                                        let input_idx = b * c_in * height * width
                                            + c * height * width
                                            + iy * width
                                            + ix;
                                        in_vals[i] = *input.add(input_idx);
                                    }
                                }

                                let input_vec = _mm256_loadu_pd(in_vals.as_ptr());
                                sum_vec = _mm256_fmadd_pd(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }

                    if let Some(bias_ptr) = bias {
                        let bias_val = *bias_ptr.add(c);
                        sum_vec = _mm256_add_pd(sum_vec, _mm256_set1_pd(bias_val));
                    }

                    let mut out_vals = [0.0f64; F64_LANES];
                    _mm256_storeu_pd(out_vals.as_mut_ptr(), sum_vec);

                    for i in 0..F64_LANES {
                        let ox = ox_base + i;
                        let output_idx = b * c_in * output_h * output_w
                            + c * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = out_vals[i];
                    }
                }

                for ox_rem in 0..ox_remainder {
                    let ox = ox_chunks * F64_LANES + ox_rem;
                    let mut sum = 0.0f64;

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;
                            let ix_signed = (ox * stride_w) as isize + (kx * dilation_w) as isize
                                - pad_left as isize;

                            if iy_signed >= 0
                                && (iy_signed as usize) < height
                                && ix_signed >= 0
                                && (ix_signed as usize) < width
                            {
                                let iy = iy_signed as usize;
                                let ix = ix_signed as usize;

                                let input_idx = b * c_in * height * width
                                    + c * height * width
                                    + iy * width
                                    + ix;
                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;

                                sum += *input.add(input_idx) * *weight.add(weight_idx);
                            }
                        }
                    }

                    if let Some(bias_ptr) = bias {
                        sum += *bias_ptr.add(c);
                    }

                    let output_idx = b * c_in * output_h * output_w
                        + c * output_h * output_w
                        + oy * output_w
                        + ox;
                    *output.add(output_idx) = sum;
                }
            }
        }
    }
}
