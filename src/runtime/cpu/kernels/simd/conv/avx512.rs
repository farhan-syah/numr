//! AVX-512 2D convolution kernels
//!
//! `conv2d` vectorizes over input channels using FMA instructions.
//! - f32: 16 channels per iteration
//! - f64: 8 channels per iteration
//!
//! `depthwise_conv2d` has one channel per group, so it vectorizes over output
//! width instead. (`conv1d` lives in the `conv1d` submodule, which vectorizes
//! over output positions for every shape.)

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use crate::ops::conv_common::Conv2dParams;

const F32_LANES: usize = 16;
const F64_LANES: usize = 8;

// ============================================================================
// Horizontal sum helpers
// ============================================================================

/// Horizontal sum of 16 f32 values in AVX-512 register
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_f32(v: __m512) -> f32 {
    // Extract low and high 256-bit halves, add them
    let low = _mm512_castps512_ps256(v);
    let high = _mm512_extractf32x8_ps::<1>(v);
    let sum256 = _mm256_add_ps(low, high);

    // Now reduce 256 bits to scalar
    let high128 = _mm256_extractf128_ps(sum256, 1);
    let low128 = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(low128, high128);
    let shuf = _mm_movehdup_ps(sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sum64, sum64);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

/// Horizontal sum of 8 f64 values in AVX-512 register
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn hsum_f64(v: __m512d) -> f64 {
    let low = _mm512_castpd512_pd256(v);
    let high = _mm512_extractf64x4_pd::<1>(v);
    let sum256 = _mm256_add_pd(low, high);

    // Reduce 256 bits
    let high128 = _mm256_extractf128_pd(sum256, 1);
    let low128 = _mm256_castpd256_pd128(sum256);
    let sum128 = _mm_add_pd(low128, high128);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(sum64)
}

// ============================================================================
// Conv2d AVX-512 kernels
// ============================================================================

/// AVX-512 conv2d for f32
#[target_feature(enable = "avx512f")]
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
                        let mut sum_vec = _mm512_setzero_ps();
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

                                    for ic_chunk in 0..ic_chunks {
                                        let ic_base = ic_chunk * F32_LANES;
                                        let c_in_idx = c_in_start + ic_base;
                                        let input_base =
                                            b * c_in * hw + c_in_idx * hw + iy * width + ix;
                                        let kw = kernel_h * kernel_w;
                                        let weight_base = c_out_idx * c_in_per_group * kw
                                            + ic_base * kw
                                            + ky * kernel_w
                                            + kx;

                                        // Gather 16 values
                                        let mut in_vals = [0.0f32; F32_LANES];
                                        for i in 0..F32_LANES {
                                            in_vals[i] = *input.add(input_base + i * hw);
                                        }
                                        let input_vec = _mm512_loadu_ps(in_vals.as_ptr());

                                        let mut w_vals = [0.0f32; F32_LANES];
                                        for i in 0..F32_LANES {
                                            w_vals[i] = *weight.add(weight_base + i * kw);
                                        }
                                        let weight_vec = _mm512_loadu_ps(w_vals.as_ptr());

                                        sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
                                    }

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

/// AVX-512 conv2d for f64
#[target_feature(enable = "avx512f")]
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
                        let mut sum_vec = _mm512_setzero_pd();
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

                                        let mut in_vals = [0.0f64; F64_LANES];
                                        for i in 0..F64_LANES {
                                            in_vals[i] = *input.add(input_base + i * hw);
                                        }
                                        let input_vec = _mm512_loadu_pd(in_vals.as_ptr());

                                        let mut w_vals = [0.0f64; F64_LANES];
                                        for i in 0..F64_LANES {
                                            w_vals[i] = *weight.add(weight_base + i * kw);
                                        }
                                        let weight_vec = _mm512_loadu_pd(w_vals.as_ptr());

                                        sum_vec = _mm512_fmadd_pd(input_vec, weight_vec, sum_vec);
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
// Depthwise Conv2d AVX-512 kernels
// ============================================================================

/// AVX-512 depthwise conv2d for f32
#[target_feature(enable = "avx512f")]
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
                // Vectorized loop over output width (16 at a time)
                for ox_chunk in 0..ox_chunks {
                    let ox_base = ox_chunk * F32_LANES;
                    let mut sum_vec = _mm512_setzero_ps();

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;

                            if iy_signed >= 0 && (iy_signed as usize) < height {
                                let iy = iy_signed as usize;

                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                                let w_val = *weight.add(weight_idx);
                                let weight_vec = _mm512_set1_ps(w_val);

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
                                }

                                let input_vec = _mm512_loadu_ps(in_vals.as_ptr());
                                sum_vec = _mm512_fmadd_ps(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }

                    if let Some(bias_ptr) = bias {
                        let bias_val = *bias_ptr.add(c);
                        sum_vec = _mm512_add_ps(sum_vec, _mm512_set1_ps(bias_val));
                    }

                    let mut out_vals = [0.0f32; F32_LANES];
                    _mm512_storeu_ps(out_vals.as_mut_ptr(), sum_vec);

                    for i in 0..F32_LANES {
                        let ox = ox_base + i;
                        let output_idx = b * c_in * output_h * output_w
                            + c * output_h * output_w
                            + oy * output_w
                            + ox;
                        *output.add(output_idx) = out_vals[i];
                    }
                }

                // Scalar remainder
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

/// AVX-512 depthwise conv2d for f64
#[target_feature(enable = "avx512f")]
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
                    let mut sum_vec = _mm512_setzero_pd();

                    for ky in 0..kernel_h {
                        for kx in 0..kernel_w {
                            let iy_signed = (oy * stride_h) as isize + (ky * dilation_h) as isize
                                - pad_top as isize;

                            if iy_signed >= 0 && (iy_signed as usize) < height {
                                let iy = iy_signed as usize;

                                let weight_idx = c * kernel_h * kernel_w + ky * kernel_w + kx;
                                let w_val = *weight.add(weight_idx);
                                let weight_vec = _mm512_set1_pd(w_val);

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

                                let input_vec = _mm512_loadu_pd(in_vals.as_ptr());
                                sum_vec = _mm512_fmadd_pd(input_vec, weight_vec, sum_vec);
                            }
                        }
                    }

                    if let Some(bias_ptr) = bias {
                        let bias_val = *bias_ptr.add(c);
                        sum_vec = _mm512_add_pd(sum_vec, _mm512_set1_pd(bias_val));
                    }

                    let mut out_vals = [0.0f64; F64_LANES];
                    _mm512_storeu_pd(out_vals.as_mut_ptr(), sum_vec);

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
