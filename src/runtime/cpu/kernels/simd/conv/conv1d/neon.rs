//! NEON 1D convolution kernels for AArch64.
//!
//! Vectorises over OUTPUT POSITIONS (see the `driver` module): the weight is a
//! scalar broadcast and, for `stride == 1`, the input is a contiguous vector
//! load. Two accumulators covering `2 * LANES` neighbouring outputs are carried
//! through the whole `(ic, kx)` reduction so two independent FMA chains stay in
//! flight.
//!
//! - f32: 4 lanes per vector, 8 output positions per unrolled iteration
//! - f64: 2 lanes per vector, 4 output positions per unrolled iteration

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::driver::conv1d_body;
use crate::ops::conv_common::Conv1dParams;

/// Packs 4 f32 spaced `stride` apart into a vector (`stride > 1` case only).
///
/// # Safety
/// - `p .. p + 3 * stride` must be readable
/// - CPU must support NEON
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gather4_f32(p: *const f32, stride: usize) -> float32x4_t {
    let mut xs = [0.0f32; 4];
    for t in 0..4 {
        xs[t] = *p.add(t * stride);
    }
    vld1q_f32(xs.as_ptr())
}

/// Packs 2 f64 spaced `stride` apart into a vector (`stride > 1` case only).
///
/// # Safety
/// - `p .. p + stride` must be readable
/// - CPU must support NEON
#[target_feature(enable = "neon")]
#[inline]
unsafe fn gather2_f64(p: *const f64, stride: usize) -> float64x2_t {
    let xs = [*p, *p.add(stride)];
    vld1q_f64(xs.as_ptr())
}

/// NEON `conv1d` for f32.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv1dParams,
) {
    conv1d_body!(
        f32,
        input,
        weight,
        bias,
        output,
        params,
        |op, ip, wp, n, nic, bv| {
            let Conv1dParams {
                length,
                kernel_size,
                stride,
                dilation,
                ..
            } = params;
            let bias_vec = vdupq_n_f32(bv);
            let mut j = 0usize;

            if stride == 1 {
                while j + 8 <= n {
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f32(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = vfmaq_f32(acc0, vld1q_f32(xb), wv);
                            acc1 = vfmaq_f32(acc1, vld1q_f32(xb.add(4)), wv);
                        }
                    }
                    vst1q_f32(op.add(j), vaddq_f32(acc0, bias_vec));
                    vst1q_f32(op.add(j + 4), vaddq_f32(acc1, bias_vec));
                    j += 8;
                }
                while j + 4 <= n {
                    let mut acc0 = vdupq_n_f32(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f32(*w_row.add(kx));
                            acc0 = vfmaq_f32(acc0, vld1q_f32(x_row.add(kx * dilation)), wv);
                        }
                    }
                    vst1q_f32(op.add(j), vaddq_f32(acc0, bias_vec));
                    j += 4;
                }
            } else {
                while j + 8 <= n {
                    let mut acc0 = vdupq_n_f32(0.0);
                    let mut acc1 = vdupq_n_f32(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f32(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = vfmaq_f32(acc0, gather4_f32(xb, stride), wv);
                            acc1 = vfmaq_f32(acc1, gather4_f32(xb.add(4 * stride), stride), wv);
                        }
                    }
                    vst1q_f32(op.add(j), vaddq_f32(acc0, bias_vec));
                    vst1q_f32(op.add(j + 4), vaddq_f32(acc1, bias_vec));
                    j += 8;
                }
                while j + 4 <= n {
                    let mut acc0 = vdupq_n_f32(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f32(*w_row.add(kx));
                            acc0 =
                                vfmaq_f32(acc0, gather4_f32(x_row.add(kx * dilation), stride), wv);
                        }
                    }
                    vst1q_f32(op.add(j), vaddq_f32(acc0, bias_vec));
                    j += 4;
                }
            }

            while j < n {
                let mut sum = 0.0f32;
                for ic in 0..nic {
                    let x_row = ip.add(ic * length + j * stride);
                    let w_row = wp.add(ic * kernel_size);
                    for kx in 0..kernel_size {
                        sum += *x_row.add(kx * dilation) * *w_row.add(kx);
                    }
                }
                *op.add(j) = sum + bv;
                j += 1;
            }
        }
    );
}

/// NEON `conv1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv1dParams,
) {
    conv1d_body!(
        f64,
        input,
        weight,
        bias,
        output,
        params,
        |op, ip, wp, n, nic, bv| {
            let Conv1dParams {
                length,
                kernel_size,
                stride,
                dilation,
                ..
            } = params;
            let bias_vec = vdupq_n_f64(bv);
            let mut j = 0usize;

            if stride == 1 {
                while j + 4 <= n {
                    let mut acc0 = vdupq_n_f64(0.0);
                    let mut acc1 = vdupq_n_f64(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f64(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = vfmaq_f64(acc0, vld1q_f64(xb), wv);
                            acc1 = vfmaq_f64(acc1, vld1q_f64(xb.add(2)), wv);
                        }
                    }
                    vst1q_f64(op.add(j), vaddq_f64(acc0, bias_vec));
                    vst1q_f64(op.add(j + 2), vaddq_f64(acc1, bias_vec));
                    j += 4;
                }
                while j + 2 <= n {
                    let mut acc0 = vdupq_n_f64(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f64(*w_row.add(kx));
                            acc0 = vfmaq_f64(acc0, vld1q_f64(x_row.add(kx * dilation)), wv);
                        }
                    }
                    vst1q_f64(op.add(j), vaddq_f64(acc0, bias_vec));
                    j += 2;
                }
            } else {
                while j + 4 <= n {
                    let mut acc0 = vdupq_n_f64(0.0);
                    let mut acc1 = vdupq_n_f64(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f64(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = vfmaq_f64(acc0, gather2_f64(xb, stride), wv);
                            acc1 = vfmaq_f64(acc1, gather2_f64(xb.add(2 * stride), stride), wv);
                        }
                    }
                    vst1q_f64(op.add(j), vaddq_f64(acc0, bias_vec));
                    vst1q_f64(op.add(j + 2), vaddq_f64(acc1, bias_vec));
                    j += 4;
                }
                while j + 2 <= n {
                    let mut acc0 = vdupq_n_f64(0.0);
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = vdupq_n_f64(*w_row.add(kx));
                            acc0 =
                                vfmaq_f64(acc0, gather2_f64(x_row.add(kx * dilation), stride), wv);
                        }
                    }
                    vst1q_f64(op.add(j), vaddq_f64(acc0, bias_vec));
                    j += 2;
                }
            }

            while j < n {
                let mut sum = 0.0f64;
                for ic in 0..nic {
                    let x_row = ip.add(ic * length + j * stride);
                    let w_row = wp.add(ic * kernel_size);
                    for kx in 0..kernel_size {
                        sum += *x_row.add(kx * dilation) * *w_row.add(kx);
                    }
                }
                *op.add(j) = sum + bv;
                j += 1;
            }
        }
    );
}
