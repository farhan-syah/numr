//! AVX-512 1D convolution kernels.
//!
//! Vectorises over OUTPUT POSITIONS (see the `driver` module): the weight is a
//! scalar broadcast and, for `stride == 1`, the input is a contiguous vector
//! load. Two accumulators covering `2 * LANES` neighbouring outputs are carried
//! through the whole `(ic, kx)` reduction so two independent FMA chains stay in
//! flight.
//!
//! - f32: 16 lanes per vector, 32 output positions per unrolled iteration
//! - f64: 8 lanes per vector, 16 output positions per unrolled iteration

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::driver::conv1d_body;
use crate::ops::conv_common::Conv1dParams;

/// Packs 16 f32 spaced `stride` apart into a vector (`stride > 1` case only).
///
/// # Safety
/// - `p .. p + 15 * stride` must be readable
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn gather16_f32(p: *const f32, stride: usize) -> __m512 {
    let mut xs = [0.0f32; 16];
    for t in 0..16 {
        xs[t] = *p.add(t * stride);
    }
    _mm512_loadu_ps(xs.as_ptr())
}

/// Packs 8 f64 spaced `stride` apart into a vector (`stride > 1` case only).
///
/// # Safety
/// - `p .. p + 7 * stride` must be readable
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
#[inline]
unsafe fn gather8_f64(p: *const f64, stride: usize) -> __m512d {
    let mut xs = [0.0f64; 8];
    for t in 0..8 {
        xs[t] = *p.add(t * stride);
    }
    _mm512_loadu_pd(xs.as_ptr())
}

/// AVX-512 `conv1d` for f32.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
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
            let bias_vec = _mm512_set1_ps(bv);
            let mut j = 0usize;

            if stride == 1 {
                while j + 32 <= n {
                    let mut acc0 = _mm512_setzero_ps();
                    let mut acc1 = _mm512_setzero_ps();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_ps(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(xb), wv, acc0);
                            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(xb.add(16)), wv, acc1);
                        }
                    }
                    _mm512_storeu_ps(op.add(j), _mm512_add_ps(acc0, bias_vec));
                    _mm512_storeu_ps(op.add(j + 16), _mm512_add_ps(acc1, bias_vec));
                    j += 32;
                }
                while j + 16 <= n {
                    let mut acc0 = _mm512_setzero_ps();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_ps(*w_row.add(kx));
                            acc0 = _mm512_fmadd_ps(
                                _mm512_loadu_ps(x_row.add(kx * dilation)),
                                wv,
                                acc0,
                            );
                        }
                    }
                    _mm512_storeu_ps(op.add(j), _mm512_add_ps(acc0, bias_vec));
                    j += 16;
                }
            } else {
                while j + 32 <= n {
                    let mut acc0 = _mm512_setzero_ps();
                    let mut acc1 = _mm512_setzero_ps();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_ps(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = _mm512_fmadd_ps(gather16_f32(xb, stride), wv, acc0);
                            acc1 = _mm512_fmadd_ps(
                                gather16_f32(xb.add(16 * stride), stride),
                                wv,
                                acc1,
                            );
                        }
                    }
                    _mm512_storeu_ps(op.add(j), _mm512_add_ps(acc0, bias_vec));
                    _mm512_storeu_ps(op.add(j + 16), _mm512_add_ps(acc1, bias_vec));
                    j += 32;
                }
                while j + 16 <= n {
                    let mut acc0 = _mm512_setzero_ps();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_ps(*w_row.add(kx));
                            acc0 = _mm512_fmadd_ps(
                                gather16_f32(x_row.add(kx * dilation), stride),
                                wv,
                                acc0,
                            );
                        }
                    }
                    _mm512_storeu_ps(op.add(j), _mm512_add_ps(acc0, bias_vec));
                    j += 16;
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

/// AVX-512 `conv1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
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
            let bias_vec = _mm512_set1_pd(bv);
            let mut j = 0usize;

            if stride == 1 {
                while j + 16 <= n {
                    let mut acc0 = _mm512_setzero_pd();
                    let mut acc1 = _mm512_setzero_pd();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_pd(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = _mm512_fmadd_pd(_mm512_loadu_pd(xb), wv, acc0);
                            acc1 = _mm512_fmadd_pd(_mm512_loadu_pd(xb.add(8)), wv, acc1);
                        }
                    }
                    _mm512_storeu_pd(op.add(j), _mm512_add_pd(acc0, bias_vec));
                    _mm512_storeu_pd(op.add(j + 8), _mm512_add_pd(acc1, bias_vec));
                    j += 16;
                }
                while j + 8 <= n {
                    let mut acc0 = _mm512_setzero_pd();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_pd(*w_row.add(kx));
                            acc0 = _mm512_fmadd_pd(
                                _mm512_loadu_pd(x_row.add(kx * dilation)),
                                wv,
                                acc0,
                            );
                        }
                    }
                    _mm512_storeu_pd(op.add(j), _mm512_add_pd(acc0, bias_vec));
                    j += 8;
                }
            } else {
                while j + 16 <= n {
                    let mut acc0 = _mm512_setzero_pd();
                    let mut acc1 = _mm512_setzero_pd();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_pd(*w_row.add(kx));
                            let xb = x_row.add(kx * dilation);
                            acc0 = _mm512_fmadd_pd(gather8_f64(xb, stride), wv, acc0);
                            acc1 =
                                _mm512_fmadd_pd(gather8_f64(xb.add(8 * stride), stride), wv, acc1);
                        }
                    }
                    _mm512_storeu_pd(op.add(j), _mm512_add_pd(acc0, bias_vec));
                    _mm512_storeu_pd(op.add(j + 8), _mm512_add_pd(acc1, bias_vec));
                    j += 16;
                }
                while j + 8 <= n {
                    let mut acc0 = _mm512_setzero_pd();
                    for ic in 0..nic {
                        let x_row = ip.add(ic * length + j * stride);
                        let w_row = wp.add(ic * kernel_size);
                        for kx in 0..kernel_size {
                            let wv = _mm512_set1_pd(*w_row.add(kx));
                            acc0 = _mm512_fmadd_pd(
                                gather8_f64(x_row.add(kx * dilation), stride),
                                wv,
                                acc0,
                            );
                        }
                    }
                    _mm512_storeu_pd(op.add(j), _mm512_add_pd(acc0, bias_vec));
                    j += 8;
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
