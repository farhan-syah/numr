//! NEON transposed 1D convolution kernels for AArch64.
//!
//! Vectorises over output positions via the polyphase decomposition described
//! in the `driver` module. The inner loop is a contiguous AXPY
//! (`acc[j] += w * x[j]`) unrolled two vectors wide so two independent FMA
//! chains are in flight, hiding FMA latency.
//!
//! - f32: 4 lanes per vector, 8 elements per unrolled iteration
//! - f64: 2 lanes per vector, 4 elements per unrolled iteration

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::driver::conv_transpose1d_body;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;

/// NEON `conv_transpose1d` for f32.
///
/// `acc` is a caller-provided scratch buffer of at least
/// `ceil(params.output_length / params.stride)` elements.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - `acc` must be large enough (see above)
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv_transpose1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: ConvTranspose1dParams,
    acc: &mut [f32],
) {
    conv_transpose1d_body!(
        f32,
        input,
        weight,
        bias,
        output,
        params,
        acc,
        |ap, xp, w, n| {
            let wv = vdupq_n_f32(w);
            let mut j = 0usize;
            while j + 8 <= n {
                let a0 = vld1q_f32(ap.add(j));
                let a1 = vld1q_f32(ap.add(j + 4));
                let x0 = vld1q_f32(xp.add(j));
                let x1 = vld1q_f32(xp.add(j + 4));
                vst1q_f32(ap.add(j), vfmaq_f32(a0, x0, wv));
                vst1q_f32(ap.add(j + 4), vfmaq_f32(a1, x1, wv));
                j += 8;
            }
            while j + 4 <= n {
                let a0 = vld1q_f32(ap.add(j));
                let x0 = vld1q_f32(xp.add(j));
                vst1q_f32(ap.add(j), vfmaq_f32(a0, x0, wv));
                j += 4;
            }
            while j < n {
                *ap.add(j) = w.mul_add(*xp.add(j), *ap.add(j));
                j += 1;
            }
        }
    );
}

/// NEON `conv_transpose1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - `acc` must hold at least `ceil(output_length / stride)` elements
/// - CPU must support NEON
#[target_feature(enable = "neon")]
pub unsafe fn conv_transpose1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: ConvTranspose1dParams,
    acc: &mut [f64],
) {
    conv_transpose1d_body!(
        f64,
        input,
        weight,
        bias,
        output,
        params,
        acc,
        |ap, xp, w, n| {
            let wv = vdupq_n_f64(w);
            let mut j = 0usize;
            while j + 4 <= n {
                let a0 = vld1q_f64(ap.add(j));
                let a1 = vld1q_f64(ap.add(j + 2));
                let x0 = vld1q_f64(xp.add(j));
                let x1 = vld1q_f64(xp.add(j + 2));
                vst1q_f64(ap.add(j), vfmaq_f64(a0, x0, wv));
                vst1q_f64(ap.add(j + 2), vfmaq_f64(a1, x1, wv));
                j += 4;
            }
            while j + 2 <= n {
                let a0 = vld1q_f64(ap.add(j));
                let x0 = vld1q_f64(xp.add(j));
                vst1q_f64(ap.add(j), vfmaq_f64(a0, x0, wv));
                j += 2;
            }
            while j < n {
                *ap.add(j) = w.mul_add(*xp.add(j), *ap.add(j));
                j += 1;
            }
        }
    );
}
