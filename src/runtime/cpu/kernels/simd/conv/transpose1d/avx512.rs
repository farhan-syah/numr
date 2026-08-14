//! AVX-512 transposed 1D convolution kernels.
//!
//! Vectorises over output positions via the polyphase decomposition described
//! in the `driver` module. The inner loop is a contiguous AXPY
//! (`acc[j] += w * x[j]`) unrolled two vectors wide so two independent FMA
//! chains are in flight, hiding the 4-5 cycle FMA latency.
//!
//! - f32: 16 lanes per vector, 32 elements per unrolled iteration
//! - f64: 8 lanes per vector, 16 elements per unrolled iteration

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::driver::conv_transpose1d_body;
use crate::ops::conv_transpose_common::ConvTranspose1dParams;

/// AVX-512 `conv_transpose1d` for f32.
///
/// `acc` is a caller-provided scratch buffer of at least
/// `ceil(params.output_length / params.stride)` elements.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - `acc` must be large enough (see above)
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
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
            let wv = _mm512_set1_ps(w);
            let mut j = 0usize;
            while j + 32 <= n {
                let a0 = _mm512_loadu_ps(ap.add(j));
                let a1 = _mm512_loadu_ps(ap.add(j + 16));
                let x0 = _mm512_loadu_ps(xp.add(j));
                let x1 = _mm512_loadu_ps(xp.add(j + 16));
                _mm512_storeu_ps(ap.add(j), _mm512_fmadd_ps(wv, x0, a0));
                _mm512_storeu_ps(ap.add(j + 16), _mm512_fmadd_ps(wv, x1, a1));
                j += 32;
            }
            while j + 16 <= n {
                let a0 = _mm512_loadu_ps(ap.add(j));
                let x0 = _mm512_loadu_ps(xp.add(j));
                _mm512_storeu_ps(ap.add(j), _mm512_fmadd_ps(wv, x0, a0));
                j += 16;
            }
            while j < n {
                *ap.add(j) = w.mul_add(*xp.add(j), *ap.add(j));
                j += 1;
            }
        }
    );
}

/// AVX-512 `conv_transpose1d` for f64.
///
/// # Safety
/// - All pointers must be valid for the shapes in `params`
/// - `acc` must hold at least `ceil(output_length / stride)` elements
/// - CPU must support AVX-512F
#[target_feature(enable = "avx512f")]
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
            let wv = _mm512_set1_pd(w);
            let mut j = 0usize;
            while j + 16 <= n {
                let a0 = _mm512_loadu_pd(ap.add(j));
                let a1 = _mm512_loadu_pd(ap.add(j + 8));
                let x0 = _mm512_loadu_pd(xp.add(j));
                let x1 = _mm512_loadu_pd(xp.add(j + 8));
                _mm512_storeu_pd(ap.add(j), _mm512_fmadd_pd(wv, x0, a0));
                _mm512_storeu_pd(ap.add(j + 8), _mm512_fmadd_pd(wv, x1, a1));
                j += 16;
            }
            while j + 8 <= n {
                let a0 = _mm512_loadu_pd(ap.add(j));
                let x0 = _mm512_loadu_pd(xp.add(j));
                _mm512_storeu_pd(ap.add(j), _mm512_fmadd_pd(wv, x0, a0));
                j += 8;
            }
            while j < n {
                *ap.add(j) = w.mul_add(*xp.add(j), *ap.add(j));
                j += 1;
            }
        }
    );
}
