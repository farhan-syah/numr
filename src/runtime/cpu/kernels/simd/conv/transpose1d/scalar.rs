//! Scalar fallbacks for transposed 1D convolution.
//!
//! These forward to the dtype-generic gather kernel so there is exactly ONE
//! algorithm: the SIMD paths are a polyphase re-ordering of the same gather,
//! never a second formulation.

use crate::ops::conv_transpose_common::ConvTranspose1dParams;

/// Scalar `conv_transpose1d` for f32.
///
/// # Safety
/// All pointers must be valid for the shapes described by `params`.
#[inline]
pub unsafe fn conv_transpose1d_scalar_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: ConvTranspose1dParams,
) {
    crate::runtime::cpu::kernels::conv_transpose::conv_transpose1d_kernel(
        input, weight, bias, output, params,
    );
}

/// Scalar `conv_transpose1d` for f64.
///
/// # Safety
/// All pointers must be valid for the shapes described by `params`.
#[inline]
pub unsafe fn conv_transpose1d_scalar_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: ConvTranspose1dParams,
) {
    crate::runtime::cpu::kernels::conv_transpose::conv_transpose1d_kernel(
        input, weight, bias, output, params,
    );
}
