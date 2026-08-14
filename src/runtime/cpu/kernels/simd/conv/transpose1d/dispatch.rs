//! Runtime `SimdLevel` dispatch for `conv_transpose1d`.
//!
//! Detection happens once here, per call, and the whole loop nest then runs
//! inside a single `#[target_feature]`-annotated function — no feature checks
//! anywhere in the hot loop.
//!
//! # Which shapes actually reach SIMD
//!
//! The vectorised axis is the OUTPUT POSITION within one polyphase lane, and
//! the run length of a contiguous AXPY is at most `min(length, n_r)` where
//! `n_r = ceil(output_length / stride) ~= length`. So the useful predictor is
//! the INPUT length, not the channel count — which is the point: the depthwise
//! case (`groups == c_in`, `c_in_per_group == 1`) still vectorises fully.
//!
//! Short sequences (`length` below the threshold) fall back to scalar, where
//! the vector prologue would cost more than it saves.

use super::super::{SIMD_THRESHOLD_F32, SIMD_THRESHOLD_F64};
use super::scalar::{conv_transpose1d_scalar_f32, conv_transpose1d_scalar_f64};
use crate::ops::conv_transpose_common::ConvTranspose1dParams;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

#[cfg(target_arch = "aarch64")]
use super::neon;
#[cfg(target_arch = "x86_64")]
use super::{avx2, avx512};

/// Scratch size for one polyphase accumulator lane.
#[inline]
fn phase_capacity(params: ConvTranspose1dParams) -> usize {
    if params.stride == 0 {
        return params.output_length;
    }
    params.output_length.div_ceil(params.stride)
}

/// SIMD `conv_transpose1d` for f32.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the shapes in `params`
#[inline]
pub unsafe fn conv_transpose1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: ConvTranspose1dParams,
) {
    let level = detect_simd();

    if params.length < SIMD_THRESHOLD_F32 || level == SimdLevel::Scalar {
        conv_transpose1d_scalar_f32(input, weight, bias, output, params);
        return;
    }

    #[allow(unused_mut, unused_variables)]
    let mut acc = vec![0.0f32; phase_capacity(params)];

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => {
            avx512::conv_transpose1d_f32(input, weight, bias, output, params, &mut acc)
        }
        SimdLevel::Avx2Fma => {
            avx2::conv_transpose1d_f32(input, weight, bias, output, params, &mut acc)
        }
        _ => conv_transpose1d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv_transpose1d_f32(input, weight, bias, output, params, &mut acc)
        }
        _ => conv_transpose1d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv_transpose1d_scalar_f32(input, weight, bias, output, params);
}

/// SIMD `conv_transpose1d` for f64.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the shapes in `params`
#[inline]
pub unsafe fn conv_transpose1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: ConvTranspose1dParams,
) {
    let level = detect_simd();

    if params.length < SIMD_THRESHOLD_F64 || level == SimdLevel::Scalar {
        conv_transpose1d_scalar_f64(input, weight, bias, output, params);
        return;
    }

    #[allow(unused_mut, unused_variables)]
    let mut acc = vec![0.0f64; phase_capacity(params)];

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => {
            avx512::conv_transpose1d_f64(input, weight, bias, output, params, &mut acc)
        }
        SimdLevel::Avx2Fma => {
            avx2::conv_transpose1d_f64(input, weight, bias, output, params, &mut acc)
        }
        _ => conv_transpose1d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv_transpose1d_f64(input, weight, bias, output, params, &mut acc)
        }
        _ => conv_transpose1d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv_transpose1d_scalar_f64(input, weight, bias, output, params);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::PaddingMode;
    use crate::ops::conv_transpose_common::validate_conv_transpose1d;

    #[allow(clippy::too_many_arguments)]
    fn compare(
        batch: usize,
        c_in: usize,
        length: usize,
        c_out_per_group: usize,
        kernel: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        padding: PaddingMode,
        output_padding: usize,
        with_bias: bool,
        label: &str,
    ) {
        let c_out = c_out_per_group * groups;
        let input: Vec<f32> = (0..(batch * c_in * length))
            .map(|x| (x as f32) * 0.013 - 0.7)
            .collect();
        let weight: Vec<f32> = (0..(c_in * c_out_per_group * kernel))
            .map(|x| (x as f32) * 0.007 - 0.3)
            .collect();
        let bias: Vec<f32> = (0..c_out).map(|x| (x as f32) * 0.5 + 1.0).collect();

        let bias_shape = [c_out];
        let params = validate_conv_transpose1d(
            &[batch, c_in, length],
            &[c_in, c_out_per_group, kernel],
            if with_bias {
                Some(&bias_shape[..])
            } else {
                None
            },
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            DType::F32,
            DType::F32,
            if with_bias { Some(DType::F32) } else { None },
        )
        .expect("valid test shapes");

        let n = batch * c_out * params.output_length;
        let mut out_simd = vec![0.0f32; n];
        let mut out_scalar = vec![0.0f32; n];
        let bias_ptr = if with_bias { Some(bias.as_ptr()) } else { None };

        unsafe {
            conv_transpose1d_f32(
                input.as_ptr(),
                weight.as_ptr(),
                bias_ptr,
                out_simd.as_mut_ptr(),
                params,
            );
            conv_transpose1d_scalar_f32(
                input.as_ptr(),
                weight.as_ptr(),
                bias_ptr,
                out_scalar.as_mut_ptr(),
                params,
            );
        }

        // numr's documented tolerance form: `diff <= atol + rtol * |expected|`.
        //
        // A pure RELATIVE bound is wrong here: the SIMD path accumulates per
        // phase while the scalar path accumulates sequentially, so the two sum
        // in different orders. With ramp test data the true result can be far
        // smaller than the intermediate terms (cancellation), which inflates a
        // ~2e-7 absolute f32 difference into a ~1e-4 relative one. The `atol`
        // term absorbs exactly that, while an indexing bug — which produces
        // O(1) differences — still fails loudly.
        const ATOL: f32 = 1e-6;
        const RTOL: f32 = 1e-4;
        let mut worst = 0.0f32;
        let mut worst_at = 0usize;
        for i in 0..n {
            let diff = (out_simd[i] - out_scalar[i]).abs();
            if diff > worst {
                worst = diff;
                worst_at = i;
            }
            assert!(
                diff <= ATOL + RTOL * out_scalar[i].abs(),
                "{label}: mismatch at {i}: simd={} scalar={} (diff {diff:e})",
                out_simd[i],
                out_scalar[i]
            );
        }
        // Surface the actual agreement so a slow drift can't hide under the bound.
        assert!(
            worst < 1e-4,
            "{label}: max abs diff {worst:e} at {worst_at} is too large for pure FP reordering"
        );
    }

    #[test]
    fn simd_matches_scalar_stride1() {
        compare(
            2,
            4,
            64,
            3,
            5,
            1,
            1,
            1,
            PaddingMode::Valid,
            0,
            true,
            "stride1",
        );
    }

    #[test]
    fn simd_matches_scalar_strided_upsample() {
        compare(
            1,
            3,
            48,
            2,
            6,
            3,
            1,
            1,
            PaddingMode::Valid,
            0,
            true,
            "stride3",
        );
    }

    /// The depthwise case the op exists for: one input channel per group, so
    /// channel vectorisation would be useless and position vectorisation is
    /// what carries the kernel.
    #[test]
    fn simd_matches_scalar_depthwise() {
        compare(
            1,
            16,
            80,
            1,
            8,
            4,
            1,
            16,
            PaddingMode::Valid,
            0,
            false,
            "depthwise",
        );
    }

    #[test]
    fn simd_matches_scalar_with_padding_and_dilation() {
        compare(
            1,
            2,
            40,
            2,
            4,
            2,
            3,
            1,
            PaddingMode::Custom(5, 3, 0, 0),
            1,
            true,
            "pad+dilation",
        );
    }

    #[test]
    fn simd_matches_scalar_grouped() {
        compare(
            2,
            8,
            32,
            2,
            3,
            2,
            1,
            4,
            PaddingMode::Same,
            0,
            true,
            "grouped-same",
        );
    }

    /// A stride far larger than the kernel leaves most phases with no
    /// contributing tap at all — those outputs must still receive bias/zero.
    #[test]
    fn simd_matches_scalar_large_stride() {
        compare(
            1,
            2,
            12,
            1,
            2,
            7,
            1,
            1,
            PaddingMode::Valid,
            0,
            false,
            "large-stride",
        );
    }
}
