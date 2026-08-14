//! Runtime `SimdLevel` dispatch for `conv1d`.
//!
//! Detection happens once here, per call, and the whole loop nest then runs
//! inside a single `#[target_feature]`-annotated function — no feature checks
//! anywhere in the hot loop.
//!
//! # Which shapes actually reach SIMD
//!
//! The vectorised axis is the OUTPUT POSITION, so the useful predictor is
//! `output_length`, not the channel count. That is the point of the rewrite:
//! a depthwise conv1d (`groups == c_in`, so `c_in_per_group == 1`) — which the
//! old channel-vectorised kernel could never touch — now vectorises fully, as
//! does any single-channel convolution.
//!
//! Short outputs (below the vector width) fall back to the scalar kernel, where
//! the vector prologue would cost more than it saves.

use super::super::scalar::{conv1d_scalar_f32, conv1d_scalar_f64};
use super::super::{SIMD_THRESHOLD_F32, SIMD_THRESHOLD_F64};
use crate::ops::conv_common::Conv1dParams;
use crate::runtime::cpu::kernels::simd::{SimdLevel, detect_simd};

#[cfg(target_arch = "aarch64")]
use super::neon;
#[cfg(target_arch = "x86_64")]
use super::{avx2, avx512};

/// SIMD `conv1d` for f32.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the shapes in `params`
#[inline]
pub unsafe fn conv1d_f32(
    input: *const f32,
    weight: *const f32,
    bias: Option<*const f32>,
    output: *mut f32,
    params: Conv1dParams,
) {
    let level = detect_simd();

    if params.output_length < SIMD_THRESHOLD_F32 || level == SimdLevel::Scalar {
        conv1d_scalar_f32(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::conv1d_f32(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::conv1d_f32(input, weight, bias, output, params),
        _ => conv1d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv1d_f32(input, weight, bias, output, params)
        }
        _ => conv1d_scalar_f32(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv1d_scalar_f32(input, weight, bias, output, params);
}

/// SIMD `conv1d` for f64.
///
/// # Safety
/// - All pointers must be valid and properly aligned
/// - Arrays must have sufficient size for the shapes in `params`
#[inline]
pub unsafe fn conv1d_f64(
    input: *const f64,
    weight: *const f64,
    bias: Option<*const f64>,
    output: *mut f64,
    params: Conv1dParams,
) {
    let level = detect_simd();

    if params.output_length < SIMD_THRESHOLD_F64 || level == SimdLevel::Scalar {
        conv1d_scalar_f64(input, weight, bias, output, params);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    match level {
        SimdLevel::Avx512 => avx512::conv1d_f64(input, weight, bias, output, params),
        SimdLevel::Avx2Fma => avx2::conv1d_f64(input, weight, bias, output, params),
        _ => conv1d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(target_arch = "aarch64")]
    match level {
        SimdLevel::Neon | SimdLevel::NeonFp16 => {
            neon::conv1d_f64(input, weight, bias, output, params)
        }
        _ => conv1d_scalar_f64(input, weight, bias, output, params),
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    conv1d_scalar_f64(input, weight, bias, output, params);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::ops::PaddingMode;
    use crate::ops::conv_common::validate_conv1d;

    /// Generates a SIMD-vs-scalar comparison harness for one dtype.
    ///
    /// The tolerance form is numr's documented `diff <= atol + rtol * |expected|`.
    /// A pure RELATIVE bound is wrong here: the SIMD path contracts the
    /// multiply-add into an FMA while the scalar path rounds the product first,
    /// so with ramp test data — where the true result can be far smaller than
    /// the intermediate terms — cancellation inflates a ~1e-7 absolute f32
    /// difference into a ~1e-3 relative one. The `atol` term absorbs exactly
    /// that, while an indexing bug (which produces O(1) differences) still fails
    /// loudly.
    macro_rules! compare_fn {
        ($name:ident, $ty:ty, $dtype:expr, $simd:path, $scalar:path, $atol:expr, $rtol:expr) => {
            #[allow(clippy::too_many_arguments)]
            fn $name(
                batch: usize,
                c_in: usize,
                length: usize,
                c_out_per_group: usize,
                kernel: usize,
                stride: usize,
                dilation: usize,
                groups: usize,
                padding: PaddingMode,
                with_bias: bool,
                label: &str,
            ) {
                let c_out = c_out_per_group * groups;
                let c_in_per_group = c_in / groups;
                // Bounded, non-monotonic test data. A plain ramp would grow with
                // the tensor size and make the absolute-drift assertion below
                // depend on the shape rather than on the kernel.
                let input: Vec<$ty> = (0..(batch * c_in * length))
                    .map(|x| (((x * 37) % 61) as $ty) * 0.031 - 0.93)
                    .collect();
                let weight: Vec<$ty> = (0..(c_out * c_in_per_group * kernel))
                    .map(|x| (((x * 23) % 47) as $ty) * 0.037 - 0.85)
                    .collect();
                let bias: Vec<$ty> = (0..c_out).map(|x| (x as $ty) * 0.5 + 1.0).collect();

                let bias_shape = [c_out];
                let params = validate_conv1d(
                    &[batch, c_in, length],
                    &[c_out, c_in_per_group, kernel],
                    if with_bias {
                        Some(&bias_shape[..])
                    } else {
                        None
                    },
                    stride,
                    padding,
                    dilation,
                    groups,
                    $dtype,
                    $dtype,
                    if with_bias { Some($dtype) } else { None },
                )
                .expect("valid test shapes");

                let n = batch * c_out * params.output_length;
                let mut out_simd = vec![0 as $ty; n];
                let mut out_scalar = vec![0 as $ty; n];
                let bias_ptr = if with_bias { Some(bias.as_ptr()) } else { None };

                unsafe {
                    $simd(
                        input.as_ptr(),
                        weight.as_ptr(),
                        bias_ptr,
                        out_simd.as_mut_ptr(),
                        params,
                    );
                    $scalar(
                        input.as_ptr(),
                        weight.as_ptr(),
                        bias_ptr,
                        out_scalar.as_mut_ptr(),
                        params,
                    );
                }

                let mut worst = 0 as $ty;
                for i in 0..n {
                    let diff = (out_simd[i] - out_scalar[i]).abs();
                    if diff > worst {
                        worst = diff;
                    }
                    assert!(
                        diff <= $atol + $rtol * out_scalar[i].abs(),
                        "{label}: mismatch at {i}: simd={} scalar={} (diff {diff:e})",
                        out_simd[i],
                        out_scalar[i]
                    );
                }
                // Surface the actual agreement so a slow drift can't hide under
                // the bound.
                assert!(
                    worst < $atol * 100.0,
                    "{label}: max abs diff {worst:e} is too large for pure FP reassociation"
                );
            }
        };
    }

    compare_fn!(
        compare,
        f32,
        DType::F32,
        conv1d_f32,
        conv1d_scalar_f32,
        1e-6f32,
        1e-4f32
    );
    compare_fn!(
        compare_f64,
        f64,
        DType::F64,
        conv1d_f64,
        conv1d_scalar_f64,
        1e-12f64,
        1e-10f64
    );

    #[test]
    fn simd_matches_scalar_stride1() {
        compare(2, 8, 64, 4, 3, 1, 1, 1, PaddingMode::Valid, true, "stride1");
    }

    /// One input channel per group: the case the old channel-vectorised kernel
    /// could not touch at all, and now the main beneficiary.
    #[test]
    fn simd_matches_scalar_depthwise() {
        compare(
            1,
            16,
            64,
            1,
            4,
            1,
            1,
            16,
            PaddingMode::Same,
            false,
            "depthwise",
        );
    }

    /// Depthwise with a stride: exercises the manual input gather.
    #[test]
    fn simd_matches_scalar_depthwise_strided() {
        compare(
            2,
            6,
            80,
            1,
            3,
            3,
            1,
            6,
            PaddingMode::Custom(2, 2, 0, 0),
            true,
            "depthwise-strided",
        );
    }

    #[test]
    fn simd_matches_scalar_strided() {
        compare(1, 4, 64, 3, 3, 2, 1, 1, PaddingMode::Valid, true, "stride2");
    }

    #[test]
    fn simd_matches_scalar_dilated() {
        compare(
            1,
            4,
            64,
            2,
            3,
            1,
            3,
            1,
            PaddingMode::Custom(3, 3, 0, 0),
            true,
            "dilation3",
        );
    }

    /// Left padding that is not a multiple of the stride: `interior_lo` must
    /// round UP, or the vector path would read before the input.
    #[test]
    fn simd_matches_scalar_padded_stride3() {
        compare(
            1,
            3,
            48,
            2,
            5,
            3,
            1,
            1,
            PaddingMode::Custom(4, 4, 0, 0),
            true,
            "pad4-stride3",
        );
    }

    #[test]
    fn simd_matches_scalar_grouped() {
        compare(
            2,
            8,
            48,
            2,
            3,
            2,
            2,
            4,
            PaddingMode::Same,
            true,
            "grouped-same",
        );
    }

    /// Dilated span wider than the padded-left input: `length + pad_left -
    /// (K-1)*dilation` goes NEGATIVE, so `interior_hi` must clamp to 0 rather
    /// than wrap around in `usize`. Every output position is then a boundary
    /// position handled by the scalar edge path.
    #[test]
    fn simd_matches_scalar_kernel_larger_than_input() {
        compare(
            1,
            2,
            2,
            2,
            5,
            1,
            1,
            1,
            PaddingMode::Custom(0, 12, 0, 0),
            true,
            "kernel>input",
        );
    }

    #[test]
    fn simd_matches_scalar_f64() {
        compare_f64(1, 4, 64, 3, 3, 1, 1, 1, PaddingMode::Same, true, "f64");
    }

    #[test]
    fn simd_matches_scalar_f64_depthwise_strided() {
        compare_f64(
            1,
            8,
            48,
            1,
            3,
            2,
            2,
            8,
            PaddingMode::Custom(3, 3, 0, 0),
            true,
            "f64-depthwise-strided",
        );
    }
}
