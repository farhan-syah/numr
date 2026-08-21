use super::*;
// The module root re-exports only the dispatch entry points; the scalar
// reference these tests compare against is reached by its own path.
use super::scalar::{conv1d_scalar_f32, conv2d_scalar_f32, depthwise_conv2d_scalar_f32};
use crate::dtype::DType;
use crate::ops::PaddingMode;
use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};

#[test]
fn test_conv1d_simd_matches_scalar() {
    // Input: (1, 16, 32) - 16 channels to trigger SIMD
    let c_in = 16;
    let length = 32;
    let c_out = 8;
    let kernel_size = 3;

    let input: Vec<f32> = (0..(c_in * length))
        .map(|x| (x as f32) * 0.01 - 0.5)
        .collect();
    let weight: Vec<f32> = (0..(c_out * c_in * kernel_size))
        .map(|x| (x as f32) * 0.001 - 0.2)
        .collect();

    let params = validate_conv1d(
        &[1, c_in, length],
        &[c_out, c_in, kernel_size],
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
        DType::F32,
        DType::F32,
        None,
    )
    .unwrap();

    let output_len = c_out * params.output_length;
    let mut out_simd = vec![0.0f32; output_len];
    let mut out_scalar = vec![0.0f32; output_len];

    unsafe {
        conv1d_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_simd.as_mut_ptr(),
            params,
        );
        conv1d_scalar_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_scalar.as_mut_ptr(),
            params,
        );
    }

    for i in 0..output_len {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        let rel_err = if out_scalar[i].abs() > 1e-6 {
            diff / out_scalar[i].abs()
        } else {
            diff
        };
        assert!(
            rel_err < 1e-5,
            "conv1d mismatch at {}: SIMD={} scalar={} (rel_err={})",
            i,
            out_simd[i],
            out_scalar[i],
            rel_err
        );
    }
}

#[test]
fn test_conv2d_simd_matches_scalar() {
    // Input: (1, 16, 8, 8) - 16 channels to trigger SIMD
    let c_in = 16;
    let (h, w) = (8, 8);
    let c_out = 4;
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..(c_in * h * w)).map(|x| (x as f32) * 0.01).collect();
    let weight: Vec<f32> = (0..(c_out * c_in * kh * kw))
        .map(|x| (x as f32) * 0.001 - 0.2)
        .collect();

    let params = validate_conv2d(
        &[1, c_in, h, w],
        &[c_out, c_in, kh, kw],
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        1,
        DType::F32,
        DType::F32,
        None,
    )
    .unwrap();

    let output_len = c_out * params.output_h * params.output_w;
    let mut out_simd = vec![0.0f32; output_len];
    let mut out_scalar = vec![0.0f32; output_len];

    unsafe {
        conv2d_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_simd.as_mut_ptr(),
            params,
        );
        conv2d_scalar_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_scalar.as_mut_ptr(),
            params,
        );
    }

    for i in 0..output_len {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        let rel_err = if out_scalar[i].abs() > 1e-6 {
            diff / out_scalar[i].abs()
        } else {
            diff
        };
        assert!(
            rel_err < 1e-4,
            "conv2d mismatch at {}: SIMD={} scalar={} (rel_err={})",
            i,
            out_simd[i],
            out_scalar[i],
            rel_err
        );
    }
}

#[test]
fn test_depthwise_conv2d_simd_matches_scalar() {
    // Input: (1, 8, 16, 16) - wide enough to trigger SIMD
    let channels = 8;
    let (h, w) = (16, 16);
    let (kh, kw) = (3, 3);

    let input: Vec<f32> = (0..(channels * h * w))
        .map(|x| (x as f32) * 0.01 - 1.0)
        .collect();
    let weight: Vec<f32> = (0..(channels * kh * kw))
        .map(|x| (x as f32) * 0.01 - 0.3)
        .collect();

    let params = validate_depthwise_conv2d(
        &[1, channels, h, w],
        &[channels, 1, kh, kw],
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
        DType::F32,
        DType::F32,
        None,
    )
    .unwrap();

    let output_len = channels * params.output_h * params.output_w;
    let mut out_simd = vec![0.0f32; output_len];
    let mut out_scalar = vec![0.0f32; output_len];

    unsafe {
        depthwise_conv2d_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_simd.as_mut_ptr(),
            params,
        );
        depthwise_conv2d_scalar_f32(
            input.as_ptr(),
            weight.as_ptr(),
            None,
            out_scalar.as_mut_ptr(),
            params,
        );
    }

    for i in 0..output_len {
        let diff = (out_simd[i] - out_scalar[i]).abs();
        let rel_err = if out_scalar[i].abs() > 1e-6 {
            diff / out_scalar[i].abs()
        } else {
            diff
        };
        assert!(
            rel_err < 1e-5,
            "depthwise conv2d mismatch at {}: SIMD={} scalar={} (rel_err={})",
            i,
            out_simd[i],
            out_scalar[i],
            rel_err
        );
    }
}
