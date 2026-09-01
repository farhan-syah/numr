// Backend parity coverage for depthwise conv1d (groups == c_in == c_out).
//
// `tests/backend_parity/conv.rs` has no case where groups equals both c_in
// and c_out, and `conv_oc4.rs` only reaches c_out_per_group >= 1 through
// small, non-depthwise group counts. Depthwise conv1d always has
// c_out_per_group = 1, so it stays on the scalar `conv1d` kernel regardless
// of channel count (`CONV1D_OC_BLOCK` in `src/runtime/cuda/kernels/conv.rs`
// is never reached). This file pins that shape ahead of any change targeting
// it, split out from `conv.rs` because the added cases would push that file
// past its 600-line integration-test limit (mirrors why `conv_oc4.rs` exists
// as its own file).

use numr::ops::PaddingMode;

use crate::backend_parity::conv_oc4::{
    assert_conv1d_parity as assert_conv1d_depthwise_parity, conv1d_bias as depthwise_bias,
    conv1d_input as depthwise_input, conv1d_weight as depthwise_weight,
};

/// Bias values an order of magnitude larger than the conv output, so a
/// dropped or misapplied bias fails outright instead of passing inside
/// tolerance.
fn depthwise_bias_dominant(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i as f64) * 10.0).collect()
}

/// Decode shape: groups == c_in == c_out == 4, K == L == 4 so the causal
/// window is exactly the kernel width and L_out == 1 (no padding needed, as
/// in a cached-state decode step).
#[test]
fn conv1d_depthwise_decode_lout1_parity() {
    let input_shape = [1usize, 4, 4];
    let weight_shape = [4usize, 1, 4];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_decode_lout1",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        4,
    );
}

/// groups == c_in == c_out == 3 with symmetric padding and a longer
/// sequence: L_out == L, the shape a full-prefill causal-ish conv exercises.
#[test]
fn conv1d_depthwise_padded_long_parity() {
    let input_shape = [1usize, 3, 8];
    let weight_shape = [3usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_padded_long",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::conv1d(1, 1),
        1,
        3,
    );
}

/// Depthwise with dilation > 1: groups == c_in == c_out == 3, K = 3,
/// dilation = 2.
#[test]
fn conv1d_depthwise_dilation2_parity() {
    let input_shape = [1usize, 3, 10];
    let weight_shape = [3usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_dilation2",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        2,
        3,
    );
}

/// Depthwise with stride > 1: groups == c_in == c_out == 3, K = 3,
/// stride = 2.
#[test]
fn conv1d_depthwise_stride2_parity() {
    let input_shape = [1usize, 3, 9];
    let weight_shape = [3usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_stride2",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        2,
        PaddingMode::Valid,
        1,
        3,
    );
}

/// Depthwise with a bias an order of magnitude larger than the conv output:
/// a dropped or misapplied bias fails outright, not inside tolerance.
#[test]
fn conv1d_depthwise_bias_dominant_parity() {
    let input_shape = [1usize, 3, 5];
    let weight_shape = [3usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    let bias = depthwise_bias_dominant(weight_shape[0]);
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_bias_dominant",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        1,
        3,
    );
}

/// Depthwise with an ordinary (non-dominant) bias, kept separate from the
/// bias-absent decode/padded/dilation/stride cases above so both "with
/// bias" and "without bias" are covered explicitly.
#[test]
fn conv1d_depthwise_bias_present_parity() {
    let input_shape = [1usize, 3, 6];
    let weight_shape = [3usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    let bias = depthwise_bias(weight_shape[0]);
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_bias_present",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        1,
        3,
    );
}

/// `c_in = c_out = groups = 5`, not a multiple of the CONV1D_OC_BLOCK (4)
/// launch-geometry block width: any future change to the launch geometry
/// that mishandles the channel tail fails loudly here.
#[test]
fn conv1d_depthwise_channel_tail_parity() {
    let input_shape = [1usize, 5, 6];
    let weight_shape = [5usize, 1, 3];
    let input = depthwise_input(input_shape.iter().product());
    let weight = depthwise_weight(weight_shape.iter().product());
    assert_conv1d_depthwise_parity(
        "conv1d_depthwise_channel_tail",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::conv1d(1, 1),
        1,
        5,
    );
}
