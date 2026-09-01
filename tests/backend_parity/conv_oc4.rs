// Backend parity coverage for CUDA's `conv1d_oc4` register-blocked kernel.
//
// `tests/backend_parity/conv.rs` has exactly one conv1d test, at c_out=1, so
// `c_out_per_group` (1) never reaches the CONV1D_OC_BLOCK threshold (4) in
// `src/runtime/cuda/kernels/conv.rs` and `conv1d_oc4_*` is never selected.
// This file exercises that kernel directly, including the ragged-tail case
// where `c_out_per_group % 4 != 0`: the kernel processes
// `ceil(c_out_per_group / 4)` chunks of 4 and masks inactive lanes by
// aliasing unused weight pointers to `w0`. A mistake there overwrites a real
// output channel with another channel's values, which only a CPU-reference
// value comparison (not just an `is_ok()` check) will catch.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Deterministic, non-repeating input values so an oc4 lane-masking bug
/// cannot cancel out by coincidence.
fn conv1d_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 11) as f64 - 5.0) * 0.3).collect()
}

/// Weight values that vary with the flat index, which starts at the c_out
/// axis: every output channel gets a distinct kernel, so a bug that reuses
/// another channel's weights (aliasing to `w0`) shows up as a value
/// mismatch rather than an accidental match.
fn conv1d_weight(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 7) as f64 - 3.0) * 0.2).collect()
}

fn conv1d_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.05 - 0.1).collect()
}

/// Runs `conv1d` on CPU and every enabled GPU backend for every float dtype
/// and asserts the GPU result matches the CPU reference.
#[allow(clippy::too_many_arguments)]
fn assert_conv1d_parity(
    label: &str,
    input: &[f64],
    input_shape: &[usize],
    weight: &[f64],
    weight_shape: &[usize],
    bias: Option<&[f64]>,
    stride: usize,
    padding: PaddingMode,
    dilation: usize,
    groups: usize,
) {
    let c_out = weight_shape[0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(input, input_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(weight, weight_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = bias.map(|b| {
            tensor_from_f64(b, &[c_out], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU bias tensor failed for {label} [{dtype:?}]: {e}"))
        });
        let cpu_result = cpu_client
            .conv1d(
                &cpu_in,
                &cpu_w,
                cpu_b.as_ref(),
                stride,
                padding,
                dilation,
                groups,
            )
            .unwrap_or_else(|e| panic!("CPU conv1d failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(input, input_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| {
                        panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let w = tensor_from_f64(weight, weight_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| {
                        panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let b = bias.map(|bd| {
                    tensor_from_f64(bd, &[c_out], dtype, &cuda_device, &cuda_client).unwrap_or_else(
                        |e| panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}"),
                    )
                });
                let result = cuda_client
                    .conv1d(&x, &w, b.as_ref(), stride, padding, dilation, groups)
                    .unwrap_or_else(|e| panic!("CUDA conv1d failed for {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(input, input_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU input tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let w = tensor_from_f64(weight, weight_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU weight tensor failed for {label} [{dtype:?}]: {e}")
                    });
                let b = bias.map(|bd| {
                    tensor_from_f64(bd, &[c_out], dtype, &wgpu_device, &wgpu_client).unwrap_or_else(
                        |e| panic!("WebGPU bias tensor failed for {label} [{dtype:?}]: {e}"),
                    )
                });
                let result = wgpu_client
                    .conv1d(&x, &w, b.as_ref(), stride, padding, dilation, groups)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU conv1d failed for {label} [{dtype:?}]: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{label} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// `c_out=8, c_in=4, groups=1` → `c_out_per_group=8`, an exact multiple of
/// CONV1D_OC_BLOCK (4): selects `conv1d_oc4` with two full 4-channel chunks
/// and no masked lanes.
#[test]
fn conv1d_oc4_exact_multiple_c_out_8_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_exact_multiple_c_out_8",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// `c_out=6, c_in=4, groups=1` → `c_out_per_group=6`, selects `conv1d_oc4`
/// with `ceil(6/4)=2` chunks, the second chunk holding only 2 real lanes and
/// 2 masked ones aliased to `w0`.
#[test]
fn conv1d_oc4_ragged_c_out_6_parity() {
    let input_shape = [1usize, 4, 9];
    let weight_shape = [6usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_ragged_c_out_6",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// `c_out=5, c_in=3, groups=1` → `c_out_per_group=5`, `ceil(5/4)=2` chunks,
/// the second chunk holding only 1 real lane and 3 masked ones.
#[test]
fn conv1d_oc4_ragged_c_out_5_parity() {
    let input_shape = [1usize, 3, 8];
    let weight_shape = [5usize, 3, 2];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_ragged_c_out_5",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// `c_out=3 < CONV1D_OC_BLOCK` → `c_out_per_group=3`, stays on the scalar
/// `conv1d` kernel. Confirms the non-oc4 path still works alongside the new
/// oc4 coverage.
#[test]
fn conv1d_scalar_c_out_3_parity() {
    let input_shape = [1usize, 2, 7];
    let weight_shape = [3usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_scalar_c_out_3",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// `groups=2, c_out=8` → `c_out_per_group=4`, an exact-multiple oc4 shape
/// inside a grouped conv, exercising the interaction between group indexing
/// and channel blocking.
#[test]
fn conv1d_oc4_grouped_groups_2_parity() {
    let input_shape = [1usize, 4, 9];
    let weight_shape = [8usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_grouped_groups_2",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        2,
    );
}

/// `groups=4, c_out=16` → `c_out_per_group=4`, same as above with more
/// groups, so each group's oc4 chunk pulls from a narrower `c_in` slice.
#[test]
fn conv1d_oc4_grouped_groups_4_parity() {
    let input_shape = [1usize, 8, 8];
    let weight_shape = [16usize, 2, 2];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_grouped_groups_4",
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

/// `groups=2, c_out=10` → `c_out_per_group=5`, ragged AND grouped: the
/// second oc4 chunk within each group has masked lanes, combined with group
/// offset arithmetic.
#[test]
fn conv1d_oc4_grouped_ragged_c_out_10_parity() {
    let input_shape = [1usize, 4, 7];
    let weight_shape = [10usize, 2, 2];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_grouped_ragged_c_out_10",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        2,
    );
}

/// `dilation=2` at an oc4-selecting shape (`c_out=8, c_in=4`): checks the
/// dilated receptive field is applied identically across the 4 register-
/// blocked output-channel lanes.
#[test]
fn conv1d_oc4_dilation_2_parity() {
    let input_shape = [1usize, 4, 12];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_dilation_2",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        2,
        1,
    );
}

/// `stride=2` at an oc4-selecting shape (`c_out=8, c_in=4`): checks the
/// output-position stride is applied identically across the 4 blocked lanes.
#[test]
fn conv1d_oc4_stride_2_parity() {
    let input_shape = [1usize, 4, 11];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_stride_2",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        2,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// `PaddingMode::Same` at an oc4-selecting shape: `resolve_padding_1d` must
/// compute the same left/right padding regardless of which kernel runs.
#[test]
fn conv1d_oc4_padding_same_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_padding_same",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Same,
        1,
        1,
    );
}

/// `PaddingMode::Custom(left, right, 0, 0)` (asymmetric) at an oc4-selecting
/// shape.
#[test]
fn conv1d_oc4_padding_custom_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_padding_custom",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Custom(1, 2, 0, 0),
        1,
        1,
    );
}

/// Bias present at an oc4-selecting shape: the bias add must land on the
/// correct output channel for every lane, including masked-chunk lanes.
#[test]
fn conv1d_oc4_bias_present_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    let bias = conv1d_bias(weight_shape[0]);
    assert_conv1d_parity(
        "conv1d_oc4_bias_present",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}

/// Bias absent (`None`) at the same oc4-selecting shape as
/// `conv1d_oc4_bias_present_parity`, so the two tests isolate the bias path.
#[test]
fn conv1d_oc4_bias_absent_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [8usize, 4, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_parity(
        "conv1d_oc4_bias_absent",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        1,
    );
}
