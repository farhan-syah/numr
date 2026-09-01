// Backend parity safety net for `conv_transpose1d` ahead of a register-
// blocking rewrite of its GPU kernel.
//
// `tests/backend_parity/conv.rs` only has two `conv_transpose1d` tests, both
// using 1-2 channels total (c_in=1/c_out=1 and c_in=2/c_out=2 depthwise).
// The upcoming rewrite blocks over output channels the same way
// `conv1d_oc4` does, so it needs coverage at real multi-channel widths and,
// crucially, at channel counts that are NOT a multiple of 4 — a channel-
// blocking bug shows up exactly as a masked/misrouted lane in the ragged
// tail chunk.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Deterministic, non-repeating input values.
fn transpose_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 13) as f64 - 6.0) * 0.25).collect()
}

/// Weight values that vary with the flat index (which spans `c_in` first,
/// then `c_out/groups`), so every output channel has a distinct kernel and a
/// channel-blocking bug that mixes lanes shows up as a value mismatch.
fn transpose_weight(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 9) as f64 - 4.0) * 0.2).collect()
}

fn transpose_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.05 - 0.075).collect()
}

/// Per-batch-distinct input: batch `b`'s slice is offset by `b * 100.0` on
/// top of the usual non-repeating pattern, so a wrong `blockIdx.z` /
/// batch-stride decode reads a different batch's numeric range instead of
/// coincidentally matching.
fn transpose_input_batched(batch: usize, per_batch: usize) -> Vec<f64> {
    (0..batch)
        .flat_map(|b| {
            (0..per_batch).map(move |i| ((i % 13) as f64 - 6.0) * 0.25 + (b as f64) * 100.0)
        })
        .collect()
}

/// Runs `conv_transpose1d` on CPU and every enabled GPU backend for every
/// float dtype and asserts the GPU result matches the CPU reference.
/// `weight_shape` is `[c_in, c_out/groups, k]`; `c_out` is derived as
/// `weight_shape[1] * groups`, matching `validate_conv_transpose1d`.
#[allow(clippy::too_many_arguments)]
fn assert_conv_transpose1d_parity(
    label: &str,
    input: &[f64],
    input_shape: &[usize],
    weight: &[f64],
    weight_shape: &[usize],
    bias: Option<&[f64]>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) {
    let c_out = weight_shape[1] * groups;

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
            .conv_transpose1d(
                &cpu_in,
                &cpu_w,
                cpu_b.as_ref(),
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            )
            .unwrap_or_else(|e| panic!("CPU conv_transpose1d failed for {label} [{dtype:?}]: {e}"));

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
                    .conv_transpose1d(
                        &x,
                        &w,
                        b.as_ref(),
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    )
                    .unwrap_or_else(|e| {
                        panic!("CUDA conv_transpose1d failed for {label} [{dtype:?}]: {e}")
                    });
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
                    .conv_transpose1d(
                        &x,
                        &w,
                        b.as_ref(),
                        stride,
                        padding,
                        output_padding,
                        dilation,
                        groups,
                    )
                    .unwrap_or_else(|e| {
                        panic!("WebGPU conv_transpose1d failed for {label} [{dtype:?}]: {e}")
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

/// `c_in=6, c_out=8`, both above the existing tests' 1-2 channel ceiling,
/// `groups=1`, no bias — isolates plain multi-channel geometry.
#[test]
fn conv_transpose1d_multichannel_c_in_6_c_out_8_parity() {
    let input_shape = [1usize, 6, 5];
    let weight_shape = [6usize, 8, 3];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_multichannel_c_in_6_c_out_8",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `c_out=6`, not a multiple of 4 — the channel-blocking rewrite will chunk
/// this into a full 4-lane group plus a 2-lane ragged tail. Bias present so
/// the tail lanes' bias add is covered too.
#[test]
fn conv_transpose1d_ragged_c_out_6_parity() {
    let input_shape = [1usize, 4, 5];
    let weight_shape = [4usize, 6, 3];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(weight_shape[1]);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_ragged_c_out_6",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `c_out=5`, not a multiple of 4 — a different ragged remainder (1 real
/// lane in the tail chunk instead of 2) than the `c_out=6` case.
#[test]
fn conv_transpose1d_ragged_c_out_5_parity() {
    let input_shape = [1usize, 3, 6];
    let weight_shape = [3usize, 5, 2];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_ragged_c_out_5",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `stride=2` combined with `dilation=2` and asymmetric `Custom` padding, at
/// a multi-channel (`c_in=4, c_out=6`) shape — the geometry the rewrite must
/// preserve exactly while also register-blocking channels.
#[test]
fn conv_transpose1d_stride_dilation_padding_parity() {
    let input_shape = [1usize, 4, 4];
    let weight_shape = [4usize, 6, 3];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_stride_dilation_padding",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        2,
        PaddingMode::Custom(1, 1, 0, 0),
        1,
        2,
        1,
    );
}

/// `groups=2, c_out_per_group=5` (`c_out=10`) — grouped AND ragged: the
/// tail chunk's masked lanes must resolve within the correct group's input
/// channels, not spill into the neighboring group.
#[test]
fn conv_transpose1d_grouped_ragged_c_out_per_group_5_parity() {
    let input_shape = [1usize, 4, 5];
    let weight_shape = [4usize, 5, 2];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(weight_shape[1] * 2);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_grouped_ragged_c_out_per_group_5",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        0,
        1,
        2,
    );
}

/// `batch=3`, `c_out/groups=4` — every prior `conv_transpose1d` test used
/// `batch=1`, so `blockIdx.z` (the oc4 kernel's batch index) was never
/// exercised. Per-batch-distinct input values turn a wrong batch stride into
/// a numeric mismatch instead of a silent pass.
#[test]
fn conv_transpose1d_oc4_batch3_parity() {
    let input_shape = [3usize, 2, 5];
    let weight_shape = [2usize, 4, 3];
    let input = transpose_input_batched(input_shape[0], input_shape[1] * input_shape[2]);
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(weight_shape[1]);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_oc4_batch3",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `batch=3`, `c_out/groups=2` (< 4) — the rewritten scalar kernel's batch
/// decode moved from `idx / (c_out * output_length)` to `blockIdx.z`; this
/// hits that path specifically, with the same per-batch-distinct data as the
/// oc4 batch case above.
#[test]
fn conv_transpose1d_scalar_batch3_parity() {
    let input_shape = [3usize, 2, 5];
    let weight_shape = [2usize, 2, 3];
    let input = transpose_input_batched(input_shape[0], input_shape[1] * input_shape[2]);
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(weight_shape[1]);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_scalar_batch3",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `groups=2` AND `dilation=2` together, `c_out_per_group=4` (oc4 path) —
/// existing tests have dilation only at `groups=1` and `groups` only at
/// `dilation=1`; the tap progression and the group's `c_in` base are
/// independent in the kernel but were never exercised jointly.
#[test]
fn conv_transpose1d_oc4_grouped_dilation_parity() {
    let input_shape = [1usize, 4, 6];
    let weight_shape = [4usize, 4, 3];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    let bias = transpose_bias(weight_shape[1] * 2);
    assert_conv_transpose1d_parity(
        "conv_transpose1d_oc4_grouped_dilation",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        1,
        PaddingMode::Valid,
        0,
        2,
        2,
    );
}

/// `stride=1` with `output_padding=1` (legal only because `dilation=2`, so
/// `output_padding < max(stride, dilation)` holds) — every other
/// `output_padding` test uses `stride=2`, so the empty-tap output positions
/// `output_padding` introduces were only ever exercised under stride 2.
#[test]
fn conv_transpose1d_oc4_output_padding_stride1_parity() {
    let input_shape = [1usize, 2, 4];
    let weight_shape = [2usize, 4, 2];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_oc4_output_padding_stride1",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        2,
        1,
    );
}

/// `output_length = 302` (> 256) — forces the oc4 kernel's block-size cap
/// and `grid.x > 1`, geometry the small shapes above never reach.
#[test]
fn conv_transpose1d_oc4_long_output_parity() {
    let input_shape = [1usize, 1, 300];
    let weight_shape = [1usize, 4, 3];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_oc4_long_output",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}

/// `kernel_size=1` — the degenerate tap-count-of-one case, at `c_out/groups
/// =4` so it still takes the oc4 path.
#[test]
fn conv_transpose1d_oc4_kernel_size_1_parity() {
    let input_shape = [1usize, 3, 5];
    let weight_shape = [3usize, 4, 1];
    let input = transpose_input(input_shape.iter().product());
    let weight = transpose_weight(weight_shape.iter().product());
    assert_conv_transpose1d_parity(
        "conv_transpose1d_oc4_kernel_size_1",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        0,
        1,
        1,
    );
}
