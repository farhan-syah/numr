// Backend parity for CUDA depthwise_conv2d (the DIRECT kernel, not a GEMM
// path — `src/runtime/cuda/kernels/conv.rs`).
//
// The only pre-existing depthwise_conv2d parity test, `test_depthwise_conv2d_parity`
// in `conv.rs`, is `[1,2,3,3] @ [2,1,2,2]` — a single square shape with stride 1,
// dilation 1, and symmetric `Valid` padding. A swapped row/column index, a swapped
// stride/dilation component, or a mis-applied pad component is invisible on that
// shape. This file covers the asymmetric cases.
//
// Because this is the direct kernel (CPU and CUDA accumulate taps in the same
// order), parity uses the ordinary elementwise tolerance (`assert_tensor_allclose`),
// not the accumulation-aware GEMM tolerance.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Deterministic, non-repeating input values that vary with the flat index,
/// so a mis-strided read (wrong batch/channel/row/col stride) changes the
/// result. NOT all-ones or a repeated constant — those hide index bugs.
fn dw_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 17) as f64 - 8.0) * 0.1).collect()
}

/// Weight values that vary with the flat index, which spans `Kw` first, then
/// `Kh`, then channel — so a transposed spatial index or a mis-strided
/// channel read changes the result.
fn dw_weight(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 11) as f64 - 5.0) * 0.05).collect()
}

fn dw_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.03 - 0.05).collect()
}

/// Runs `depthwise_conv2d` on CPU and CUDA and asserts they agree within the
/// ordinary elementwise tolerance. `weight_shape` is `[channels, 1, kh, kw]`.
#[allow(clippy::too_many_arguments)]
fn assert_depthwise_conv2d_parity(
    label: &str,
    input: &[f64],
    input_shape: &[usize],
    weight: &[f64],
    weight_shape: &[usize],
    bias: Option<&[f64]>,
    stride: (usize, usize),
    padding: PaddingMode,
    dilation: (usize, usize),
) {
    let channels = weight_shape[0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(input, input_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(weight, weight_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = bias.map(|b| {
            tensor_from_f64(b, &[channels], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU bias tensor failed for {label} [{dtype:?}]: {e}"))
        });
        let cpu_result = cpu_client
            .depthwise_conv2d(&cpu_in, &cpu_w, cpu_b.as_ref(), stride, padding, dilation)
            .unwrap_or_else(|e| panic!("CPU depthwise_conv2d failed for {label} [{dtype:?}]: {e}"));

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
                    tensor_from_f64(bd, &[channels], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}")
                        })
                });
                let result = cuda_client
                    .depthwise_conv2d(&x, &w, b.as_ref(), stride, padding, dilation)
                    .unwrap_or_else(|e| {
                        panic!("CUDA depthwise_conv2d failed for {label} [{dtype:?}]: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Asymmetric spatial geometry: `kh != kw` AND `height != width`, so
/// `output_h != output_w`. A swapped row/column index in the direct kernel
/// is invisible on square shapes — this is the single most valuable case.
#[test]
fn depthwise_conv2d_asymmetric_hw_parity() {
    let input_shape = [1usize, 4, 10, 7];
    let weight_shape = [4usize, 1, 5, 3];
    let input = dw_input(input_shape.iter().product());
    let weight = dw_weight(weight_shape.iter().product());
    assert_depthwise_conv2d_parity(
        "depthwise_conv2d_asymmetric_hw",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
    );
}

/// `stride_h != stride_w` and `dilation_h != dilation_w` on a non-square
/// input, so a swap of the two stride or two dilation parameters is caught.
#[test]
fn depthwise_conv2d_asymmetric_stride_dilation_parity() {
    let input_shape = [1usize, 4, 13, 9];
    let weight_shape = [4usize, 1, 3, 3];
    let input = dw_input(input_shape.iter().product());
    let weight = dw_weight(weight_shape.iter().product());
    assert_depthwise_conv2d_parity(
        "depthwise_conv2d_asymmetric_stride_dilation",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        (2, 1),
        PaddingMode::Valid,
        (1, 2),
    );
}

/// `PaddingMode::Custom` with four different pad values, plus bias, so a
/// swapped or mis-applied pad component is caught, and per-channel bias
/// addition is confirmed.
#[test]
fn depthwise_conv2d_asymmetric_padding_parity() {
    let input_shape = [1usize, 3, 8, 6];
    let weight_shape = [3usize, 1, 3, 3];
    let input = dw_input(input_shape.iter().product());
    let weight = dw_weight(weight_shape.iter().product());
    let bias = dw_bias(weight_shape[0]);
    assert_depthwise_conv2d_parity(
        "depthwise_conv2d_asymmetric_padding",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        (1, 1),
        PaddingMode::Custom(2, 0, 1, 3),
        (1, 1),
    );
}

/// `batch = 2` with more channels (8) and a non-square input, plus bias, so
/// a wrong batch or channel stride in the direct kernel is caught.
#[test]
fn depthwise_conv2d_batch_multichannel_parity() {
    let input_shape = [2usize, 8, 11, 9];
    let weight_shape = [8usize, 1, 3, 3];
    let input = dw_input(input_shape.iter().product());
    let weight = dw_weight(weight_shape.iter().product());
    let bias = dw_bias(weight_shape[0]);
    assert_depthwise_conv2d_parity(
        "depthwise_conv2d_batch_multichannel",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        (1, 1),
        PaddingMode::Valid,
        (1, 1),
    );
}
