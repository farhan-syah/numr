// Backend parity for the CUDA conv2d im2col + GEMM fast path
// (`src/ops/cuda/conv2d_im2col.rs`, `use_conv2d_im2col`).
//
// The only pre-existing conv2d CUDA parity test, `test_conv2d_box_blur_parity`
// in `conv.rs`, is `[1,1,3,3] @ [1,1,2,2]` — contraction 4, far below
// `MIN_CONTRACTION`, so it runs the direct kernel and never touches this path.
//
// Every shape here MUST clear `MIN_CONTRACTION`, `MIN_C_OUT`, `MAX_COL_ELEMENTS`
// and `groups == 1` in `src/ops/cuda/conv2d_im2col.rs`. If those shapes drop
// below the gate, the test does not fail — it silently starts running the
// direct kernel again and stops covering this path at all.
//
// The GEMM reassociates the sum the direct kernel accumulates tap-outer /
// channel-inner, so the result is within tolerance rather than bit-identical.
// The tolerance is therefore accumulation-aware (`gemm_long_k_tolerance`), not
// output-relative — matching `conv_transpose1d_gemm.rs`.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance,
    is_dtype_supported, parity_dtypes,
};

/// Deterministic, non-repeating, small-magnitude input values. The
/// contraction spans hundreds of terms, so large operands would let the
/// narrow dtypes drift purely from accumulation.
fn gemm_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 13) as f64 - 6.0) * 0.05).collect()
}

/// Weight values that vary with the flat index, which spans `Kw` first, then
/// `Kh`, then `C_in`, then `C_out` — so a transposed spatial index or a
/// mis-strided channel read changes the result.
fn gemm_weight(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 9) as f64 - 4.0) * 0.03).collect()
}

fn gemm_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * 0.01 - 0.02).collect()
}

/// Largest absolute operand value, which drives the accumulation-aware bound.
fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0f64, |acc, v| acc.max(v.abs()))
}

/// Runs `conv2d` on CPU and CUDA and asserts they agree within an
/// accumulation-aware bound. `weight_shape` is `[c_out, c_in, kh, kw]`; this
/// path is `groups == 1` only.
#[allow(clippy::too_many_arguments)]
fn assert_conv2d_im2col_parity(
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
    let c_out = weight_shape[0];
    let contraction = weight_shape[1] * weight_shape[2] * weight_shape[3];
    let operand_scale = max_abs(input).max(max_abs(weight));

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
            .conv2d(
                &cpu_in,
                &cpu_w,
                cpu_b.as_ref(),
                stride,
                padding,
                dilation,
                1,
            )
            .unwrap_or_else(|e| panic!("CPU conv2d failed for {label} [{dtype:?}]: {e}"));

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
                    .conv2d(&x, &w, b.as_ref(), stride, padding, dilation, 1)
                    .unwrap_or_else(|e| panic!("CUDA conv2d failed for {label} [{dtype:?}]: {e}"));
                let (rtol, atol) = gemm_long_k_tolerance(dtype, contraction, operand_scale);
                assert_tensor_allclose_tol(
                    &result,
                    &cpu_result,
                    rtol,
                    atol,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Plain geometry, simplest shape that clears every floor.
///
/// `c_in=16, kh=4, kw=4` gives contraction `16*4*4 = 256`, exactly
/// `MIN_CONTRACTION`. `c_out = 4` meets `MIN_C_OUT`. `groups = 1`.
/// `H=W=6` with `Valid` padding, stride 1, dilation 1 gives
/// `output_h=output_w=3`, so `spatial = 9` and
/// `col_elements = 1*256*9 = 2304`, far under `MAX_COL_ELEMENTS`.
#[test]
fn conv2d_im2col_plain_parity() {
    let input_shape = [1usize, 16, 6, 6];
    let weight_shape = [4usize, 16, 4, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    let bias = gemm_bias(weight_shape[0]);
    assert_conv2d_im2col_parity(
        "conv2d_im2col_plain",
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

/// Asymmetric spatial geometry: `Kh != Kw` and `H != W`, so a transposed row
/// vs column index in the gather is caught. A square shape would hide this —
/// this is the single most valuable case.
///
/// `c_in=16, kh=8, kw=4` gives contraction `16*8*4 = 512`, past
/// `MIN_CONTRACTION`. `c_out=4` meets `MIN_C_OUT`, `groups=1`. `H=10, W=8`
/// with `Valid` padding gives `output_h=3, output_w=5`, `spatial=15`, so
/// `col_elements = 1*512*15 = 7680`.
#[test]
fn conv2d_im2col_asymmetric_spatial_parity() {
    let input_shape = [1usize, 16, 10, 8];
    let weight_shape = [4usize, 16, 8, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    assert_conv2d_im2col_parity(
        "conv2d_im2col_asymmetric_spatial",
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

/// Stride and dilation both greater than 1, with asymmetric `Custom`
/// padding, so out-of-range taps are exercised in both spatial dimensions.
///
/// `c_in=16, kh=4, kw=4` gives contraction `16*4*4 = 256`, exactly
/// `MIN_CONTRACTION`. `c_out=4` meets `MIN_C_OUT`, `groups=1`. `stride=(2,3)`,
/// `dilation=(2,2)`, padding `Custom(top=1, bottom=0, left=2, right=1)`,
/// `H=9, W=11` give `output_h=2, output_w=3`, `spatial=6`, so
/// `col_elements = 1*256*6 = 1536`.
#[test]
fn conv2d_im2col_stride_dilation_asymmetric_padding_parity() {
    let input_shape = [1usize, 16, 9, 11];
    let weight_shape = [4usize, 16, 4, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    let bias = gemm_bias(weight_shape[0]);
    assert_conv2d_im2col_parity(
        "conv2d_im2col_stride_dilation_asymmetric_padding",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        (2, 3),
        PaddingMode::Custom(1, 0, 2, 1),
        (2, 2),
    );
}

/// Batched input, `batch > 1`, so the per-sample GEMM offset in the batched
/// dispatch is exercised.
///
/// `c_in=16, kh=4, kw=4` gives contraction `256`, exactly `MIN_CONTRACTION`.
/// `c_out=4` meets `MIN_C_OUT`, `groups=1`. `batch=2`, `H=W=6`, `Valid`
/// padding, stride 1, dilation 1 give `output_h=output_w=3`, `spatial=9`, so
/// `col_elements = 2*256*9 = 4608`.
#[test]
fn conv2d_im2col_batched_parity() {
    let input_shape = [2usize, 16, 6, 6];
    let weight_shape = [4usize, 16, 4, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    assert_conv2d_im2col_parity(
        "conv2d_im2col_batched",
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

/// `H_out * W_out` not a multiple of a common GEMM tile size (16, 32, 64),
/// to exercise the partial-tile path on the spatial axis.
///
/// `c_in=16, kh=4, kw=4` gives contraction `256`, exactly `MIN_CONTRACTION`.
/// `c_out=4` meets `MIN_C_OUT`, `groups=1`. `H=6, W=10` with `Valid` padding
/// give `output_h=3, output_w=7`, so `spatial = 21` — not a multiple of 16,
/// 32 or 64 — and `col_elements = 1*256*21 = 5376`.
#[test]
fn conv2d_im2col_partial_tile_parity() {
    let input_shape = [1usize, 16, 6, 10];
    let weight_shape = [4usize, 16, 4, 4];
    let input = gemm_input(input_shape.iter().product());
    let weight = gemm_weight(weight_shape.iter().product());
    let bias = gemm_bias(weight_shape[0]);
    assert_conv2d_im2col_parity(
        "conv2d_im2col_partial_tile",
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
