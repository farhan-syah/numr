// Backend parity coverage for conv1d autograd (grad_input, grad_weight,
// grad_bias).
//
// `src/autograd/var_ops/conv1d.rs` has 4 unit tests, all CPU-only. CUDA
// conv1d forward can take either the direct kernel or the im2col + GEMM fast
// path (`src/ops/cuda/conv1d_im2col.rs`, `use_conv1d_im2col`), so the forward
// a gradient is computed from now depends on shape. Forward parity is
// covered by `conv.rs` / `conv_oc4.rs` / `conv1d_depthwise.rs`; this file
// covers backward, on both sides of the im2col dispatch.

use numr::autograd::var_ops::{var_conv1d, var_sum};
use numr::autograd::{Var, backward};
use numr::ops::PaddingMode;

use crate::backend_parity::conv_oc4::{conv1d_bias, conv1d_input, conv1d_weight};
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Bias an order of magnitude larger than the conv output, so a dropped or
/// misrouted bias gradient fails outright instead of passing inside
/// tolerance.
fn bias_dominant(n: usize) -> Vec<f64> {
    (0..n).map(|i| 100.0 + (i as f64) * 10.0).collect()
}

/// Runs conv1d forward + backward on CPU and every enabled GPU backend from
/// identical inputs, and compares grad_input, grad_weight, and grad_bias
/// against the CPU reference.
#[allow(clippy::too_many_arguments)]
fn assert_conv1d_backward_parity(
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
        let cpu_input = Var::new(
            tensor_from_f64(input, input_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU input tensor failed for {label} [{dtype:?}]: {e}")),
            true,
        );
        let cpu_weight = Var::new(
            tensor_from_f64(weight, weight_shape, dtype, &cpu_device, &cpu_client).unwrap_or_else(
                |e| panic!("CPU weight tensor failed for {label} [{dtype:?}]: {e}"),
            ),
            true,
        );
        let cpu_bias = bias.map(|b| {
            Var::new(
                tensor_from_f64(b, &[c_out], dtype, &cpu_device, &cpu_client).unwrap_or_else(|e| {
                    panic!("CPU bias tensor failed for {label} [{dtype:?}]: {e}")
                }),
                true,
            )
        });
        let cpu_out = var_conv1d(
            &cpu_input,
            &cpu_weight,
            cpu_bias.as_ref(),
            stride,
            padding,
            dilation,
            groups,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU conv1d forward failed for {label} [{dtype:?}]: {e}"));
        let cpu_loss = var_sum(&cpu_out, &[], false, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU sum failed for {label} [{dtype:?}]: {e}"));
        let cpu_grads = backward(&cpu_loss, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU backward failed for {label} [{dtype:?}]: {e}"));
        let cpu_grad_input = cpu_grads
            .get(cpu_input.id())
            .expect("CPU input gradient missing")
            .contiguous()
            .expect("CPU input gradient contiguous failed");
        let cpu_grad_weight = cpu_grads
            .get(cpu_weight.id())
            .expect("CPU weight gradient missing")
            .contiguous()
            .expect("CPU weight gradient contiguous failed");
        let cpu_grad_bias = cpu_bias.as_ref().map(|b| {
            cpu_grads
                .get(b.id())
                .expect("CPU bias gradient missing")
                .contiguous()
                .expect("CPU bias gradient contiguous failed")
        });

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let in_v = Var::new(
                    tensor_from_f64(input, input_shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA input tensor failed for {label} [{dtype:?}]: {e}")
                        }),
                    true,
                );
                let w_v = Var::new(
                    tensor_from_f64(weight, weight_shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA weight tensor failed for {label} [{dtype:?}]: {e}")
                        }),
                    true,
                );
                let b_v = bias.map(|b| {
                    Var::new(
                        tensor_from_f64(b, &[c_out], dtype, &cuda_device, &cuda_client)
                            .unwrap_or_else(|e| {
                                panic!("CUDA bias tensor failed for {label} [{dtype:?}]: {e}")
                            }),
                        true,
                    )
                });
                let out = var_conv1d(
                    &in_v,
                    &w_v,
                    b_v.as_ref(),
                    stride,
                    padding,
                    dilation,
                    groups,
                    &cuda_client,
                )
                .unwrap_or_else(|e| {
                    panic!("CUDA conv1d forward failed for {label} [{dtype:?}]: {e}")
                });
                let loss = var_sum(&out, &[], false, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA sum failed for {label} [{dtype:?}]: {e}"));
                let grads = backward(&loss, &cuda_client).unwrap_or_else(|e| {
                    panic!("CUDA backward failed for {label} [{dtype:?}]: {e}")
                });

                let grad_input = grads
                    .get(in_v.id())
                    .expect("CUDA input gradient missing")
                    .contiguous()
                    .expect("CUDA input gradient contiguous failed");
                assert_tensor_allclose(
                    &grad_input,
                    &cpu_grad_input,
                    dtype,
                    &format!("{label} grad_input CUDA vs CPU [{dtype:?}]"),
                );

                let grad_weight = grads
                    .get(w_v.id())
                    .expect("CUDA weight gradient missing")
                    .contiguous()
                    .expect("CUDA weight gradient contiguous failed");
                assert_tensor_allclose(
                    &grad_weight,
                    &cpu_grad_weight,
                    dtype,
                    &format!("{label} grad_weight CUDA vs CPU [{dtype:?}]"),
                );

                if let (Some(b_v), Some(cpu_gb)) = (&b_v, &cpu_grad_bias) {
                    let grad_bias = grads
                        .get(b_v.id())
                        .expect("CUDA bias gradient missing")
                        .contiguous()
                        .expect("CUDA bias gradient contiguous failed");
                    assert_tensor_allclose(
                        &grad_bias,
                        cpu_gb,
                        dtype,
                        &format!("{label} grad_bias CUDA vs CPU [{dtype:?}]"),
                    );
                }
            });
        }
    }
}

/// `L_out = (L + pad_left + pad_right - dilation*(K-1) - 1) / stride + 1`,
/// the same formula `compute_output_size` uses.
fn output_length(
    length: usize,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
) -> usize {
    let effective_kernel = dilation * (kernel_size - 1) + 1;
    let padded = length + pad_left + pad_right;
    (padded - effective_kernel) / stride + 1
}

/// `groups=1`, contraction `c_in*K = 8*8 = 64` (meets the 64 floor exactly),
/// `c_out=4` (meets the floor exactly), `output_length=16` (meets the floor
/// exactly): lands on the im2col + GEMM path.
#[test]
fn conv1d_backward_im2col_boundary_parity() {
    let stride = 1;
    let dilation = 1;
    let kernel_size = 8usize;
    let out_len = output_length(23, kernel_size, stride, dilation, 0, 0);
    assert_eq!(out_len, 16);
    let input_shape = [1usize, 8, 23];
    let weight_shape = [4usize, 8, kernel_size];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_im2col_boundary",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        stride,
        PaddingMode::Valid,
        dilation,
        1,
    );
}

/// `groups=1`, contraction `4*16=64`, `c_out=8`, `output_length=20`,
/// `batch=3`: comfortably past every im2col floor, with a batch dimension
/// so the im2col path's reshape across `(N, groups)` is exercised, and an
/// ordinary (non-dominant) bias.
#[test]
fn conv1d_backward_im2col_batched_bias_parity() {
    let stride = 1;
    let dilation = 1;
    let kernel_size = 16usize;
    let out_len = output_length(35, kernel_size, stride, dilation, 0, 0);
    assert_eq!(out_len, 20);
    let input_shape = [3usize, 4, 35];
    let weight_shape = [8usize, 4, kernel_size];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    let bias = conv1d_bias(weight_shape[0]);
    assert_conv1d_backward_parity(
        "conv1d_backward_im2col_batched_bias",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        Some(&bias),
        stride,
        PaddingMode::Valid,
        dilation,
        1,
    );
}

/// Same channel/kernel shape as the boundary case (`c_in*K=64`, `c_out=4`,
/// `groups=1`), but `output_length=8` (< 16): fails the im2col predicate
/// only on `output_length`, so it stays on the direct kernel.
#[test]
fn conv1d_backward_direct_output_length_too_short_parity() {
    let stride = 1;
    let dilation = 1;
    let kernel_size = 8usize;
    let out_len = output_length(15, kernel_size, stride, dilation, 0, 0);
    assert_eq!(out_len, 8);
    let input_shape = [1usize, 8, 15];
    let weight_shape = [4usize, 8, kernel_size];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_direct_output_length_too_short",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        stride,
        PaddingMode::Valid,
        dilation,
        1,
    );
}

/// `groups=2`, `c_in_per_group*K = 8*8 = 64`, `c_out_per_group = 4`,
/// `output_length=16`: every im2col floor is met per-group, but `groups != 1`
/// fails the predicate on its own, so this stays on the direct kernel.
#[test]
fn conv1d_backward_direct_grouped_parity() {
    let stride = 1;
    let dilation = 1;
    let kernel_size = 8usize;
    let out_len = output_length(23, kernel_size, stride, dilation, 0, 0);
    assert_eq!(out_len, 16);
    let input_shape = [1usize, 16, 23];
    let weight_shape = [8usize, 8, kernel_size];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_direct_grouped",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        stride,
        PaddingMode::Valid,
        dilation,
        2,
    );
}

/// `groups == c_in == c_out == 4`: depthwise conv1d, `c_out_per_group = 1`
/// always, so this never reaches the im2col floor regardless of channel
/// count.
#[test]
fn conv1d_backward_depthwise_parity() {
    let input_shape = [1usize, 4, 10];
    let weight_shape = [4usize, 1, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_depthwise",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::Valid,
        1,
        // groups == c_in == c_out: the weight above is [c_out, 1, K].
        4,
    );
}

/// Bias an order of magnitude larger than the conv output: a dropped or
/// misrouted `grad_bias` fails outright instead of passing inside tolerance.
#[test]
fn conv1d_backward_bias_dominant_parity() {
    let input_shape = [1usize, 2, 6];
    let weight_shape = [3usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    let bias = bias_dominant(weight_shape[0]);
    assert_conv1d_backward_parity(
        "conv1d_backward_bias_dominant",
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

/// `stride=2`, small direct-kernel shape.
#[test]
fn conv1d_backward_stride2_parity() {
    let input_shape = [1usize, 2, 9];
    let weight_shape = [3usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_stride2",
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

/// `dilation=2`, small direct-kernel shape.
#[test]
fn conv1d_backward_dilation2_parity() {
    let input_shape = [1usize, 2, 10];
    let weight_shape = [3usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_dilation2",
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

/// Asymmetric padding (`pad_left=1, pad_right=2`): `resolve_padding_1d`
/// treats each side independently, so a bug that assumes symmetric padding
/// in the backward pass only shows up when the two sides differ.
#[test]
fn conv1d_backward_asymmetric_padding_parity() {
    let input_shape = [1usize, 2, 8];
    let weight_shape = [3usize, 2, 3];
    let input = conv1d_input(input_shape.iter().product());
    let weight = conv1d_weight(weight_shape.iter().product());
    assert_conv1d_backward_parity(
        "conv1d_backward_asymmetric_padding",
        &input,
        &input_shape,
        &weight,
        &weight_shape,
        None,
        1,
        PaddingMode::conv1d(1, 2),
        1,
        1,
    );
}
