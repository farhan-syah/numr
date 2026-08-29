// Backend parity tests for ConvOps
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::ops::{ConvOps, PaddingMode};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

#[test]
fn test_conv1d_moving_average_parity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let weight = vec![1.0, 1.0, 1.0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 1, 5], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv1d(&cpu_in, &cpu_w, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap_or_else(|e| panic!("CPU conv1d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 1, 5], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
                    .unwrap_or_else(|e| panic!("CUDA conv1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv1d_moving_average CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 1, 5], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv1d(&x, &w, None, 1, PaddingMode::Valid, 1, 1)
                    .unwrap_or_else(|e| panic!("WebGPU conv1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv1d_moving_average WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_conv2d_box_blur_parity() {
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = vec![1.0; 4];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap_or_else(|e| panic!("CPU conv2d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
                    .unwrap_or_else(|e| panic!("CUDA conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv2d_box_blur CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 1, 3, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
                    .unwrap_or_else(|e| panic!("WebGPU conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv2d_box_blur WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_depthwise_conv2d_parity() {
    let input = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    let weight = vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .depthwise_conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap_or_else(|e| panic!("CPU depthwise_conv2d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                    .unwrap_or_else(|e| panic!("CUDA depthwise_conv2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("depthwise_conv2d CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .depthwise_conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1))
                    .unwrap_or_else(|e| {
                        panic!("WebGPU depthwise_conv2d failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("depthwise_conv2d WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_conv2d_invalid_groups_parity() {
    let input_data = vec![0.0; 5 * 8 * 8];
    let weight_data = vec![0.0; 10 * 3 * 3 * 3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input_data, &[1, 5, 8, 8], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(
            &weight_data,
            &[10, 3, 3, 3],
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        assert!(
            cpu_client
                .conv2d(&cpu_in, &cpu_w, None, (1, 1), PaddingMode::Valid, (1, 1), 2,)
                .is_err()
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(
                    &input_data,
                    &[1, 5, 8, 8],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(
                    &weight_data,
                    &[10, 3, 3, 3],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                assert!(
                    cuda_client
                        .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                        .is_err()
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(
                    &input_data,
                    &[1, 5, 8, 8],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(
                    &weight_data,
                    &[10, 3, 3, 3],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                assert!(
                    wgpu_client
                        .conv2d(&x, &w, None, (1, 1), PaddingMode::Valid, (1, 1), 2)
                        .is_err()
                );
            });
        }
    }
}

/// `conv_transpose1d` with stride 2 — the upsampling case alias-free
/// resamplers and vocoder decoders are built on.
///
/// The gather-form GPU kernels index differently from the CPU scatter loop, so
/// this is the test that catches a transposed-conv index error on one backend.
#[test]
fn test_conv_transpose1d_upsample_parity() {
    let input = vec![1.0, -2.0, 0.5, 3.0];
    // weight [c_in=1, c_out=1, k=3]
    let weight = vec![0.5, -1.5, 2.0];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 1, 4], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv_transpose1d(&cpu_in, &cpu_w, None, 2, PaddingMode::Valid, 0, 1, 1)
            .unwrap_or_else(|e| panic!("CPU conv_transpose1d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 1, 4], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv_transpose1d(&x, &w, None, 2, PaddingMode::Valid, 0, 1, 1)
                    .unwrap_or_else(|e| panic!("CUDA conv_transpose1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv_transpose1d_upsample CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 1, 4], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[1, 1, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv_transpose1d(&x, &w, None, 2, PaddingMode::Valid, 0, 1, 1)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU conv_transpose1d failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv_transpose1d_upsample WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Grouped (depthwise) transposed conv with bias and dilation — exercises the
/// group indexing and the `[c_in, c_out/groups, k]` weight layout, which is
/// where a backend is most likely to diverge.
#[test]
fn test_conv_transpose1d_grouped_bias_parity() {
    // 2 channels, depthwise (groups = 2), each with its own 2-tap kernel.
    let input = vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.5];
    let weight = vec![1.0, 0.5, -2.0, 3.0];
    let bias = vec![0.25, -0.75];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = tensor_from_f64(&input, &[1, 2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_w = tensor_from_f64(&weight, &[2, 1, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_b = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .conv_transpose1d(
                &cpu_in,
                &cpu_w,
                Some(&cpu_b),
                2,
                PaddingMode::Valid,
                0,
                2,
                2,
            )
            .unwrap_or_else(|e| panic!("CPU conv_transpose1d failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let bs = tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .conv_transpose1d(&x, &w, Some(&bs), 2, PaddingMode::Valid, 0, 2, 2)
                    .unwrap_or_else(|e| panic!("CUDA conv_transpose1d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv_transpose1d_grouped CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input, &[1, 2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let w = tensor_from_f64(&weight, &[2, 1, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let bs = tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .conv_transpose1d(&x, &w, Some(&bs), 2, PaddingMode::Valid, 0, 2, 2)
                    .unwrap_or_else(|e| {
                        panic!("WebGPU conv_transpose1d failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("conv_transpose1d_grouped WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}
