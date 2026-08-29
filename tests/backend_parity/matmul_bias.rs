// Backend parity tests for MatmulOps::matmul_bias
//
// This module tests matmul_bias across all supported dtypes and backends,
// ensuring numerical consistency across CPU, CUDA, and WebGPU.

use numr::ops::{BinaryOps, MatmulOps};
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::assert_parity_f32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Test matmul_bias with 2D matrices across all supported dtypes and backends
#[test]
fn test_matmul_bias_2d_parity() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![5.0f64, 6.0, 7.0, 8.0];
    let bias = vec![1.0f64, 2.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("matmul_bias_2d CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("matmul_bias_2d WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Test matmul_bias with batched 3D tensors across all supported dtypes and backends
#[test]
fn test_matmul_bias_batched_parity() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![1.0f64, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0];
    let bias = vec![0.5f64, 1.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("matmul_bias_batched CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("matmul_bias_batched WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// CPU-only reference test: verify matmul_bias matches matmul + add pattern
///
/// This test is F32-only (not parameterized) because it verifies the mathematical
/// identity of the fused operation against the reference implementation.
#[test]
fn test_matmul_bias_matches_matmul_plus_bias() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &cpu_device).unwrap();
    let b = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &cpu_device).unwrap();
    let bias = Tensor::from_slice(&[0.1f32, 0.2], &[2], &cpu_device).unwrap();
    let fused: Vec<f32> = cpu_client.matmul_bias(&a, &b, &bias).unwrap().to_vec();
    let reference: Vec<f32> = cpu_client
        .add(
            &cpu_client.matmul(&a, &b).unwrap(),
            &bias.broadcast_to(&[2, 2]).unwrap(),
        )
        .unwrap()
        .to_vec();
    assert_parity_f32(&fused, &reference, "matmul_bias_matches_reference_cpu");
}

/// CPU-only test: verify matmul_bias parallelism configuration doesn't affect results
///
/// This test is F32-only (not parameterized) because it verifies that different
/// parallelism configurations produce identical numerical results on CPU.
#[test]
fn test_cpu_matmul_bias_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    let a_shape = [4, 20, 16];
    let b_shape = [4, 16, 10];
    let bias_shape = [10];
    let a_numel: usize = a_shape.iter().product();
    let b_numel: usize = b_shape.iter().product();
    let bias_numel: usize = bias_shape.iter().product();

    let a_data: Vec<f32> = (0..a_numel)
        .map(|i| (i as f32 * 0.009).sin() + (i as f32 * 0.004).cos())
        .collect();
    let b_data: Vec<f32> = (0..b_numel)
        .map(|i| (i as f32 * 0.015).cos() - (i as f32 * 0.006).sin())
        .collect();
    let bias_data: Vec<f32> = (0..bias_numel).map(|i| (i as f32 * 0.021).sin()).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &a_shape, &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &b_shape, &device).unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&bias_data, &bias_shape, &device).unwrap();

    let base: Vec<f32> = default_client.matmul_bias(&a, &b, &bias).unwrap().to_vec();
    let cfg: Vec<f32> = configured_client
        .matmul_bias(&a, &b, &bias)
        .unwrap()
        .to_vec();
    assert_parity_f32(&base, &cfg, "cpu_matmul_bias_parallelism_config");
}
