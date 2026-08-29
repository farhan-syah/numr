// Backend parity tests for fused add+normalization operations (NormalizationOps trait)
//
// Tests: fused_add_rms_norm, fused_add_layer_norm (forward)
//        fused_add_rms_norm_bwd, fused_add_layer_norm_bwd (backward)
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.

use numr::dtype::DType;
use numr::ops::NormalizationOps;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// Test Data
// ============================================================================

struct FusedNormTestCase {
    x: Vec<f64>,
    residual: Vec<f64>,
    weight: Vec<f64>,
    bias: Vec<f64>,
    shape: Vec<usize>,
    hidden_size: usize,
}

fn test_cases() -> Vec<FusedNormTestCase> {
    vec![
        // [4, 8] - simple 2D
        FusedNormTestCase {
            x: (0..32).map(|i| (i as f64) * 0.1 - 1.6).collect(),
            residual: (0..32).map(|i| (i as f64) * 0.05 + 0.1).collect(),
            weight: vec![1.0, 0.5, 2.0, 1.5, 0.8, 1.2, 0.7, 1.1],
            bias: vec![0.1, -0.1, 0.2, 0.0, -0.2, 0.3, 0.0, 0.1],
            shape: vec![4, 8],
            hidden_size: 8,
        },
        // [2, 3, 16] - 3D batched
        FusedNormTestCase {
            x: (0..96).map(|i| ((i as f64) * 0.07 - 3.0).sin()).collect(),
            residual: (0..96).map(|i| ((i as f64) * 0.13 + 1.0).cos()).collect(),
            weight: (0..16).map(|i| 0.5 + (i as f64) * 0.1).collect(),
            bias: (0..16).map(|i| -0.5 + (i as f64) * 0.05).collect(),
            shape: vec![2, 3, 16],
            hidden_size: 16,
        },
        // [1, 64] - single batch, larger hidden
        FusedNormTestCase {
            x: (0..64).map(|i| (i as f64) * 0.03 - 1.0).collect(),
            residual: (0..64).map(|i| (i as f64) * 0.02 + 0.5).collect(),
            weight: vec![1.0; 64],
            bias: vec![0.0; 64],
            shape: vec![1, 64],
            hidden_size: 64,
        },
    ]
}

// ============================================================================
// Fused Add + RMS Norm Forward
// ============================================================================

fn test_fused_add_rms_norm_parity_impl(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = test_cases();
    let eps = 1e-5f32;

    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = cases
        .iter()
        .map(|tc| {
            let x = tensor_from_f64(&tc.x, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let res =
                tensor_from_f64(&tc.residual, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let w = tensor_from_f64(
                &tc.weight,
                &[tc.hidden_size],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap();
            cpu_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let (out, pre_norm) = cuda_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("fused_add_rms_norm output CUDA vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &pre_norm,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("fused_add_rms_norm pre_norm CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let (out, pre_norm) = wgpu_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("fused_add_rms_norm output WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &pre_norm,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("fused_add_rms_norm pre_norm WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_fused_add_rms_norm_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_fused_add_rms_norm_parity_impl(dtype);
    }
}

// ============================================================================
// Fused Add + Layer Norm Forward
// ============================================================================

fn test_fused_add_layer_norm_parity_impl(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = test_cases();
    let eps = 1e-5f32;

    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = cases
        .iter()
        .map(|tc| {
            let x = tensor_from_f64(&tc.x, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let res =
                tensor_from_f64(&tc.residual, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let w = tensor_from_f64(
                &tc.weight,
                &[tc.hidden_size],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap();
            let b = tensor_from_f64(&tc.bias, &[tc.hidden_size], dtype, &cpu_device, &cpu_client)
                .unwrap();
            cpu_client
                .fused_add_layer_norm(&x, &res, &w, &b, eps)
                .unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let b = tensor_from_f64(
                    &tc.bias,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let (out, pre_norm) = cuda_client
                    .fused_add_layer_norm(&x, &res, &w, &b, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("fused_add_layer_norm output CUDA vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &pre_norm,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("fused_add_layer_norm pre_norm CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let b = tensor_from_f64(
                    &tc.bias,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let (out, pre_norm) = wgpu_client
                    .fused_add_layer_norm(&x, &res, &w, &b, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("fused_add_layer_norm output WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &pre_norm,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("fused_add_layer_norm pre_norm WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_fused_add_layer_norm_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_fused_add_layer_norm_parity_impl(dtype);
    }
}

// ============================================================================
// Fused Add + RMS Norm Backward
// ============================================================================

fn test_fused_add_rms_norm_bwd_parity_impl(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = test_cases();
    let eps = 1e-5f32;

    // First compute pre_norm via forward, then test backward
    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = cases
        .iter()
        .map(|tc| {
            let x = tensor_from_f64(&tc.x, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let res =
                tensor_from_f64(&tc.residual, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let w = tensor_from_f64(
                &tc.weight,
                &[tc.hidden_size],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap();
            let (_out, pre_norm) = cpu_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap();
            let grad_data: Vec<f64> = (0..tc.x.len())
                .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                .collect();
            let grad =
                tensor_from_f64(&grad_data, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            cpu_client
                .fused_add_rms_norm_bwd(&grad, &pre_norm, &w, eps)
                .unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let (_out, pre_norm) = cuda_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap();
                let grad_data: Vec<f64> = (0..tc.x.len())
                    .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                    .collect();
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let (d_input_res, d_weight) = cuda_client
                    .fused_add_rms_norm_bwd(&grad, &pre_norm, &w, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_res,
                    &cpu_results[idx].0,
                    dtype,
                    &format!(
                        "fused_add_rms_norm_bwd d_input_residual CUDA vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_weight,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("fused_add_rms_norm_bwd d_weight CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let (_out, pre_norm) = wgpu_client.fused_add_rms_norm(&x, &res, &w, eps).unwrap();
                let grad_data: Vec<f64> = (0..tc.x.len())
                    .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                    .collect();
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let (d_input_res, d_weight) = wgpu_client
                    .fused_add_rms_norm_bwd(&grad, &pre_norm, &w, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_res,
                    &cpu_results[idx].0,
                    dtype,
                    &format!(
                        "fused_add_rms_norm_bwd d_input_residual WebGPU vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_weight,
                    &cpu_results[idx].1,
                    dtype,
                    &format!(
                        "fused_add_rms_norm_bwd d_weight WebGPU vs CPU [{dtype:?}] case {idx}"
                    ),
                );
            }
        });
    }
}

#[test]
fn test_fused_add_rms_norm_bwd_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_fused_add_rms_norm_bwd_parity_impl(dtype);
    }
}

// ============================================================================
// Fused Add + Layer Norm Backward
// ============================================================================

fn test_fused_add_layer_norm_bwd_parity_impl(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = test_cases();
    let eps = 1e-5f32;

    let cpu_results: Vec<(
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
        Tensor<numr::runtime::cpu::CpuRuntime>,
    )> = cases
        .iter()
        .map(|tc| {
            let x = tensor_from_f64(&tc.x, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let res =
                tensor_from_f64(&tc.residual, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let w = tensor_from_f64(
                &tc.weight,
                &[tc.hidden_size],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap();
            let b = tensor_from_f64(&tc.bias, &[tc.hidden_size], dtype, &cpu_device, &cpu_client)
                .unwrap();
            let (_out, pre_norm) = cpu_client
                .fused_add_layer_norm(&x, &res, &w, &b, eps)
                .unwrap();
            let grad_data: Vec<f64> = (0..tc.x.len())
                .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                .collect();
            let grad =
                tensor_from_f64(&grad_data, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            cpu_client
                .fused_add_layer_norm_bwd(&grad, &pre_norm, &w, &b, eps)
                .unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let b = tensor_from_f64(
                    &tc.bias,
                    &[tc.hidden_size],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap();
                let (_out, pre_norm) = cuda_client
                    .fused_add_layer_norm(&x, &res, &w, &b, eps)
                    .unwrap();
                let grad_data: Vec<f64> = (0..tc.x.len())
                    .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                    .collect();
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap();
                let (d_input_res, d_weight, d_bias) = cuda_client
                    .fused_add_layer_norm_bwd(&grad, &pre_norm, &w, &b, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_res,
                    &cpu_results[idx].0,
                    dtype,
                    &format!(
                        "fused_add_layer_norm_bwd d_input_residual CUDA vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_weight,
                    &cpu_results[idx].1,
                    dtype,
                    &format!(
                        "fused_add_layer_norm_bwd d_weight CUDA vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_bias,
                    &cpu_results[idx].2,
                    dtype,
                    &format!("fused_add_layer_norm_bwd d_bias CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let x =
                    tensor_from_f64(&tc.x, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let res =
                    tensor_from_f64(&tc.residual, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let w = tensor_from_f64(
                    &tc.weight,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let b = tensor_from_f64(
                    &tc.bias,
                    &[tc.hidden_size],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap();
                let (_out, pre_norm) = wgpu_client
                    .fused_add_layer_norm(&x, &res, &w, &b, eps)
                    .unwrap();
                let grad_data: Vec<f64> = (0..tc.x.len())
                    .map(|i| ((i as f64) * 0.1).sin() + 0.5)
                    .collect();
                let grad =
                    tensor_from_f64(&grad_data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap();
                let (d_input_res, d_weight, d_bias) = wgpu_client
                    .fused_add_layer_norm_bwd(&grad, &pre_norm, &w, &b, eps)
                    .unwrap();
                assert_tensor_allclose(
                    &d_input_res,
                    &cpu_results[idx].0,
                    dtype,
                    &format!(
                        "fused_add_layer_norm_bwd d_input_residual WebGPU vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_weight,
                    &cpu_results[idx].1,
                    dtype,
                    &format!(
                        "fused_add_layer_norm_bwd d_weight WebGPU vs CPU [{dtype:?}] case {idx}"
                    ),
                );
                assert_tensor_allclose(
                    &d_bias,
                    &cpu_results[idx].2,
                    dtype,
                    &format!(
                        "fused_add_layer_norm_bwd d_bias WebGPU vs CPU [{dtype:?}] case {idx}"
                    ),
                );
            }
        });
    }
}

#[test]
fn test_fused_add_layer_norm_bwd_parity() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_fused_add_layer_norm_bwd_parity_impl(dtype);
    }
}
