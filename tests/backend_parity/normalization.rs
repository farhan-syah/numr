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

// ============================================================================
// Flat shape-per-test coverage
// ============================================================================
//
// The table-driven cases above pin ordinary data. These pin the two regimes
// that data cannot reach:
//
//   - Cancellation: `pre_norm = x + residual` centred near 2.0 with a spread of
//     ~2e-3. Subtracting the mean then discards most of the mantissa unless the
//     kernel shifts by a reference element first. Only layer_norm subtracts a
//     mean, so only it needs these; rms_norm normalizes by RMS.
//   - Block reduction: the CUDA block is `min(256, hidden_size)`, and a tree
//     reduction starting at `blockDim.x / 2` silently drops the entries above
//     the largest power of two below the block size. Every hidden_size above is
//     a power of two, so nothing else here would notice.

/// Values centred on 1.0 with a controllable spread, so `x + residual` lands
/// near 2.0 and the deviation from the mean is `spread`-sized.
fn flat_input(n: usize, spread: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i as f64) * 0.017).sin() * spread)
        .collect()
}

fn flat_residual(n: usize, spread: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i as f64) * 0.023).cos() * spread)
        .collect()
}

fn flat_weight(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.75 + ((i as f64) * 0.011).cos() * 0.25)
        .collect()
}

fn flat_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.013).sin() * 0.1).collect()
}

fn flat_grad(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.1).sin() + 0.5).collect()
}

/// Runs `fused_add_layer_norm` on CPU and each GPU backend and asserts the
/// output AND the `pre_norm` residual stream agree.
/// Dtypes whose resolution can represent a given spread around 1.0.
///
/// These cases centre inputs on 1.0 with a small spread so the variance is
/// tiny and `eps` is not negligible. F16 spacing near 1.0 is 9.77e-4, so a
/// spread of 2e-3 is barely two representable steps — the input cannot encode
/// the regime under test and the normalized output becomes quantization noise
/// rather than a measurement of the kernel. Narrow dtypes are therefore
/// excluded from the tiny-spread cases, not given a looser tolerance.
fn dtypes_resolving_spread(spread: f64) -> Vec<DType> {
    parity_dtypes(DTypeDomain::FloatsOnly, "cpu")
        .into_iter()
        .filter(|dtype| {
            let steps = match dtype {
                DType::F16 | DType::BF16 => spread / 9.77e-4,
                _ => f64::INFINITY,
            };
            steps >= 64.0
        })
        .collect()
}

fn assert_fused_add_layer_norm_parity(label: &str, batch: usize, hidden: usize, spread: f64) {
    let shape = [batch, hidden];
    let n = batch * hidden;
    let (x_data, r_data) = (flat_input(n, spread), flat_residual(n, spread));
    let (w_data, b_data) = (flat_weight(hidden), flat_bias(hidden));
    let eps = 1e-5f32;

    for dtype in dtypes_resolving_spread(spread) {
        let (cpu_client, cpu_device) = create_cpu_client();
        // A macro, not a closure: the same builder is used against the CPU
        // runtime and each GPU runtime, and a closure fixes its parameter types
        // at the first call site.
        macro_rules! mk {
            ($d:expr, $s:expr, $dev:expr, $cl:expr) => {
                tensor_from_f64($d, $s, dtype, $dev, $cl)
                    .unwrap_or_else(|e| panic!("{label} [{dtype:?}]: tensor build failed: {e}"))
            };
        }
        let cpu_x = mk!(&x_data, &shape, &cpu_device, &cpu_client);
        let cpu_r = mk!(&r_data, &shape, &cpu_device, &cpu_client);
        let cpu_w = mk!(&w_data, &[hidden], &cpu_device, &cpu_client);
        let cpu_b = mk!(&b_data, &[hidden], &cpu_device, &cpu_client);
        let (cpu_out, cpu_pn) = cpu_client
            .fused_add_layer_norm(&cpu_x, &cpu_r, &cpu_w, &cpu_b, eps)
            .unwrap_or_else(|e| panic!("CPU {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let x = mk!(&x_data, &shape, &device, &client);
                let r = mk!(&r_data, &shape, &device, &client);
                let w = mk!(&w_data, &[hidden], &device, &client);
                let b = mk!(&b_data, &[hidden], &device, &client);
                let (out, pn) = client
                    .fused_add_layer_norm(&x, &r, &w, &b, eps)
                    .unwrap_or_else(|e| panic!("CUDA {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("{label} out CUDA vs CPU"));
                assert_tensor_allclose(
                    &pn,
                    &cpu_pn,
                    dtype,
                    &format!("{label} pre_norm CUDA vs CPU"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let x = mk!(&x_data, &shape, &device, &client);
                let r = mk!(&r_data, &shape, &device, &client);
                let w = mk!(&w_data, &[hidden], &device, &client);
                let b = mk!(&b_data, &[hidden], &device, &client);
                let (out, pn) = client
                    .fused_add_layer_norm(&x, &r, &w, &b, eps)
                    .unwrap_or_else(|e| panic!("WGPU {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(
                    &out,
                    &cpu_out,
                    dtype,
                    &format!("{label} out WebGPU vs CPU"),
                );
                assert_tensor_allclose(
                    &pn,
                    &cpu_pn,
                    dtype,
                    &format!("{label} pre_norm WebGPU vs CPU"),
                );
            });
        }
    }
}

/// Same for the backward pass, which recomputes mean and variance from
/// `pre_norm` rather than consuming a saved statistic.
fn assert_fused_add_layer_norm_bwd_parity(label: &str, batch: usize, hidden: usize, spread: f64) {
    let shape = [batch, hidden];
    let n = batch * hidden;
    let (x_data, r_data) = (flat_input(n, spread), flat_residual(n, spread));
    let (w_data, b_data) = (flat_weight(hidden), flat_bias(hidden));
    let g_data = flat_grad(n);
    let eps = 1e-5f32;

    for dtype in dtypes_resolving_spread(spread) {
        let (cpu_client, cpu_device) = create_cpu_client();
        // A macro, not a closure: the same builder is used against the CPU
        // runtime and each GPU runtime, and a closure fixes its parameter types
        // at the first call site.
        macro_rules! mk {
            ($d:expr, $s:expr, $dev:expr, $cl:expr) => {
                tensor_from_f64($d, $s, dtype, $dev, $cl)
                    .unwrap_or_else(|e| panic!("{label} [{dtype:?}]: tensor build failed: {e}"))
            };
        }
        let cpu_x = mk!(&x_data, &shape, &cpu_device, &cpu_client);
        let cpu_r = mk!(&r_data, &shape, &cpu_device, &cpu_client);
        let cpu_w = mk!(&w_data, &[hidden], &cpu_device, &cpu_client);
        let cpu_b = mk!(&b_data, &[hidden], &cpu_device, &cpu_client);
        let cpu_g = mk!(&g_data, &shape, &cpu_device, &cpu_client);
        let (_, cpu_pn) = cpu_client
            .fused_add_layer_norm(&cpu_x, &cpu_r, &cpu_w, &cpu_b, eps)
            .unwrap_or_else(|e| panic!("CPU {label} forward [{dtype:?}]: {e}"));
        let (cpu_dir, cpu_dw, cpu_db) = cpu_client
            .fused_add_layer_norm_bwd(&cpu_g, &cpu_pn, &cpu_w, &cpu_b, eps)
            .unwrap_or_else(|e| panic!("CPU {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let w = mk!(&w_data, &[hidden], &device, &client);
                let b = mk!(&b_data, &[hidden], &device, &client);
                let g = mk!(&g_data, &shape, &device, &client);
                let (_, pn) = client
                    .fused_add_layer_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &w,
                        &b,
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("CUDA {label} forward [{dtype:?}]: {e}"));
                let (dir, dw, db) = client
                    .fused_add_layer_norm_bwd(&g, &pn, &w, &b, eps)
                    .unwrap_or_else(|e| panic!("CUDA {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(&dir, &cpu_dir, dtype, &format!("{label} d_ir CUDA vs CPU"));
                assert_tensor_allclose(&dw, &cpu_dw, dtype, &format!("{label} d_w CUDA vs CPU"));
                assert_tensor_allclose(&db, &cpu_db, dtype, &format!("{label} d_b CUDA vs CPU"));
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let w = mk!(&w_data, &[hidden], &device, &client);
                let b = mk!(&b_data, &[hidden], &device, &client);
                let g = mk!(&g_data, &shape, &device, &client);
                let (_, pn) = client
                    .fused_add_layer_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &w,
                        &b,
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("WGPU {label} forward [{dtype:?}]: {e}"));
                let (dir, dw, db) = client
                    .fused_add_layer_norm_bwd(&g, &pn, &w, &b, eps)
                    .unwrap_or_else(|e| panic!("WGPU {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(
                    &dir,
                    &cpu_dir,
                    dtype,
                    &format!("{label} d_ir WebGPU vs CPU"),
                );
                assert_tensor_allclose(&dw, &cpu_dw, dtype, &format!("{label} d_w WebGPU vs CPU"));
                assert_tensor_allclose(&db, &cpu_db, dtype, &format!("{label} d_b WebGPU vs CPU"));
            });
        }
    }
}

/// `fused_add_rms_norm` has no mean to subtract, so these shapes exist for the
/// block reduction alone.
fn assert_fused_add_rms_norm_parity(
    label: &str,
    batch: usize,
    hidden: usize,
    spread: f64,
    scale: f64,
) {
    let shape = [batch, hidden];
    let n = batch * hidden;
    // `scale` shrinks the magnitude, not the spread. rms_norm divides by
    // `sqrt(mean(x^2) + eps)`, so eps is only non-negligible when the values
    // themselves are small — the opposite knob from layer_norm, where eps sits
    // beside the variance and a small SPREAD is what exposes it.
    let x_data: Vec<f64> = flat_input(n, spread).iter().map(|v| v * scale).collect();
    let r_data: Vec<f64> = flat_residual(n, spread).iter().map(|v| v * scale).collect();
    let w_data = flat_weight(hidden);
    let eps = 1e-5f32;

    for dtype in dtypes_resolving_spread(spread) {
        let (cpu_client, cpu_device) = create_cpu_client();
        // A macro, not a closure: the same builder is used against the CPU
        // runtime and each GPU runtime, and a closure fixes its parameter types
        // at the first call site.
        macro_rules! mk {
            ($d:expr, $s:expr, $dev:expr, $cl:expr) => {
                tensor_from_f64($d, $s, dtype, $dev, $cl)
                    .unwrap_or_else(|e| panic!("{label} [{dtype:?}]: tensor build failed: {e}"))
            };
        }
        let cpu_x = mk!(&x_data, &shape, &cpu_device, &cpu_client);
        let cpu_r = mk!(&r_data, &shape, &cpu_device, &cpu_client);
        let cpu_w = mk!(&w_data, &[hidden], &cpu_device, &cpu_client);
        let (cpu_out, cpu_pn) = cpu_client
            .fused_add_rms_norm(&cpu_x, &cpu_r, &cpu_w, eps)
            .unwrap_or_else(|e| panic!("CPU {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let (out, pn) = client
                    .fused_add_rms_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &mk!(&w_data, &[hidden], &device, &client),
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("CUDA {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(&out, &cpu_out, dtype, &format!("{label} out CUDA vs CPU"));
                assert_tensor_allclose(
                    &pn,
                    &cpu_pn,
                    dtype,
                    &format!("{label} pre_norm CUDA vs CPU"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let (out, pn) = client
                    .fused_add_rms_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &mk!(&w_data, &[hidden], &device, &client),
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("WGPU {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(
                    &out,
                    &cpu_out,
                    dtype,
                    &format!("{label} out WebGPU vs CPU"),
                );
                assert_tensor_allclose(
                    &pn,
                    &cpu_pn,
                    dtype,
                    &format!("{label} pre_norm WebGPU vs CPU"),
                );
            });
        }
    }
}

fn assert_fused_add_rms_norm_bwd_parity(
    label: &str,
    batch: usize,
    hidden: usize,
    spread: f64,
    scale: f64,
) {
    let shape = [batch, hidden];
    let n = batch * hidden;
    // See `assert_fused_add_rms_norm_parity` for why rms scales magnitude.
    let x_data: Vec<f64> = flat_input(n, spread).iter().map(|v| v * scale).collect();
    let r_data: Vec<f64> = flat_residual(n, spread).iter().map(|v| v * scale).collect();
    let w_data = flat_weight(hidden);
    let g_data = flat_grad(n);
    let eps = 1e-5f32;

    for dtype in dtypes_resolving_spread(spread) {
        let (cpu_client, cpu_device) = create_cpu_client();
        // A macro, not a closure: the same builder is used against the CPU
        // runtime and each GPU runtime, and a closure fixes its parameter types
        // at the first call site.
        macro_rules! mk {
            ($d:expr, $s:expr, $dev:expr, $cl:expr) => {
                tensor_from_f64($d, $s, dtype, $dev, $cl)
                    .unwrap_or_else(|e| panic!("{label} [{dtype:?}]: tensor build failed: {e}"))
            };
        }
        let cpu_w = mk!(&w_data, &[hidden], &cpu_device, &cpu_client);
        let cpu_g = mk!(&g_data, &shape, &cpu_device, &cpu_client);
        let (_, cpu_pn) = cpu_client
            .fused_add_rms_norm(
                &mk!(&x_data, &shape, &cpu_device, &cpu_client),
                &mk!(&r_data, &shape, &cpu_device, &cpu_client),
                &cpu_w,
                eps,
            )
            .unwrap_or_else(|e| panic!("CPU {label} forward [{dtype:?}]: {e}"));
        let (cpu_dir, cpu_dw) = cpu_client
            .fused_add_rms_norm_bwd(&cpu_g, &cpu_pn, &cpu_w, eps)
            .unwrap_or_else(|e| panic!("CPU {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let w = mk!(&w_data, &[hidden], &device, &client);
                let (_, pn) = client
                    .fused_add_rms_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &w,
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("CUDA {label} forward [{dtype:?}]: {e}"));
                let (dir, dw) = client
                    .fused_add_rms_norm_bwd(&mk!(&g_data, &shape, &device, &client), &pn, &w, eps)
                    .unwrap_or_else(|e| panic!("CUDA {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(&dir, &cpu_dir, dtype, &format!("{label} d_ir CUDA vs CPU"));
                assert_tensor_allclose(&dw, &cpu_dw, dtype, &format!("{label} d_w CUDA vs CPU"));
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let w = mk!(&w_data, &[hidden], &device, &client);
                let (_, pn) = client
                    .fused_add_rms_norm(
                        &mk!(&x_data, &shape, &device, &client),
                        &mk!(&r_data, &shape, &device, &client),
                        &w,
                        eps,
                    )
                    .unwrap_or_else(|e| panic!("WGPU {label} forward [{dtype:?}]: {e}"));
                let (dir, dw) = client
                    .fused_add_rms_norm_bwd(&mk!(&g_data, &shape, &device, &client), &pn, &w, eps)
                    .unwrap_or_else(|e| panic!("WGPU {label} [{dtype:?}]: {e}"));
                assert_tensor_allclose(
                    &dir,
                    &cpu_dir,
                    dtype,
                    &format!("{label} d_ir WebGPU vs CPU"),
                );
                assert_tensor_allclose(&dw, &cpu_dw, dtype, &format!("{label} d_w WebGPU vs CPU"));
            });
        }
    }
}

// ---------------------------------------------------------------------------
// fused_add_layer_norm forward
// ---------------------------------------------------------------------------

/// Ordinary variance, so the shift is a no-op numerically. Pins the normal path.
#[test]
fn fused_add_layer_norm_ordinary_variance_parity() {
    assert_fused_add_layer_norm_parity("faln_ordinary_variance", 2, 256, 1.0);
}

/// `pre_norm` near 2.0 with a ~2e-3 spread: subtracting the mean discards most
/// of the mantissa unless the kernel shifts first.
#[test]
fn fused_add_layer_norm_tiny_variance_parity() {
    assert_fused_add_layer_norm_parity("faln_tiny_variance", 2, 32, 2e-3);
}

/// Block size 24, not a power of two, so the reduction must not start at
/// `blockDim.x / 2`.
#[test]
fn fused_add_layer_norm_odd_block_parity() {
    assert_fused_add_layer_norm_parity("faln_odd_block", 2, 24, 2e-3);
}

/// Block size 96: not a power of two either, and wide enough that the CPU takes
/// its SIMD path rather than the scalar fallback.
#[test]
fn fused_add_layer_norm_wide_odd_block_parity() {
    assert_fused_add_layer_norm_parity("faln_wide_odd_block", 2, 96, 2e-3);
}

// ---------------------------------------------------------------------------
// fused_add_layer_norm backward
// ---------------------------------------------------------------------------

#[test]
fn fused_add_layer_norm_bwd_ordinary_variance_parity() {
    assert_fused_add_layer_norm_bwd_parity("faln_bwd_ordinary_variance", 2, 256, 1.0);
}

/// Reduced spread AND a non-power-of-two block, so one shape pins both the
/// reference shift and the reduction. The backward pass recomputes mean and
/// variance from `pre_norm`, so it carries the forward cancellation exactly.
///
/// The spread here is 0.05, not the 2e-3 the forward tests use, and that is a
/// numerical requirement rather than a preference. `d_input_residual` is
/// `inv_std * (gs - mean_gs - normalized * mean_gsn)`: an already-cancelling
/// difference scaled by `inv_std`. At a 2e-3 spread `inv_std` is ~300, so the
/// intermediates run two to three orders above the smallest outputs and the
/// rounding both backends carry tracks the tensor scale, not each element's own
/// magnitude — which an output-relative tolerance cannot express. 0.05 keeps
/// `inv_std` near 30, where the outputs stay well conditioned and the shift is
/// still worth ~100x the tolerance.
#[test]
fn fused_add_layer_norm_bwd_odd_block_parity() {
    assert_fused_add_layer_norm_bwd_parity("faln_bwd_odd_block", 2, 24, 0.05);
}

/// Same regime, wide enough that the CPU takes its SIMD path.
#[test]
fn fused_add_layer_norm_bwd_wide_odd_block_parity() {
    assert_fused_add_layer_norm_bwd_parity("faln_bwd_wide_odd_block", 2, 96, 0.05);
}

// ---------------------------------------------------------------------------
// fused_add_rms_norm (no mean, so these pin the reduction only)
// ---------------------------------------------------------------------------

#[test]
fn fused_add_rms_norm_ordinary_variance_parity() {
    assert_fused_add_rms_norm_parity("farn_ordinary_variance", 2, 256, 1.0, 1.0);
}

#[test]
fn fused_add_rms_norm_odd_block_parity() {
    assert_fused_add_rms_norm_parity("farn_odd_block", 2, 24, 1.0, 1.0);
}

#[test]
fn fused_add_rms_norm_wide_odd_block_parity() {
    assert_fused_add_rms_norm_parity("farn_wide_odd_block", 2, 96, 1.0, 1.0);
}

#[test]
fn fused_add_rms_norm_bwd_ordinary_variance_parity() {
    assert_fused_add_rms_norm_bwd_parity("farn_bwd_ordinary_variance", 2, 256, 1.0, 1.0);
}

#[test]
fn fused_add_rms_norm_bwd_odd_block_parity() {
    assert_fused_add_rms_norm_bwd_parity("farn_bwd_odd_block", 2, 24, 1.0, 1.0);
}

#[test]
fn fused_add_rms_norm_bwd_wide_odd_block_parity() {
    assert_fused_add_rms_norm_bwd_parity("farn_bwd_wide_odd_block", 2, 96, 1.0, 1.0);
}

/// Magnitude ~1e-3, so `mean(x^2)` lands near eps and `1 / sqrt(mean + eps)`
/// changes by ~2x if eps is dropped or applied outside the root. Nothing above
/// discriminates that for rms in F32: at magnitude 1.0 the mean square is ~4 and
/// a 1e-5 eps moves the result by ~1e-6 relative, under the F32 tolerance.
#[test]
fn fused_add_rms_norm_eps_dominant_parity() {
    assert_fused_add_rms_norm_parity("farn_eps_dominant", 2, 32, 1.0, 1e-3);
}

#[test]
fn fused_add_rms_norm_bwd_eps_dominant_parity() {
    assert_fused_add_rms_norm_bwd_parity("farn_bwd_eps_dominant", 2, 32, 1.0, 1e-3);
}
