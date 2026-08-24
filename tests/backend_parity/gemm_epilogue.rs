// Backend parity tests for GemmEpilogueOps
//
// This module tests matmul_bias_activation, matmul_bias_residual, and
// matmul_bias_activation_bwd across all supported dtypes and backends,
// ensuring numerical consistency across CPU, CUDA, and WebGPU.

use numr::ops::{ActivationOps, BinaryOps, GemmActivation, GemmEpilogueOps, MatmulOps};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// matmul_bias_activation: 2D parity across activations, dtypes, backends
// ============================================================================

#[test]
fn test_gemm_bias_activation_none_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::None, "gemm_bias_act_none_2d");
}

#[test]
fn test_gemm_bias_activation_relu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::ReLU, "gemm_bias_act_relu_2d");
}

#[test]
fn test_gemm_bias_activation_gelu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::GELU, "gemm_bias_act_gelu_2d");
}

#[test]
fn test_gemm_bias_activation_silu_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::SiLU, "gemm_bias_act_silu_2d");
}

#[test]
fn test_gemm_bias_activation_sigmoid_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::Sigmoid, "gemm_bias_act_sigmoid_2d");
}

#[test]
fn test_gemm_bias_activation_tanh_2d_parity() {
    gemm_bias_activation_2d_parity(GemmActivation::Tanh, "gemm_bias_act_tanh_2d");
}

fn gemm_bias_activation_2d_parity(activation: GemmActivation, label: &str) {
    // [2, 3] @ [3, 2] + [2] -> [2, 2]
    let a = vec![1.0f64, 2.0, -1.0, 3.0, -2.0, 4.0];
    let b = vec![0.5f64, -0.3, 0.1, 0.7, -0.2, 0.4];
    let bias = vec![-0.1f64, 0.2];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap();
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
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, activation)
                    .unwrap();
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

// ============================================================================
// matmul_bias_activation: batched 3D parity
// ============================================================================

#[test]
fn test_gemm_bias_activation_batched_3d_parity() {
    // [2, 2, 3] @ [2, 3, 2] + [2] -> [2, 2, 2]
    let a = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = vec![
        0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let bias = vec![0.01f64, 0.02];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_act_batched CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_activation(&a_t, &b_t, &bias_t, GemmActivation::ReLU)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_act_batched WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_residual: 2D parity across dtypes and backends
// ============================================================================

#[test]
fn test_gemm_bias_residual_2d_parity() {
    let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![0.5f64, -0.3, 0.1, 0.7, -0.2, 0.4];
    let bias = vec![-0.1f64, 0.2];
    let residual = vec![1.0f64, 2.0, 3.0, 4.0];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let res_t = tensor_from_f64(&residual, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client
            .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let res_t =
                    tensor_from_f64(&residual, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client
                    .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_residual_2d CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a_t = tensor_from_f64(&a, &[2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                let b_t = tensor_from_f64(&b, &[3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let res_t =
                    tensor_from_f64(&residual, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client
                    .matmul_bias_residual(&a_t, &b_t, &bias_t, &res_t)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("gemm_bias_residual_2d WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_activation_bwd: parity across dtypes and backends
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_none_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::None, "gemm_bias_act_bwd_none");
}

#[test]
fn test_gemm_bias_activation_bwd_relu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::ReLU, "gemm_bias_act_bwd_relu");
}

#[test]
fn test_gemm_bias_activation_bwd_sigmoid_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::Sigmoid, "gemm_bias_act_bwd_sigmoid");
}

#[test]
fn test_gemm_bias_activation_bwd_tanh_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::Tanh, "gemm_bias_act_bwd_tanh");
}

#[test]
fn test_gemm_bias_activation_bwd_silu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::SiLU, "gemm_bias_act_bwd_silu");
}

#[test]
fn test_gemm_bias_activation_bwd_gelu_parity() {
    gemm_bias_activation_bwd_parity(GemmActivation::GELU, "gemm_bias_act_bwd_gelu");
}

fn gemm_bias_activation_bwd_parity(activation: GemmActivation, label: &str) {
    let a = vec![1.0f64, 2.0, 3.0, 4.0];
    let b = vec![0.5f64, 0.3, -0.1, 0.7];
    let bias = vec![0.0f64, 0.0];
    let grad = vec![1.0f64, 1.0, 1.0, 1.0];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
        let grad_t = tensor_from_f64(&grad, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
        let (cpu_da, cpu_db, cpu_dbias) = cpu_client
            .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
            .unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                let grad_t =
                    tensor_from_f64(&grad, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let (da, db, dbias) = cuda_client
                    .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &da,
                    &cpu_da,
                    dtype,
                    &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &db,
                    &cpu_db,
                    dtype,
                    &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &dbias,
                    &cpu_dbias,
                    dtype,
                    &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
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
                let grad_t =
                    tensor_from_f64(&grad, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                let (da, db, dbias) = wgpu_client
                    .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                    .unwrap();
                assert_tensor_allclose(
                    &da,
                    &cpu_da,
                    dtype,
                    &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &db,
                    &cpu_db,
                    dtype,
                    &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                );
                assert_tensor_allclose(
                    &dbias,
                    &cpu_dbias,
                    dtype,
                    &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// matmul_bias_activation_bwd: batched 3D parity
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_batched_3d_parity() {
    let a = vec![
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let b = vec![
        0.1f64, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
    ];
    let bias = vec![0.01f64, 0.02];
    let grad = vec![1.0f64; 8];

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::SiLU,
    ] {
        for dtype in supported_dtypes("cpu") {
            let (cpu_client, cpu_device) = create_cpu_client();
            let a_t = tensor_from_f64(&a, &[2, 2, 3], dtype, &cpu_device, &cpu_client).unwrap();
            let b_t = tensor_from_f64(&b, &[2, 3, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
            let grad_t =
                tensor_from_f64(&grad, &[2, 2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let (cpu_da, cpu_db, cpu_dbias) = cpu_client
                .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                .unwrap();

            assert_eq!(cpu_da.shape(), &[2, 2, 3]);
            assert_eq!(cpu_db.shape(), &[2, 3, 2]);
            assert_eq!(cpu_dbias.shape(), &[2]);

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2, 3], dtype, &cuda_device, &cuda_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 3, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2, 2], dtype, &cuda_device, &cuda_client)
                            .unwrap();
                    let label = format!("bwd_batched_{activation:?}");
                    let (da, db, dbias) = cuda_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, wgpu_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2, 3], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 3, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2, 2], dtype, &wgpu_device, &wgpu_client)
                            .unwrap();
                    let label = format!("bwd_batched_{activation:?}");
                    let (da, db, dbias) = wgpu_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                    );
                });
            }
        }
    }
}

// ============================================================================
// Helpers
// ============================================================================

/// Assert all elements of a tensor are finite, reading as the correct native dtype.
fn assert_finite<R: numr::runtime::Runtime>(
    tensor: &numr::tensor::Tensor<R>,
    dtype: numr::dtype::DType,
    label: &str,
) {
    use numr::dtype::DType;
    macro_rules! check {
        ($T:ty) => {
            for (i, val) in tensor.to_vec::<$T>().iter().enumerate() {
                let v = *val as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        };
    }
    match dtype {
        DType::F64 => check!(f64),
        DType::F32 => check!(f32),
        #[cfg(feature = "f16")]
        DType::F16 => {
            for (i, val) in tensor.to_vec::<half::f16>().iter().enumerate() {
                let v = f32::from(*val) as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        }
        #[cfg(feature = "f16")]
        DType::BF16 => {
            for (i, val) in tensor.to_vec::<half::bf16>().iter().enumerate() {
                let v = f32::from(*val) as f64;
                assert!(
                    v.is_finite(),
                    "non-finite {label} [{dtype:?}] at index {i}: {v}"
                );
            }
        }
        _ => {} // integer/bool dtypes are always "finite"
    }
}

// ============================================================================
// matmul_bias_activation_bwd: negative values / edge cases
// ============================================================================

#[test]
fn test_gemm_bias_activation_bwd_negative_values_parity() {
    let a = vec![-1.0f64, 2.0, 3.0, -4.0];
    let b = vec![-1.0f64, 0.5, 0.5, -1.0];
    let bias = vec![-0.5f64, 0.5];
    let grad = vec![1.0f64, 1.0, 1.0, 1.0];

    for activation in [
        GemmActivation::None,
        GemmActivation::ReLU,
        GemmActivation::Sigmoid,
        GemmActivation::Tanh,
        GemmActivation::SiLU,
        GemmActivation::GELU,
    ] {
        for dtype in supported_dtypes("cpu") {
            let (cpu_client, cpu_device) = create_cpu_client();
            let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let bias_t = tensor_from_f64(&bias, &[2], dtype, &cpu_device, &cpu_client).unwrap();
            let grad_t = tensor_from_f64(&grad, &[2, 2], dtype, &cpu_device, &cpu_client).unwrap();
            let (cpu_da, cpu_db, cpu_dbias) = cpu_client
                .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                .unwrap();

            // Verify finiteness on CPU reference (must read as native dtype,
            // not f64, because to_vec is a raw byte copy with no conversion)
            assert_finite(&cpu_da, dtype, &format!("d_a for {activation:?}"));
            assert_finite(&cpu_db, dtype, &format!("d_b for {activation:?}"));
            assert_finite(&cpu_dbias, dtype, &format!("d_bias for {activation:?}"));

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &cuda_device, &cuda_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                    let label = format!("bwd_neg_{activation:?}");
                    let (da, db, dbias) = cuda_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b CUDA vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias CUDA vs CPU [{dtype:?}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, wgpu_device| {
                    let a_t =
                        tensor_from_f64(&a, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let b_t =
                        tensor_from_f64(&b, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let bias_t =
                        tensor_from_f64(&bias, &[2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let grad_t =
                        tensor_from_f64(&grad, &[2, 2], dtype, &wgpu_device, &wgpu_client).unwrap();
                    let label = format!("bwd_neg_{activation:?}");
                    let (da, db, dbias) = wgpu_client
                        .matmul_bias_activation_bwd(&grad_t, &a_t, &b_t, &bias_t, activation)
                        .unwrap();
                    assert_tensor_allclose(
                        &da,
                        &cpu_da,
                        dtype,
                        &format!("{label} d_a WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &db,
                        &cpu_db,
                        dtype,
                        &format!("{label} d_b WebGPU vs CPU [{dtype:?}]"),
                    );
                    assert_tensor_allclose(
                        &dbias,
                        &cpu_dbias,
                        dtype,
                        &format!("{label} d_bias WebGPU vs CPU [{dtype:?}]"),
                    );
                });
            }
        }
    }
}

// ============================================================================
// CPU-only reference tests: fused == unfused
// ============================================================================

#[test]
fn test_gemm_bias_activation_none_matches_matmul_bias() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.1f32, 0.2], &[2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_activation(&a, &b, &bias, GemmActivation::None)
        .unwrap()
        .to_vec();
    let reference: Vec<f32> = client.matmul_bias(&a, &b, &bias).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &reference,
        "gemm_bias_act_none_matches_matmul_bias",
    );
}

#[test]
fn test_gemm_bias_activation_relu_matches_unfused() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, -1.0, 3.0, -2.0, 4.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.5f32, 0.3], &[2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_activation(&a, &b, &bias, GemmActivation::ReLU)
        .unwrap()
        .to_vec();
    let pre = client.matmul_bias(&a, &b, &bias).unwrap();
    let unfused: Vec<f32> = client.relu(&pre).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &unfused,
        "gemm_bias_act_relu_matches_unfused",
    );
}

#[test]
fn test_gemm_bias_residual_matches_unfused() {
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    let (client, dev) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &dev)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0.5f32, -0.3, 0.1, 0.7, -0.2, 0.4], &[3, 2], &dev)
        .unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[-0.1f32, 0.2], &[2], &dev).unwrap();
    let residual =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &dev).unwrap();

    let fused: Vec<f32> = client
        .matmul_bias_residual(&a, &b, &bias, &residual)
        .unwrap()
        .to_vec();
    let pre = client.matmul_bias(&a, &b, &bias).unwrap();
    let unfused: Vec<f32> = client.add(&pre, &residual).unwrap().to_vec();

    crate::backend_parity::helpers::assert_parity_f32(
        &fused,
        &unfused,
        "gemm_bias_residual_matches_unfused",
    );
}
