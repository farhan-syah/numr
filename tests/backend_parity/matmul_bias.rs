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
    DTypeDomain, assert_tensor_allclose, assert_tensor_allclose_tol, create_cpu_client,
    gemm_long_k_tolerance, is_dtype_supported, parity_dtypes, values_close,
};
use numr::ops::matmul_output_dtype;

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
        let bias_dtype = matmul_output_dtype(dtype);
        let bias_t = tensor_from_f64(&bias, &[2], bias_dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t = tensor_from_f64(&a, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t = tensor_from_f64(&b, &[2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], bias_dtype, &cuda_device, &cuda_client).unwrap();
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
                    tensor_from_f64(&bias, &[2], bias_dtype, &wgpu_device, &wgpu_client).unwrap();
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
        let bias_dtype = matmul_output_dtype(dtype);
        let bias_t = tensor_from_f64(&bias, &[2], bias_dtype, &cpu_device, &cpu_client).unwrap();
        let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a_t =
                    tensor_from_f64(&a, &[2, 2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let b_t =
                    tensor_from_f64(&b, &[2, 2, 2], dtype, &cuda_device, &cuda_client).unwrap();
                let bias_t =
                    tensor_from_f64(&bias, &[2], bias_dtype, &cuda_device, &cuda_client).unwrap();
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
                    tensor_from_f64(&bias, &[2], bias_dtype, &wgpu_device, &wgpu_client).unwrap();
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

// ============================================================================
// 128x128x8 tile coverage (F32): the 2x2/2x2x2 shapes above all select the
// 64x64x32 tile (`f32_batched_tile_config`: m<=64 || n<=64), so none of them
// ever exercise the 128x128x8 specialized kernel. These do.
// ============================================================================

/// Deterministic F32 fill, distinct phase per operand so a/b/bias/residual
/// don't correlate into an accidental zero.
fn deterministic_f32(n: usize, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            ((i as f32 * 0.013 + phase).sin() * 0.5) + ((i as f32 * 0.0047 + phase).cos() * 0.3)
        })
        .collect()
}

/// Exact multiples of the 128x128x8 tile: no ragged edge, isolates the
/// specialized kernel's steady-state accumulation from partial-tile handling.
#[test]
fn matmul_bias_f32_128_tile_exact_multiples_match_cpu() {
    let (m, k, n) = (256usize, 128usize, 256usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_f32_128_tile_exact_multiples CUDA vs CPU",
            );
        });
    }
}

/// M, N, and K all fail to divide the 128 tile: catches an out-of-range
/// `bias[col]` read or a mishandled partial tile at the boundary.
#[test]
fn matmul_bias_f32_128_tile_ragged_dims_match_cpu() {
    let (m, k, n) = (130usize, 70usize, 130usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_f32_128_tile_ragged_dims CUDA vs CPU",
            );
        });
    }
}

/// Ragged at a larger size: several full tile crossings plus a partial tail,
/// where an accumulated-offset bug would compound instead of showing once.
#[test]
fn matmul_bias_f32_128_tile_ragged_large_match_cpu() {
    let (m, k, n) = (500usize, 500usize, 500usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    // K=500 dot products: CPU seeds its FMA chain with bias and CUDA appends
    // bias after the reduction (see `gemm_long_k_tolerance`'s doc comment) —
    // both are correct, but their partial sums round differently. An
    // output-relative bound is the wrong instrument once the true result
    // cancels far below the ~sqrt(K)*|operand| scale the rounding tracked.
    let operand_scale = a_data
        .iter()
        .chain(b_data.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs() as f64));
    let (gemm_rtol, gemm_atol) = gemm_long_k_tolerance(numr::dtype::DType::F32, k, operand_scale);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose_tol(
                &result,
                &cpu_result,
                gemm_rtol,
                gemm_atol,
                "matmul_bias_f32_128_tile_ragged_large CUDA vs CPU",
            );
        });
    }
}

/// Both CPU and CUDA F32 results checked against an F64 ground truth,
/// computed from the SAME inputs (cast up, never regenerated) at the SAME
/// shape as `matmul_bias_f32_128_tile_ragged_large_match_cpu`.
///
/// This is a stronger statement than CPU-vs-CUDA agreement: two backends
/// could in principle agree with each other while both being wrong. Diagnosed
/// while investigating that test's failure under `tolerance_for_dtype`
/// (output-relative, K-independent) — this is the permanent record of that
/// diagnosis, not a throwaway. `gemm_long_k_tolerance` is the same
/// accumulation-scaled bound used there; a real accumulation bug (as opposed
/// to a benign bias-fold-order reassociation) would blow through it, because
/// it targets f64 truth rather than another F32 backend that could share the
/// same bug.
#[test]
fn matmul_bias_f32_128_tile_ragged_large_match_f64_truth() {
    let (m, k, n) = (500usize, 500usize, 500usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);
    let operand_scale = a_data
        .iter()
        .chain(b_data.iter())
        .fold(0.0f64, |acc, &v| acc.max(v.abs() as f64));
    let (_, gemm_atol) = gemm_long_k_tolerance(numr::dtype::DType::F32, k, operand_scale);

    let a_f64: Vec<f64> = a_data.iter().map(|&v| v as f64).collect();
    let b_f64: Vec<f64> = b_data.iter().map(|&v| v as f64).collect();
    let bias_f64: Vec<f64> = bias_data.iter().map(|&v| v as f64).collect();

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t64 = Tensor::<CpuRuntime>::from_slice(&a_f64, &[m, k], &cpu_device).unwrap();
    let b_t64 = Tensor::<CpuRuntime>::from_slice(&b_f64, &[k, n], &cpu_device).unwrap();
    let bias_t64 = Tensor::<CpuRuntime>::from_slice(&bias_f64, &[n], &cpu_device).unwrap();
    let truth: Vec<f64> = cpu_client
        .matmul_bias(&a_t64, &b_t64, &bias_t64)
        .unwrap()
        .to_vec();

    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_f32: Vec<f32> = cpu_client
        .matmul_bias(&a_t, &b_t, &bias_t)
        .unwrap()
        .to_vec();

    for (i, (&c, &t)) in cpu_f32.iter().zip(truth.iter()).enumerate() {
        assert!(
            values_close(c as f64, t, 0.0, gemm_atol),
            "matmul_bias_f32_128_tile_ragged_large CPU vs F64 truth: element {i} differs: {c} vs {t} (tol={gemm_atol:.2e})"
        );
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let cuda_f32: Vec<f32> = cuda_client
                .matmul_bias(&a_t, &b_t, &bias_t)
                .unwrap()
                .to_vec();
            for (i, (&c, &t)) in cuda_f32.iter().zip(truth.iter()).enumerate() {
                assert!(
                    values_close(c as f64, t, 0.0, gemm_atol),
                    "matmul_bias_f32_128_tile_ragged_large CUDA vs F64 truth: element {i} differs: {c} vs {t} (tol={gemm_atol:.2e})"
                );
            }
        });
    }
}

/// K % 4 == 0 && N % 4 == 0: exercises the kernel's float4 vectorized tile
/// load path at the 128 tile.
#[test]
fn matmul_bias_f32_128_tile_vectorized_load_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_f32_128_tile_vectorized_load CUDA vs CPU",
            );
        });
    }
}

/// K % 4 != 0 at a 128-tile size: forces the scalar tile load path, the
/// counterpart to the float4 path above.
#[test]
fn matmul_bias_f32_128_tile_scalar_load_match_cpu() {
    let (m, k, n) = (128usize, 130usize, 128usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_f32_128_tile_scalar_load CUDA vs CPU",
            );
        });
    }
}

/// `a_batch_count < batch` at a 128-tile size: the specialized batched kernel
/// must replicate A via `b % a_batch_count`, matching the generic kernel.
#[test]
fn matmul_bias_batched_f32_128_tile_a_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 128usize, 128usize, 128usize);
    let a_data = deterministic_f32(m * k, 0.0);
    let b_data = deterministic_f32(batch * k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[1, m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[batch, k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[1, m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[batch, k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_batched_f32_128_tile_a_broadcast CUDA vs CPU",
            );
        });
    }
}

/// `b_batch_count < batch` at a 128-tile size: mirror of the A-broadcast case
/// above, on the B operand.
#[test]
fn matmul_bias_batched_f32_128_tile_b_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 128usize, 128usize, 128usize);
    let a_data = deterministic_f32(batch * m * k, 0.0);
    let b_data = deterministic_f32(k * n, 1.7);
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[batch, m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[1, k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[batch, m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[1, k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_batched_f32_128_tile_b_broadcast CUDA vs CPU",
            );
        });
    }
}

/// K == 0 at a 128-tile size: the tiled kernel used to leave C unwritten for
/// K==0 and now must write the epilogue (zeros plus bias), same as CPU.
#[test]
fn matmul_bias_f32_128_tile_k_zero_match_cpu() {
    let (m, k, n) = (100usize, 0usize, 100usize);
    let a_data: Vec<f32> = Vec::new();
    let b_data: Vec<f32> = Vec::new();
    let bias_data = deterministic_f32(n, 3.1);

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &cpu_device).unwrap();
    let b_t = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &cpu_device).unwrap();
    let bias_t = Tensor::<CpuRuntime>::from_slice(&bias_data, &[n], &cpu_device).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", numr::dtype::DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &a_data,
                &[m, k],
                &cuda_device,
            )
            .unwrap();
            let b_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &b_data,
                &[k, n],
                &cuda_device,
            )
            .unwrap();
            let bias_t = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &bias_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(
                &result,
                &cpu_result,
                numr::dtype::DType::F32,
                "matmul_bias_f32_128_tile_k_zero CUDA vs CPU",
            );
        });
    }
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
