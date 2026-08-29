// Backend parity tests for LinearAlgebraAlgorithms trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::algorithm::linalg::LinearAlgebraAlgorithms;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

#[test]
fn test_pinverse_parity() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let shape = vec![4, 3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .pinverse(&cpu_tensor, None)
            .unwrap_or_else(|e| panic!("CPU pinverse failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .pinverse(&cuda_tensor, None)
                    .unwrap_or_else(|e| panic!("CUDA pinverse failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("pinverse CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_result = wgpu_client
                    .pinverse(&wgpu_tensor, None)
                    .unwrap_or_else(|e| panic!("WebGPU pinverse failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_result,
                    &cpu_result,
                    dtype,
                    &format!("pinverse WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_cond_parity() {
    let data = vec![4.0, 2.0, 2.0, 3.0];
    let shape = vec![2, 2];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .cond(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU cond failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .cond(&cuda_tensor)
                    .unwrap_or_else(|e| panic!("CUDA cond failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("cond CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_result = wgpu_client
                    .cond(&wgpu_tensor)
                    .unwrap_or_else(|e| panic!("WebGPU cond failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_result,
                    &cpu_result,
                    dtype,
                    &format!("cond WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_cov_parity() {
    let data = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let shape = vec![3, 3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .cov(&cpu_tensor, Some(1))
            .unwrap_or_else(|e| panic!("CPU cov failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .cov(&cuda_tensor, Some(1))
                    .unwrap_or_else(|e| panic!("CUDA cov failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("cov CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_result = wgpu_client
                    .cov(&wgpu_tensor, Some(1))
                    .unwrap_or_else(|e| panic!("WebGPU cov failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_result,
                    &cpu_result,
                    dtype,
                    &format!("cov WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_corrcoef_parity() {
    let data = vec![1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    let shape = vec![3, 3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .corrcoef(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU corrcoef failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .corrcoef(&cuda_tensor)
                    .unwrap_or_else(|e| panic!("CUDA corrcoef failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("corrcoef CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_result = wgpu_client
                    .corrcoef(&wgpu_tensor)
                    .unwrap_or_else(|e| panic!("WebGPU corrcoef failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_result,
                    &cpu_result,
                    dtype,
                    &format!("corrcoef WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_corrcoef_zero_variance_parity() {
    let data = vec![1.0, 2.0, 1.0, 3.0, 1.0, 4.0];
    let shape = vec![3, 2];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .corrcoef(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU corrcoef failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .corrcoef(&cuda_tensor)
                    .unwrap_or_else(|e| panic!("CUDA corrcoef failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("corrcoef zero-variance CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_result = wgpu_client
                    .corrcoef(&wgpu_tensor)
                    .unwrap_or_else(|e| panic!("WebGPU corrcoef failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_result,
                    &cpu_result,
                    dtype,
                    &format!("corrcoef zero-variance WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}
