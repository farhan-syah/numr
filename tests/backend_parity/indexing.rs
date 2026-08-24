// Backend parity tests for IndexingOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Index tensors remain as I32/I64 (not parameterized), only data tensors vary by dtype.

use numr::error::Error;
use numr::ops::IndexingOps;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// masked_select / masked_fill tests
// ============================================================================

#[test]
fn test_masked_select_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Test case 1: 2D tensor with row mask
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let mask_row_cpu = Tensor::from_slice(&[1u8, 0, 1], &[1, 3], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .masked_select(&a_cpu, &mask_row_cpu)
            .unwrap_or_else(|e| panic!("CPU masked_select failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask_row = Tensor::from_slice(&[1u8, 0, 1], &[1, 3], &cuda_device).unwrap();

                let result = cuda_client
                    .masked_select(&a, &mask_row)
                    .unwrap_or_else(|e| panic!("CUDA masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select row CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask_row = Tensor::from_slice(&[1u32, 0, 1], &[1, 3], &wgpu_device).unwrap();

                let result = wgpu_client
                    .masked_select(&a, &mask_row)
                    .unwrap_or_else(|e| panic!("WebGPU masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select row WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_masked_select_column_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let mask_col_cpu = Tensor::from_slice(&[1u8, 0], &[2, 1], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .masked_select(&a_cpu, &mask_col_cpu)
            .unwrap_or_else(|e| panic!("CPU masked_select failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask_col = Tensor::from_slice(&[1u8, 0], &[2, 1], &cuda_device).unwrap();

                let result = cuda_client
                    .masked_select(&a, &mask_col)
                    .unwrap_or_else(|e| panic!("CUDA masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select column CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask_col = Tensor::from_slice(&[1u32, 0], &[2, 1], &wgpu_device).unwrap();

                let result = wgpu_client
                    .masked_select(&a, &mask_col)
                    .unwrap_or_else(|e| panic!("WebGPU masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select column WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_masked_select_3d_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let mask_cpu = Tensor::from_slice(&[1u8, 0], &[1, 2, 1], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .masked_select(&a_cpu, &mask_cpu)
            .unwrap_or_else(|e| panic!("CPU masked_select failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u8, 0], &[1, 2, 1], &cuda_device).unwrap();

                let result = cuda_client
                    .masked_select(&a, &mask)
                    .unwrap_or_else(|e| panic!("CUDA masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select 3D CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u32, 0], &[1, 2, 1], &wgpu_device).unwrap();

                let result = wgpu_client
                    .masked_select(&a, &mask)
                    .unwrap_or_else(|e| panic!("WebGPU masked_select failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_select 3D WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_masked_fill_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let mask_cpu = Tensor::from_slice(&[1u8, 0, 1], &[1, 3], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .masked_fill(&a_cpu, &mask_cpu, -1.0)
            .unwrap_or_else(|e| panic!("CPU masked_fill failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u8, 0, 1], &[1, 3], &cuda_device).unwrap();

                let result = cuda_client
                    .masked_fill(&a, &mask, -1.0)
                    .unwrap_or_else(|e| panic!("CUDA masked_fill failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_fill CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u32, 0, 1], &[1, 3], &wgpu_device).unwrap();

                let result = wgpu_client
                    .masked_fill(&a, &mask, -1.0)
                    .unwrap_or_else(|e| panic!("WebGPU masked_fill failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_fill WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_masked_fill_column_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let mask_cpu = Tensor::from_slice(&[1u8, 0], &[2, 1], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .masked_fill(&a_cpu, &mask_cpu, 99.0)
            .unwrap_or_else(|e| panic!("CPU masked_fill failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u8, 0], &[2, 1], &cuda_device).unwrap();

                let result = cuda_client
                    .masked_fill(&a, &mask, 99.0)
                    .unwrap_or_else(|e| panic!("CUDA masked_fill failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_fill column CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let mask = Tensor::from_slice(&[1u32, 0], &[2, 1], &wgpu_device).unwrap();

                let result = wgpu_client
                    .masked_fill(&a, &mask, 99.0)
                    .unwrap_or_else(|e| panic!("WebGPU masked_fill failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("masked_fill column WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// take / put tests (I32 indices)
// ============================================================================

#[test]
fn test_take_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx_cpu = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .take(&a_cpu, &idx_cpu)
            .unwrap_or_else(|e| panic!("CPU take failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &cuda_device).unwrap();

                let result = cuda_client
                    .take(&a, &idx)
                    .unwrap_or_else(|e| panic!("CUDA take failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("take CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &wgpu_device).unwrap();

                let result = wgpu_client
                    .take(&a, &idx)
                    .unwrap_or_else(|e| panic!("WebGPU take failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("take WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_put_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx_cpu = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &cpu_device).unwrap();
        let put_values_data = vec![1.0, 2.0, 3.0, 4.0];
        let put_values_cpu =
            tensor_from_f64(&put_values_data, &[2, 2], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_result = cpu_client
            .put(&a_cpu, &idx_cpu, &put_values_cpu)
            .unwrap_or_else(|e| panic!("CPU put failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &cuda_device).unwrap();
                let put_values =
                    tensor_from_f64(&put_values_data, &[2, 2], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let result = cuda_client
                    .put(&a, &idx, &put_values)
                    .unwrap_or_else(|e| panic!("CUDA put failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("put CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i32, 0, 2, 4], &[2, 2], &wgpu_device).unwrap();
                let put_values =
                    tensor_from_f64(&put_values_data, &[2, 2], dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let result = wgpu_client
                    .put(&a, &idx, &put_values)
                    .unwrap_or_else(|e| panic!("WebGPU put failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("put WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// take / put tests (I64 indices)
// ============================================================================

#[test]
fn test_take_i64_indices_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx_cpu = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &cpu_device).unwrap();

        let cpu_result = cpu_client
            .take(&a_cpu, &idx_cpu)
            .unwrap_or_else(|e| panic!("CPU take failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &cuda_device).unwrap();

                let result = cuda_client
                    .take(&a, &idx)
                    .unwrap_or_else(|e| panic!("CUDA take failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("take I64 indices CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &wgpu_device).unwrap();

                let result = wgpu_client
                    .take(&a, &idx)
                    .unwrap_or_else(|e| panic!("WebGPU take failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("take I64 indices WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_put_i64_indices_parity() {
    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let a_data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let a_cpu = tensor_from_f64(&a_data, &[2, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx_cpu = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &cpu_device).unwrap();
        let put_values_data = vec![1.0, 2.0, 3.0, 4.0];
        let put_values_cpu =
            tensor_from_f64(&put_values_data, &[2, 2], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_result = cpu_client
            .put(&a_cpu, &idx_cpu, &put_values_cpu)
            .unwrap_or_else(|e| panic!("CPU put failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &cuda_device).unwrap();
                let put_values =
                    tensor_from_f64(&put_values_data, &[2, 2], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let result = cuda_client
                    .put(&a, &idx, &put_values)
                    .unwrap_or_else(|e| panic!("CUDA put failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("put I64 indices CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&a_data, &[2, 3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let idx = Tensor::from_slice(&[5i64, 0, 2, 4], &[2, 2], &wgpu_device).unwrap();
                let put_values =
                    tensor_from_f64(&put_values_data, &[2, 2], dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let result = wgpu_client
                    .put(&a, &idx, &put_values)
                    .unwrap_or_else(|e| panic!("WebGPU put failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("put I64 indices WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// Error handling tests (not dtype-parameterized)
// ============================================================================

#[test]
fn test_take_put_reject_non_integer_indices() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &cpu_device,
    )
    .unwrap();
    let idx_cpu = Tensor::from_slice(&[0.0f32, 2.0], &[2], &cpu_device).unwrap();
    let put_values_cpu = Tensor::from_slice(&[1.0f32, 2.0], &[2], &cpu_device).unwrap();

    let take_err = cpu_client.take(&a_cpu, &idx_cpu).unwrap_err();
    match take_err {
        Error::InvalidArgument { arg, reason } => {
            assert_eq!(arg, "indices");
            assert!(reason.contains("I32 or I64"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }

    let put_err = cpu_client
        .put(&a_cpu, &idx_cpu, &put_values_cpu)
        .unwrap_err();
    match put_err {
        Error::InvalidArgument { arg, reason } => {
            assert_eq!(arg, "indices");
            assert!(reason.contains("I32 or I64"));
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}
