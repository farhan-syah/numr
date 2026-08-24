// Backend parity tests for StatisticalOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::StatisticalOps;
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
// Test Utilities
// ============================================================================

/// Helper to check if dtype is floating-point (for statistical ops that require it)
fn is_float_dtype(dtype: DType) -> bool {
    matches!(dtype, DType::F16 | DType::BF16 | DType::F32 | DType::F64)
}

/// Helper to get floating-point dtypes only
fn float_dtypes(backend: &str) -> Vec<DType> {
    supported_dtypes(backend)
        .into_iter()
        .filter(|&dtype| is_float_dtype(dtype))
        .collect()
}

// ============================================================================
// Covariance Tests
// ============================================================================

#[test]
fn test_cov_basic_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Test case: [[1, 4], [2, 5], [3, 6]] -> cov should be [[1, 1], [1, 1]]
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let shape = vec![3, 2];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_result = cpu_client
            .cov(&cpu_tensor, None)
            .unwrap_or_else(|e| panic!("CPU cov failed for {dtype:?}: {e}"));

        // Expected result: [[1.0, 1.0], [1.0, 1.0]]
        let expected_data = vec![1.0, 1.0, 1.0, 1.0];
        let expected_shape = vec![2, 2];
        let expected = tensor_from_f64(
            &expected_data,
            &expected_shape,
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();

        assert_tensor_allclose(
            &cpu_result,
            &expected,
            dtype,
            &format!("cov CPU [{dtype:?}]"),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let cuda_result = cuda_client
                    .cov(&cuda_tensor, None)
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
                    .cov(&wgpu_tensor, None)
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

// ============================================================================
// Correlation Coefficient Tests
// ============================================================================

#[test]
fn test_corrcoef_range_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1.0, 5.0, 2.0, 3.0, 4.0, 1.0, 5.0, 2.0, 3.0, 4.0, 6.0, 7.0];
        let shape = vec![4, 3];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_result = cpu_client
            .corrcoef(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU corrcoef failed for {dtype:?}: {e}"));

        // Verify CPU result is in valid range [-1, 1]
        let cpu_data: Vec<f64> = match dtype {
            DType::F64 => cpu_result.to_vec::<f64>(),
            DType::F32 => cpu_result
                .to_vec::<f32>()
                .iter()
                .map(|&x| x as f64)
                .collect(),
            #[cfg(feature = "f16")]
            DType::F16 => cpu_result
                .to_vec::<half::f16>()
                .iter()
                .map(|&x| x.to_f64())
                .collect(),
            #[cfg(feature = "f16")]
            DType::BF16 => cpu_result
                .to_vec::<half::bf16>()
                .iter()
                .map(|&x| x.to_f64())
                .collect(),
            _ => panic!("Unsupported dtype for corrcoef: {dtype:?}"),
        };

        for (i, &v) in cpu_data.iter().enumerate() {
            assert!(
                (-1.1..=1.1).contains(&v),
                "corrcoef CPU[{i}]={v} out of range for {dtype:?}"
            );
        }

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

// ============================================================================
// Skewness and Kurtosis Tests
// ============================================================================

#[test]
fn test_skew_kurtosis_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        // Symmetric data: skew should be close to 0
        let sym_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sym_shape = vec![5];

        let sym_tensor = tensor_from_f64(&sym_data, &sym_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_skew = cpu_client
            .skew(&sym_tensor, &[], false, 0)
            .unwrap_or_else(|e| panic!("CPU skew failed for {dtype:?}: {e}"));

        // Verify skew is near 0 for symmetric data
        let skew_val: f64 = match dtype {
            DType::F64 => cpu_skew.to_vec::<f64>()[0],
            DType::F32 => cpu_skew.to_vec::<f32>()[0] as f64,
            #[cfg(feature = "f16")]
            DType::F16 => cpu_skew.to_vec::<half::f16>()[0].to_f64(),
            #[cfg(feature = "f16")]
            DType::BF16 => cpu_skew.to_vec::<half::bf16>()[0].to_f64(),
            _ => panic!("Unsupported dtype for skew: {dtype:?}"),
        };
        assert!(
            skew_val.abs() < 0.2,
            "Symmetric skew should be near 0, got {skew_val} for {dtype:?}"
        );

        // Heavy-tailed data: kurtosis should be positive
        let heavy_data = vec![-100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0];
        let heavy_shape = vec![10];

        let heavy_tensor =
            tensor_from_f64(&heavy_data, &heavy_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_kurt = cpu_client
            .kurtosis(&heavy_tensor, &[], false, 0)
            .unwrap_or_else(|e| panic!("CPU kurtosis failed for {dtype:?}: {e}"));

        // Verify kurtosis is positive for heavy-tailed data
        let kurt_val: f64 = match dtype {
            DType::F64 => cpu_kurt.to_vec::<f64>()[0],
            DType::F32 => cpu_kurt.to_vec::<f32>()[0] as f64,
            #[cfg(feature = "f16")]
            DType::F16 => cpu_kurt.to_vec::<half::f16>()[0].to_f64(),
            #[cfg(feature = "f16")]
            DType::BF16 => cpu_kurt.to_vec::<half::bf16>()[0].to_f64(),
            _ => panic!("Unsupported dtype for kurtosis: {dtype:?}"),
        };
        assert!(
            kurt_val > 0.0,
            "Heavy-tail kurtosis should be positive, got {kurt_val} for {dtype:?}"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                // Test skew
                let cuda_sym =
                    tensor_from_f64(&sym_data, &sym_shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let cuda_skew = cuda_client
                    .skew(&cuda_sym, &[], false, 0)
                    .unwrap_or_else(|e| panic!("CUDA skew failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_skew,
                    &cpu_skew,
                    dtype,
                    &format!("skew CUDA vs CPU [{dtype:?}]"),
                );

                // Test kurtosis
                let cuda_heavy =
                    tensor_from_f64(&heavy_data, &heavy_shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let cuda_kurt = cuda_client
                    .kurtosis(&cuda_heavy, &[], false, 0)
                    .unwrap_or_else(|e| panic!("CUDA kurtosis failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_kurt,
                    &cpu_kurt,
                    dtype,
                    &format!("kurtosis CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                // Test skew
                let wgpu_sym =
                    tensor_from_f64(&sym_data, &sym_shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let wgpu_skew = wgpu_client
                    .skew(&wgpu_sym, &[], false, 0)
                    .unwrap_or_else(|e| panic!("WebGPU skew failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_skew,
                    &cpu_skew,
                    dtype,
                    &format!("skew WebGPU vs CPU [{dtype:?}]"),
                );

                // Test kurtosis
                let wgpu_heavy =
                    tensor_from_f64(&heavy_data, &heavy_shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });

                let wgpu_kurt = wgpu_client
                    .kurtosis(&wgpu_heavy, &[], false, 0)
                    .unwrap_or_else(|e| panic!("WebGPU kurtosis failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_kurt,
                    &cpu_kurt,
                    dtype,
                    &format!("kurtosis WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// Mode Tests (supports all dtypes)
// ============================================================================

#[test]
fn test_mode_parity_float() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1.0, 2.0, 2.0, 2.0, 3.0];
        let shape = vec![5];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let (cpu_values, cpu_counts) = cpu_client
            .mode(&cpu_tensor, Some(0), false)
            .unwrap_or_else(|e| panic!("CPU mode failed for {dtype:?}: {e}"));

        // Expected: mode value = 2.0, count = 3
        let expected_value = vec![2.0];
        let expected_shape = vec![];
        let expected = tensor_from_f64(
            &expected_value,
            &expected_shape,
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();

        assert_tensor_allclose(
            &cpu_values,
            &expected,
            dtype,
            &format!("mode values CPU [{dtype:?}]"),
        );

        let counts_data: Vec<i64> = cpu_counts.to_vec();
        assert_eq!(
            counts_data[0], 3,
            "mode count mismatch for {dtype:?}: expected 3, got {}",
            counts_data[0]
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let (cuda_values, cuda_counts) = cuda_client
                    .mode(&cuda_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("CUDA mode failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_values,
                    &cpu_values,
                    dtype,
                    &format!("mode values CUDA vs CPU [{dtype:?}]"),
                );

                let cuda_counts_data: Vec<i64> = cuda_counts.to_vec();
                assert_eq!(
                    cuda_counts_data[0], counts_data[0],
                    "mode count CUDA vs CPU mismatch for {dtype:?}"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let (wgpu_values, wgpu_counts) = wgpu_client
                    .mode(&wgpu_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("WebGPU mode failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_values,
                    &cpu_values,
                    dtype,
                    &format!("mode values WebGPU vs CPU [{dtype:?}]"),
                );

                let wgpu_counts_data: Vec<i64> = wgpu_counts.to_vec();
                assert_eq!(
                    wgpu_counts_data[0], counts_data[0],
                    "mode count WebGPU vs CPU mismatch for {dtype:?}"
                );
            });
        }
    }
}

#[test]
fn test_mode_parity_i32() {
    for dtype in supported_dtypes("cpu") {
        if !matches!(dtype, DType::I32) {
            continue;
        }

        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1i32, 2, 2, 3, 2];
        let cpu_tensor = Tensor::try_from_slice(&data, &[5], &cpu_device).unwrap();

        let (cpu_values, cpu_counts) = cpu_client
            .mode(&cpu_tensor, Some(0), false)
            .unwrap_or_else(|e| panic!("CPU mode failed for I32: {e}"));

        let values_data: Vec<i32> = cpu_values.to_vec();
        let counts_data: Vec<i64> = cpu_counts.to_vec();

        assert_eq!(values_data[0], 2, "mode value mismatch for I32");
        assert_eq!(counts_data[0], 3, "mode count mismatch for I32");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = Tensor::try_from_slice(&data, &[5], &cuda_device).unwrap();

                let (cuda_values, cuda_counts) = cuda_client
                    .mode(&cuda_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("CUDA mode failed for I32: {e}"));

                let cuda_values_data: Vec<i32> = cuda_values.to_vec();
                let cuda_counts_data: Vec<i64> = cuda_counts.to_vec();

                assert_eq!(
                    cuda_values_data[0], values_data[0],
                    "mode value CUDA vs CPU mismatch for I32"
                );
                assert_eq!(
                    cuda_counts_data[0], counts_data[0],
                    "mode count CUDA vs CPU mismatch for I32"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = Tensor::try_from_slice(&data, &[5], &wgpu_device).unwrap();

                let (wgpu_values, wgpu_counts) = wgpu_client
                    .mode(&wgpu_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("WebGPU mode failed for I32: {e}"));

                let wgpu_values_data: Vec<i32> = wgpu_values.to_vec();
                let wgpu_counts_data: Vec<i64> = wgpu_counts.to_vec();

                assert_eq!(
                    wgpu_values_data[0], values_data[0],
                    "mode value WebGPU vs CPU mismatch for I32"
                );
                assert_eq!(
                    wgpu_counts_data[0], counts_data[0],
                    "mode count WebGPU vs CPU mismatch for I32"
                );
            });
        }
    }
}

// ============================================================================
// Quantile, Percentile, Median Tests
// ============================================================================

#[test]
fn test_quantile_percentile_median_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let shape = vec![4];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        // Test quantile (0.5 -> 2.5)
        let cpu_quantile = cpu_client
            .quantile(&cpu_tensor, 0.5, Some(0), false, "linear")
            .unwrap_or_else(|e| panic!("CPU quantile failed for {dtype:?}: {e}"));

        let expected_value = vec![2.5];
        let expected_shape = vec![];
        let expected = tensor_from_f64(
            &expected_value,
            &expected_shape,
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap();

        assert_tensor_allclose(
            &cpu_quantile,
            &expected,
            dtype,
            &format!("quantile CPU [{dtype:?}]"),
        );

        // Test percentile (50.0 -> 2.5)
        let cpu_percentile = cpu_client
            .percentile(&cpu_tensor, 50.0, Some(0), false)
            .unwrap_or_else(|e| panic!("CPU percentile failed for {dtype:?}: {e}"));

        assert_tensor_allclose(
            &cpu_percentile,
            &expected,
            dtype,
            &format!("percentile CPU [{dtype:?}]"),
        );

        // Test median (-> 2.5)
        let cpu_median = cpu_client
            .median(&cpu_tensor, Some(0), false)
            .unwrap_or_else(|e| panic!("CPU median failed for {dtype:?}: {e}"));

        assert_tensor_allclose(
            &cpu_median,
            &expected,
            dtype,
            &format!("median CPU [{dtype:?}]"),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let cuda_quantile = cuda_client
                    .quantile(&cuda_tensor, 0.5, Some(0), false, "linear")
                    .unwrap_or_else(|e| panic!("CUDA quantile failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_quantile,
                    &cpu_quantile,
                    dtype,
                    &format!("quantile CUDA vs CPU [{dtype:?}]"),
                );

                let cuda_percentile = cuda_client
                    .percentile(&cuda_tensor, 50.0, Some(0), false)
                    .unwrap_or_else(|e| panic!("CUDA percentile failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_percentile,
                    &cpu_percentile,
                    dtype,
                    &format!("percentile CUDA vs CPU [{dtype:?}]"),
                );

                let cuda_median = cuda_client
                    .median(&cuda_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("CUDA median failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &cuda_median,
                    &cpu_median,
                    dtype,
                    &format!("median CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let wgpu_quantile = wgpu_client
                    .quantile(&wgpu_tensor, 0.5, Some(0), false, "linear")
                    .unwrap_or_else(|e| panic!("WebGPU quantile failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_quantile,
                    &cpu_quantile,
                    dtype,
                    &format!("quantile WebGPU vs CPU [{dtype:?}]"),
                );

                let wgpu_percentile = wgpu_client
                    .percentile(&wgpu_tensor, 50.0, Some(0), false)
                    .unwrap_or_else(|e| panic!("WebGPU percentile failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_percentile,
                    &cpu_percentile,
                    dtype,
                    &format!("percentile WebGPU vs CPU [{dtype:?}]"),
                );

                let wgpu_median = wgpu_client
                    .median(&wgpu_tensor, Some(0), false)
                    .unwrap_or_else(|e| panic!("WebGPU median failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &wgpu_median,
                    &cpu_median,
                    dtype,
                    &format!("median WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// Invalid Input Tests
// ============================================================================

#[test]
fn test_quantile_invalid_inputs_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        // Test invalid quantile values
        assert!(
            cpu_client
                .quantile(&cpu_tensor, -0.1, Some(0), false, "linear")
                .is_err(),
            "quantile q<0 should error for {dtype:?}"
        );

        assert!(
            cpu_client
                .quantile(&cpu_tensor, 1.1, Some(0), false, "linear")
                .is_err(),
            "quantile q>1 should error for {dtype:?}"
        );

        // Test invalid percentile values
        assert!(
            cpu_client
                .percentile(&cpu_tensor, -1.0, Some(0), false)
                .is_err(),
            "percentile p<0 should error for {dtype:?}"
        );

        assert!(
            cpu_client
                .percentile(&cpu_tensor, 101.0, Some(0), false)
                .is_err(),
            "percentile p>100 should error for {dtype:?}"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                assert!(
                    cuda_client
                        .quantile(&cuda_tensor, -0.1, Some(0), false, "linear")
                        .is_err(),
                    "CUDA quantile q<0 should error for {dtype:?}"
                );

                assert!(
                    cuda_client
                        .quantile(&cuda_tensor, 1.1, Some(0), false, "linear")
                        .is_err(),
                    "CUDA quantile q>1 should error for {dtype:?}"
                );

                assert!(
                    cuda_client
                        .percentile(&cuda_tensor, -1.0, Some(0), false)
                        .is_err(),
                    "CUDA percentile p<0 should error for {dtype:?}"
                );

                assert!(
                    cuda_client
                        .percentile(&cuda_tensor, 101.0, Some(0), false)
                        .is_err(),
                    "CUDA percentile p>100 should error for {dtype:?}"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                assert!(
                    wgpu_client
                        .quantile(&wgpu_tensor, -0.1, Some(0), false, "linear")
                        .is_err(),
                    "WebGPU quantile q<0 should error for {dtype:?}"
                );

                assert!(
                    wgpu_client
                        .quantile(&wgpu_tensor, 1.1, Some(0), false, "linear")
                        .is_err(),
                    "WebGPU quantile q>1 should error for {dtype:?}"
                );

                assert!(
                    wgpu_client
                        .percentile(&wgpu_tensor, -1.0, Some(0), false)
                        .is_err(),
                    "WebGPU percentile p<0 should error for {dtype:?}"
                );

                assert!(
                    wgpu_client
                        .percentile(&wgpu_tensor, 101.0, Some(0), false)
                        .is_err(),
                    "WebGPU percentile p>100 should error for {dtype:?}"
                );
            });
        }
    }
}

// ============================================================================
// Histogram Tests
// ============================================================================

#[test]
fn test_histogram_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![0.5, 1.5, 2.5, 3.5, 4.5];
        let shape = vec![5];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let (cpu_hist, cpu_edges) = cpu_client
            .histogram(&cpu_tensor, 5, Some((0.0, 5.0)))
            .unwrap_or_else(|e| panic!("CPU histogram failed for {dtype:?}: {e}"));

        assert_eq!(
            cpu_hist.shape(),
            &[5],
            "histogram shape mismatch for {dtype:?}"
        );
        assert_eq!(
            cpu_edges.shape(),
            &[6],
            "histogram edges shape mismatch for {dtype:?}"
        );

        let hist_data: Vec<i64> = cpu_hist.to_vec();
        assert_eq!(
            hist_data,
            vec![1, 1, 1, 1, 1],
            "histogram counts mismatch for {dtype:?}"
        );

        // Verify edges
        let edges_data: Vec<f64> = match dtype {
            DType::F64 => cpu_edges.to_vec::<f64>(),
            DType::F32 => cpu_edges
                .to_vec::<f32>()
                .iter()
                .map(|&x| x as f64)
                .collect(),
            #[cfg(feature = "f16")]
            DType::F16 => cpu_edges
                .to_vec::<half::f16>()
                .iter()
                .map(|&x| x.to_f64())
                .collect(),
            #[cfg(feature = "f16")]
            DType::BF16 => cpu_edges
                .to_vec::<half::bf16>()
                .iter()
                .map(|&x| x.to_f64())
                .collect(),
            _ => panic!("Unsupported dtype for histogram: {dtype:?}"),
        };

        assert!(
            (edges_data[0] - 0.0).abs() < 1e-5,
            "histogram first edge mismatch for {dtype:?}"
        );
        assert!(
            (edges_data[5] - 5.0).abs() < 1e-5,
            "histogram last edge mismatch for {dtype:?}"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let (cuda_hist, cuda_edges) = cuda_client
                    .histogram(&cuda_tensor, 5, Some((0.0, 5.0)))
                    .unwrap_or_else(|e| panic!("CUDA histogram failed for {dtype:?}: {e}"));

                // Compare histogram counts (i64)
                let cuda_hist_data: Vec<i64> = cuda_hist.to_vec();
                assert_eq!(
                    cuda_hist_data, hist_data,
                    "histogram counts CUDA vs CPU mismatch for {dtype:?}"
                );

                // Compare edges (use assert_tensor_allclose)
                assert_tensor_allclose(
                    &cuda_edges,
                    &cpu_edges,
                    dtype,
                    &format!("histogram edges CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let (wgpu_hist, wgpu_edges) = wgpu_client
                    .histogram(&wgpu_tensor, 5, Some((0.0, 5.0)))
                    .unwrap_or_else(|e| panic!("WebGPU histogram failed for {dtype:?}: {e}"));

                // Compare histogram counts (i64)
                let wgpu_hist_data: Vec<i64> = wgpu_hist.to_vec();
                assert_eq!(
                    wgpu_hist_data, hist_data,
                    "histogram counts WebGPU vs CPU mismatch for {dtype:?}"
                );

                // Compare edges (use assert_tensor_allclose)
                assert_tensor_allclose(
                    &wgpu_edges,
                    &cpu_edges,
                    dtype,
                    &format!("histogram edges WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_histogram_invalid_inputs_parity() {
    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data = vec![1.0, 2.0, 3.0];
        let shape = vec![3];

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        // Test invalid bins
        assert!(
            cpu_client.histogram(&cpu_tensor, 0, None).is_err(),
            "histogram bins=0 should error for {dtype:?}"
        );

        // Test invalid range (min == max)
        assert!(
            cpu_client
                .histogram(&cpu_tensor, 5, Some((5.0, 5.0)))
                .is_err(),
            "histogram invalid range (min==max) should error for {dtype:?}"
        );

        // Test invalid descending range
        assert!(
            cpu_client
                .histogram(&cpu_tensor, 5, Some((10.0, 5.0)))
                .is_err(),
            "histogram invalid descending range should error for {dtype:?}"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                assert!(
                    cuda_client.histogram(&cuda_tensor, 0, None).is_err(),
                    "CUDA histogram bins=0 should error for {dtype:?}"
                );

                assert!(
                    cuda_client
                        .histogram(&cuda_tensor, 5, Some((5.0, 5.0)))
                        .is_err(),
                    "CUDA histogram invalid range should error for {dtype:?}"
                );

                assert!(
                    cuda_client
                        .histogram(&cuda_tensor, 5, Some((10.0, 5.0)))
                        .is_err(),
                    "CUDA histogram invalid descending range should error for {dtype:?}"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                assert!(
                    wgpu_client.histogram(&wgpu_tensor, 0, None).is_err(),
                    "WebGPU histogram bins=0 should error for {dtype:?}"
                );

                assert!(
                    wgpu_client
                        .histogram(&wgpu_tensor, 5, Some((5.0, 5.0)))
                        .is_err(),
                    "WebGPU histogram invalid range should error for {dtype:?}"
                );

                assert!(
                    wgpu_client
                        .histogram(&wgpu_tensor, 5, Some((10.0, 5.0)))
                        .is_err(),
                    "WebGPU histogram invalid descending range should error for {dtype:?}"
                );
            });
        }
    }
}
