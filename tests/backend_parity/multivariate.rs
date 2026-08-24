// Backend parity tests for MultivariateRandomOps trait
//
// Multivariate distributions produce stochastic samples - we validate:
// - Shape correctness
// - Dtype correctness
// - Statistical properties (mean, variance, sum constraints)
// - Consistency with the mathematical definition

use numr::dtype::DType;
use numr::ops::MultivariateRandomOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{create_cpu_client, is_dtype_supported};

// ============================================================================
// Helpers
// ============================================================================

/// Check that all values in a slice are finite (no NaN/Inf)
fn assert_all_finite_f32(vals: &[f32], name: &str) {
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            v.is_finite(),
            "{name} value at index {i} is not finite: {v}"
        );
    }
}

/// Check that the rows of a 2D slice (n_samples × k) each sum to approximately `expected_sum`
fn assert_rows_sum_to_f32(vals: &[f32], k: usize, expected_sum: f32, tol: f32, name: &str) {
    let n = vals.len() / k;
    for i in 0..n {
        let row_sum: f32 = vals[i * k..(i + 1) * k].iter().sum();
        assert!(
            (row_sum - expected_sum).abs() < tol,
            "{name} row {i} sum = {row_sum}, expected {expected_sum} ± {tol}"
        );
    }
}

/// Check that all values are non-negative
fn assert_all_non_negative_f32(vals: &[f32], name: &str) {
    for (i, &v) in vals.iter().enumerate() {
        assert!(v >= 0.0, "{name} value at index {i} is negative: {v}");
    }
}

/// Check approximate mean across columns of a 2D matrix (n_samples × k)
fn check_column_mean_f32(vals: &[f32], k: usize, expected_means: &[f32], tol: f32, name: &str) {
    let n = (vals.len() / k) as f32;
    for (j, &expected) in expected_means.iter().enumerate().take(k) {
        let col_mean: f32 = vals.iter().skip(j).step_by(k).sum::<f32>() / n;
        assert!(
            (col_mean - expected).abs() < tol,
            "{name} column {j} mean = {col_mean}, expected {expected} ± {tol}"
        );
    }
}

// ============================================================================
// multivariate_normal tests
// ============================================================================

/// Test multivariate_normal produces correct shape, dtype, and finite values on all backends
#[test]
fn test_multivariate_normal_shape_and_dtype() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let mean = Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &cpu_device).unwrap();
    let cov = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &cpu_device)
        .unwrap();
    let n_samples = 100usize;

    let result = cpu_client
        .multivariate_normal(&mean, &cov, n_samples)
        .unwrap_or_else(|e| panic!("CPU multivariate_normal failed: {e}"));

    assert_eq!(
        result.shape(),
        &[100, 2],
        "multivariate_normal shape mismatch"
    );
    assert_eq!(
        result.dtype(),
        DType::F32,
        "multivariate_normal dtype mismatch"
    );
    let vals: Vec<f32> = result.to_vec();
    assert_all_finite_f32(&vals, "multivariate_normal CPU");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let mean_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &cuda_device).unwrap();
            let cov_cuda = Tensor::<CudaRuntime>::try_from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                &[2, 2],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .multivariate_normal(&mean_cuda, &cov_cuda, n_samples)
                .unwrap_or_else(|e| panic!("CUDA multivariate_normal failed: {e}"));
            assert_eq!(result.shape(), &[100, 2]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "multivariate_normal CUDA");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", DType::F32) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            use numr::runtime::wgpu::WgpuRuntime;
            let mean_wgpu =
                Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32, 0.0], &[2], &wgpu_device).unwrap();
            let cov_wgpu = Tensor::<WgpuRuntime>::try_from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                &[2, 2],
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .multivariate_normal(&mean_wgpu, &cov_wgpu, n_samples)
                .unwrap_or_else(|e| panic!("WebGPU multivariate_normal failed: {e}"));
            assert_eq!(result.shape(), &[100, 2]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "multivariate_normal WebGPU");
        });
    }
}

/// Test multivariate_normal statistical properties: sample mean converges to true mean
#[test]
fn test_multivariate_normal_statistical_properties() {
    let true_mean = [2.0f32, -1.0f32];
    // With 5000 samples and identity cov, sample mean should be within ~0.1 of true mean

    let (cpu_client, cpu_device) = create_cpu_client();
    let mean = Tensor::<CpuRuntime>::try_from_slice(&true_mean, &[2], &cpu_device).unwrap();
    let cov = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &cpu_device)
        .unwrap();
    let result = cpu_client
        .multivariate_normal(&mean, &cov, 5000)
        .unwrap_or_else(|e| panic!("CPU multivariate_normal statistical test failed: {e}"));
    let vals: Vec<f32> = result.to_vec();
    check_column_mean_f32(&vals, 2, &true_mean, 0.1, "multivariate_normal CPU");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let mean_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&true_mean, &[2], &cuda_device).unwrap();
            let cov_cuda = Tensor::<CudaRuntime>::try_from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                &[2, 2],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .multivariate_normal(&mean_cuda, &cov_cuda, 5000)
                .unwrap_or_else(|e| {
                    panic!("CUDA multivariate_normal statistical test failed: {e}")
                });
            let vals: Vec<f32> = result.to_vec();
            check_column_mean_f32(&vals, 2, &true_mean, 0.1, "multivariate_normal CUDA");
        });
    }
}

/// Test multivariate_normal with F64 dtype
#[test]
fn test_multivariate_normal_f64() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let mean = Tensor::<CpuRuntime>::try_from_slice(&[0.0f64, 0.0], &[2], &cpu_device).unwrap();
    let cov = Tensor::<CpuRuntime>::try_from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], &cpu_device)
        .unwrap();

    let result = cpu_client
        .multivariate_normal(&mean, &cov, 100)
        .unwrap_or_else(|e| panic!("CPU multivariate_normal F64 failed: {e}"));
    assert_eq!(result.shape(), &[100, 2]);
    assert_eq!(result.dtype(), DType::F64);
    let vals: Vec<f64> = result.to_vec();
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "f64 value at index {i} is not finite: {v}");
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F64) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let mean_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[0.0f64, 0.0], &[2], &cuda_device).unwrap();
            let cov_cuda = Tensor::<CudaRuntime>::try_from_slice(
                &[1.0f64, 0.0, 0.0, 1.0],
                &[2, 2],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .multivariate_normal(&mean_cuda, &cov_cuda, 100)
                .unwrap_or_else(|e| panic!("CUDA multivariate_normal F64 failed: {e}"));
            assert_eq!(result.shape(), &[100, 2]);
            assert_eq!(result.dtype(), DType::F64);
            let vals: Vec<f64> = result.to_vec();
            for (i, &v) in vals.iter().enumerate() {
                assert!(
                    v.is_finite(),
                    "CUDA f64 value at index {i} is not finite: {v}"
                );
            }
        });
    }
}

// ============================================================================
// dirichlet tests
// ============================================================================

/// Test dirichlet produces correct shape, dtype, non-negativity, and row sums on all backends
#[test]
fn test_dirichlet_shape_and_constraints() {
    let n_samples = 200usize;

    let (cpu_client, cpu_device) = create_cpu_client();
    let alpha =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &cpu_device).unwrap();

    let result = cpu_client
        .dirichlet(&alpha, n_samples)
        .unwrap_or_else(|e| panic!("CPU dirichlet failed: {e}"));

    assert_eq!(result.shape(), &[200, 3], "dirichlet shape mismatch");
    assert_eq!(result.dtype(), DType::F32);
    let vals: Vec<f32> = result.to_vec();
    assert_all_finite_f32(&vals, "dirichlet CPU");
    assert_all_non_negative_f32(&vals, "dirichlet CPU");
    assert_rows_sum_to_f32(&vals, 3, 1.0, 1e-5, "dirichlet CPU");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let alpha_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &cuda_device)
                    .unwrap();
            let result = cuda_client
                .dirichlet(&alpha_cuda, n_samples)
                .unwrap_or_else(|e| panic!("CUDA dirichlet failed: {e}"));
            assert_eq!(result.shape(), &[200, 3]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "dirichlet CUDA");
            assert_all_non_negative_f32(&vals, "dirichlet CUDA");
            assert_rows_sum_to_f32(&vals, 3, 1.0, 1e-5, "dirichlet CUDA");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", DType::F32) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            use numr::runtime::wgpu::WgpuRuntime;
            let alpha_wgpu =
                Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 1.0, 1.0], &[3], &wgpu_device)
                    .unwrap();
            let result = wgpu_client
                .dirichlet(&alpha_wgpu, n_samples)
                .unwrap_or_else(|e| panic!("WebGPU dirichlet failed: {e}"));
            assert_eq!(result.shape(), &[200, 3]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "dirichlet WebGPU");
            assert_all_non_negative_f32(&vals, "dirichlet WebGPU");
            assert_rows_sum_to_f32(&vals, 3, 1.0, 1e-5, "dirichlet WebGPU");
        });
    }
}

/// Test dirichlet statistical properties: sample mean converges to alpha_i / sum(alpha)
#[test]
fn test_dirichlet_concentrated_mean() {
    // alpha = [10, 10, 10] -> symmetric, expected mean [1/3, 1/3, 1/3]
    let expected_means = [1.0f32 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let alpha =
        Tensor::<CpuRuntime>::try_from_slice(&[10.0f32, 10.0, 10.0], &[3], &cpu_device).unwrap();
    let result = cpu_client
        .dirichlet(&alpha, 2000)
        .unwrap_or_else(|e| panic!("CPU dirichlet concentrated mean test failed: {e}"));
    let vals: Vec<f32> = result.to_vec();
    check_column_mean_f32(
        &vals,
        3,
        &expected_means,
        0.05,
        "dirichlet CPU concentrated",
    );

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let alpha_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[10.0f32, 10.0, 10.0], &[3], &cuda_device)
                    .unwrap();
            let result = cuda_client
                .dirichlet(&alpha_cuda, 2000)
                .unwrap_or_else(|e| panic!("CUDA dirichlet concentrated mean test failed: {e}"));
            let vals: Vec<f32> = result.to_vec();
            check_column_mean_f32(
                &vals,
                3,
                &expected_means,
                0.05,
                "dirichlet CUDA concentrated",
            );
        });
    }
}

// ============================================================================
// multinomial_samples tests
// ============================================================================

/// Test multinomial_samples produces correct shape, dtype, non-negativity, and row sums on all backends
#[test]
fn test_multinomial_samples_shape_and_constraints() {
    let n_trials = 50usize;
    let n_samples = 100usize;

    let (cpu_client, cpu_device) = create_cpu_client();
    let probs =
        Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, 0.3, 0.2], &[3], &cpu_device).unwrap();

    let result = cpu_client
        .multinomial_samples(&probs, n_trials, n_samples)
        .unwrap_or_else(|e| panic!("CPU multinomial_samples failed: {e}"));

    assert_eq!(result.shape(), &[100, 3], "multinomial shape mismatch");
    assert_eq!(result.dtype(), DType::F32);
    let vals: Vec<f32> = result.to_vec();
    assert_all_finite_f32(&vals, "multinomial CPU");
    assert_all_non_negative_f32(&vals, "multinomial CPU");
    assert_rows_sum_to_f32(&vals, 3, n_trials as f32, 1e-4, "multinomial CPU");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let probs_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[0.5f32, 0.3, 0.2], &[3], &cuda_device)
                    .unwrap();
            let result = cuda_client
                .multinomial_samples(&probs_cuda, n_trials, n_samples)
                .unwrap_or_else(|e| panic!("CUDA multinomial_samples failed: {e}"));
            assert_eq!(result.shape(), &[100, 3]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "multinomial CUDA");
            assert_all_non_negative_f32(&vals, "multinomial CUDA");
            assert_rows_sum_to_f32(&vals, 3, n_trials as f32, 1e-4, "multinomial CUDA");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", DType::F32) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            use numr::runtime::wgpu::WgpuRuntime;
            let probs_wgpu =
                Tensor::<WgpuRuntime>::try_from_slice(&[0.5f32, 0.3, 0.2], &[3], &wgpu_device)
                    .unwrap();
            let result = wgpu_client
                .multinomial_samples(&probs_wgpu, n_trials, n_samples)
                .unwrap_or_else(|e| panic!("WebGPU multinomial_samples failed: {e}"));
            assert_eq!(result.shape(), &[100, 3]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "multinomial WebGPU");
            assert_all_non_negative_f32(&vals, "multinomial WebGPU");
            assert_rows_sum_to_f32(&vals, 3, n_trials as f32, 1e-4, "multinomial WebGPU");
        });
    }
}

/// Test multinomial_samples statistical properties: mean counts proportional to probs
#[test]
fn test_multinomial_mean_proportional_to_probs() {
    // Expected mean for each category = n_trials * p_i
    let n_trials = 100usize;
    let expected_means = [50.0f32, 30.0, 20.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let probs =
        Tensor::<CpuRuntime>::try_from_slice(&[0.5f32, 0.3, 0.2], &[3], &cpu_device).unwrap();
    let result = cpu_client
        .multinomial_samples(&probs, n_trials, 2000)
        .unwrap_or_else(|e| panic!("CPU multinomial mean test failed: {e}"));
    let vals: Vec<f32> = result.to_vec();
    check_column_mean_f32(&vals, 3, &expected_means, 2.0, "multinomial CPU mean");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let probs_cuda =
                Tensor::<CudaRuntime>::try_from_slice(&[0.5f32, 0.3, 0.2], &[3], &cuda_device)
                    .unwrap();
            let result = cuda_client
                .multinomial_samples(&probs_cuda, n_trials, 2000)
                .unwrap_or_else(|e| panic!("CUDA multinomial mean test failed: {e}"));
            let vals: Vec<f32> = result.to_vec();
            check_column_mean_f32(&vals, 3, &expected_means, 2.0, "multinomial CUDA mean");
        });
    }
}

// ============================================================================
// wishart tests
// ============================================================================

/// Test wishart produces correct shape, dtype, and positive diagonal elements on all backends
#[test]
fn test_wishart_shape_and_positivity() {
    let df = 5usize;
    let n_samples = 50usize;

    let (cpu_client, cpu_device) = create_cpu_client();
    let scale =
        Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2], &cpu_device)
            .unwrap();

    let result = cpu_client
        .wishart(&scale, df, n_samples)
        .unwrap_or_else(|e| panic!("CPU wishart failed: {e}"));

    assert_eq!(result.shape(), &[50, 2, 2], "wishart shape mismatch");
    assert_eq!(result.dtype(), DType::F32);
    let vals: Vec<f32> = result.to_vec();
    assert_all_finite_f32(&vals, "wishart CPU");
    // Diagonal elements (variances) must be positive
    for i in 0..n_samples {
        let base = i * 4; // 2x2 matrix
        assert!(
            vals[base] > 0.0,
            "wishart CPU sample {i}: [0,0] diagonal not positive: {}",
            vals[base]
        );
        assert!(
            vals[base + 3] > 0.0,
            "wishart CPU sample {i}: [1,1] diagonal not positive: {}",
            vals[base + 3]
        );
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let scale_cuda = Tensor::<CudaRuntime>::try_from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                &[2, 2],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .wishart(&scale_cuda, df, n_samples)
                .unwrap_or_else(|e| panic!("CUDA wishart failed: {e}"));
            assert_eq!(result.shape(), &[50, 2, 2]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "wishart CUDA");
            for i in 0..n_samples {
                let base = i * 4;
                assert!(
                    vals[base] > 0.0,
                    "wishart CUDA sample {i}: [0,0] not positive"
                );
                assert!(
                    vals[base + 3] > 0.0,
                    "wishart CUDA sample {i}: [1,1] not positive"
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", DType::F32) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            use numr::runtime::wgpu::WgpuRuntime;
            let scale_wgpu = Tensor::<WgpuRuntime>::try_from_slice(
                &[1.0f32, 0.0, 0.0, 1.0],
                &[2, 2],
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .wishart(&scale_wgpu, df, n_samples)
                .unwrap_or_else(|e| panic!("WebGPU wishart failed: {e}"));
            assert_eq!(result.shape(), &[50, 2, 2]);
            assert_eq!(result.dtype(), DType::F32);
            let vals: Vec<f32> = result.to_vec();
            assert_all_finite_f32(&vals, "wishart WebGPU");
            for i in 0..n_samples {
                let base = i * 4;
                assert!(
                    vals[base] > 0.0,
                    "wishart WebGPU sample {i}: [0,0] not positive"
                );
                assert!(
                    vals[base + 3] > 0.0,
                    "wishart WebGPU sample {i}: [1,1] not positive"
                );
            }
        });
    }
}
