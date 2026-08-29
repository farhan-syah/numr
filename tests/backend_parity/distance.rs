// Backend parity tests for DistanceOps trait
//
// Tests: cdist, pdist, squareform, squareform_inverse
// CPU is the reference implementation; CUDA and WebGPU must match.

use numr::dtype::DType;
use numr::ops::{DistanceMetric, DistanceOps};

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// cdist
// ============================================================================

struct CdistCase {
    x: Vec<f64>,
    x_shape: Vec<usize>,
    y: Vec<f64>,
    y_shape: Vec<usize>,
    metric: DistanceMetric,
}

impl CdistCase {
    fn new(
        x: Vec<f64>,
        x_shape: Vec<usize>,
        y: Vec<f64>,
        y_shape: Vec<usize>,
        metric: DistanceMetric,
    ) -> Self {
        Self {
            x,
            x_shape,
            y,
            y_shape,
            metric,
        }
    }
}

fn cdist_test_cases() -> Vec<CdistCase> {
    // Points in 2D
    let x = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]; // 3 points in 2D
    let y = vec![1.0, 1.0, 2.0, 0.0]; // 2 points in 2D

    vec![
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Euclidean,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::SquaredEuclidean,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Manhattan,
        ),
        CdistCase::new(
            x.clone(),
            vec![3, 2],
            y.clone(),
            vec![2, 2],
            DistanceMetric::Chebyshev,
        ),
        // 3D points
        CdistCase::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            vec![3, 3],
            DistanceMetric::Euclidean,
        ),
    ]
}

fn test_cdist_parity(dtype: DType) {
    let cases = cdist_test_cases();
    let (cpu_client, cpu_device) = create_cpu_client();

    for (idx, tc) in cases.iter().enumerate() {
        let cpu_x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU x tensor failed");
        let cpu_y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU y tensor failed");
        let cpu_result = cpu_client
            .cdist(&cpu_x, &cpu_y, tc.metric)
            .unwrap_or_else(|e| panic!("CPU cdist {:?} failed for {dtype:?}: {e}", tc.metric));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA x tensor failed");
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA y tensor failed");
                let result = cuda_client
                    .cdist(&x, &y, tc.metric)
                    .unwrap_or_else(|e| panic!("CUDA cdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("cdist {:?} CUDA vs CPU [{dtype:?}] case {idx}", tc.metric),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU x tensor failed");
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU y tensor failed");
                let result = wgpu_client
                    .cdist(&x, &y, tc.metric)
                    .unwrap_or_else(|e| panic!("WebGPU cdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("cdist {:?} WebGPU vs CPU [{dtype:?}] case {idx}", tc.metric),
                );
            });
        }
    }
}

#[test]
fn test_cdist_parity_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_cdist_parity(dtype);
    }
}

// ============================================================================
// pdist
// ============================================================================

fn test_pdist_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    // 4 points in 2D
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let shape = vec![4, 2];

    let metrics = vec![
        DistanceMetric::Euclidean,
        DistanceMetric::SquaredEuclidean,
        DistanceMetric::Manhattan,
        DistanceMetric::Chebyshev,
    ];

    for metric in &metrics {
        let cpu_x = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU tensor failed");
        let cpu_result = cpu_client
            .pdist(&cpu_x, *metric)
            .unwrap_or_else(|e| panic!("CPU pdist {metric:?} failed: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA tensor failed");
                let result = cuda_client
                    .pdist(&x, *metric)
                    .unwrap_or_else(|e| panic!("CUDA pdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("pdist {metric:?} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU tensor failed");
                let result = wgpu_client
                    .pdist(&x, *metric)
                    .unwrap_or_else(|e| panic!("WebGPU pdist failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("pdist {metric:?} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_pdist_parity_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_pdist_parity(dtype);
    }
}

// ============================================================================
// squareform roundtrip
// ============================================================================

#[test]
fn test_squareform_roundtrip_parity() {
    let dtype = DType::F32;
    let (cpu_client, cpu_device) = create_cpu_client();

    // 4 points in 2D
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let shape = vec![4, 2];
    let n = 4usize;

    let cpu_x =
        tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let cpu_condensed = cpu_client
        .pdist(&cpu_x, DistanceMetric::Euclidean)
        .expect("pdist failed");
    let cpu_square = cpu_client
        .squareform(&cpu_condensed, n)
        .expect("squareform failed");
    let cpu_back = cpu_client
        .squareform_inverse(&cpu_square)
        .expect("squareform_inverse failed");

    // Verify roundtrip: condensed -> square -> condensed
    assert_tensor_allclose(&cpu_back, &cpu_condensed, dtype, "squareform roundtrip CPU");

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let x = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
            .expect("tensor failed");
        let condensed = wgpu_client
            .pdist(&x, DistanceMetric::Euclidean)
            .expect("pdist failed");
        let square = wgpu_client
            .squareform(&condensed, n)
            .expect("squareform failed");

        assert_tensor_allclose(&square, &cpu_square, dtype, "squareform WebGPU vs CPU");

        let back = wgpu_client
            .squareform_inverse(&square)
            .expect("squareform_inverse failed");
        assert_tensor_allclose(
            &back,
            &cpu_condensed,
            dtype,
            "squareform_inverse WebGPU vs CPU",
        );
    });
}

// ============================================================================
// cosine distance
// ============================================================================

#[test]
fn test_cdist_cosine_parity() {
    let dtype = DType::F32;
    let (cpu_client, cpu_device) = create_cpu_client();

    let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 points in 2D
    let y = vec![1.0, 0.0, 0.0, 1.0]; // 2 points in 2D

    let cpu_x =
        tensor_from_f64(&x, &[3, 2], dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let cpu_y =
        tensor_from_f64(&y, &[2, 2], dtype, &cpu_device, &cpu_client).expect("tensor failed");
    let _cpu_result = cpu_client
        .cdist(&cpu_x, &cpu_y, DistanceMetric::Cosine)
        .expect("CPU cosine cdist failed");

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wx =
            tensor_from_f64(&x, &[3, 2], dtype, &wgpu_device, &wgpu_client).expect("tensor failed");
        let wy =
            tensor_from_f64(&y, &[2, 2], dtype, &wgpu_device, &wgpu_client).expect("tensor failed");
        let result = wgpu_client
            .cdist(&wx, &wy, DistanceMetric::Cosine)
            .expect("WebGPU cosine cdist failed");
        assert_tensor_allclose(&result, &_cpu_result, dtype, "cdist Cosine WebGPU vs CPU");
    });
}
