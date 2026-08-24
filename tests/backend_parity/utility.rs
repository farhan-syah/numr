// Backend parity tests for UtilityOps trait
//
// Tests: clamp, fill, arange, linspace, eye, one_hot
// CPU is the reference implementation; CUDA and WebGPU must match.

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};
use numr::dtype::DType;
use numr::ops::UtilityOps;
use numr::tensor::Tensor;

// ============================================================================
// clamp
// ============================================================================

fn test_clamp_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let data = vec![-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0];
    let shape = vec![8];
    let min_val = 0.0;
    let max_val = 3.0;

    let cpu_input = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
        .expect("CPU tensor creation failed");
    let cpu_result = cpu_client
        .clamp(&cpu_input, min_val, max_val)
        .expect("CPU clamp failed");

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                .expect("CUDA tensor creation failed");
            let result = cuda_client
                .clamp(&input, min_val, max_val)
                .expect("CUDA clamp failed");
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!("clamp CUDA vs CPU [{dtype:?}]"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let input = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                .expect("WebGPU tensor creation failed");
            let result = wgpu_client
                .clamp(&input, min_val, max_val)
                .expect("WebGPU clamp failed");
            assert_tensor_allclose(
                &result,
                &cpu_result,
                dtype,
                &format!("clamp WebGPU vs CPU [{dtype:?}]"),
            );
        });
    }
}

#[test]
fn test_clamp_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_clamp_parity(dtype);
    }
}

// ============================================================================
// fill
// ============================================================================

fn test_fill_parity(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    let shapes: Vec<Vec<usize>> = vec![vec![4], vec![2, 3], vec![2, 2, 2]];
    let values = vec![0.0, 1.0, 42.0, -3.5];

    for shape in &shapes {
        for &value in &values {
            let cpu_result = cpu_client
                .fill(shape, value, dtype)
                .expect("CPU fill failed");

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, _cuda_device| {
                    let result = cuda_client
                        .fill(shape, value, dtype)
                        .expect("CUDA fill failed");
                    assert_tensor_allclose(
                        &result,
                        &cpu_result,
                        dtype,
                        &format!("fill({value}) CUDA vs CPU [{dtype:?}] shape {shape:?}"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend(|wgpu_client, _wgpu_device| {
                    let result = wgpu_client
                        .fill(shape, value, dtype)
                        .expect("WebGPU fill failed");
                    assert_tensor_allclose(
                        &result,
                        &cpu_result,
                        dtype,
                        &format!("fill({value}) WebGPU vs CPU [{dtype:?}] shape {shape:?}"),
                    );
                });
            }
        }
    }
}

#[test]
fn test_fill_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_fill_parity(dtype);
    }
}

// ============================================================================
// arange
// ============================================================================

fn test_arange_parity(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    let cases: Vec<(f64, f64, f64)> = vec![
        (0.0, 5.0, 1.0),  // [0, 1, 2, 3, 4]
        (0.0, 6.0, 2.0),  // [0, 2, 4]
        (1.0, 10.0, 3.0), // [1, 4, 7]
        (5.0, 0.0, -1.0), // [5, 4, 3, 2, 1]
        (0.0, 1.0, 0.25), // [0, 0.25, 0.5, 0.75]
    ];

    for (start, stop, step) in &cases {
        let cpu_result = cpu_client
            .arange(*start, *stop, *step, dtype)
            .expect("CPU arange failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client
                    .arange(*start, *stop, *step, dtype)
                    .expect("CUDA arange failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("arange({start},{stop},{step}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _wgpu_device| {
                let result = wgpu_client
                    .arange(*start, *stop, *step, dtype)
                    .expect("WebGPU arange failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("arange({start},{stop},{step}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_arange_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_arange_parity(dtype);
    }
}

// ============================================================================
// linspace
// ============================================================================

fn test_linspace_parity(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    let cases: Vec<(f64, f64, usize)> = vec![
        (0.0, 10.0, 5),   // [0, 2.5, 5, 7.5, 10]
        (0.0, 1.0, 3),    // [0, 0.5, 1]
        (-1.0, 1.0, 5),   // [-1, -0.5, 0, 0.5, 1]
        (0.0, 100.0, 11), // [0, 10, 20, ..., 100]
        (5.0, 5.0, 3),    // [5, 5, 5]
    ];

    for (start, stop, steps) in &cases {
        let cpu_result = cpu_client
            .linspace(*start, *stop, *steps, dtype)
            .expect("CPU linspace failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client
                    .linspace(*start, *stop, *steps, dtype)
                    .expect("CUDA linspace failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("linspace({start},{stop},{steps}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _wgpu_device| {
                let result = wgpu_client
                    .linspace(*start, *stop, *steps, dtype)
                    .expect("WebGPU linspace failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("linspace({start},{stop},{steps}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_linspace_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_linspace_parity(dtype);
    }
}

// ============================================================================
// eye
// ============================================================================

fn test_eye_parity(dtype: DType) {
    let (cpu_client, _cpu_device) = create_cpu_client();

    let cases: Vec<(usize, Option<usize>)> = vec![
        (3, None),    // 3x3 identity
        (4, None),    // 4x4 identity
        (2, Some(4)), // 2x4 rectangular
        (4, Some(2)), // 4x2 rectangular
        (1, None),    // 1x1 identity
    ];

    for (n, m) in &cases {
        let cpu_result = cpu_client.eye(*n, *m, dtype).expect("CPU eye failed");

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, _cuda_device| {
                let result = cuda_client.eye(*n, *m, dtype).expect("CUDA eye failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("eye({n},{m:?}) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, _wgpu_device| {
                let result = wgpu_client.eye(*n, *m, dtype).expect("WebGPU eye failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("eye({n},{m:?}) WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_eye_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_eye_parity(dtype);
    }
}

// ============================================================================
// one_hot
// ============================================================================

#[test]
fn test_one_hot_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cases: Vec<(Vec<i32>, Vec<usize>, usize)> = vec![
        (vec![0, 1, 2], vec![3], 3),       // Simple 1D
        (vec![0, 2, 1, 3], vec![4], 5),    // With num_classes > max index
        (vec![0, 1, 2, 3], vec![2, 2], 4), // 2D indices
    ];

    for (data, shape, num_classes) in &cases {
        let cpu_indices =
            Tensor::<numr::runtime::cpu::CpuRuntime>::try_from_slice(data, shape, &cpu_device)
                .unwrap();
        let cpu_result = cpu_client
            .one_hot(&cpu_indices, *num_classes)
            .expect("CPU one_hot failed");

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let indices = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
                data,
                shape,
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .one_hot(&indices, *num_classes)
                .expect("CUDA one_hot failed");
            assert_tensor_allclose(
                &result,
                &cpu_result,
                DType::F32,
                &format!("one_hot CUDA vs CPU shape {shape:?} classes {num_classes}"),
            );
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let indices = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
                data,
                shape,
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .one_hot(&indices, *num_classes)
                .expect("WebGPU one_hot failed");
            assert_tensor_allclose(
                &result,
                &cpu_result,
                DType::F32,
                &format!("one_hot WebGPU vs CPU shape {shape:?} classes {num_classes}"),
            );
        });
    }
}
