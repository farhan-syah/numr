// Backend parity tests for ConditionalOps trait (where_cond)
//
// Dtype-parameterized: each test runs for all supported dtypes.
// CPU is the reference implementation; CUDA and WebGPU must match.

use numr::dtype::DType;
use numr::ops::{CompareOps, ConditionalOps};
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

struct WhereTestCase {
    cond: Vec<f64>,
    cond_shape: Vec<usize>,
    x: Vec<f64>,
    x_shape: Vec<usize>,
    y: Vec<f64>,
    y_shape: Vec<usize>,
}

impl WhereTestCase {
    fn new(
        cond: Vec<f64>,
        cond_shape: Vec<usize>,
        x: Vec<f64>,
        x_shape: Vec<usize>,
        y: Vec<f64>,
        y_shape: Vec<usize>,
    ) -> Self {
        Self {
            cond,
            cond_shape,
            x,
            x_shape,
            y,
            y_shape,
        }
    }
}

// The condition is a boolean mask, never the parameterised dtype: only x and y
// carry it. A backend takes the mask in the boolean carrier its `where_cond`
// kernels accept — a byte on CPU and CUDA, a 32-bit word on WebGPU, which has
// no narrower type.
fn mask_u8(cond: &[f64]) -> Vec<u8> {
    cond.iter().map(|&v| u8::from(v != 0.0)).collect()
}

#[cfg(feature = "wgpu")]
fn mask_u32(cond: &[f64]) -> Vec<u32> {
    cond.iter().map(|&v| u32::from(v != 0.0)).collect()
}

fn test_where_cond_parity(test_cases: &[WhereTestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let cond = Tensor::from_slice(&mask_u8(&tc.cond), &tc.cond_shape, &cpu_device)
                .unwrap_or_else(|e| panic!("CPU cond tensor failed for {dtype:?}: {e}"));
            let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU x tensor failed for {dtype:?}: {e}"));
            let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU y tensor failed for {dtype:?}: {e}"));

            cpu_client
                .where_cond(&cond, &x, &y)
                .unwrap_or_else(|e| panic!("CPU where_cond failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let cond = Tensor::from_slice(&mask_u8(&tc.cond), &tc.cond_shape, &cuda_device)
                    .unwrap_or_else(|e| panic!("CUDA cond tensor failed for {dtype:?}: {e}"));
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA x tensor failed for {dtype:?}: {e}"));
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA y tensor failed for {dtype:?}: {e}"));

                let result = cuda_client
                    .where_cond(&cond, &x, &y)
                    .unwrap_or_else(|e| panic!("CUDA where_cond failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("where_cond CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let cond = Tensor::from_slice(&mask_u32(&tc.cond), &tc.cond_shape, &wgpu_device)
                    .unwrap_or_else(|e| panic!("WebGPU cond tensor failed for {dtype:?}: {e}"));
                let x = tensor_from_f64(&tc.x, &tc.x_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU x tensor failed for {dtype:?}: {e}"));
                let y = tensor_from_f64(&tc.y, &tc.y_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU y tensor failed for {dtype:?}: {e}"));

                let result = wgpu_client
                    .where_cond(&cond, &x, &y)
                    .unwrap_or_else(|e| panic!("WebGPU where_cond failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("where_cond WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

fn where_test_cases() -> Vec<WhereTestCase> {
    vec![
        // 1D: simple mask
        WhereTestCase::new(
            vec![1.0, 0.0, 1.0, 0.0],
            vec![4],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![4],
            vec![100.0, 200.0, 300.0, 400.0],
            vec![4],
        ),
        // 2D: all true
        WhereTestCase::new(
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
        ),
        // 2D: all false
        WhereTestCase::new(
            vec![0.0, 0.0, 0.0, 0.0],
            vec![2, 2],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
        ),
        // 1D: alternating
        WhereTestCase::new(
            vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            vec![6],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![6],
            vec![100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            vec![6],
        ),
        // 3D tensor
        WhereTestCase::new(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            vec![2, 2, 2],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            vec![2, 2, 2],
        ),
    ]
}

#[test]
fn test_where_cond_parity_all_dtypes() {
    let cases = where_test_cases();
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        test_where_cond_parity(&cases, dtype);
    }
}

// Test where_cond with condition from comparison ops
#[test]
fn test_where_cond_from_compare_parity() {
    let (cpu_client, cpu_device) = create_cpu_client();
    let dtype = DType::F32;

    let a = tensor_from_f64(&[1.0, 5.0, 3.0, 7.0], &[4], dtype, &cpu_device, &cpu_client)
        .expect("tensor creation failed");
    let threshold = tensor_from_f64(&[3.0, 3.0, 3.0, 3.0], &[4], dtype, &cpu_device, &cpu_client)
        .expect("tensor creation failed");
    let x = tensor_from_f64(
        &[10.0, 20.0, 30.0, 40.0],
        &[4],
        dtype,
        &cpu_device,
        &cpu_client,
    )
    .expect("tensor creation failed");
    let y = tensor_from_f64(
        &[100.0, 200.0, 300.0, 400.0],
        &[4],
        dtype,
        &cpu_device,
        &cpu_client,
    )
    .expect("tensor creation failed");

    let mask = cpu_client.gt(&a, &threshold).expect("gt failed");
    let _cpu_result = cpu_client
        .where_cond(&mask, &x, &y)
        .expect("where_cond failed");

    #[cfg(feature = "wgpu")]
    {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let a_w = tensor_from_f64(
                &[1.0, 5.0, 3.0, 7.0],
                &[4],
                dtype,
                &wgpu_device,
                &wgpu_client,
            )
            .expect("tensor creation failed");
            let t_w = tensor_from_f64(
                &[3.0, 3.0, 3.0, 3.0],
                &[4],
                dtype,
                &wgpu_device,
                &wgpu_client,
            )
            .expect("tensor creation failed");
            let x_w = tensor_from_f64(
                &[10.0, 20.0, 30.0, 40.0],
                &[4],
                dtype,
                &wgpu_device,
                &wgpu_client,
            )
            .expect("tensor creation failed");
            let y_w = tensor_from_f64(
                &[100.0, 200.0, 300.0, 400.0],
                &[4],
                dtype,
                &wgpu_device,
                &wgpu_client,
            )
            .expect("tensor creation failed");

            let mask_w = wgpu_client.gt(&a_w, &t_w).expect("gt failed");
            let result = wgpu_client
                .where_cond(&mask_w, &x_w, &y_w)
                .expect("where_cond failed");

            assert_tensor_allclose(
                &result,
                &_cpu_result,
                dtype,
                "where_cond(gt mask) WebGPU vs CPU",
            );
        });
    }
}
