// Backend parity tests for SortOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::ops::SortingOps;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

#[test]
fn test_sort_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let shape = vec![8];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_sorted = cpu_client
            .sort(&cpu_tensor, 0, false)
            .unwrap_or_else(|e| panic!("CPU sort failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_sorted = cuda_client
                    .sort(&cuda_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("CUDA sort failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_sorted,
                    &cpu_sorted,
                    dtype,
                    &format!("sort CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_sorted = wgpu_client
                    .sort(&wgpu_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("WebGPU sort failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_sorted,
                    &cpu_sorted,
                    dtype,
                    &format!("sort WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_argsort_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
    let shape = vec![5];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_indices = cpu_client
            .argsort(&cpu_tensor, 0, false)
            .unwrap_or_else(|e| panic!("CPU argsort failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_indices = cuda_client
                    .argsort(&cuda_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("CUDA argsort failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "argsort CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_indices = wgpu_client
                    .argsort(&wgpu_tensor, 0, false)
                    .unwrap_or_else(|e| panic!("WebGPU argsort failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "argsort WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_global_argsort_is_stable_past_shared_memory_limit() {
    const LEN: usize = 4097;
    let data: Vec<f64> = (0..LEN).map(|index| (index % 17) as f64).collect();
    let shape = vec![LEN];

    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let tensor = tensor_from_f64(
            &data,
            &shape,
            numr::dtype::DType::U32,
            &wgpu_device,
            &wgpu_client,
        )
        .expect("create duplicate-heavy WGPU input");
        let indices: Vec<i32> = wgpu_client
            .argsort(&tensor, 0, false)
            .expect("global WGPU argsort")
            .to_vec();

        let mut expected: Vec<i32> = (0..LEN as i32).collect();
        expected.sort_by_key(|&index| (index as usize % 17, index));
        assert_eq!(indices, expected);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_global_sort_family_matches_cpu_on_arbitrary_axis() {
    use numr::dtype::DType;

    let shape = vec![2, 513, 3];
    for dtype in [DType::U32, DType::I32, DType::F32] {
        let data: Vec<f64> = (0..shape.iter().product())
            .map(|index| {
                let value = ((index * 37 + index / 11) % 101) as f64;
                if dtype == DType::U32 {
                    value
                } else {
                    value - 50.0
                }
            })
            .collect();
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .expect("create CPU global-sort input");

        for descending in [false, true] {
            let cpu_values = cpu_client
                .sort(&cpu_tensor, 1, descending)
                .expect("CPU sort");
            let cpu_argsort: Vec<i64> = cpu_client
                .argsort(&cpu_tensor, 1, descending)
                .expect("CPU argsort")
                .to_vec();
            let (cpu_values_with_indices, cpu_indices): (_, Vec<i64>) = {
                let (values, indices) = cpu_client
                    .sort_with_indices(&cpu_tensor, 1, descending)
                    .expect("CPU sort_with_indices");
                (values, indices.to_vec())
            };

            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("create WGPU global-sort input");
                let wgpu_values = wgpu_client
                    .sort(&wgpu_tensor, 1, descending)
                    .expect("WGPU global sort");
                let wgpu_argsort: Vec<i32> = wgpu_client
                    .argsort(&wgpu_tensor, 1, descending)
                    .expect("WGPU global argsort")
                    .to_vec();
                let (wgpu_values_with_indices, wgpu_indices) = wgpu_client
                    .sort_with_indices(&wgpu_tensor, 1, descending)
                    .expect("WGPU global sort_with_indices");
                let wgpu_indices: Vec<i32> = wgpu_indices.to_vec();

                assert_tensor_allclose(
                    &wgpu_values,
                    &cpu_values,
                    dtype,
                    &format!("global sort WGPU vs CPU [{dtype:?}, descending={descending}]"),
                );
                assert_tensor_allclose(
                    &wgpu_values_with_indices,
                    &cpu_values_with_indices,
                    dtype,
                    &format!(
                        "global sort_with_indices WGPU vs CPU [{dtype:?}, descending={descending}]"
                    ),
                );
                assert_eq!(
                    wgpu_argsort
                        .iter()
                        .map(|&value| i64::from(value))
                        .collect::<Vec<_>>(),
                    cpu_argsort,
                    "global argsort indices [{dtype:?}, descending={descending}]"
                );
                assert_eq!(
                    wgpu_indices
                        .iter()
                        .map(|&value| i64::from(value))
                        .collect::<Vec<_>>(),
                    cpu_indices,
                    "global sort_with_indices indices [{dtype:?}, descending={descending}]"
                );
            });
        }
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_global_f32_orders_nans_and_stabilizes_signed_zero() {
    use numr::tensor::Tensor;

    let mut data: Vec<f32> = (0..513).map(|index| (index % 23) as f32 - 11.0).collect();
    data[3] = f32::NAN;
    data[200] = f32::from_bits(0xffc0_0001);
    data[17] = -0.0;
    data[41] = 0.0;

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = Tensor::from_slice(&data, &[data.len()], &cpu_device);
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_tensor = Tensor::from_slice(&data, &[data.len()], &wgpu_device);
        for descending in [false, true] {
            let expected: Vec<i64> = cpu_client
                .argsort(&cpu_tensor, 0, descending)
                .expect("CPU global f32 argsort")
                .to_vec();
            let actual: Vec<i64> = wgpu_client
                .argsort(&wgpu_tensor, 0, descending)
                .expect("WGPU global f32 argsort")
                .to_vec::<i32>()
                .into_iter()
                .map(i64::from)
                .collect();
            assert_eq!(
                actual, expected,
                "f32 global argsort WGPU vs CPU descending={descending}"
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
#[ignore = "large physical-GPU validation"]
fn test_wgpu_global_sort_one_million_elements() {
    use numr::tensor::Tensor;

    const LEN: usize = 1_000_003;
    let data: Vec<u32> = (0..LEN as u32).rev().collect();
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let tensor = Tensor::from_slice(&data, &[LEN], &wgpu_device);
        let sorted: Vec<u32> = wgpu_client
            .sort(&tensor, 0, false)
            .expect("one-million-element WGPU global sort")
            .to_vec();
        assert_eq!(sorted.len(), LEN);
        assert!(
            sorted
                .iter()
                .enumerate()
                .all(|(index, &value)| value == index as u32)
        );
    });
}

#[test]
fn test_topk_parity() {
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let shape = vec![8];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let (cpu_vals, cpu_indices) = cpu_client
            .topk(&cpu_tensor, 3, 0, true, true)
            .unwrap_or_else(|e| panic!("CPU topk failed for {dtype:?}: {e}"));
        let cpu_i: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let (cuda_vals, cuda_indices) = cuda_client
                    .topk(&cuda_tensor, 3, 0, true, true)
                    .unwrap_or_else(|e| panic!("CUDA topk failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_vals,
                    &cpu_vals,
                    dtype,
                    &format!("topk values CUDA vs CPU [{dtype:?}]"),
                );
                let cuda_i: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_i, cuda_i,
                    "topk indices CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let (wgpu_vals, wgpu_indices) = wgpu_client
                    .topk(&wgpu_tensor, 3, 0, true, true)
                    .unwrap_or_else(|e| panic!("WebGPU topk failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_vals,
                    &cpu_vals,
                    dtype,
                    &format!("topk values WebGPU vs CPU [{dtype:?}]"),
                );
                let wgpu_i: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_i.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_i, wgpu_as_i64,
                    "topk indices WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[test]
fn test_unique_parity() {
    let data = vec![1.0, 2.0, 2.0, 3.0, 1.0, 4.0];
    let shape = vec![6];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_unique = cpu_client
            .unique(&cpu_tensor, true)
            .unwrap_or_else(|e| panic!("CPU unique failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_unique = cuda_client
                    .unique(&cuda_tensor, true)
                    .unwrap_or_else(|e| panic!("CUDA unique failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_unique,
                    &cpu_unique,
                    dtype,
                    &format!("unique CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_unique = wgpu_client
                    .unique(&wgpu_tensor, true)
                    .unwrap_or_else(|e| panic!("WebGPU unique failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_unique,
                    &cpu_unique,
                    dtype,
                    &format!("unique WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_nonzero_parity() {
    let data = vec![0.0, 1.0, 0.0, 2.0, 3.0];
    let shape = vec![5];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_indices = cpu_client
            .nonzero(&cpu_tensor)
            .unwrap_or_else(|e| panic!("CPU nonzero failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_indices = cuda_client
                    .nonzero(&cuda_tensor)
                    .unwrap_or_else(|e| panic!("CUDA nonzero failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "nonzero CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_indices = wgpu_client
                    .nonzero(&wgpu_tensor)
                    .unwrap_or_else(|e| panic!("WebGPU nonzero failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "nonzero WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}

#[test]
fn test_searchsorted_parity() {
    let sorted_data = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let values_data = vec![2.0, 4.0, 6.0, 8.0];
    let sorted_shape = vec![5];
    let values_shape = vec![4];

    for dtype in supported_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_sorted =
            tensor_from_f64(&sorted_data, &sorted_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| {
                    panic!("CPU tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
        let cpu_values =
            tensor_from_f64(&values_data, &values_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| {
                    panic!("CPU tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
        let cpu_indices = cpu_client
            .searchsorted(&cpu_sorted, &cpu_values, false)
            .unwrap_or_else(|e| panic!("CPU searchsorted failed for {dtype:?}: {e}"));
        let cpu_data: Vec<i64> = cpu_indices.to_vec();

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_sorted = tensor_from_f64(
                    &sorted_data,
                    &sorted_shape,
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| {
                    panic!("CUDA tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
                let cuda_values = tensor_from_f64(
                    &values_data,
                    &values_shape,
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| {
                    panic!("CUDA tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
                let cuda_indices = cuda_client
                    .searchsorted(&cuda_sorted, &cuda_values, false)
                    .unwrap_or_else(|e| panic!("CUDA searchsorted failed for {dtype:?}: {e}"));
                let cuda_data: Vec<i64> = cuda_indices.to_vec();
                assert_eq!(
                    cpu_data, cuda_data,
                    "searchsorted CUDA vs CPU [{dtype:?}] mismatch"
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let wgpu_sorted = tensor_from_f64(
                    &sorted_data,
                    &sorted_shape,
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| {
                    panic!("WebGPU tensor_from_f64 (sorted) failed for {dtype:?}: {e}")
                });
                let wgpu_values = tensor_from_f64(
                    &values_data,
                    &values_shape,
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| {
                    panic!("WebGPU tensor_from_f64 (values) failed for {dtype:?}: {e}")
                });
                let wgpu_indices = wgpu_client
                    .searchsorted(&wgpu_sorted, &wgpu_values, false)
                    .unwrap_or_else(|e| panic!("WebGPU searchsorted failed for {dtype:?}: {e}"));
                let wgpu_data: Vec<i32> = wgpu_indices.to_vec();
                let wgpu_as_i64: Vec<i64> = wgpu_data.iter().map(|&x| x as i64).collect();
                assert_eq!(
                    cpu_data, wgpu_as_i64,
                    "searchsorted WebGPU vs CPU [{dtype:?}] mismatch"
                );
            });
        }
    }
}
