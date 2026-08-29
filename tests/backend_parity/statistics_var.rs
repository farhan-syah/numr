// Backend parity tests for multi-dimension variance and standard deviation.
//
// Variance over several dims is one reduction against one mean, with the
// correction applied once against the total reduced count. Chaining a
// single-dim variance per dim computes the variance OF the variances instead,
// which is a different quantity. These tests pin the correct value and check
// every backend against CPU.

use numr::dtype::DType;
use numr::ops::StatisticalOps;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Variance carries a mean, so it needs a float. FP8's 4-bit mantissa cannot
/// hold an accumulated moment, so a mismatch there reports precision, not a
/// parity break.
fn float_dtypes(backend: &str) -> Vec<DType> {
    parity_dtypes(DTypeDomain::FloatsOnly, backend)
        .into_iter()
        .filter(|dtype| !matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2))
        .collect()
}

/// `[[1, 2], [3, 4]]` over both dims with correction 0. The mean is 2.5 and the
/// variance is `5 / 4 = 1.25`. Chaining per-dim variances answered 0 here.
#[test]
fn test_multi_dim_var_counterexample_parity() {
    let data = vec![1.0, 2.0, 3.0, 4.0];
    let shape = vec![2usize, 2];
    let expected_data = vec![1.25];
    let expected_shape = vec![1usize];

    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_var = cpu_client
            .var(&cpu_tensor, &[0, 1], true, 0)
            .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
        // keepdim gives [1, 1]; compare against a [1] tensor by value.
        let cpu_var_flat = cpu_var
            .reshape(&expected_shape)
            .unwrap_or_else(|e| panic!("CPU reshape failed for {dtype:?}: {e}"));

        let expected = tensor_from_f64(
            &expected_data,
            &expected_shape,
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        assert_tensor_allclose(
            &cpu_var_flat,
            &expected,
            dtype,
            &format!("multi-dim var CPU vs analytic [{dtype:?}]"),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_var = cuda_client
                    .var(&cuda_tensor, &[0, 1], true, 0)
                    .unwrap_or_else(|e| panic!("CUDA var failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_var,
                    &cpu_var,
                    dtype,
                    &format!("multi-dim var CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let wgpu_var = wgpu_client
                    .var(&wgpu_tensor, &[0, 1], true, 0)
                    .unwrap_or_else(|e| panic!("WebGPU var failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_var,
                    &cpu_var,
                    dtype,
                    &format!("multi-dim var WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Partial multi-dim reduction on a 3-D tensor, plus `std`, for both
/// corrections. Values stay small so F16 carries them exactly.
#[test]
fn test_partial_multi_dim_var_std_parity() {
    let data: Vec<f64> = (0..24).map(|v| ((v % 7) as f64) - 3.0).collect();
    let shape = vec![2usize, 3, 4];

    for dtype in float_dtypes("cpu") {
        for correction in [0usize, 1] {
            let (cpu_client, cpu_device) = create_cpu_client();

            let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

            let cpu_var = cpu_client
                .var(&cpu_tensor, &[0, 2], false, correction)
                .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
            assert_eq!(cpu_var.shape(), &[3]);

            let cpu_std = cpu_client
                .std(&cpu_tensor, &[0, 2], false, correction)
                .unwrap_or_else(|e| panic!("CPU std failed for {dtype:?}: {e}"));

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                    let cuda_var = cuda_client
                        .var(&t, &[0, 2], false, correction)
                        .unwrap_or_else(|e| panic!("CUDA var failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &cuda_var,
                        &cpu_var,
                        dtype,
                        &format!("partial var CUDA vs CPU [{dtype:?}, correction {correction}]"),
                    );
                    let cuda_std = cuda_client
                        .std(&t, &[0, 2], false, correction)
                        .unwrap_or_else(|e| panic!("CUDA std failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &cuda_std,
                        &cpu_std,
                        dtype,
                        &format!("partial std CUDA vs CPU [{dtype:?}, correction {correction}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                    let wgpu_var = wgpu_client
                        .var(&t, &[0, 2], false, correction)
                        .unwrap_or_else(|e| panic!("WebGPU var failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &wgpu_var,
                        &cpu_var,
                        dtype,
                        &format!("partial var WebGPU vs CPU [{dtype:?}, correction {correction}]"),
                    );
                    let wgpu_std = wgpu_client
                        .std(&t, &[0, 2], false, correction)
                        .unwrap_or_else(|e| panic!("WebGPU std failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &wgpu_std,
                        &cpu_std,
                        dtype,
                        &format!("partial std WebGPU vs CPU [{dtype:?}, correction {correction}]"),
                    );
                });
            }
        }
    }
}

/// Reducing over every dim by naming them must equal `dims = []` on every
/// backend, for both corrections. `correction` applies once against the total
/// reduced count, so the two spellings cannot disagree.
#[test]
fn test_all_dims_matches_empty_dims_parity() {
    let data: Vec<f64> = (0..24).map(|v| ((v % 5) as f64) - 2.0).collect();
    let shape = vec![2usize, 3, 4];

    for dtype in float_dtypes("cpu") {
        for correction in [0usize, 1] {
            let (cpu_client, cpu_device) = create_cpu_client();

            let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

            let cpu_empty = cpu_client
                .var(&cpu_tensor, &[], false, correction)
                .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
            let cpu_all = cpu_client
                .var(&cpu_tensor, &[0, 1, 2], false, correction)
                .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &cpu_all,
                &cpu_empty,
                dtype,
                &format!("var dims=[0,1,2] vs dims=[] CPU [{dtype:?}, correction {correction}]"),
            );

            #[cfg(feature = "cuda")]
            if is_dtype_supported("cuda", dtype) {
                with_cuda_backend(|cuda_client, cuda_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                    let cuda_all = cuda_client
                        .var(&t, &[0, 1, 2], false, correction)
                        .unwrap_or_else(|e| panic!("CUDA var failed for {dtype:?}: {e}"));
                    let cuda_empty = cuda_client
                        .var(&t, &[], false, correction)
                        .unwrap_or_else(|e| panic!("CUDA var failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &cuda_all,
                        &cuda_empty,
                        dtype,
                        &format!("var all-dims vs empty-dims CUDA [{dtype:?}, {correction}]"),
                    );
                    assert_tensor_allclose(
                        &cuda_all,
                        &cpu_all,
                        dtype,
                        &format!("var all-dims CUDA vs CPU [{dtype:?}, {correction}]"),
                    );
                });
            }

            #[cfg(feature = "wgpu")]
            if is_dtype_supported("wgpu", dtype) {
                with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                    let t = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                    let wgpu_all = wgpu_client
                        .var(&t, &[0, 1, 2], false, correction)
                        .unwrap_or_else(|e| panic!("WebGPU var failed for {dtype:?}: {e}"));
                    let wgpu_empty = wgpu_client
                        .var(&t, &[], false, correction)
                        .unwrap_or_else(|e| panic!("WebGPU var failed for {dtype:?}: {e}"));
                    assert_tensor_allclose(
                        &wgpu_all,
                        &wgpu_empty,
                        dtype,
                        &format!("var all-dims vs empty-dims WebGPU [{dtype:?}, {correction}]"),
                    );
                    assert_tensor_allclose(
                        &wgpu_all,
                        &cpu_all,
                        dtype,
                        &format!("var all-dims WebGPU vs CPU [{dtype:?}, {correction}]"),
                    );
                });
            }
        }
    }
}

/// A transposed view must give the same variance as its materialized
/// equivalent on every backend.
#[test]
fn test_non_contiguous_multi_dim_var_parity() {
    let data: Vec<f64> = (0..12).map(|v| ((v % 5) as f64) - 2.0).collect();
    let shape = vec![3usize, 4];

    for dtype in float_dtypes("cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_tensor: Tensor<_> = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_view = cpu_tensor
            .transpose(0, 1)
            .unwrap_or_else(|e| panic!("CPU transpose failed for {dtype:?}: {e}"));
        let cpu_materialized = cpu_view
            .contiguous()
            .unwrap_or_else(|e| panic!("CPU contiguous failed for {dtype:?}: {e}"));

        let cpu_from_view = cpu_client
            .var(&cpu_view, &[0, 1], false, 1)
            .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
        let cpu_from_contig = cpu_client
            .var(&cpu_materialized, &[0, 1], false, 1)
            .unwrap_or_else(|e| panic!("CPU var failed for {dtype:?}: {e}"));
        assert_tensor_allclose(
            &cpu_from_view,
            &cpu_from_contig,
            dtype,
            &format!("var transposed vs contiguous CPU [{dtype:?}]"),
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let t = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let view = t
                    .transpose(0, 1)
                    .unwrap_or_else(|e| panic!("CUDA transpose failed for {dtype:?}: {e}"));
                let cuda_var = cuda_client
                    .var(&view, &[0, 1], false, 1)
                    .unwrap_or_else(|e| panic!("CUDA var failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &cuda_var,
                    &cpu_from_view,
                    dtype,
                    &format!("var transposed CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                let t = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let view = t
                    .transpose(0, 1)
                    .unwrap_or_else(|e| panic!("WebGPU transpose failed for {dtype:?}: {e}"));
                let wgpu_var = wgpu_client
                    .var(&view, &[0, 1], false, 1)
                    .unwrap_or_else(|e| panic!("WebGPU var failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &wgpu_var,
                    &cpu_from_view,
                    dtype,
                    &format!("var transposed WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}
