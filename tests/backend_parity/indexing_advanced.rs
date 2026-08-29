// Backend parity tests for advanced indexing operations
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Index tensors remain as I32 (not parameterized), only data tensors are dtype-parameterized.

use numr::ops::IndexingOps;
use numr::ops::ScatterReduceOp;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

#[test]
fn test_index_select_parity() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let indices = [2i32, 0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_x = tensor_from_f64(&input_data, &[3, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_i = Tensor::from_slice(&indices, &[2], &cpu_device).unwrap();
        let cpu_result = cpu_client
            .index_select(&cpu_x, 0, &cpu_i)
            .unwrap_or_else(|e| panic!("CPU index_select failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input_data, &[3, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[2], &cuda_device).unwrap();
                let result = cuda_client
                    .index_select(&x, 0, &i)
                    .unwrap_or_else(|e| panic!("CUDA index_select failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("index_select CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input_data, &[3, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[2], &wgpu_device).unwrap();
                let result = wgpu_client
                    .index_select(&x, 0, &i)
                    .unwrap_or_else(|e| panic!("WGPU index_select failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("index_select WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_i32_indices_parity() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let scatter_src_data = vec![1.0, 2.0, 3.0, 4.0];
    let index_put_values_data = vec![10.0, 11.0, 12.0, 13.0];
    let nd_input_data = vec![0.0, 1.0, 2.0, 3.0];
    let emb_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let g2d_input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let scatter_reduce_dst_data = vec![0.0, 0.0, 0.0, 0.0];
    let scatter_reduce_src_data = vec![1.0, 2.0, 3.0];
    let scatter_dst_data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let input = tensor_from_f64(&input_data, &[3, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx_1d = Tensor::from_slice(&[2i32, 0], &[2], &cpu_device).unwrap();
        let idx_2d = Tensor::from_slice(&[0i32, 2, 1, 0], &[2, 2], &cpu_device).unwrap();

        let cpu_index_select = cpu_client
            .index_select(&input, 0, &idx_1d)
            .unwrap_or_else(|e| panic!("CPU index_select failed for {dtype:?}: {e}"));
        let cpu_gather = cpu_client
            .gather(&input, 0, &idx_2d)
            .unwrap_or_else(|e| panic!("CPU gather failed for {dtype:?}: {e}"));

        let scatter_dst =
            tensor_from_f64(&scatter_dst_data, &[3, 2], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let scatter_src =
            tensor_from_f64(&scatter_src_data, &[2, 2], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_scatter = cpu_client
            .scatter(&scatter_dst, 0, &idx_2d, &scatter_src)
            .unwrap_or_else(|e| panic!("CPU scatter failed for {dtype:?}: {e}"));

        let index_put_values = tensor_from_f64(
            &index_put_values_data,
            &[2, 2],
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_index_put = cpu_client
            .index_put(&input, 0, &idx_1d, &index_put_values)
            .unwrap_or_else(|e| panic!("CPU index_put failed for {dtype:?}: {e}"));

        let nd_input = tensor_from_f64(&nd_input_data, &[2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let nd_idx = Tensor::from_slice(&[0i32, 0, 1, 1], &[2, 2], &cpu_device).unwrap();
        let cpu_gather_nd = cpu_client
            .gather_nd(&nd_input, &nd_idx)
            .unwrap_or_else(|e| panic!("CPU gather_nd failed for {dtype:?}: {e}"));

        let emb = tensor_from_f64(&emb_data, &[4, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let emb_idx = Tensor::from_slice(&[3i32, 0, 1], &[3], &cpu_device).unwrap();
        let cpu_emb = cpu_client
            .embedding_lookup(&emb, &emb_idx)
            .unwrap_or_else(|e| panic!("CPU embedding_lookup failed for {dtype:?}: {e}"));

        let g2d_input = tensor_from_f64(&g2d_input_data, &[3, 3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let g2d_rows = Tensor::from_slice(&[0i32, 1, 2, 0], &[4], &cpu_device).unwrap();
        let g2d_cols = Tensor::from_slice(&[0i32, 1, 2, 2], &[4], &cpu_device).unwrap();
        let cpu_g2d = cpu_client
            .gather_2d(&g2d_input, &g2d_rows, &g2d_cols)
            .unwrap_or_else(|e| panic!("CPU gather_2d failed for {dtype:?}: {e}"));

        let scatter_reduce_dst = tensor_from_f64(
            &scatter_reduce_dst_data,
            &[4],
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let scatter_reduce_idx = Tensor::from_slice(&[0i32, 0, 2], &[3], &cpu_device).unwrap();
        let scatter_reduce_src = tensor_from_f64(
            &scatter_reduce_src_data,
            &[3],
            dtype,
            &cpu_device,
            &cpu_client,
        )
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_scatter_reduce = cpu_client
            .scatter_reduce(
                &scatter_reduce_dst,
                0,
                &scatter_reduce_idx,
                &scatter_reduce_src,
                ScatterReduceOp::Sum,
                false,
            )
            .unwrap_or_else(|e| panic!("CPU scatter_reduce failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let input =
                    tensor_from_f64(&input_data, &[3, 2], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let idx_1d = Tensor::from_slice(&[2i32, 0], &[2], &cuda_device).unwrap();
                let idx_2d = Tensor::from_slice(&[0i32, 2, 1, 0], &[2, 2], &cuda_device).unwrap();

                let result_index_select = cuda_client
                    .index_select(&input, 0, &idx_1d)
                    .unwrap_or_else(|e| panic!("CUDA index_select failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_index_select,
                    &cpu_index_select,
                    dtype,
                    &format!("index_select CUDA vs CPU [{dtype:?}]"),
                );

                let result_gather = cuda_client
                    .gather(&input, 0, &idx_2d)
                    .unwrap_or_else(|e| panic!("CUDA gather failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather,
                    &cpu_gather,
                    dtype,
                    &format!("gather CUDA vs CPU [{dtype:?}]"),
                );

                let scatter_dst = tensor_from_f64(
                    &scatter_dst_data,
                    &[3, 2],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let scatter_src = tensor_from_f64(
                    &scatter_src_data,
                    &[2, 2],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter = cuda_client
                    .scatter(&scatter_dst, 0, &idx_2d, &scatter_src)
                    .unwrap_or_else(|e| panic!("CUDA scatter failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter,
                    &cpu_scatter,
                    dtype,
                    &format!("scatter CUDA vs CPU [{dtype:?}]"),
                );

                let index_put_values = tensor_from_f64(
                    &index_put_values_data,
                    &[2, 2],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_index_put = cuda_client
                    .index_put(&input, 0, &idx_1d, &index_put_values)
                    .unwrap_or_else(|e| panic!("CUDA index_put failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_index_put,
                    &cpu_index_put,
                    dtype,
                    &format!("index_put CUDA vs CPU [{dtype:?}]"),
                );

                let nd_input =
                    tensor_from_f64(&nd_input_data, &[2, 2], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let nd_idx = Tensor::from_slice(&[0i32, 0, 1, 1], &[2, 2], &cuda_device).unwrap();
                let result_gather_nd = cuda_client
                    .gather_nd(&nd_input, &nd_idx)
                    .unwrap_or_else(|e| panic!("CUDA gather_nd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather_nd,
                    &cpu_gather_nd,
                    dtype,
                    &format!("gather_nd CUDA vs CPU [{dtype:?}]"),
                );

                let emb = tensor_from_f64(&emb_data, &[4, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let emb_idx = Tensor::from_slice(&[3i32, 0, 1], &[3], &cuda_device).unwrap();
                let result_emb = cuda_client
                    .embedding_lookup(&emb, &emb_idx)
                    .unwrap_or_else(|e| panic!("CUDA embedding_lookup failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_emb,
                    &cpu_emb,
                    dtype,
                    &format!("embedding_lookup CUDA vs CPU [{dtype:?}]"),
                );

                let g2d_input =
                    tensor_from_f64(&g2d_input_data, &[3, 3], dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let g2d_rows = Tensor::from_slice(&[0i32, 1, 2, 0], &[4], &cuda_device).unwrap();
                let g2d_cols = Tensor::from_slice(&[0i32, 1, 2, 2], &[4], &cuda_device).unwrap();
                let result_g2d = cuda_client
                    .gather_2d(&g2d_input, &g2d_rows, &g2d_cols)
                    .unwrap_or_else(|e| panic!("CUDA gather_2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_g2d,
                    &cpu_g2d,
                    dtype,
                    &format!("gather_2d CUDA vs CPU [{dtype:?}]"),
                );

                let scatter_reduce_dst = tensor_from_f64(
                    &scatter_reduce_dst_data,
                    &[4],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let scatter_reduce_idx =
                    Tensor::from_slice(&[0i32, 0, 2], &[3], &cuda_device).unwrap();
                let scatter_reduce_src = tensor_from_f64(
                    &scatter_reduce_src_data,
                    &[3],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter_reduce = cuda_client
                    .scatter_reduce(
                        &scatter_reduce_dst,
                        0,
                        &scatter_reduce_idx,
                        &scatter_reduce_src,
                        ScatterReduceOp::Sum,
                        false,
                    )
                    .unwrap_or_else(|e| panic!("CUDA scatter_reduce failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter_reduce,
                    &cpu_scatter_reduce,
                    dtype,
                    &format!("scatter_reduce CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let input =
                    tensor_from_f64(&input_data, &[3, 2], dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let idx_1d = Tensor::from_slice(&[2i32, 0], &[2], &wgpu_device).unwrap();
                let idx_2d = Tensor::from_slice(&[0i32, 2, 1, 0], &[2, 2], &wgpu_device).unwrap();

                let result_index_select = wgpu_client
                    .index_select(&input, 0, &idx_1d)
                    .unwrap_or_else(|e| panic!("WGPU index_select failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_index_select,
                    &cpu_index_select,
                    dtype,
                    &format!("index_select WGPU vs CPU [{dtype:?}]"),
                );

                let result_gather = wgpu_client
                    .gather(&input, 0, &idx_2d)
                    .unwrap_or_else(|e| panic!("WGPU gather failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather,
                    &cpu_gather,
                    dtype,
                    &format!("gather WGPU vs CPU [{dtype:?}]"),
                );

                let scatter_dst = tensor_from_f64(
                    &scatter_dst_data,
                    &[3, 2],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let scatter_src = tensor_from_f64(
                    &scatter_src_data,
                    &[2, 2],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter = wgpu_client
                    .scatter(&scatter_dst, 0, &idx_2d, &scatter_src)
                    .unwrap_or_else(|e| panic!("WGPU scatter failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter,
                    &cpu_scatter,
                    dtype,
                    &format!("scatter WGPU vs CPU [{dtype:?}]"),
                );

                let index_put_values = tensor_from_f64(
                    &index_put_values_data,
                    &[2, 2],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_index_put = wgpu_client
                    .index_put(&input, 0, &idx_1d, &index_put_values)
                    .unwrap_or_else(|e| panic!("WGPU index_put failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_index_put,
                    &cpu_index_put,
                    dtype,
                    &format!("index_put WGPU vs CPU [{dtype:?}]"),
                );

                let nd_input =
                    tensor_from_f64(&nd_input_data, &[2, 2], dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let nd_idx = Tensor::from_slice(&[0i32, 0, 1, 1], &[2, 2], &wgpu_device).unwrap();
                let result_gather_nd = wgpu_client
                    .gather_nd(&nd_input, &nd_idx)
                    .unwrap_or_else(|e| panic!("WGPU gather_nd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather_nd,
                    &cpu_gather_nd,
                    dtype,
                    &format!("gather_nd WGPU vs CPU [{dtype:?}]"),
                );

                let emb = tensor_from_f64(&emb_data, &[4, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let emb_idx = Tensor::from_slice(&[3i32, 0, 1], &[3], &wgpu_device).unwrap();
                let result_emb = wgpu_client
                    .embedding_lookup(&emb, &emb_idx)
                    .unwrap_or_else(|e| panic!("WGPU embedding_lookup failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_emb,
                    &cpu_emb,
                    dtype,
                    &format!("embedding_lookup WGPU vs CPU [{dtype:?}]"),
                );

                let g2d_input =
                    tensor_from_f64(&g2d_input_data, &[3, 3], dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let g2d_rows = Tensor::from_slice(&[0i32, 1, 2, 0], &[4], &wgpu_device).unwrap();
                let g2d_cols = Tensor::from_slice(&[0i32, 1, 2, 2], &[4], &wgpu_device).unwrap();
                let result_g2d = wgpu_client
                    .gather_2d(&g2d_input, &g2d_rows, &g2d_cols)
                    .unwrap_or_else(|e| panic!("WGPU gather_2d failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_g2d,
                    &cpu_g2d,
                    dtype,
                    &format!("gather_2d WGPU vs CPU [{dtype:?}]"),
                );

                let scatter_reduce_dst = tensor_from_f64(
                    &scatter_reduce_dst_data,
                    &[4],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let scatter_reduce_idx =
                    Tensor::from_slice(&[0i32, 0, 2], &[3], &wgpu_device).unwrap();
                let scatter_reduce_src = tensor_from_f64(
                    &scatter_reduce_src_data,
                    &[3],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter_reduce = wgpu_client
                    .scatter_reduce(
                        &scatter_reduce_dst,
                        0,
                        &scatter_reduce_idx,
                        &scatter_reduce_src,
                        ScatterReduceOp::Sum,
                        false,
                    )
                    .unwrap_or_else(|e| panic!("WGPU scatter_reduce failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter_reduce,
                    &cpu_scatter_reduce,
                    dtype,
                    &format!("scatter_reduce WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_gather_scatter_parity() {
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gather_indices = [0i32, 2, 1, 0];
    let src_data = vec![1.0, 2.0, 3.0, 4.0];
    let dst_data = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let cpu_x = tensor_from_f64(&input_data, &[3, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_i = Tensor::from_slice(&gather_indices, &[2, 2], &cpu_device).unwrap();
        let cpu_gather = cpu_client
            .gather(&cpu_x, 0, &cpu_i)
            .unwrap_or_else(|e| panic!("CPU gather failed for {dtype:?}: {e}"));

        let cpu_dst = tensor_from_f64(&dst_data, &[3, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_src = tensor_from_f64(&src_data, &[2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_scatter = cpu_client
            .scatter(&cpu_dst, 0, &cpu_i, &cpu_src)
            .unwrap_or_else(|e| panic!("CPU scatter failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input_data, &[3, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&gather_indices, &[2, 2], &cuda_device).unwrap();
                let result_gather = cuda_client
                    .gather(&x, 0, &i)
                    .unwrap_or_else(|e| panic!("CUDA gather failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather,
                    &cpu_gather,
                    dtype,
                    &format!("gather CUDA vs CPU [{dtype:?}]"),
                );

                let dst = tensor_from_f64(&dst_data, &[3, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let src_t = tensor_from_f64(&src_data, &[2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter = cuda_client
                    .scatter(&dst, 0, &i, &src_t)
                    .unwrap_or_else(|e| panic!("CUDA scatter failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter,
                    &cpu_scatter,
                    dtype,
                    &format!("scatter CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input_data, &[3, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&gather_indices, &[2, 2], &wgpu_device).unwrap();
                let result_gather = wgpu_client
                    .gather(&x, 0, &i)
                    .unwrap_or_else(|e| panic!("WGPU gather failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_gather,
                    &cpu_gather,
                    dtype,
                    &format!("gather WGPU vs CPU [{dtype:?}]"),
                );

                let dst = tensor_from_f64(&dst_data, &[3, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let src_t = tensor_from_f64(&src_data, &[2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result_scatter = wgpu_client
                    .scatter(&dst, 0, &i, &src_t)
                    .unwrap_or_else(|e| panic!("WGPU scatter failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_scatter,
                    &cpu_scatter,
                    dtype,
                    &format!("scatter WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_gather_nd_bincount_embedding_parity() {
    let input_data = vec![0.0, 1.0, 2.0, 3.0];
    let nd_indices_i32 = [0i32, 0, 1, 1];
    let bins_input_i64 = [0i64, 1, 1, 3, 2, 1, 3];
    let emb_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let emb_idx_i64 = [3i64, 0, 1];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let input = tensor_from_f64(&input_data, &[2, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let nd_idx = Tensor::from_slice(&nd_indices_i32, &[2, 2], &cpu_device).unwrap();
        let cpu_nd = cpu_client
            .gather_nd(&input, &nd_idx)
            .unwrap_or_else(|e| panic!("CPU gather_nd failed for {dtype:?}: {e}"));

        // bincount operates on i64 indices, returns i64 counts (not parameterized)
        let bins_input = Tensor::from_slice(&bins_input_i64, &[7], &cpu_device).unwrap();
        let cpu_bins: Vec<i64> = cpu_client
            .bincount(&bins_input, None, 0)
            .unwrap_or_else(|e| panic!("CPU bincount failed: {e}"))
            .to_vec();

        let emb = tensor_from_f64(&emb_data, &[4, 2], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let emb_idx = Tensor::from_slice(&emb_idx_i64, &[3], &cpu_device).unwrap();
        let cpu_emb = cpu_client
            .embedding_lookup(&emb, &emb_idx)
            .unwrap_or_else(|e| panic!("CPU embedding_lookup failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let x = tensor_from_f64(&input_data, &[2, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&nd_indices_i32, &[2, 2], &cuda_device).unwrap();
                let result_nd = cuda_client
                    .gather_nd(&x, &i)
                    .unwrap_or_else(|e| panic!("CUDA gather_nd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_nd,
                    &cpu_nd,
                    dtype,
                    &format!("gather_nd CUDA vs CPU [{dtype:?}]"),
                );

                let b_in = Tensor::from_slice(&bins_input_i64, &[7], &cuda_device).unwrap();
                let bins: Vec<i64> = cuda_client
                    .bincount(&b_in, None, 0)
                    .unwrap_or_else(|e| panic!("CUDA bincount failed: {e}"))
                    .to_vec();
                assert_eq!(cpu_bins, bins, "bincount CUDA vs CPU mismatch");

                let e = tensor_from_f64(&emb_data, &[4, 2], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let ei = Tensor::from_slice(&emb_idx_i64, &[3], &cuda_device).unwrap();
                let result_emb = cuda_client
                    .embedding_lookup(&e, &ei)
                    .unwrap_or_else(|e| panic!("CUDA embedding_lookup failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_emb,
                    &cpu_emb,
                    dtype,
                    &format!("embedding_lookup CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let x = tensor_from_f64(&input_data, &[2, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&nd_indices_i32, &[2, 2], &wgpu_device).unwrap();
                let result_nd = wgpu_client
                    .gather_nd(&x, &i)
                    .unwrap_or_else(|e| panic!("WGPU gather_nd failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_nd,
                    &cpu_nd,
                    dtype,
                    &format!("gather_nd WGPU vs CPU [{dtype:?}]"),
                );

                let b_in = Tensor::from_slice(&bins_input_i64, &[7], &wgpu_device).unwrap();
                let bins: Vec<i64> = wgpu_client
                    .bincount(&b_in, None, 0)
                    .unwrap_or_else(|e| panic!("WGPU bincount failed: {e}"))
                    .to_vec();
                assert_eq!(cpu_bins, bins, "bincount WGPU vs CPU mismatch");

                let e = tensor_from_f64(&emb_data, &[4, 2], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let ei = Tensor::from_slice(&emb_idx_i64, &[3], &wgpu_device).unwrap();
                let result_emb = wgpu_client
                    .embedding_lookup(&e, &ei)
                    .unwrap_or_else(|e| panic!("WGPU embedding_lookup failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result_emb,
                    &cpu_emb,
                    dtype,
                    &format!("embedding_lookup WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_scatter_reduce_sum_parity() {
    let dst_data = vec![0.0, 0.0, 0.0, 0.0];
    let indices = [0i32, 0, 2];
    let src_data = vec![1.0, 2.0, 3.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let dst = tensor_from_f64(&dst_data, &[4], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx = Tensor::from_slice(&indices, &[3], &cpu_device).unwrap();
        let src = tensor_from_f64(&src_data, &[3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Sum, false)
            .unwrap_or_else(|e| panic!("CPU scatter_reduce failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let d = tensor_from_f64(&dst_data, &[4], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[3], &cuda_device).unwrap();
                let s = tensor_from_f64(&src_data, &[3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
                    .unwrap_or_else(|e| panic!("CUDA scatter_reduce failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("scatter_reduce_sum CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let d = tensor_from_f64(&dst_data, &[4], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[3], &wgpu_device).unwrap();
                let s = tensor_from_f64(&src_data, &[3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Sum, false)
                    .unwrap_or_else(|e| panic!("WGPU scatter_reduce failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("scatter_reduce_sum WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

#[test]
fn test_scatter_reduce_mean_prod_parity() {
    let dst_data = vec![10.0, 20.0, 30.0, 40.0];
    let indices = [0i32, 0, 2];
    let src_data = vec![2.0, 4.0, 8.0];

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();

        let dst = tensor_from_f64(&dst_data, &[4], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let idx = Tensor::from_slice(&indices, &[3], &cpu_device).unwrap();
        let src = tensor_from_f64(&src_data, &[3], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

        let cpu_mean = cpu_client
            .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Mean, true)
            .unwrap_or_else(|e| panic!("CPU scatter_reduce Mean failed for {dtype:?}: {e}"));
        let cpu_prod = cpu_client
            .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Prod, true)
            .unwrap_or_else(|e| panic!("CPU scatter_reduce Prod failed for {dtype:?}: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let d = tensor_from_f64(&dst_data, &[4], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[3], &cuda_device).unwrap();
                let s = tensor_from_f64(&src_data, &[3], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));

                let result_mean = cuda_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Mean, true)
                    .unwrap_or_else(|e| {
                        panic!("CUDA scatter_reduce Mean failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result_mean,
                    &cpu_mean,
                    dtype,
                    &format!("scatter_reduce_mean CUDA vs CPU [{dtype:?}]"),
                );

                let result_prod = cuda_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Prod, true)
                    .unwrap_or_else(|e| {
                        panic!("CUDA scatter_reduce Prod failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result_prod,
                    &cpu_prod,
                    dtype,
                    &format!("scatter_reduce_prod CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let d = tensor_from_f64(&dst_data, &[4], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let i = Tensor::from_slice(&indices, &[3], &wgpu_device).unwrap();
                let s = tensor_from_f64(&src_data, &[3], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WGPU tensor_from_f64 failed for {dtype:?}: {e}"));

                let result_mean = wgpu_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Mean, true)
                    .unwrap_or_else(|e| {
                        panic!("WGPU scatter_reduce Mean failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result_mean,
                    &cpu_mean,
                    dtype,
                    &format!("scatter_reduce_mean WGPU vs CPU [{dtype:?}]"),
                );

                let result_prod = wgpu_client
                    .scatter_reduce(&d, 0, &i, &s, ScatterReduceOp::Prod, true)
                    .unwrap_or_else(|e| {
                        panic!("WGPU scatter_reduce Prod failed for {dtype:?}: {e}")
                    });
                assert_tensor_allclose(
                    &result_prod,
                    &cpu_prod,
                    dtype,
                    &format!("scatter_reduce_prod WGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Regression: `scatter_reduce` with `inner_size > 1` (2D `dst`, scatter along
/// `dim=0`). The WGPU shader previously indexed the index buffer per-slice (as
/// if it were 1D), producing wrong results whenever `inner_size > 1`. The
/// canonical (CPU/CUDA/PyTorch) semantics are element-wise: `index.shape ==
/// src.shape`, and `index[e]` gives the `dim` coordinate for source element `e`.
#[test]
fn test_scatter_reduce_2d_inner_parity() {
    use numr::dtype::DType;

    // dst [4, 3] of zeros; src [2, 3] = 1..6. Source row 0 → dst row 0,
    // source row 1 → dst row 2 (element-wise index, same shape as src).
    let dst_data = vec![0.0f64; 4 * 3];
    let src_data: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let idx_data = [0i32, 0, 0, 2, 2, 2];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_dst =
        tensor_from_f64(&dst_data, &[4, 3], DType::F32, &cpu_device, &cpu_client).unwrap();
    let cpu_idx = Tensor::from_slice(&idx_data, &[2, 3], &cpu_device).unwrap();
    let cpu_src =
        tensor_from_f64(&src_data, &[2, 3], DType::F32, &cpu_device, &cpu_client).unwrap();
    let cpu_out = cpu_client
        .scatter_reduce(&cpu_dst, 0, &cpu_idx, &cpu_src, ScatterReduceOp::Sum, true)
        .unwrap();
    // Expected: dst[0] = [1,2,3], dst[2] = [4,5,6], dst[1] = dst[3] = 0.
    assert_eq!(
        cpu_out.to_vec::<f32>(),
        vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0]
    );

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", DType::F32) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let dst = tensor_from_f64(&dst_data, &[4, 3], DType::F32, &cuda_device, &cuda_client)
                .unwrap();
            let idx = Tensor::from_slice(&idx_data, &[2, 3], &cuda_device).unwrap();
            let src = tensor_from_f64(&src_data, &[2, 3], DType::F32, &cuda_device, &cuda_client)
                .unwrap();
            let out = cuda_client
                .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Sum, true)
                .unwrap();
            assert_tensor_allclose(&out, &cpu_out, DType::F32, "scatter_reduce 2D CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", DType::F32) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let dst = tensor_from_f64(&dst_data, &[4, 3], DType::F32, &wgpu_device, &wgpu_client)
                .unwrap();
            let idx = Tensor::from_slice(&idx_data, &[2, 3], &wgpu_device).unwrap();
            let src = tensor_from_f64(&src_data, &[2, 3], DType::F32, &wgpu_device, &wgpu_client)
                .unwrap();
            let out = wgpu_client
                .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Sum, true)
                .unwrap();
            assert_tensor_allclose(&out, &cpu_out, DType::F32, "scatter_reduce 2D WGPU vs CPU");
        });
    }
}
