// Backend parity tests for `bincount`: output sizing across the full index
// range, and the error variant each backend returns for a rejected input.
//
// Output length is `max(input) + 1`, so sizing it correctly depends on the max
// reduction being exact. A backend that takes the max in F32 loses precision
// past 2^24 and sizes the output wrongly, which no value-comparison test
// catches because the counts it does produce are still right.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::IndexingOps;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::create_cpu_client;

/// Smallest integer an F32 cannot represent exactly. A max reduction taken in
/// F32 rounds it back to 2^24 and undersizes the output by one bin.
const ABOVE_F32_MANTISSA: i64 = 16_777_217;

#[test]
fn test_bincount_output_len_above_2_pow_24_parity() {
    // Output length is max + 1, so this allocates ~16.7M bins per backend.
    // That size is inherent to the defect: the precision loss only appears
    // above 2^24, so no smaller input can exercise it.
    let input_data = [0i64, 1, ABOVE_F32_MANTISSA];
    let expected_len = (ABOVE_F32_MANTISSA as usize) + 1;

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_input = Tensor::from_slice(&input_data, &[3], &cpu_device).unwrap();
    let cpu_out = cpu_client
        .bincount(&cpu_input, None, 0)
        .unwrap_or_else(|e| panic!("CPU bincount failed: {e}"));
    assert_eq!(
        cpu_out.shape(),
        &[expected_len][..],
        "CPU bincount output length"
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::from_slice(&input_data, &[3], &cuda_device).unwrap();
        let out = cuda_client
            .bincount(&input, None, 0)
            .unwrap_or_else(|e| panic!("CUDA bincount failed: {e}"));
        assert_eq!(
            out.shape(),
            &[expected_len][..],
            "bincount CUDA vs CPU output length"
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let input = Tensor::from_slice(&input_data, &[3], &wgpu_device).unwrap();
        let out = wgpu_client
            .bincount(&input, None, 0)
            .unwrap_or_else(|e| panic!("WGPU bincount failed: {e}"));
        assert_eq!(
            out.shape(),
            &[expected_len][..],
            "bincount WGPU vs CPU output length"
        );
    });
}

/// Assert a rejected non-1-D input yields `ShapeMismatch` on `backend`.
fn assert_rank_rejection(backend: &str, err: Error) {
    match err {
        Error::ShapeMismatch { got, .. } => {
            assert_eq!(got, vec![2, 2], "{backend} bincount rank rejection payload");
        }
        other => panic!("{backend} bincount rank: want ShapeMismatch, got {other:?}"),
    }
}

/// Assert a rejected non-integer input yields `DTypeMismatch` on `backend`.
fn assert_dtype_rejection(backend: &str, err: Error) {
    match err {
        Error::DTypeMismatch { rhs, .. } => {
            assert_eq!(
                rhs,
                DType::F32,
                "{backend} bincount dtype rejection payload"
            );
        }
        other => panic!("{backend} bincount dtype: want DTypeMismatch, got {other:?}"),
    }
}

/// Assert a rejected mismatched-shape `weights` yields `ShapeMismatch` on `backend`.
fn assert_weights_shape_rejection(backend: &str, err: Error) {
    match err {
        Error::ShapeMismatch { expected, got } => {
            assert_eq!(
                expected,
                vec![4],
                "{backend} bincount weights rejection expected payload"
            );
            assert_eq!(
                got,
                vec![3],
                "{backend} bincount weights rejection got payload"
            );
        }
        other => panic!("{backend} bincount weights shape: want ShapeMismatch, got {other:?}"),
    }
}

#[test]
fn test_bincount_weights_shape_rejection_parity() {
    // WebGPU accepted a `weights` tensor without checking its shape against
    // the input, so a mismatched length silently read out of bounds instead
    // of being rejected. CPU and CUDA both reject with ShapeMismatch.
    let input_data = [0i64, 1, 2, 3];
    let weights_data = [1.0f32, 2.0, 3.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_input = Tensor::from_slice(&input_data, &[4], &cpu_device).unwrap();
    let cpu_weights = Tensor::from_slice(&weights_data, &[3], &cpu_device).unwrap();
    assert_weights_shape_rejection(
        "CPU",
        cpu_client
            .bincount(&cpu_input, Some(&cpu_weights), 0)
            .expect_err("CPU bincount must reject mismatched weights shape"),
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::from_slice(&input_data, &[4], &cuda_device).unwrap();
        let weights = Tensor::from_slice(&weights_data, &[3], &cuda_device).unwrap();
        assert_weights_shape_rejection(
            "CUDA",
            cuda_client
                .bincount(&input, Some(&weights), 0)
                .expect_err("CUDA bincount must reject mismatched weights shape"),
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let input = Tensor::from_slice(&input_data, &[4], &wgpu_device).unwrap();
        let weights = Tensor::from_slice(&weights_data, &[3], &wgpu_device).unwrap();
        assert_weights_shape_rejection(
            "WGPU",
            wgpu_client
                .bincount(&input, Some(&weights), 0)
                .expect_err("WGPU bincount must reject mismatched weights shape"),
        );
    });
}

#[test]
fn test_bincount_rejection_error_variant_parity() {
    // Asserting the variant, not merely that an error occurred: the backends
    // previously all rejected these inputs, but WebGPU used InvalidArgument
    // where CPU and CUDA used ShapeMismatch/DTypeMismatch, so a caller matching
    // on the error had to special-case the backend.
    let rank_data = [0i64, 1, 2, 3];
    let float_data = [0.0f32, 1.0, 2.0, 3.0];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_rank = Tensor::from_slice(&rank_data, &[2, 2], &cpu_device).unwrap();
    assert_rank_rejection(
        "CPU",
        cpu_client
            .bincount(&cpu_rank, None, 0)
            .expect_err("CPU bincount must reject a 2-D input"),
    );
    let cpu_float = Tensor::from_slice(&float_data, &[4], &cpu_device).unwrap();
    assert_dtype_rejection(
        "CPU",
        cpu_client
            .bincount(&cpu_float, None, 0)
            .expect_err("CPU bincount must reject an F32 input"),
    );

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let rank = Tensor::from_slice(&rank_data, &[2, 2], &cuda_device).unwrap();
        assert_rank_rejection(
            "CUDA",
            cuda_client
                .bincount(&rank, None, 0)
                .expect_err("CUDA bincount must reject a 2-D input"),
        );
        let float = Tensor::from_slice(&float_data, &[4], &cuda_device).unwrap();
        assert_dtype_rejection(
            "CUDA",
            cuda_client
                .bincount(&float, None, 0)
                .expect_err("CUDA bincount must reject an F32 input"),
        );
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
        let rank = Tensor::from_slice(&rank_data, &[2, 2], &wgpu_device).unwrap();
        assert_rank_rejection(
            "WGPU",
            wgpu_client
                .bincount(&rank, None, 0)
                .expect_err("WGPU bincount must reject a 2-D input"),
        );
        let float = Tensor::from_slice(&float_data, &[4], &wgpu_device).unwrap();
        assert_dtype_rejection(
            "WGPU",
            wgpu_client
                .bincount(&float, None, 0)
                .expect_err("WGPU bincount must reject an F32 input"),
        );
    });
}
