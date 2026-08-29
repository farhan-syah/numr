// Backend parity tests for ShapeOps trait
//
// Tests verify that ShapeOps operations produce identical results across
// CPU, CUDA, and WebGPU backends, with full dtype coverage.
//
// Migrated from scattered cuda_parity/wgpu_parity modules in shape_ops.rs.

use numr::dtype::DType;
use numr::ops::ShapeOps;
use numr::runtime::Runtime;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

fn test_repeat_on_backends(data: &[f64], shape: &[usize], repeats: &[usize], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.repeat(&cpu_tensor, repeats).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = cuda_client.repeat(&tensor, repeats).unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(&result, &cpu_result, dtype, "repeat CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = wgpu_client.repeat(&tensor, repeats).unwrap();
            assert_eq!(cpu_result.shape(), result.shape());
            assert_tensor_allclose(&result, &cpu_result, dtype, "repeat WebGPU vs CPU");
        });
    }
}

fn test_cat_on_backends(
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    dim: isize,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let b_cpu = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.cat(&[&a_cpu, &b_cpu], dim).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client.cat(&[&a, &b], dim).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "cat CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let a = tensor_from_f64(a_data, a_shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(b_data, b_shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client.cat(&[&a, &b], dim).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(&wgpu_result, &cpu_result, dtype, "cat WebGPU vs CPU");
        });
    }
}

fn test_stack_on_backends(
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    dim: isize,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let b_cpu = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.stack(&[&a_cpu, &b_cpu], dim).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client.stack(&[&a, &b], dim).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "stack CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let a = tensor_from_f64(a_data, a_shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(b_data, b_shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client.stack(&[&a, &b], dim).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(&wgpu_result, &cpu_result, dtype, "stack WebGPU vs CPU");
        });
    }
}

fn test_split_on_backends(
    data: &[f64],
    shape: &[usize],
    split_size: usize,
    dim: isize,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_chunks = cpu_client.split(&cpu_tensor, split_size, dim).unwrap();
    let cpu_shapes: Vec<Vec<usize>> = cpu_chunks.iter().map(|t| t.shape().to_vec()).collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let chunks = cuda_client.split(&tensor, split_size, dim).unwrap();
            assert_eq!(cpu_chunks.len(), chunks.len());
            for (idx, chunk) in chunks.iter().enumerate() {
                assert_eq!(cpu_shapes[idx], chunk.shape().to_vec());
                assert_tensor_allclose(
                    &chunk.contiguous().unwrap(),
                    &cpu_chunks[idx].contiguous().unwrap(),
                    dtype,
                    &format!("split CUDA vs CPU chunk {}", idx),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let chunks = wgpu_client.split(&tensor, split_size, dim).unwrap();
            assert_eq!(cpu_chunks.len(), chunks.len());
            for (idx, chunk) in chunks.iter().enumerate() {
                assert_eq!(cpu_shapes[idx], chunk.shape().to_vec());
                assert_tensor_allclose(
                    &chunk.contiguous().unwrap(),
                    &cpu_chunks[idx].contiguous().unwrap(),
                    dtype,
                    &format!("split WebGPU vs CPU chunk {}", idx),
                );
            }
        });
    }
}

fn test_chunk_on_backends(data: &[f64], shape: &[usize], chunks: usize, dim: isize, dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_chunks = cpu_client.chunk(&cpu_tensor, chunks, dim).unwrap();
    let cpu_shapes: Vec<Vec<usize>> = cpu_chunks.iter().map(|t| t.shape().to_vec()).collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let got = cuda_client.chunk(&tensor, chunks, dim).unwrap();
            assert_eq!(cpu_chunks.len(), got.len());
            for (idx, chunk) in got.iter().enumerate() {
                assert_eq!(cpu_shapes[idx], chunk.shape().to_vec());
                assert_tensor_allclose(
                    &chunk.contiguous().unwrap(),
                    &cpu_chunks[idx].contiguous().unwrap(),
                    dtype,
                    &format!("chunk CUDA vs CPU chunk {}", idx),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let got = wgpu_client.chunk(&tensor, chunks, dim).unwrap();
            assert_eq!(cpu_chunks.len(), got.len());
            for (idx, chunk) in got.iter().enumerate() {
                assert_eq!(cpu_shapes[idx], chunk.shape().to_vec());
                assert_tensor_allclose(
                    &chunk.contiguous().unwrap(),
                    &cpu_chunks[idx].contiguous().unwrap(),
                    dtype,
                    &format!("chunk WebGPU vs CPU chunk {}", idx),
                );
            }
        });
    }
}

fn test_pad_on_backends(
    data: &[f64],
    shape: &[usize],
    padding: &[usize],
    value: f64,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.pad(&cpu_tensor, padding, value).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client.pad(&cuda_tensor, padding, value).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "pad CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client.pad(&wgpu_tensor, padding, value).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(&wgpu_result, &cpu_result, dtype, "pad WebGPU vs CPU");
        });
    }
}

fn test_roll_on_backends(data: &[f64], shape: &[usize], shift: isize, dim: isize, dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.roll(&cpu_tensor, shift, dim).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client.roll(&cuda_tensor, shift, dim).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(&cuda_result, &cpu_result, dtype, "roll CUDA vs CPU");
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client.roll(&wgpu_tensor, shift, dim).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(&wgpu_result, &cpu_result, dtype, "roll WebGPU vs CPU");
        });
    }
}

fn test_unfold_on_backends(
    data: &[f64],
    shape: &[usize],
    dim: isize,
    size: usize,
    step: usize,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client.unfold(&cpu_tensor, dim, size, step).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client.unfold(&cuda_tensor, dim, size, step).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(
                &cuda_result.contiguous().unwrap(),
                &cpu_result.contiguous().unwrap(),
                dtype,
                "unfold CUDA vs CPU",
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client.unfold(&wgpu_tensor, dim, size, step).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(
                &wgpu_result.contiguous().unwrap(),
                &cpu_result.contiguous().unwrap(),
                dtype,
                "unfold WebGPU vs CPU",
            );
        });
    }
}

fn test_repeat_interleave_on_backends(
    data: &[f64],
    shape: &[usize],
    repeats: usize,
    dim: Option<isize>,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_tensor = tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_client
        .repeat_interleave(&cpu_tensor, repeats, dim)
        .unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_client
                .repeat_interleave(&cuda_tensor, repeats, dim)
                .unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(
                &cuda_result,
                &cpu_result,
                dtype,
                "repeat_interleave CUDA vs CPU",
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_client
                .repeat_interleave(&wgpu_tensor, repeats, dim)
                .unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(
                &wgpu_result,
                &cpu_result,
                dtype,
                "repeat_interleave WebGPU vs CPU",
            );
        });
    }
}

fn test_flip_on_backends(data: &[f64], shape: &[usize], dim: isize, dtype: DType) {
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};
    let cpu_device = CpuDevice::new();
    let cpu_tensor = tensor_from_f64(
        data,
        shape,
        dtype,
        &cpu_device,
        &CpuRuntime::default_client(&cpu_device),
    )
    .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = cpu_tensor.flip(dim).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = cuda_tensor.flip(dim).unwrap();
            assert_eq!(cpu_result.shape(), cuda_result.shape());
            assert_tensor_allclose(
                &cuda_result.contiguous().unwrap(),
                &cpu_result.contiguous().unwrap(),
                dtype,
                "flip CUDA vs CPU",
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = wgpu_tensor.flip(dim).unwrap();
            assert_eq!(cpu_result.shape(), wgpu_result.shape());
            assert_tensor_allclose(
                &wgpu_result.contiguous().unwrap(),
                &cpu_result.contiguous().unwrap(),
                dtype,
                "flip WebGPU vs CPU",
            );
        });
    }
}

// ============================================================================
// Shape Operation Parity Tests
// ============================================================================

#[test]
fn test_cat_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0];
        test_cat_on_backends(&a, &[2, 2], &b, &[2, 1], -1, dtype);
    }
}

#[test]
fn test_stack_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 20.0, 30.0, 40.0];
        test_stack_on_backends(&a, &[2, 2], &b, &[2, 2], -1, dtype);
    }
}

#[test]
fn test_split_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        test_split_on_backends(&data, &[2, 5], 2, -1, dtype);
    }
}

#[test]
fn test_chunk_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        test_chunk_on_backends(&data, &[2, 5], 3, -1, dtype);
    }
}

#[test]
fn test_repeat_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_repeat_on_backends(&data, &[2, 3], &[2, 3], dtype);
    }
}

#[test]
fn test_pad_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // Pad last dim by (1, 2), second-to-last by (1, 1)
        test_pad_on_backends(&data, &[2, 3], &[1, 2, 1, 1], 0.0, dtype);
    }
}

#[test]
fn test_roll_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_roll_on_backends(&data, &[2, 3], 2, 1, dtype);
    }
}

#[test]
fn test_roll_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_roll_on_backends(&data, &[2, 3], -1, -1, dtype);
    }
}

#[test]
fn test_flip_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_flip_on_backends(&data, &[2, 3], 1, dtype);
    }
}

#[test]
fn test_flip_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_flip_on_backends(&data, &[2, 3], -1, dtype);
    }
}

#[test]
fn test_unfold_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_unfold_on_backends(&data, &[2, 3], 1, 2, 1, dtype);
    }
}

#[test]
fn test_unfold_parity_dim0() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_unfold_on_backends(&data, &[2, 3], 0, 2, 1, dtype);
    }
}

#[test]
fn test_unfold_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        test_unfold_on_backends(&data, &[2, 3], -1, 2, 1, dtype);
    }
}

#[test]
fn test_repeat_interleave_parity() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0];
        test_repeat_interleave_on_backends(&data, &[2, 2], 2, Some(1), dtype);
    }
}

#[test]
fn test_repeat_interleave_parity_negative_dim() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0];
        test_repeat_interleave_on_backends(&data, &[2, 2], 2, Some(-1), dtype);
    }
}

#[test]
fn test_repeat_interleave_parity_flattened() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let data = [1.0, 2.0, 3.0, 4.0];
        test_repeat_interleave_on_backends(&data, &[2, 2], 2, None, dtype);
    }
}
