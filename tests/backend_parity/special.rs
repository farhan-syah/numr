// Backend parity tests for SpecialFunctions
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::SpecialFunctions;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

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

fn apply_special_unary<R: Runtime>(
    client: &impl SpecialFunctions<R>,
    op: &str,
    tensor: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "erf" => client.erf(tensor),
        "gamma" => client.gamma(tensor),
        _ => panic!("Unknown special unary op: {}", op),
    }
}

fn apply_special_binary<R: Runtime>(
    client: &impl SpecialFunctions<R>,
    op: &str,
    a: &Tensor<R>,
    x: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "gammainc" => client.gammainc(a, x),
        "gammaincc" => client.gammaincc(a, x),
        _ => panic!("Unknown special binary op: {}", op),
    }
}

fn test_special_unary_parity(op: &str, data: Vec<f64>, shape: Vec<usize>, dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
    let cpu_result = apply_special_unary(&cpu_client, op, &cpu_tensor)
        .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
            let cuda_result = apply_special_unary(&cuda_client, op, &cuda_tensor)
                .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &cuda_result,
                &cpu_result,
                dtype,
                &format!("{op} CUDA vs CPU [{dtype:?}]"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_tensor = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let wgpu_result = apply_special_unary(&wgpu_client, op, &wgpu_tensor)
                .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &wgpu_result,
                &cpu_result,
                dtype,
                &format!("{op} WebGPU vs CPU [{dtype:?}]"),
            );
        });
    }
}

fn test_special_binary_parity(
    op: &str,
    a_data: Vec<f64>,
    x_data: Vec<f64>,
    shape: Vec<usize>,
    dtype: DType,
) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_a = tensor_from_f64(&a_data, &shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 (a) failed for {dtype:?}: {e}"));
    let cpu_x = tensor_from_f64(&x_data, &shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 (x) failed for {dtype:?}: {e}"));
    let cpu_result = apply_special_binary(&cpu_client, op, &cpu_a, &cpu_x)
        .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_a = tensor_from_f64(&a_data, &shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 (a) failed for {dtype:?}: {e}"));
            let cuda_x = tensor_from_f64(&x_data, &shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 (x) failed for {dtype:?}: {e}"));
            let cuda_result = apply_special_binary(&cuda_client, op, &cuda_a, &cuda_x)
                .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &cuda_result,
                &cpu_result,
                dtype,
                &format!("{op} CUDA vs CPU [{dtype:?}]"),
            );
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_a = tensor_from_f64(&a_data, &shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 (a) failed for {dtype:?}: {e}"));
            let wgpu_x = tensor_from_f64(&x_data, &shape, dtype, &wgpu_device, &wgpu_client)
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 (x) failed for {dtype:?}: {e}"));
            let wgpu_result = apply_special_binary(&wgpu_client, op, &wgpu_a, &wgpu_x)
                .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
            assert_tensor_allclose(
                &wgpu_result,
                &cpu_result,
                dtype,
                &format!("{op} WebGPU vs CPU [{dtype:?}]"),
            );
        });
    }
}

// ============================================================================
// Special Function Parity Tests
// ============================================================================

#[test]
fn test_erf_parity() {
    let data = vec![0.0, 0.5, 1.0, 2.0];
    let shape = vec![4];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_special_unary_parity("erf", data.clone(), shape.clone(), dtype);
    }
}

#[test]
fn test_gamma_parity() {
    let data = vec![0.5, 1.0, 2.0, 3.0];
    let shape = vec![4];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_special_unary_parity("gamma", data.clone(), shape.clone(), dtype);
    }
}

#[test]
fn test_gammainc_parity() {
    let a_data = vec![2.0, 3.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0];
    let shape = vec![3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_special_binary_parity(
            "gammainc",
            a_data.clone(),
            x_data.clone(),
            shape.clone(),
            dtype,
        );
    }
}

#[test]
fn test_gammaincc_parity() {
    let a_data = vec![2.0, 3.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0];
    let shape = vec![3];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        test_special_binary_parity(
            "gammaincc",
            a_data.clone(),
            x_data.clone(),
            shape.clone(),
            dtype,
        );
    }
}

#[test]
fn test_incomplete_gamma_complement() {
    // Verify that gammainc + gammaincc = 1 (CPU only, F64 for precision)
    let a_data = vec![2.0, 3.0, 5.0];
    let x_data = vec![1.0, 2.0, 3.0];
    let shape = vec![3];
    let dtype = DType::F64;

    let (cpu_client, cpu_device) = create_cpu_client();
    let a = tensor_from_f64(&a_data, &shape, dtype, &cpu_device, &cpu_client)
        .expect("tensor_from_f64 failed");
    let x = tensor_from_f64(&x_data, &shape, dtype, &cpu_device, &cpu_client)
        .expect("tensor_from_f64 failed");

    let p: Vec<f64> = cpu_client.gammainc(&a, &x).unwrap().to_vec();
    let q: Vec<f64> = cpu_client.gammaincc(&a, &x).unwrap().to_vec();

    for i in 0..3 {
        let sum = p[i] + q[i];
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "CPU P+Q != 1 at {}: P={}, Q={}, sum={}",
            i,
            p[i],
            q[i],
            sum
        );
    }
}
