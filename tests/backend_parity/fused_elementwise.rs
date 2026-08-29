// Backend parity tests for fused elementwise operations
//
// Tests: fused_mul_add, fused_add_mul (BinaryOps), fused_mul_add_scalar (ScalarOps)
// Dtype-parameterized: runs for all supported dtypes across all backends.

use numr::dtype::DType;
use numr::ops::{BinaryOps, ScalarOps};
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
// Ternary test cases (a, b, c)
// ============================================================================

#[derive(Clone)]
struct TernaryCase {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    shape: Vec<usize>,
}

impl TernaryCase {
    fn new(a: Vec<f64>, b: Vec<f64>, c: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { a, b, c, shape }
    }
}

fn ternary_cases() -> Vec<TernaryCase> {
    vec![
        TernaryCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![0.5, 1.0, 1.5, 2.0],
            vec![4],
        ),
        TernaryCase::new(
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
            vec![2, 3],
        ),
        TernaryCase::new(
            vec![-1.0, 0.0, 1.0, 2.0],
            vec![3.0, 3.0, 3.0, 3.0],
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
        ),
    ]
}

// ============================================================================
// fused_mul_add: out = a * b + c
// ============================================================================

fn test_fused_mul_add_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = ternary_cases();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let c = tensor_from_f64(&tc.c, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            cpu_client.fused_mul_add(&a, &b, &c).unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let c =
                    tensor_from_f64(&tc.c, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client.fused_mul_add(&a, &b, &c).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_mul_add CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let c =
                    tensor_from_f64(&tc.c, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client.fused_mul_add(&a, &b, &c).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_mul_add WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_fused_mul_add_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        test_fused_mul_add_parity(dtype);
    }
}

// ============================================================================
// fused_add_mul: out = (a + b) * c
// ============================================================================

fn test_fused_add_mul_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = ternary_cases();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let b = tensor_from_f64(&tc.b, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            let c = tensor_from_f64(&tc.c, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            cpu_client.fused_add_mul(&a, &b, &c).unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let c =
                    tensor_from_f64(&tc.c, &tc.shape, dtype, &cuda_device, &cuda_client).unwrap();
                let result = cuda_client.fused_add_mul(&a, &b, &c).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_add_mul CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let c =
                    tensor_from_f64(&tc.c, &tc.shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let result = wgpu_client.fused_add_mul(&a, &b, &c).unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_add_mul WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_fused_add_mul_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        test_fused_add_mul_parity(dtype);
    }
}

// ============================================================================
// fused_mul_add_scalar: out = a * scale + bias
// ============================================================================

#[derive(Clone)]
struct ScalarFmaCase {
    data: Vec<f64>,
    shape: Vec<usize>,
    scale: f64,
    bias: f64,
}

impl ScalarFmaCase {
    fn new(data: Vec<f64>, shape: Vec<usize>, scale: f64, bias: f64) -> Self {
        Self {
            data,
            shape,
            scale,
            bias,
        }
    }
}

fn scalar_fma_cases() -> Vec<ScalarFmaCase> {
    vec![
        ScalarFmaCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 2.5, -1.0),
        ScalarFmaCase::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], vec![2, 3], 10.0, 0.5),
        ScalarFmaCase::new(vec![-2.0, -1.0, 0.0, 1.0], vec![2, 2], 0.5, 3.0),
    ]
}

fn test_fused_mul_add_scalar_parity(dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cases = scalar_fma_cases();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client).unwrap();
            cpu_client
                .fused_mul_add_scalar(&a, tc.scale, tc.bias)
                .unwrap()
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                    .unwrap();
                let result = cuda_client
                    .fused_mul_add_scalar(&a, tc.scale, tc.bias)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_mul_add_scalar CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap();
                let result = wgpu_client
                    .fused_mul_add_scalar(&a, tc.scale, tc.bias)
                    .unwrap();
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("fused_mul_add_scalar WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_fused_mul_add_scalar_all_dtypes() {
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        test_fused_mul_add_scalar_parity(dtype);
    }
}
