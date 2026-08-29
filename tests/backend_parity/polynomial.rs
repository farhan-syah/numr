// Backend parity tests for PolynomialAlgorithms trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::algorithm::polynomial::PolynomialAlgorithms;
use numr::dtype::DType;
use numr::runtime::cpu::CpuRuntime;
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

#[derive(Clone)]
struct PolymulTest {
    a: Vec<f64>,
    b: Vec<f64>,
}

impl PolymulTest {
    fn new(a: Vec<f64>, b: Vec<f64>) -> Self {
        PolymulTest { a, b }
    }
}

#[derive(Clone)]
struct PolyvalTest {
    coeffs: Vec<f64>,
    x: Vec<f64>,
}

impl PolyvalTest {
    fn new(coeffs: Vec<f64>, x: Vec<f64>) -> Self {
        PolyvalTest { coeffs, x }
    }
}

#[derive(Clone)]
struct PolyrootsTest {
    coeffs: Vec<f64>,
}

impl PolyrootsTest {
    fn new(coeffs: Vec<f64>) -> Self {
        PolyrootsTest { coeffs }
    }
}

#[derive(Clone)]
struct PolyfromrootsTest {
    roots_real: Vec<f64>,
    roots_imag: Vec<f64>,
}

impl PolyfromrootsTest {
    fn new(roots_real: Vec<f64>, roots_imag: Vec<f64>) -> Self {
        PolyfromrootsTest {
            roots_real,
            roots_imag,
        }
    }
}

// ============================================================================
// Polymul Parity Tests
// ============================================================================

fn run_polymul_parity(test_cases: &[PolymulTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &[tc.a.len()], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &[tc.b.len()], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            cpu_client
                .polymul(&a, &b)
                .unwrap_or_else(|e| panic!("CPU polymul failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &[tc.a.len()], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &[tc.b.len()], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .polymul(&a, &b)
                    .unwrap_or_else(|e| panic!("CUDA polymul failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polymul CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &[tc.a.len()], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &[tc.b.len()], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .polymul(&a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU polymul failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polymul WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_polymul_parity() {
    let test_cases = &[
        PolymulTest::new(vec![1.0, 2.0], vec![3.0, 4.0]),
        PolymulTest::new(vec![1.0, 0.0, 1.0], vec![1.0, 1.0]),
        PolymulTest::new(vec![2.0, 3.0, 1.0], vec![1.0, -1.0]),
        PolymulTest::new(vec![1.0], vec![5.0, 6.0, 7.0]),
    ];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        run_polymul_parity(test_cases, dtype);
    }
}

// ============================================================================
// Polyval Parity Tests
// ============================================================================

fn run_polyval_parity(test_cases: &[PolyvalTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let coeffs = tensor_from_f64(
                &tc.coeffs,
                &[tc.coeffs.len()],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let x = tensor_from_f64(&tc.x, &[tc.x.len()], dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            cpu_client
                .polyval(&coeffs, &x)
                .unwrap_or_else(|e| panic!("CPU polyval failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let coeffs = tensor_from_f64(
                    &tc.coeffs,
                    &[tc.coeffs.len()],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let x = tensor_from_f64(&tc.x, &[tc.x.len()], dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .polyval(&coeffs, &x)
                    .unwrap_or_else(|e| panic!("CUDA polyval failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polyval CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let coeffs = tensor_from_f64(
                    &tc.coeffs,
                    &[tc.coeffs.len()],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let x = tensor_from_f64(&tc.x, &[tc.x.len()], dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .polyval(&coeffs, &x)
                    .unwrap_or_else(|e| panic!("WebGPU polyval failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polyval WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_polyval_parity() {
    let test_cases = &[
        PolyvalTest::new(vec![1.0, 2.0, 3.0], vec![0.5, 1.5, 2.5]),
        PolyvalTest::new(vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 2.0]),
        PolyvalTest::new(vec![5.0, -3.0, 2.0, 1.0], vec![-1.0, 0.0, 1.0, 2.0]),
    ];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        run_polyval_parity(test_cases, dtype);
    }
}

// ============================================================================
// Polyroots Parity Tests
// ============================================================================

fn run_polyroots_parity(test_cases: &[PolyrootsTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<(Tensor<CpuRuntime>, Tensor<CpuRuntime>)> = test_cases
        .iter()
        .map(|tc| {
            let coeffs = tensor_from_f64(
                &tc.coeffs,
                &[tc.coeffs.len()],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let roots = cpu_client
                .polyroots(&coeffs)
                .unwrap_or_else(|e| panic!("CPU polyroots failed for {dtype:?}: {e}"));
            (roots.roots_real, roots.roots_imag)
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let coeffs = tensor_from_f64(
                    &tc.coeffs,
                    &[tc.coeffs.len()],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let roots = cuda_client
                    .polyroots(&coeffs)
                    .unwrap_or_else(|e| panic!("CUDA polyroots failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &roots.roots_real,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("polyroots real CUDA vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &roots.roots_imag,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("polyroots imag CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let coeffs = tensor_from_f64(
                    &tc.coeffs,
                    &[tc.coeffs.len()],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let roots = wgpu_client
                    .polyroots(&coeffs)
                    .unwrap_or_else(|e| panic!("WebGPU polyroots failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &roots.roots_real,
                    &cpu_results[idx].0,
                    dtype,
                    &format!("polyroots real WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
                assert_tensor_allclose(
                    &roots.roots_imag,
                    &cpu_results[idx].1,
                    dtype,
                    &format!("polyroots imag WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_polyroots_parity() {
    let test_cases = &[
        PolyrootsTest::new(vec![6.0, -5.0, 1.0]), // (x-2)(x-3) = x^2 - 5x + 6
        PolyrootsTest::new(vec![2.0, -3.0, 1.0]), // (x-1)(x-2) = x^2 - 3x + 2
        PolyrootsTest::new(vec![0.0, 0.0, 1.0]),  // x^2
    ];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        run_polyroots_parity(test_cases, dtype);
    }
}

// ============================================================================
// Polyfromroots Parity Tests
// ============================================================================

fn run_polyfromroots_parity(test_cases: &[PolyfromrootsTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let roots_real = tensor_from_f64(
                &tc.roots_real,
                &[tc.roots_real.len()],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let roots_imag = tensor_from_f64(
                &tc.roots_imag,
                &[tc.roots_imag.len()],
                dtype,
                &cpu_device,
                &cpu_client,
            )
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            cpu_client
                .polyfromroots(&roots_real, &roots_imag)
                .unwrap_or_else(|e| panic!("CPU polyfromroots failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let roots_real = tensor_from_f64(
                    &tc.roots_real,
                    &[tc.roots_real.len()],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let roots_imag = tensor_from_f64(
                    &tc.roots_imag,
                    &[tc.roots_imag.len()],
                    dtype,
                    &cuda_device,
                    &cuda_client,
                )
                .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = cuda_client
                    .polyfromroots(&roots_real, &roots_imag)
                    .unwrap_or_else(|e| panic!("CUDA polyfromroots failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polyfromroots CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let roots_real = tensor_from_f64(
                    &tc.roots_real,
                    &[tc.roots_real.len()],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let roots_imag = tensor_from_f64(
                    &tc.roots_imag,
                    &[tc.roots_imag.len()],
                    dtype,
                    &wgpu_device,
                    &wgpu_client,
                )
                .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = wgpu_client
                    .polyfromroots(&roots_real, &roots_imag)
                    .unwrap_or_else(|e| panic!("WebGPU polyfromroots failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("polyfromroots WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_polyfromroots_parity() {
    let test_cases = &[
        PolyfromrootsTest::new(vec![2.0, 3.0], vec![0.0, 0.0]), // Real roots: 2, 3
        PolyfromrootsTest::new(vec![1.0, 2.0], vec![0.0, 0.0]), // Real roots: 1, 2
        PolyfromrootsTest::new(vec![0.0, 0.0], vec![0.0, 0.0]), // Double root at 0
        PolyfromrootsTest::new(vec![1.0, 1.0], vec![1.0, -1.0]), // Complex pair: 1±i
    ];

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        run_polyfromroots_parity(test_cases, dtype);
    }
}
