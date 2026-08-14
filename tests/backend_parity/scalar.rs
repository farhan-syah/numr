// Backend parity tests for ScalarOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.
// Comparison reads back in native dtype via assert_tensor_allclose.

use numr::dtype::DType;
use numr::ops::ScalarOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    ToF64, assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

#[derive(Clone)]
struct ScalarTest {
    data: Vec<f64>,
    shape: Vec<usize>,
    scalar: f64,
}

impl ScalarTest {
    fn new(data: Vec<f64>, shape: Vec<usize>, scalar: f64) -> Self {
        ScalarTest {
            data,
            shape,
            scalar,
        }
    }
}

fn apply_scalar_op<R: Runtime>(
    client: &impl ScalarOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    scalar: f64,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "add_scalar" => client.add_scalar(tensor, scalar),
        "sub_scalar" => client.sub_scalar(tensor, scalar),
        "mul_scalar" => client.mul_scalar(tensor, scalar),
        "div_scalar" => client.div_scalar(tensor, scalar),
        "pow_scalar" => client.pow_scalar(tensor, scalar),
        "rsub_scalar" => client.rsub_scalar(tensor, scalar),
        _ => panic!("Unknown scalar op: {}", op),
    }
}

/// Read back a tensor as `Vec<f64>` regardless of its native dtype.
fn tensor_to_f64_vec<R: Runtime<DType = DType>>(tensor: &Tensor<R>, dtype: DType) -> Vec<f64> {
    macro_rules! readback {
        ($T:ty) => {
            tensor
                .to_vec::<$T>()
                .iter()
                .map(|x| <$T as ToF64>::to_f64(*x))
                .collect()
        };
    }

    match dtype {
        DType::F64 => readback!(f64),
        DType::F32 => readback!(f32),
        #[cfg(feature = "f16")]
        DType::F16 => readback!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => readback!(half::bf16),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => readback!(numr::dtype::FP8E4M3),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => readback!(numr::dtype::FP8E5M2),
        _ => panic!("tensor_to_f64_vec: unsupported dtype {dtype:?}"),
    }
}

fn test_scalar_parity(op: &str, test_cases: &[ScalarTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensor = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            apply_scalar_op(&cpu_client, op, &tensor, tc.scalar)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"))
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .unwrap_or_else(|e| {
                            panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_scalar_op(&cuda_client, op, &tensor, tc.scalar)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .unwrap_or_else(|e| {
                            panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                        });
                let result = apply_scalar_op(&wgpu_client, op, &tensor, tc.scalar)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! scalar_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_scalar_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Scalar Operation Parity Tests
// ============================================================================

scalar_case!(
    test_add_scalar_parity,
    "add_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 5.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -2.5),
        ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 10.0),
    ]
);

scalar_case!(
    test_sub_scalar_parity,
    "sub_scalar",
    &[
        ScalarTest::new(vec![5.0, 6.0, 7.0, 8.0], vec![4], 2.0),
        ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], 5.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 0.5),
    ]
);

scalar_case!(
    test_mul_scalar_parity,
    "mul_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 2.0),
        ScalarTest::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], 0.5),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], -3.0),
    ]
);

scalar_case!(
    test_div_scalar_parity,
    "div_scalar",
    &[
        ScalarTest::new(vec![10.0, 20.0, 30.0, 40.0], vec![4], 2.0),
        ScalarTest::new(vec![100.0, 200.0, 300.0, 400.0], vec![2, 2], 10.0),
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], 4.0),
    ]
);

scalar_case!(
    test_pow_scalar_parity,
    "pow_scalar",
    &[
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![4], 2.0),
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 3.0),
        ScalarTest::new(vec![4.0, 9.0, 16.0, 25.0], vec![2, 2], 0.5),
    ]
);

scalar_case!(
    test_rsub_scalar_parity,
    "rsub_scalar",
    &[
        ScalarTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 10.0),
        ScalarTest::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2], 20.0),
        ScalarTest::new(vec![0.5, 1.5, 2.5, 3.5], vec![2, 2], 5.0),
    ]
);

// ============================================================================
// pow_scalar: negative base with integral exponent
//
// Regression test for the CUDA --use_fast_math bug where powf(x, y) lowers to
// exp2f(y * log2f(x)), which is NaN for ANY negative base x - even when y is
// an integer and IEEE/CPU semantics define a real result. CPU uses Rust's
// f64::powf, which is IEEE-correct: (-3.0).powf(2.0) == 9.0. This test pins
// down the exact expected values (not just CPU==CUDA parity) so a regression
// that breaks both backends identically would still be caught.
// ============================================================================

#[test]
fn test_pow_scalar_negative_base_cpu_exact() {
    let (cpu_client, cpu_device) = create_cpu_client();

    for dtype in supported_dtypes("cpu") {
        if !dtype.is_float() {
            continue;
        }

        // (-3.0)^2 == 9.0 (negative base, EVEN integer exponent -> positive)
        let neg_even = tensor_from_f64(&[-3.0], &[1], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("tensor_from_f64 failed for {dtype:?}: {e}"));
        let result = cpu_client
            .pow_scalar(&neg_even, 2.0)
            .unwrap_or_else(|e| panic!("pow_scalar failed for {dtype:?}: {e}"));
        assert_tensor_allclose(
            &result,
            &tensor_from_f64(&[9.0], &[1], dtype, &cpu_device, &cpu_client).unwrap(),
            dtype,
            &format!("pow_scalar(-3, 2) CPU [{dtype:?}]"),
        );

        // (-3.0)^3 == -27.0 (negative base, ODD integer exponent -> negative)
        let neg_odd = tensor_from_f64(&[-3.0], &[1], dtype, &cpu_device, &cpu_client).unwrap();
        let result = cpu_client
            .pow_scalar(&neg_odd, 3.0)
            .unwrap_or_else(|e| panic!("pow_scalar failed for {dtype:?}: {e}"));
        assert_tensor_allclose(
            &result,
            &tensor_from_f64(&[-27.0], &[1], dtype, &cpu_device, &cpu_client).unwrap(),
            dtype,
            &format!("pow_scalar(-3, 3) CPU [{dtype:?}]"),
        );

        // 2.0^2 == 4.0 (positive base, unaffected baseline)
        let pos = tensor_from_f64(&[2.0], &[1], dtype, &cpu_device, &cpu_client).unwrap();
        let result = cpu_client
            .pow_scalar(&pos, 2.0)
            .unwrap_or_else(|e| panic!("pow_scalar failed for {dtype:?}: {e}"));
        assert_tensor_allclose(
            &result,
            &tensor_from_f64(&[4.0], &[1], dtype, &cpu_device, &cpu_client).unwrap(),
            dtype,
            &format!("pow_scalar(2, 2) CPU [{dtype:?}]"),
        );
    }
}

#[test]
fn test_pow_scalar_negative_base_parity() {
    // Vector mixing positive and negative bases, squared: CUDA must match CPU
    // exactly (within tolerance) and must be finite everywhere CPU is finite.
    let data = vec![-3.0, 2.0, -5.0, 4.0, -1.0, 0.0];
    let shape = vec![data.len()];

    for dtype in supported_dtypes("cpu") {
        if !dtype.is_float() {
            continue;
        }

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_tensor = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = cpu_client
            .pow_scalar(&cpu_tensor, 2.0)
            .unwrap_or_else(|e| panic!("CPU pow_scalar failed for {dtype:?}: {e}"));
        let cpu_vals = tensor_to_f64_vec(&cpu_result, dtype);
        assert!(
            cpu_vals.iter().all(|v| v.is_finite()),
            "CPU pow_scalar([-3,2,-5,4,-1,0], 2) should be finite for {dtype:?}, got {cpu_vals:?}"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let cuda_tensor = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let cuda_result = cuda_client
                    .pow_scalar(&cuda_tensor, 2.0)
                    .unwrap_or_else(|e| panic!("CUDA pow_scalar failed for {dtype:?}: {e}"));
                let cuda_vals = tensor_to_f64_vec(&cuda_result, dtype);
                assert!(
                    cuda_vals.iter().all(|v| v.is_finite()),
                    "CUDA pow_scalar([-3,2,-5,4,-1,0], 2) should be finite wherever CPU is \
                     finite for {dtype:?} (fast-math negative-base regression), got {cuda_vals:?}"
                );
                assert_tensor_allclose(
                    &cuda_result,
                    &cpu_result,
                    dtype,
                    &format!("pow_scalar(mixed, 2) CUDA vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}
