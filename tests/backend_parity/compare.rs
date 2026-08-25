// Backend parity tests for CompareOps trait
//
// Dtype-parameterized: each test runs for all supported input dtypes across all backends.
// Compare ops return boolean masks - output dtype may differ by backend (u8 vs u32),
// so we read back as u32 for uniform comparison.

use numr::dtype::DType;
use numr::ops::CompareOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
use crate::backend_parity::helpers::assert_parity_u32;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{create_cpu_client, is_dtype_supported, supported_dtypes};

// ============================================================================
// Test Utilities
// ============================================================================

#[derive(Clone)]
struct CompareTest {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
}

impl CompareTest {
    fn new(a: Vec<f64>, a_shape: Vec<usize>, b: Vec<f64>, b_shape: Vec<usize>) -> Self {
        CompareTest {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn apply_compare_op<R: Runtime>(
    client: &impl CompareOps<R>,
    op: &str,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "eq" => client.eq(a, b),
        "ne" => client.ne(a, b),
        "lt" => client.lt(a, b),
        "le" => client.le(a, b),
        "gt" => client.gt(a, b),
        "ge" => client.ge(a, b),
        _ => panic!("Unknown compare op: {}", op),
    }
}

/// Read back a compare result as Vec<u32> regardless of backend output dtype.
/// Some backends return Bool (u8), some U32, some keep the input dtype
/// where nonzero = true, zero = false.
fn readback_as_u32<R: Runtime<DType = DType>>(tensor: &Tensor<R>) -> Vec<u32> {
    use crate::common::ToF64;

    macro_rules! via_f64 {
        ($T:ty) => {
            tensor
                .to_vec::<$T>()
                .iter()
                .map(|x| {
                    if <$T as ToF64>::to_f64(*x) != 0.0 {
                        1u32
                    } else {
                        0u32
                    }
                })
                .collect()
        };
    }

    match tensor.dtype() {
        DType::Bool => tensor.to_vec::<u8>().iter().map(|&x| x as u32).collect(),
        DType::U32 => tensor
            .to_vec::<u32>()
            .iter()
            .map(|&x| if x != 0 { 1 } else { 0 })
            .collect(),
        DType::I32 => tensor
            .to_vec::<i32>()
            .iter()
            .map(|&x| if x != 0 { 1 } else { 0 })
            .collect(),
        DType::F32 => via_f64!(f32),
        DType::F64 => via_f64!(f64),
        #[cfg(feature = "f16")]
        DType::F16 => via_f64!(half::f16),
        #[cfg(feature = "f16")]
        DType::BF16 => via_f64!(half::bf16),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => via_f64!(numr::dtype::FP8E4M3),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => via_f64!(numr::dtype::FP8E5M2),
        other => panic!("Unexpected compare output dtype: {other:?}"),
    }
}

fn test_compare_parity(op: &str, test_cases: &[CompareTest], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Vec<u32>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let result = apply_compare_op(&cpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));
            readback_as_u32(&result)
        })
        .collect();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_compare_op(&cuda_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_parity_u32(
                    &cpu_results[idx],
                    &readback_as_u32(&result),
                    &format!("{op} CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply_compare_op(&wgpu_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_parity_u32(
                    &cpu_results[idx],
                    &readback_as_u32(&result),
                    &format!("{op} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

macro_rules! compare_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_compare_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Compare Operation Parity Tests
// ============================================================================

compare_case!(
    test_eq_parity,
    "eq",
    &[
        CompareTest::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![1.0, 2.0, 0.0, 4.0],
            vec![4],
        ),
        CompareTest::new(
            vec![5.0, 5.0, 5.0, 5.0],
            vec![2, 2],
            vec![5.0, 5.0, 5.0, 5.0],
            vec![2, 2],
        ),
    ]
);

compare_case!(
    test_ne_parity,
    "ne",
    &[
        CompareTest::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![1.0, 2.0, 0.0, 4.0],
            vec![4],
        ),
        CompareTest::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            vec![5.0, 0.0, 7.0, 0.0],
            vec![2, 2],
        ),
    ]
);

compare_case!(
    test_lt_parity,
    "lt",
    &[
        CompareTest::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 5.0],
            vec![4],
        ),
        CompareTest::new(
            vec![1.0, 5.0, 3.0, 7.0],
            vec![2, 2],
            vec![2.0, 4.0, 3.0, 8.0],
            vec![2, 2],
        ),
    ]
);

compare_case!(
    test_le_parity,
    "le",
    &[
        CompareTest::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 5.0],
            vec![4],
        ),
        CompareTest::new(
            vec![1.0, 5.0, 3.0, 7.0],
            vec![2, 2],
            vec![2.0, 4.0, 3.0, 8.0],
            vec![2, 2],
        ),
    ]
);

compare_case!(
    test_gt_parity,
    "gt",
    &[
        CompareTest::new(
            vec![3.0, 2.0, 1.0, 5.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 4.0],
            vec![4],
        ),
        CompareTest::new(
            vec![5.0, 3.0, 4.0, 2.0],
            vec![2, 2],
            vec![2.0, 4.0, 3.0, 1.0],
            vec![2, 2],
        ),
    ]
);

compare_case!(
    test_ge_parity,
    "ge",
    &[
        CompareTest::new(
            vec![3.0, 2.0, 1.0, 5.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 4.0],
            vec![4],
        ),
        CompareTest::new(
            vec![5.0, 3.0, 4.0, 2.0],
            vec![2, 2],
            vec![2.0, 4.0, 3.0, 1.0],
            vec![2, 2],
        ),
    ]
);

/// `native_compare_op` shares the zero-grid exposure the other elementwise
/// dispatch helpers had: with `numel == 0`, `elementwise_launch_config` used to
/// return `grid.x == 0`, which the CUDA driver rejects outright
/// (`CUDA_ERROR_INVALID_VALUE`). An empty comparison must return an empty mask,
/// matching CPU, not fail to launch.
#[cfg(feature = "cuda")]
#[test]
fn test_eq_empty_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let b = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let result = cuda_client.eq(&a, &b);
        assert!(
            result.is_ok(),
            "eq on empty CUDA tensors should succeed, got {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.shape(), &[0]);
        // Deliberately not pinned to a dtype: this file's header notes that a
        // compare op's mask dtype differs by backend (CUDA returns the input
        // dtype here, others use u8/u32). Shape and emptiness are the parity
        // properties that matter for an empty input.
        assert_eq!(result.numel(), 0);
    });
}
