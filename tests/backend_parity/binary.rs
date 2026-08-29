// Backend parity tests for BinaryOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Tensors are created in f64 then cast to target dtype via tensor_from_f64().
// Comparison reads back in native dtype - no unnecessary f64 conversion.

use numr::dtype::DType;
use numr::ops::BinaryOps;
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

#[derive(Clone, Copy, Debug)]
enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Maximum,
    Minimum,
    Atan2,
}

#[derive(Clone)]
struct TestCase {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
}

impl TestCase {
    fn new(a: Vec<f64>, a_shape: Vec<usize>, b: Vec<f64>, b_shape: Vec<usize>) -> Self {
        Self {
            a,
            a_shape,
            b,
            b_shape,
        }
    }
}

fn apply_binary_op<R: Runtime>(
    client: &impl BinaryOps<R>,
    op: BinaryOp,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        BinaryOp::Add => client.add(a, b),
        BinaryOp::Sub => client.sub(a, b),
        BinaryOp::Mul => client.mul(a, b),
        BinaryOp::Div => client.div(a, b),
        BinaryOp::Pow => client.pow(a, b),
        BinaryOp::Maximum => client.maximum(a, b),
        BinaryOp::Minimum => client.minimum(a, b),
        BinaryOp::Atan2 => client.atan2(a, b),
    }
}

fn test_binary_parity(op: BinaryOp, test_cases: &[TestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    // Compute CPU baseline results (kept as tensors for native comparison)
    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
            let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client)
                .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));

            apply_binary_op(&cpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?} failed for {dtype:?}: {e}"))
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

                let result = apply_binary_op(&cuda_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("CUDA {op:?} failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} CUDA vs CPU [{dtype:?}] case {idx}"),
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

                let result = apply_binary_op(&wgpu_client, op, &a, &b)
                    .unwrap_or_else(|e| panic!("WebGPU {op:?} failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op:?} WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

// The domain is per-op, not per-module: this module mixes plain arithmetic with
// `atan2`, whose result is an angle.
macro_rules! binary_case {
    ($name:ident, $op:expr, $cases:expr, $domain:expr) => {
        #[test]
        fn $name() {
            for dtype in parity_dtypes($domain, "cpu") {
                test_binary_parity($op, $cases, dtype);
            }
        }
    };
}

binary_case!(
    test_add_parity,
    BinaryOp::Add,
    &[
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4]
        ),
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![2, 2]
        ),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![10.0], vec![1]),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![5.0], vec![]),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_sub_parity,
    BinaryOp::Sub,
    &[
        TestCase::new(
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4],
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![1.0, 1.0, 1.0, 1.0],
            vec![2, 2]
        ),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_mul_parity,
    BinaryOp::Mul,
    &[
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![2.0, 3.0, 4.0, 5.0],
            vec![4]
        ),
        TestCase::new(
            vec![0.5, 1.5, 2.5, 3.5],
            vec![2, 2],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![2, 2]
        ),
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![2.0], vec![]),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_div_parity,
    BinaryOp::Div,
    &[
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![4],
            vec![2.0, 4.0, 5.0, 8.0],
            vec![4]
        ),
        TestCase::new(
            vec![100.0, 200.0, 300.0, 400.0],
            vec![2, 2],
            vec![2.0, 4.0, 5.0, 8.0],
            vec![2, 2],
        ),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_pow_parity,
    BinaryOp::Pow,
    &[
        TestCase::new(
            vec![2.0, 3.0, 4.0, 5.0],
            vec![4],
            vec![2.0, 2.0, 2.0, 2.0],
            vec![4]
        ),
        TestCase::new(
            vec![2.0, 3.0, 4.0, 5.0],
            vec![2, 2],
            vec![0.0, 1.0, 2.0, 3.0],
            vec![2, 2]
        ),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_maximum_parity,
    BinaryOp::Maximum,
    &[
        TestCase::new(
            vec![1.0, 5.0, 3.0, 2.0],
            vec![4],
            vec![3.0, 2.0, 5.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![15.0, 15.0, 15.0, 15.0],
            vec![2, 2],
        ),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_minimum_parity,
    BinaryOp::Minimum,
    &[
        TestCase::new(
            vec![1.0, 5.0, 3.0, 2.0],
            vec![4],
            vec![3.0, 2.0, 5.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![10.0, 20.0, 30.0, 40.0],
            vec![2, 2],
            vec![15.0, 15.0, 15.0, 15.0],
            vec![2, 2],
        ),
    ],
    DTypeDomain::AllNumeric
);

binary_case!(
    test_atan2_parity,
    BinaryOp::Atan2,
    &[
        TestCase::new(
            vec![0.0, 1.0, 1.0, 0.0],
            vec![4],
            vec![1.0, 0.0, 1.0, 1.0],
            vec![4]
        ),
        TestCase::new(
            vec![1.0, -1.0, -1.0, 1.0],
            vec![2, 2],
            vec![1.0, 1.0, -1.0, -1.0],
            vec![2, 2]
        ),
    ],
    DTypeDomain::FloatsOnly
);

// Destination-passing `add_into`: must match the allocating `add` on every
// backend (CPU reference), including the broadcast path.
fn test_add_into_parity(test_cases: &[TestCase], dtype: DType) {
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client).unwrap();
            let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client).unwrap();
            // Independent reference: the allocating `add`. Using `add` (not
            // `add_into`) means a CPU-only run genuinely validates that CPU
            // `add_into` produces `a + b`, instead of comparing it to itself.
            cpu_client.add(&a, &b).unwrap()
        })
        .collect();

    // Verify CPU `add_into` writes into the destination and matches the
    // independent `add` reference (runs on every build, including CPU-only).
    for (idx, tc) in test_cases.iter().enumerate() {
        let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client).unwrap();
        let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client).unwrap();
        let out = Tensor::<numr::runtime::cpu::CpuRuntime>::zeros(
            cpu_results[idx].shape(),
            dtype,
            &cpu_device,
        )
        .unwrap();
        cpu_client.add_into(&out, &a, &b).unwrap();
        assert_tensor_allclose(
            &out,
            &cpu_results[idx],
            dtype,
            &format!("add_into CPU vs add [{dtype:?}] case {idx}"),
        );
    }

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cuda_device, &cuda_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cuda_device, &cuda_client).unwrap();
                let out = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(
                    cpu_results[idx].shape(),
                    dtype,
                    &cuda_device,
                )
                .unwrap();
                cuda_client.add_into(&out, &a, &b).unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx],
                    dtype,
                    &format!("add_into CUDA vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }

    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let a =
                    tensor_from_f64(&tc.a, &tc.a_shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let b =
                    tensor_from_f64(&tc.b, &tc.b_shape, dtype, &wgpu_device, &wgpu_client).unwrap();
                let out = Tensor::<numr::runtime::wgpu::WgpuRuntime>::zeros(
                    cpu_results[idx].shape(),
                    dtype,
                    &wgpu_device,
                )
                .unwrap();
                wgpu_client.add_into(&out, &a, &b).unwrap();
                assert_tensor_allclose(
                    &out,
                    &cpu_results[idx],
                    dtype,
                    &format!("add_into WebGPU vs CPU [{dtype:?}] case {idx}"),
                );
            }
        });
    }
}

#[test]
fn test_add_into_matches_add() {
    let cases = [
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![4],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4],
        ),
        TestCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![0.5, 0.5, 0.5, 0.5],
            vec![2, 2],
        ),
        // Broadcast path (a [4] + b [1]).
        TestCase::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], vec![10.0], vec![1]),
    ];
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        test_add_into_parity(&cases, dtype);
    }
}

// ============================================================================
// Empty tensor - CUDA parity (native_binary_op / native_binary_op_into)
// ============================================================================

/// Before the fix, `elementwise_launch_config(0)` produced `grid.x == 0`,
/// which is an invalid CUDA launch. `add` on same-shape `[0]` F32 tensors must
/// succeed and return an empty tensor, matching the CPU backend's natural
/// no-op.
#[cfg(feature = "cuda")]
#[test]
fn test_add_empty_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let b = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let result = cuda_client.add(&a, &b);
        assert!(
            result.is_ok(),
            "add on empty CUDA tensors should succeed, got {:?}",
            result.err()
        );
        let result = result.unwrap();
        assert_eq!(result.shape(), &[0]);
        assert_eq!(result.dtype(), DType::F32);
        assert_eq!(result.numel(), 0);
    });
}

/// Same as `test_add_empty_cuda` but through the destination-passing
/// `add_into` path (`native_binary_op_into`), which writes into a
/// caller-provided `out` tensor instead of allocating.
#[cfg(feature = "cuda")]
#[test]
fn test_add_into_empty_cuda() {
    with_cuda_backend(|cuda_client, cuda_device| {
        let a = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let b = Tensor::from_slice::<f32>(&[], &[0], &cuda_device).unwrap();
        let out = Tensor::<numr::runtime::cuda::CudaRuntime>::zeros(&[0], DType::F32, &cuda_device)
            .unwrap();
        let result = cuda_client.add_into(&out, &a, &b);
        assert!(
            result.is_ok(),
            "add_into on empty CUDA tensors should succeed, got {:?}",
            result.err()
        );
        assert_eq!(out.shape(), &[0]);
        assert_eq!(out.dtype(), DType::F32);
        assert_eq!(out.numel(), 0);
    });
}

// ============================================================================
// FP8 broadcast binary ops - CUDA parity
//
// `binary.rs` builds `format!("{}_broadcast_{}_inline", op, dtype_str)` for
// the general broadcast path and looks it up with a hard `?`. Before the fix,
// no `*_broadcast_fp8_e4m3_inline` / `*_broadcast_fp8_e5m2_inline` kernels
// existed, so every broadcast binary op on an FP8 CUDA tensor failed with a
// kernel-not-found error. Shapes [2,3] and [3] genuinely broadcast.
// Values are small powers of two (and their sums), exactly representable in
// both FP8E4M3 and FP8E5M2, so results are compared with the dtype's normal
// parity tolerance (see `tolerance_for_dtype`) rather than a hand-waved one.
// ============================================================================

#[cfg(all(feature = "fp8", feature = "cuda"))]
#[test]
fn test_fp8_broadcast_parity() {
    let cases = &[TestCase::new(
        vec![1.0, 2.0, 4.0, 8.0, 0.5, 2.0],
        vec![2, 3],
        vec![1.0, 2.0, 0.5],
        vec![3],
    )];

    for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Div,
            BinaryOp::Pow,
            BinaryOp::Maximum,
            BinaryOp::Minimum,
        ] {
            test_binary_parity(op, cases, dtype);
        }
    }
}
