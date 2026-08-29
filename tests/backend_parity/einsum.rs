// Backend parity tests for EinsumOps trait
//
// Dtype-parameterized: each test runs for all supported dtypes (F32, F64, F16, BF16, FP8).
// Tensors are created in f64 then cast to target dtype via tensor_from_f64().
// Comparison reads back in native dtype - no unnecessary f64 conversion.

use numr::dtype::DType;
use numr::ops::EinsumOps;
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

struct EinsumTest {
    notation: &'static str,
    inputs: Vec<(Vec<f64>, Vec<usize>)>,
}

impl EinsumTest {
    fn new(notation: &'static str, inputs: Vec<(Vec<f64>, Vec<usize>)>) -> Self {
        EinsumTest { notation, inputs }
    }
}

fn test_einsum_parity(test_cases: &[EinsumTest], dtype: DType) {
    // CPU baseline
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensors: Vec<_> = tc
                .inputs
                .iter()
                .map(|(data, shape)| {
                    tensor_from_f64(data, shape, dtype, &cpu_device, &cpu_client)
                        .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"))
                })
                .collect();
            let tensor_refs: Vec<_> = tensors.iter().collect();
            cpu_client
                .einsum(tc.notation, &tensor_refs)
                .unwrap_or_else(|e| panic!("CPU einsum failed for {dtype:?}: {e}"))
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensors: Vec<_> = tc
                    .inputs
                    .iter()
                    .map(|(data, shape)| {
                        tensor_from_f64(data, shape, dtype, &cuda_device, &cuda_client)
                            .unwrap_or_else(|e| {
                                panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}")
                            })
                    })
                    .collect();
                let tensor_refs: Vec<_> = tensors.iter().collect();

                let result = cuda_client
                    .einsum(tc.notation, &tensor_refs)
                    .unwrap_or_else(|e| panic!("CUDA einsum failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("einsum {} CUDA vs CPU [{dtype:?}]", tc.notation),
                );
            }
        });
    }

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensors: Vec<_> = tc
                    .inputs
                    .iter()
                    .map(|(data, shape)| {
                        tensor_from_f64(data, shape, dtype, &wgpu_device, &wgpu_client)
                            .unwrap_or_else(|e| {
                                panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}")
                            })
                    })
                    .collect();
                let tensor_refs: Vec<_> = tensors.iter().collect();

                let result = wgpu_client
                    .einsum(tc.notation, &tensor_refs)
                    .unwrap_or_else(|e| panic!("WebGPU einsum failed for {dtype:?}: {e}"));

                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("einsum {} WebGPU vs CPU [{dtype:?}]", tc.notation),
                );
            }
        });
    }
}

// ============================================================================
// Einsum Parity Tests
// ============================================================================

macro_rules! einsum_case {
    ($name:ident, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
                test_einsum_parity($cases, dtype);
            }
        }
    };
}

einsum_case!(
    test_einsum_matmul_parity,
    &[EinsumTest::new(
        "ij,jk->ik",
        vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            (vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![3, 2])
        ],
    )]
);

einsum_case!(
    test_einsum_batched_matmul_parity,
    &[EinsumTest::new(
        "bij,bjk->bik",
        vec![
            (
                vec![
                    // Batch 0
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
                    2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                ],
                vec![2, 2, 3]
            ),
            (
                vec![
                    // Batch 0
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Batch 1
                    2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                ],
                vec![2, 3, 2]
            )
        ],
    )]
);

einsum_case!(
    test_einsum_transpose_parity,
    &[EinsumTest::new(
        "ij->ji",
        vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])]
    )]
);

einsum_case!(
    test_einsum_outer_product_parity,
    &[EinsumTest::new(
        "i,j->ij",
        vec![
            (vec![1.0, 2.0, 3.0], vec![3]),
            (vec![4.0, 5.0, 6.0, 7.0], vec![4])
        ],
    )]
);

einsum_case!(
    test_einsum_trace_parity,
    &[EinsumTest::new(
        "ii->",
        vec![(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3]
        )]
    )]
);

einsum_case!(
    test_einsum_elementwise_parity,
    &[EinsumTest::new(
        "ij,ij->ij",
        vec![
            (vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]),
            (vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3])
        ],
    )]
);

einsum_case!(
    test_einsum_sum_parity,
    &[EinsumTest::new(
        "ij->",
        vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])]
    )]
);

einsum_case!(
    test_einsum_reduction_parity,
    &[EinsumTest::new(
        "ij->i",
        vec![(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])]
    )]
);
