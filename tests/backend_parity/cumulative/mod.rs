// Backend parity tests for CumulativeOps trait
//
// Tests verify that all CumulativeOps operations produce identical results across
// CPU, CUDA, and WebGPU backends, for all supported dtypes.
//
// `supported_dtypes("cpu")` below never yields I32/U32/I64/U64, so the
// macro-driven tests in this file never exercise integer cumsum/cumprod.
// `wgpu` holds the hand-built WebGPU-vs-CPU I32/U32 coverage, and `cuda`
// holds the hand-built CUDA-vs-CPU I32/U32/I64/U64 coverage, that fill that
// hole.

pub mod cuda;
pub mod wgpu;

use numr::dtype::DType;
use numr::ops::CumulativeOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};

// ============================================================================
// Test Utilities
// ============================================================================

struct CumulativeTest {
    data: Vec<f64>,
    shape: Vec<usize>,
    dim: isize,
}

impl CumulativeTest {
    fn new(data: Vec<f64>, shape: Vec<usize>, dim: isize) -> Self {
        CumulativeTest { data, shape, dim }
    }
}

fn apply_cumulative_op<R: Runtime>(
    client: &impl CumulativeOps<R>,
    op: &str,
    tensor: &Tensor<R>,
    dim: isize,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "cumsum" => client.cumsum(tensor, dim),
        "cumprod" => client.cumprod(tensor, dim),
        "logsumexp" => {
            // Convert negative dim to positive index
            let ndim = tensor.shape().len() as isize;
            let dim_usize = if dim < 0 {
                (ndim + dim) as usize
            } else {
                dim as usize
            };
            client.logsumexp(tensor, &[dim_usize], false)
        }
        _ => panic!("Unknown cumulative op: {}", op),
    }
}

fn test_cumulative_parity(op: &str, test_cases: Vec<CumulativeTest>, dtype: DType) {
    // CPU baseline - store as Tensor<CpuRuntime> for comparison
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Tensor<numr::runtime::cpu::CpuRuntime>> = test_cases
        .iter()
        .map(|tc| {
            let tensor = tensor_from_f64(&tc.data, &tc.shape, dtype, &cpu_device, &cpu_client)
                .expect("tensor creation failed");
            apply_cumulative_op(&cpu_client, op, &tensor, tc.dim).expect("CPU operation failed")
        })
        .collect();

    // CUDA parity
    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &cuda_device, &cuda_client)
                        .expect("tensor creation failed");
                let result = apply_cumulative_op(&cuda_client, op, &tensor, tc.dim)
                    .expect("CUDA operation failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op}_cuda_dtype_{dtype:?}_case_{idx}"),
                );
            }
        });
    }

    // WebGPU parity
    #[cfg(feature = "wgpu")]
    if is_dtype_supported("wgpu", dtype) {
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            for (idx, tc) in test_cases.iter().enumerate() {
                let tensor =
                    tensor_from_f64(&tc.data, &tc.shape, dtype, &wgpu_device, &wgpu_client)
                        .expect("tensor creation failed");
                let result = apply_cumulative_op(&wgpu_client, op, &tensor, tc.dim)
                    .expect("WebGPU operation failed");
                assert_tensor_allclose(
                    &result,
                    &cpu_results[idx],
                    dtype,
                    &format!("{op}_wgpu_dtype_{dtype:?}_case_{idx}"),
                );
            }
        });
    }
}

// ============================================================================
// Test Macro for DType Parameterization
// ============================================================================

macro_rules! cumulative_case {
    ($name:ident, $op:expr, $cases:expr) => {
        #[test]
        fn $name() {
            for dtype in supported_dtypes("cpu") {
                test_cumulative_parity($op, $cases, dtype);
            }
        }
    };
}

// ============================================================================
// Cumulative Operation Parity Tests
// ============================================================================

cumulative_case!(
    test_cumsum_parity,
    "cumsum",
    vec![
        // 1D cumsum
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
        // 2D cumsum along rows
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 0),
        // 2D cumsum along columns
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 1),
        // 3D cumsum
        CumulativeTest::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
            1,
        ),
    ]
);

cumulative_case!(
    test_cumprod_parity,
    "cumprod",
    vec![
        // 1D cumprod
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
        // 2D cumprod along rows
        CumulativeTest::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3], 0),
        // 2D cumprod along columns
        CumulativeTest::new(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0], vec![2, 3], 1),
    ]
);

cumulative_case!(
    test_logsumexp_parity,
    "logsumexp",
    vec![
        // 1D logsumexp
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], 0),
        // 2D logsumexp along rows
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 0),
        // 2D logsumexp along columns
        CumulativeTest::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], 1),
    ]
);
