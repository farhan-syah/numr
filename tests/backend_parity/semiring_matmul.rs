// Backend parity tests for SemiringMatmulOps trait
//
// Tests: semiring_matmul with MinPlus, MaxPlus, MaxMin, MinMax, OrAnd
// CPU is the reference implementation; CUDA and WebGPU must match.

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    assert_tensor_allclose, create_cpu_client, is_dtype_supported, supported_dtypes,
};
use numr::dtype::DType;
use numr::ops::{SemiringMatmulOps, SemiringOp};

struct SemiringCase {
    a: Vec<f64>,
    a_shape: Vec<usize>,
    b: Vec<f64>,
    b_shape: Vec<usize>,
    op: SemiringOp,
}

impl SemiringCase {
    fn new(
        a: Vec<f64>,
        a_shape: Vec<usize>,
        b: Vec<f64>,
        b_shape: Vec<usize>,
        op: SemiringOp,
    ) -> Self {
        Self {
            a,
            a_shape,
            b,
            b_shape,
            op,
        }
    }
}

fn semiring_test_cases() -> Vec<SemiringCase> {
    // 2x3 @ 3x2 matrices
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];

    vec![
        // MinPlus: shortest path semantics
        SemiringCase::new(
            a.clone(),
            vec![2, 3],
            b.clone(),
            vec![3, 2],
            SemiringOp::MinPlus,
        ),
        // MaxPlus: longest path semantics
        SemiringCase::new(
            a.clone(),
            vec![2, 3],
            b.clone(),
            vec![3, 2],
            SemiringOp::MaxPlus,
        ),
        // MaxMin: bottleneck path
        SemiringCase::new(
            a.clone(),
            vec![2, 3],
            b.clone(),
            vec![3, 2],
            SemiringOp::MaxMin,
        ),
        // MinMax: fuzzy relations
        SemiringCase::new(
            a.clone(),
            vec![2, 3],
            b.clone(),
            vec![3, 2],
            SemiringOp::MinMax,
        ),
        // Smaller matrices
        SemiringCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![2, 2],
            SemiringOp::MinPlus,
        ),
        // 1x4 @ 4x1 (vector inner product)
        SemiringCase::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 4],
            vec![5.0, 6.0, 7.0, 8.0],
            vec![4, 1],
            SemiringOp::MaxPlus,
        ),
    ]
}

fn test_semiring_parity(dtype: DType) {
    let cases = semiring_test_cases();
    let (cpu_client, cpu_device) = create_cpu_client();

    for (idx, tc) in cases.iter().enumerate() {
        let cpu_a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU a tensor failed");
        let cpu_b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cpu_device, &cpu_client)
            .expect("CPU b tensor failed");
        let cpu_result = cpu_client
            .semiring_matmul(&cpu_a, &cpu_b, tc.op)
            .unwrap_or_else(|e| panic!("CPU semiring {:?} failed for {dtype:?}: {e}", tc.op));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA a tensor failed");
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &cuda_device, &cuda_client)
                    .expect("CUDA b tensor failed");
                let result = cuda_client
                    .semiring_matmul(&a, &b, tc.op)
                    .unwrap_or_else(|e| panic!("CUDA semiring failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("semiring {:?} CUDA vs CPU [{dtype:?}] case {idx}", tc.op),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|wgpu_client, wgpu_device| {
                let a = tensor_from_f64(&tc.a, &tc.a_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU a tensor failed");
                let b = tensor_from_f64(&tc.b, &tc.b_shape, dtype, &wgpu_device, &wgpu_client)
                    .expect("WebGPU b tensor failed");
                let result = wgpu_client
                    .semiring_matmul(&a, &b, tc.op)
                    .unwrap_or_else(|e| panic!("WebGPU semiring failed: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("semiring {:?} WebGPU vs CPU [{dtype:?}] case {idx}", tc.op),
                );
            });
        }
    }
}

#[test]
fn test_semiring_matmul_parity_all_dtypes() {
    for dtype in supported_dtypes("cpu") {
        test_semiring_parity(dtype);
    }
}

// OrAnd operates on Bool tensors (u8: 0/1 values)
#[test]
fn test_semiring_or_and_parity() {
    use numr::tensor::Tensor;

    let (cpu_client, cpu_device) = create_cpu_client();

    // Boolean adjacency matrices
    let a: Vec<u8> = vec![1, 0, 1, 0, 1, 1, 0, 0, 1];
    let b: Vec<u8> = vec![0, 1, 0, 1, 0, 1, 1, 1, 0];

    let cpu_a =
        Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&a, &[3, 3], &cpu_device).unwrap();
    let cpu_b =
        Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&b, &[3, 3], &cpu_device).unwrap();
    #[allow(unused_variables)]
    let cpu_result = cpu_client
        .semiring_matmul(&cpu_a, &cpu_b, SemiringOp::OrAnd)
        .expect("CPU OrAnd failed");

    // WebGPU skipped: OrAnd requires Bool dtype, WebGPU is 32-bit only

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cpu_vals = cpu_result.to_vec::<u8>();
        let ca = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&a, &[3, 3], &cuda_device)
            .unwrap();
        let cb = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&b, &[3, 3], &cuda_device)
            .unwrap();
        let result = cuda_client
            .semiring_matmul(&ca, &cb, SemiringOp::OrAnd)
            .expect("CUDA OrAnd failed");
        let cuda_vals = result.to_vec::<u8>();
        assert_eq!(cpu_vals, cuda_vals, "OrAnd CUDA vs CPU");
    });
}
