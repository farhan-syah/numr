// Backend parity tests for LogicalOps trait
//
// Logical ops work on U8 tensors (0 = false, non-zero = true).
// CPU is the reference implementation; CUDA and WebGPU must match.

use numr::ops::LogicalOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{create_cpu_client, readback_as_bool};

#[derive(Clone, Copy, Debug)]
enum LogicalOp {
    And,
    Or,
    Xor,
}

fn apply_logical_op<R: Runtime>(
    client: &impl LogicalOps<R>,
    op: LogicalOp,
    a: &Tensor<R>,
    b: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        LogicalOp::And => client.logical_and(a, b),
        LogicalOp::Or => client.logical_or(a, b),
        LogicalOp::Xor => client.logical_xor(a, b),
    }
}

struct BinaryLogicalCase {
    a: Vec<u8>,
    b: Vec<u8>,
    shape: Vec<usize>,
}

impl BinaryLogicalCase {
    fn new(a: Vec<u8>, b: Vec<u8>, shape: Vec<usize>) -> Self {
        Self { a, b, shape }
    }
}

fn binary_logical_cases() -> Vec<BinaryLogicalCase> {
    vec![
        // Basic 1D
        BinaryLogicalCase::new(vec![1, 0, 1, 0], vec![1, 1, 0, 0], vec![4]),
        // All true
        BinaryLogicalCase::new(vec![1, 1, 1, 1], vec![1, 1, 1, 1], vec![4]),
        // All false
        BinaryLogicalCase::new(vec![0, 0, 0, 0], vec![0, 0, 0, 0], vec![4]),
        // 2D
        BinaryLogicalCase::new(vec![1, 0, 0, 1, 1, 0], vec![0, 1, 1, 0, 1, 1], vec![2, 3]),
        // Non-zero values treated as true
        BinaryLogicalCase::new(vec![5, 0, 255, 0], vec![0, 3, 0, 1], vec![4]),
    ]
}

fn test_binary_logical_parity(op: LogicalOp) {
    let cases = binary_logical_cases();
    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Vec<bool>> = cases
        .iter()
        .map(|tc| {
            let a =
                Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&tc.a, &tc.shape, &cpu_device)
                    .unwrap();
            let b =
                Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(&tc.b, &tc.shape, &cpu_device)
                    .unwrap();
            let result = apply_logical_op(&cpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CPU {op:?} failed: {e}"));
            readback_as_bool(&result)
        })
        .collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, tc) in cases.iter().enumerate() {
            let a = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &tc.a,
                &tc.shape,
                &cuda_device,
            )
            .unwrap();
            let b = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &tc.b,
                &tc.shape,
                &cuda_device,
            )
            .unwrap();
            let result = apply_logical_op(&cuda_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("CUDA {op:?} failed: {e}"));
            let cuda_bools = readback_as_bool(&result);
            assert_eq!(
                cuda_bools, cpu_results[idx],
                "{op:?} CUDA vs CPU case {idx}"
            );
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, tc) in cases.iter().enumerate() {
            // WebGPU uses U32 for bool-like tensors
            let a_u32: Vec<u32> = tc.a.iter().map(|&v| v as u32).collect();
            let b_u32: Vec<u32> = tc.b.iter().map(|&v| v as u32).collect();
            let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &a_u32,
                &tc.shape,
                &wgpu_device,
            )
            .unwrap();
            let b = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &b_u32,
                &tc.shape,
                &wgpu_device,
            )
            .unwrap();
            let result = apply_logical_op(&wgpu_client, op, &a, &b)
                .unwrap_or_else(|e| panic!("WebGPU {op:?} failed: {e}"));
            let wgpu_bools = readback_as_bool(&result);
            assert_eq!(
                wgpu_bools, cpu_results[idx],
                "{op:?} WebGPU vs CPU case {idx}"
            );
        }
    });
}

fn test_not_parity() {
    let cases: Vec<(Vec<u8>, Vec<usize>)> = vec![
        (vec![1, 0, 1, 0], vec![4]),
        (vec![0, 0, 0, 0], vec![4]),
        (vec![1, 1, 1, 1], vec![4]),
        (vec![5, 0, 255, 0, 1, 0], vec![2, 3]),
    ];

    let (cpu_client, cpu_device) = create_cpu_client();

    let cpu_results: Vec<Vec<bool>> = cases
        .iter()
        .map(|(data, shape)| {
            let a = Tensor::<numr::runtime::cpu::CpuRuntime>::from_slice(data, shape, &cpu_device)
                .unwrap();
            let result = cpu_client
                .logical_not(&a)
                .unwrap_or_else(|e| panic!("CPU NOT failed: {e}"));
            readback_as_bool(&result)
        })
        .collect();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        for (idx, (data, shape)) in cases.iter().enumerate() {
            let a =
                Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(data, shape, &cuda_device)
                    .unwrap();
            let result = cuda_client
                .logical_not(&a)
                .unwrap_or_else(|e| panic!("CUDA NOT failed: {e}"));
            let cuda_bools = readback_as_bool(&result);
            assert_eq!(cuda_bools, cpu_results[idx], "NOT CUDA vs CPU case {idx}");
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        for (idx, (data, shape)) in cases.iter().enumerate() {
            let data_u32: Vec<u32> = data.iter().map(|&v| v as u32).collect();
            let a = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &data_u32,
                shape,
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .logical_not(&a)
                .unwrap_or_else(|e| panic!("WebGPU NOT failed: {e}"));
            let wgpu_bools = readback_as_bool(&result);
            assert_eq!(wgpu_bools, cpu_results[idx], "NOT WebGPU vs CPU case {idx}");
        }
    });
}

#[test]
fn test_logical_and_parity() {
    test_binary_logical_parity(LogicalOp::And);
}

#[test]
fn test_logical_or_parity() {
    test_binary_logical_parity(LogicalOp::Or);
}

#[test]
fn test_logical_xor_parity() {
    test_binary_logical_parity(LogicalOp::Xor);
}

#[test]
fn test_logical_not_parity() {
    test_not_parity();
}
