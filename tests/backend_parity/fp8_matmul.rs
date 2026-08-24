//! Backend parity tests for FP8 matrix multiplication operations.
//!
//! Tests verify that CUDA FP8 matmul produces results matching CPU reference
//! (cast FP8→F32, matmul, scale, cast to output dtype) within FP tolerance.

use crate::common::create_cpu_client;
use numr::dtype::DType;
use numr::ops::{Fp8MatmulOps, TypeConversionOps};
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Create FP8E4M3 tensor from f32 data on the given backend.
fn create_fp8e4m3_tensor<R: numr::runtime::Runtime<DType = DType>>(
    data: &[f32],
    shape: &[usize],
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> numr::error::Result<Tensor<R>> {
    let f32_tensor = Tensor::from_slice(data, shape, device).unwrap();
    client.cast(&f32_tensor, DType::FP8E4M3)
}

/// Create FP8E5M2 tensor from f32 data on the given backend.
fn create_fp8e5m2_tensor<R: numr::runtime::Runtime<DType = DType>>(
    data: &[f32],
    shape: &[usize],
    device: &R::Device,
    client: &impl TypeConversionOps<R>,
) -> numr::error::Result<Tensor<R>> {
    let f32_tensor = Tensor::from_slice(data, shape, device).unwrap();
    client.cast(&f32_tensor, DType::FP8E5M2)
}

/// Compare f32 results with relaxed tolerance for FP8 (limited precision).
fn assert_fp8_parity(cpu: &[f32], other: &[f32], op: &str) {
    let rtol = 0.1f32; // FP8 has very low precision, ~10% relative tolerance
    let atol = 0.5f32; // Absolute tolerance for small values
    assert_eq!(
        cpu.len(),
        other.len(),
        "fp8_parity[{}]: length mismatch: {} vs {}",
        op,
        cpu.len(),
        other.len()
    );
    for (i, (c, o)) in cpu.iter().zip(other.iter()).enumerate() {
        let diff = (c - o).abs();
        let tol = atol + rtol * c.abs();
        if diff > tol {
            panic!(
                "fp8_parity[{}] at index {}: cpu={} vs other={} (diff={}, tol={})",
                op, i, c, o, diff, tol
            );
        }
    }
}

// ============================================================================
// CPU Tests (baseline)
// ============================================================================

#[test]
fn test_fp8_matmul_e4m3_cpu_f32_output() {
    let (client, device) = create_cpu_client();
    // Small values to stay within FP8E4M3 range
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    let a = create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 3], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[3, 2], &device, &client).unwrap();

    let result = client.fp8_matmul(&a, &b, 1.0, 1.0, DType::F32).unwrap();
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(result.shape(), &[2, 2]);

    let vals = result.to_vec::<f32>();
    // Expected: [1*1+2*3+3*5, 1*2+2*4+3*6, 4*1+5*3+6*5, 4*2+5*4+6*6]
    //         = [22, 28, 49, 64]
    assert_fp8_parity(&[22.0, 28.0, 49.0, 64.0], &vals, "fp8_e4m3_cpu_f32");
}

#[test]
fn test_fp8_matmul_e4m3_cpu_with_scaling() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let a = create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &device, &client).unwrap();

    let result = client.fp8_matmul(&a, &b, 2.0, 0.5, DType::F32).unwrap();
    let vals = result.to_vec::<f32>();
    // scale_a * scale_b = 1.0, so same as unscaled
    // [1*1+2*3, 1*2+2*4, 3*1+4*3, 3*2+4*4] = [7, 10, 15, 22]
    assert_fp8_parity(&[7.0, 10.0, 15.0, 22.0], &vals, "fp8_e4m3_cpu_scaled");
}

#[test]
fn test_fp8_matmul_e5m2_cpu() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

    let a = create_fp8e5m2_tensor::<CpuRuntime>(&a_data, &[2, 2], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &device, &client).unwrap();

    let result = client
        .fp8_matmul_e5m2(&a, &b, 1.0, 1.0, DType::F32)
        .unwrap();
    assert_eq!(result.dtype(), DType::F32);
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_fp8_matmul_e4m3_cpu_f16_output() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let a = create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &device, &client).unwrap();

    let result = client.fp8_matmul(&a, &b, 1.0, 1.0, DType::F16).unwrap();
    assert_eq!(result.dtype(), DType::F16);
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_fp8_matmul_e4m3_cpu_bf16_output() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

    let a = create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &device, &client).unwrap();

    let result = client.fp8_matmul(&a, &b, 1.0, 1.0, DType::BF16).unwrap();
    assert_eq!(result.dtype(), DType::BF16);
    assert_eq!(result.shape(), &[2, 2]);
}

#[test]
fn test_fp8_matmul_dtype_validation() {
    let (client, device) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap();
    let b_data: Vec<f32> = vec![1.0, 2.0];
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 1], &device, &client).unwrap();

    // a is F32, not FP8E4M3 — should fail
    let result = client.fp8_matmul(&a, &b, 1.0, 1.0, DType::F32);
    assert!(result.is_err());
}

#[test]
fn test_fp8_matmul_invalid_output_dtype() {
    let (client, device) = create_cpu_client();
    let a_data: Vec<f32> = vec![1.0, 2.0];
    let b_data: Vec<f32> = vec![1.0, 2.0];

    let a = create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[1, 2], &device, &client).unwrap();
    let b = create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 1], &device, &client).unwrap();

    // I32 is not a valid output dtype for FP8 matmul
    let result = client.fp8_matmul(&a, &b, 1.0, 1.0, DType::I32);
    assert!(result.is_err());
}

// ============================================================================
// CUDA Parity Tests
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use crate::backend_parity::helpers::with_cuda_backend;
    use numr::ops::TypeConversionOps;
    use numr::runtime::cuda::CudaRuntime;

    #[test]
    fn test_fp8_matmul_e4m3_cuda_parity_f32() {
        let (cpu_client, cpu_device) = create_cpu_client();
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
            let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

            let a_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 3], &cpu_device, &cpu_client)
                    .unwrap();
            let b_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[3, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let cpu_result = cpu_client
                .fp8_matmul(&a_cpu, &b_cpu, 1.0, 1.0, DType::F32)
                .unwrap();

            let a_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&a_data, &[2, 3], &cuda_device, &cuda_client)
                    .unwrap();
            let b_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&b_data, &[3, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let cuda_result = cuda_client
                .fp8_matmul(&a_cuda, &b_cuda, 1.0, 1.0, DType::F32)
                .unwrap();

            let cpu_vals = cpu_result.to_vec::<f32>();
            let cuda_f32 = cuda_client.cast(&cuda_result, DType::F32).unwrap();
            let cuda_vals = cuda_f32.to_vec::<f32>();
            assert_fp8_parity(&cpu_vals, &cuda_vals, "fp8_e4m3_cuda_f32");
        });
    }

    #[test]
    fn test_fp8_matmul_e4m3_cuda_parity_with_scaling() {
        let (cpu_client, cpu_device) = create_cpu_client();
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let scale_a = 2.0f32;
            let scale_b = 0.5f32;

            let a_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let b_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let cpu_result = cpu_client
                .fp8_matmul(&a_cpu, &b_cpu, scale_a, scale_b, DType::F32)
                .unwrap();

            let a_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&a_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let b_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&b_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let cuda_result = cuda_client
                .fp8_matmul(&a_cuda, &b_cuda, scale_a, scale_b, DType::F32)
                .unwrap();

            let cpu_vals = cpu_result.to_vec::<f32>();
            let cuda_f32 = cuda_client.cast(&cuda_result, DType::F32).unwrap();
            let cuda_vals = cuda_f32.to_vec::<f32>();
            assert_fp8_parity(&cpu_vals, &cuda_vals, "fp8_e4m3_cuda_scaled");
        });
    }

    #[test]
    fn test_fp8_matmul_e5m2_cuda_parity() {
        let (cpu_client, cpu_device) = create_cpu_client();
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];

            let a_cpu =
                create_fp8e5m2_tensor::<CpuRuntime>(&a_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let b_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let cpu_result = cpu_client
                .fp8_matmul_e5m2(&a_cpu, &b_cpu, 1.0, 1.0, DType::F32)
                .unwrap();

            let a_cuda =
                create_fp8e5m2_tensor::<CudaRuntime>(&a_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let b_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&b_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let cuda_result = cuda_client
                .fp8_matmul_e5m2(&a_cuda, &b_cuda, 1.0, 1.0, DType::F32)
                .unwrap();

            let cpu_vals = cpu_result.to_vec::<f32>();
            let cuda_f32 = cuda_client.cast(&cuda_result, DType::F32).unwrap();
            let cuda_vals = cuda_f32.to_vec::<f32>();
            assert_fp8_parity(&cpu_vals, &cuda_vals, "fp8_e5m2_cuda");
        });
    }

    #[test]
    fn test_fp8_matmul_e4m3_cuda_parity_f16_output() {
        let (cpu_client, cpu_device) = create_cpu_client();
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
            let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

            let a_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let b_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let cpu_result = cpu_client
                .fp8_matmul(&a_cpu, &b_cpu, 1.0, 1.0, DType::F16)
                .unwrap();
            let cpu_f32 = cpu_client.cast(&cpu_result, DType::F32).unwrap();

            let a_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&a_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let b_cuda =
                create_fp8e4m3_tensor::<CudaRuntime>(&b_data, &[2, 2], &cuda_device, &cuda_client)
                    .unwrap();
            let cuda_result = cuda_client
                .fp8_matmul(&a_cuda, &b_cuda, 1.0, 1.0, DType::F16)
                .unwrap();
            let cuda_f32 = cuda_client.cast(&cuda_result, DType::F32).unwrap();

            let cpu_vals = cpu_f32.to_vec::<f32>();
            let cuda_vals = cuda_f32.to_vec::<f32>();
            assert_fp8_parity(&cpu_vals, &cuda_vals, "fp8_e4m3_cuda_f16");
        });
    }

    #[test]
    fn test_fp8_matmul_e4m3_cuda_batched_parity() {
        let (cpu_client, cpu_device) = create_cpu_client();
        with_cuda_backend(|cuda_client, cuda_device| {
            // [2, 2, 2] x [2, 2, 2] batched matmul
            let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 2.0, 3.0, 4.0];

            let a_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&a_data, &[2, 2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let b_cpu =
                create_fp8e4m3_tensor::<CpuRuntime>(&b_data, &[2, 2, 2], &cpu_device, &cpu_client)
                    .unwrap();
            let cpu_result = cpu_client
                .fp8_matmul(&a_cpu, &b_cpu, 1.0, 1.0, DType::F32)
                .unwrap();

            let a_cuda = create_fp8e4m3_tensor::<CudaRuntime>(
                &a_data,
                &[2, 2, 2],
                &cuda_device,
                &cuda_client,
            )
            .unwrap();
            let b_cuda = create_fp8e4m3_tensor::<CudaRuntime>(
                &b_data,
                &[2, 2, 2],
                &cuda_device,
                &cuda_client,
            )
            .unwrap();
            let cuda_result = cuda_client
                .fp8_matmul(&a_cuda, &b_cuda, 1.0, 1.0, DType::F32)
                .unwrap();

            let cpu_vals = cpu_result.to_vec::<f32>();
            let cuda_f32 = cuda_client.cast(&cuda_result, DType::F32).unwrap();
            let cuda_vals = cuda_f32.to_vec::<f32>();
            assert_fp8_parity(&cpu_vals, &cuda_vals, "fp8_e4m3_cuda_batched");
        });
    }
}
