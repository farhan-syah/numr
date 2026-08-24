//! Integration tests for CPU runtime
//!
//! These tests verify the public API of the CPU runtime implementation.

use numr::dtype::DType;
use numr::ops::{
    ActivationOps, BinaryOps, CompareOps, ConditionalOps, IndexingOps, LogicalOps, MatmulOps,
    NormalizationOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps, UtilityOps,
};
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::{Allocator, Runtime, RuntimeClient};
use numr::tensor::Tensor;

#[test]
fn test_allocate_deallocate() {
    let device = CpuDevice::new();
    let ptr = CpuRuntime::allocate(1024, &device).unwrap();
    assert_ne!(ptr, 0);
    CpuRuntime::deallocate(ptr, 1024, &device);
}

#[test]
fn test_copy_roundtrip() {
    let device = CpuDevice::new();
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let ptr = CpuRuntime::allocate(data.len(), &device).unwrap();
    CpuRuntime::copy_to_device(&data, ptr, &device).unwrap();

    let mut result = vec![0u8; data.len()];
    CpuRuntime::copy_from_device(ptr, &mut result, &device).unwrap();

    assert_eq!(data, result);

    CpuRuntime::deallocate(ptr, data.len(), &device);
}

#[test]
fn test_copy_within_device() {
    let device = CpuDevice::new();
    let data: Vec<u8> = vec![1, 2, 3, 4, 5, 6, 7, 8];

    let src = CpuRuntime::allocate(data.len(), &device).unwrap();
    let dst = CpuRuntime::allocate(data.len(), &device).unwrap();

    CpuRuntime::copy_to_device(&data, src, &device).unwrap();
    CpuRuntime::copy_within_device(src, dst, data.len(), &device).unwrap();

    let mut result = vec![0u8; data.len()];
    CpuRuntime::copy_from_device(dst, &mut result, &device).unwrap();

    assert_eq!(data, result);

    CpuRuntime::deallocate(src, data.len(), &device);
    CpuRuntime::deallocate(dst, data.len(), &device);
}

#[test]
fn test_zero_allocation() {
    let device = CpuDevice::new();
    let ptr = CpuRuntime::allocate(0, &device).unwrap();
    assert_eq!(ptr, 0);
    CpuRuntime::deallocate(ptr, 0, &device); // Should not panic
}

#[test]
fn test_client_allocator() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let ptr = client.allocator().allocate(256).unwrap();
    assert_ne!(ptr, 0);
    client.allocator().deallocate(ptr, 256);
}

#[test]
fn test_raw_handle() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    let _handle: &() = CpuRuntime::raw_handle(&client);
    // For CPU, handle is just ()
}

// ===== TensorOps Integration Tests =====

#[test]
fn test_tensor_add() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_tensor_sub() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let c = client.sub(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [9.0, 18.0, 27.0, 36.0]);
}

#[test]
fn test_tensor_mul() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0, 5.0], &[4], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_tensor_div() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 5.0, 8.0], &[4], &device).unwrap();

    let c = client.div(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [5.0, 5.0, 6.0, 5.0]);
}

#[test]
fn test_tensor_neg() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], &device).unwrap();

    let b = client.neg(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_tensor_sqrt() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 4.0, 9.0, 16.0], &[4], &device).unwrap();

    let b = client.sqrt(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_exp() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device).unwrap();

    let b = client.exp(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert!((result[0] - 1.0).abs() < 1e-6); // e^0 = 1
    // e^1 should be in range (2.7, 2.72)
    assert!(result[1] > 2.7 && result[1] < 2.72); // e^1 = e ≈ 2.718
}

#[test]
fn test_tensor_matmul_2x2() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A = [[1, 2], [3, 4]]
    // B = [[5, 6], [7, 8]]
    // C = A @ B = [[19, 22], [43, 50]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

    let c = client.matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_tensor_matmul_3x2_2x4() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // A = [[1, 2], [3, 4], [5, 6]] (3x2)
    // B = [[1, 2, 3, 4], [5, 6, 7, 8]] (2x4)
    // C = A @ B (3x4)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        &[2, 4],
        &device,
    )
    .unwrap();

    let c = client.matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3, 4]);
    let result: Vec<f32> = c.to_vec();
    // Row 0: [1*1+2*5, 1*2+2*6, 1*3+2*7, 1*4+2*8] = [11, 14, 17, 20]
    // Row 1: [3*1+4*5, 3*2+4*6, 3*3+4*7, 3*4+4*8] = [23, 30, 37, 44]
    // Row 2: [5*1+6*5, 5*2+6*6, 5*3+6*7, 5*4+6*8] = [35, 46, 57, 68]
    assert_eq!(
        result,
        [
            11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
        ]
    );
}

#[test]
fn test_tensor_sum_last_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 3] -> sum over dim 1 -> shape [2]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();

    let b = client.sum(&a, &[1], false).unwrap();

    assert_eq!(b.shape(), &[2]);
    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [6.0, 15.0]); // [1+2+3, 4+5+6]
}

#[test]
fn test_tensor_mean_last_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 10.0, 20.0, 30.0], &[2, 3], &device)
            .unwrap();

    let b = client.mean(&a, &[1], false).unwrap();

    assert_eq!(b.shape(), &[2]);
    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [2.0, 20.0]); // [6/3, 60/3]
}

#[test]
fn test_tensor_max_last_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 2.0, 8.0, 4.0], &[2, 3], &device)
        .unwrap();

    let b = client.max(&a, &[1], false).unwrap();

    assert_eq!(b.shape(), &[2]);
    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [5.0, 8.0]);
}

#[test]
fn test_tensor_relu() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, -2.0], &[4], &device).unwrap();

    let b = client.relu(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);
}

#[test]
fn test_tensor_sigmoid() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();

    let b = client.sigmoid(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert!((result[0] - 0.5).abs() < 1e-6);
}

#[test]
fn test_tensor_silu() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device).unwrap();
    let b = client.silu(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    // SiLU(x) = x / (1 + exp(-x))
    assert!((result[2] - 0.0).abs() < 1e-5); // SiLU(0) = 0
    assert!((result[3] - 0.7310586).abs() < 1e-4); // SiLU(1) ≈ 0.731
    assert!((result[1] - (-0.2689414)).abs() < 1e-4); // SiLU(-1) ≈ -0.269
}

#[test]
fn test_tensor_gelu() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device).unwrap();
    let b = client.gelu(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    // GELU(0) = 0
    assert!((result[2] - 0.0).abs() < 1e-5); // GELU(0) = 0
    assert!((result[3] - 0.8413).abs() < 0.01); // GELU(1) ≈ 0.841
    assert!((result[4] - 1.9545).abs() < 0.01); // GELU(2) ≈ 1.955
}

#[test]
fn test_tensor_rms_norm() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: 2 rows, 4 features each
    let input = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();

    let out = client.rms_norm(&input, &weight, 1e-5).unwrap();
    let result: Vec<f32> = out.to_vec();

    // Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.739
    // Normalized: [0.365, 0.730, 1.095, 1.460]
    let rms1 = (30.0f32 / 4.0 + 1e-5).sqrt();
    assert!((result[0] - 1.0 / rms1).abs() < 1e-4);
    assert!((result[1] - 2.0 / rms1).abs() < 1e-4);
    assert!((result[2] - 3.0 / rms1).abs() < 1e-4);
    assert!((result[3] - 4.0 / rms1).abs() < 1e-4);

    // Row 2: [2, 4, 6, 8], values are 2x row 1, so RMS is also 2x
    let rms2 = (120.0f32 / 4.0 + 1e-5).sqrt();
    assert!((result[4] - 2.0 / rms2).abs() < 1e-4);
}

#[test]
fn test_tensor_layer_norm() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input: 2 rows, 4 features each
    let input = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
        &[2, 4],
        &device,
    )
    .unwrap();
    let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();
    let bias = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap();

    let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();
    let result: Vec<f32> = out.to_vec();

    // Row 1: [1, 2, 3, 4], mean = 2.5, var = 1.25, std = 1.118
    // Normalized: (x - mean) / std = [-1.342, -0.447, 0.447, 1.342]
    let mean1 = 2.5f32;
    let var1 = ((1.0 - mean1).powi(2)
        + (2.0 - mean1).powi(2)
        + (3.0 - mean1).powi(2)
        + (4.0 - mean1).powi(2))
        / 4.0;
    let std1 = (var1 + 1e-5).sqrt();
    assert!((result[0] - (1.0 - mean1) / std1).abs() < 1e-4);
    assert!((result[1] - (2.0 - mean1) / std1).abs() < 1e-4);
    assert!((result[2] - (3.0 - mean1) / std1).abs() < 1e-4);
    assert!((result[3] - (4.0 - mean1) / std1).abs() < 1e-4);

    // Verify normalized outputs sum to approximately 0 (zero-centered)
    let row1_sum: f32 = result[0..4].iter().sum();
    assert!(row1_sum.abs() < 1e-4);
}

#[test]
fn test_tensor_argmax() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D tensor: [[1, 5, 3], [4, 2, 6]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
        .unwrap();

    // argmax along dim=1 (find max index in each row)
    let out = client.argmax(&a, 1, false).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(result, [1, 2]); // Row 0: max at index 1 (5.0), Row 1: max at index 2 (6.0)

    // argmax along dim=0 (find max index in each column)
    let out = client.argmax(&a, 0, false).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(result, [1, 0, 1]); // Col 0: max at 1 (4.0), Col 1: max at 0 (5.0), Col 2: max at 1 (6.0)

    // Test keepdim=true
    let out = client.argmax(&a, 1, true).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(result, [1, 2]);
}

#[test]
fn test_tensor_argmin() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D tensor: [[1, 5, 3], [4, 2, 6]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
        .unwrap();

    // argmin along dim=1 (find min index in each row)
    let out = client.argmin(&a, 1, false).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[2]);
    assert_eq!(result, [0, 1]); // Row 0: min at index 0 (1.0), Row 1: min at index 1 (2.0)

    // argmin along dim=0 (find min index in each column)
    let out = client.argmin(&a, 0, false).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[3]);
    assert_eq!(result, [0, 1, 0]); // Col 0: min at 0 (1.0), Col 1: min at 1 (2.0), Col 2: min at 0 (3.0)

    // Test keepdim=true
    let out = client.argmin(&a, 1, true).unwrap();
    let result: Vec<i64> = out.to_vec();
    assert_eq!(out.shape(), &[2, 1]);
    assert_eq!(result, [0, 1]);
}

#[test]
fn test_tensor_softmax_last_dim() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();

    let b = client.softmax(&a, -1).unwrap();

    let result: Vec<f32> = b.to_vec();
    // Check that outputs sum to 1
    let sum: f32 = result.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // Check monotonicity: result[0] < result[1] < result[2]
    assert!(result[0] < result[1]);
    assert!(result[1] < result[2]);
}

#[test]
fn test_tensor_ops_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[5i32, 6, 7, 8], &[4], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    let result: Vec<i32> = c.to_vec();
    assert_eq!(result, [6, 8, 10, 12]);
}

#[test]
fn test_tensor_dtype_mismatch() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[2], &device).unwrap();

    let result = client.add(&a, &b);
    assert!(result.is_err());
}

// ===== New Unary Operations Tests =====

#[test]
fn test_tensor_tan() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Use 0.5 radians to avoid clippy::approx_constant warnings
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.5], &[2], &device).unwrap();
    let b = client.tan(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert!((result[0] - 0.0).abs() < 1e-6); // tan(0) = 0
    // tan(0.5) ≈ 0.5463
    assert!((result[1] - 0.5463).abs() < 1e-3);
}

#[test]
fn test_tensor_recip() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 4.0, 5.0], &[4], &device).unwrap();
    let b = client.recip(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 0.5, 0.25, 0.2]);
}

#[test]
fn test_tensor_square() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, -4.0], &[4], &device).unwrap();
    let b = client.square(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_tensor_floor() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.9, -1.5, -2.9], &[4], &device).unwrap();
    let b = client.floor(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 2.0, -2.0, -3.0]);
}

#[test]
fn test_tensor_ceil() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.1, -1.5, -2.1], &[4], &device).unwrap();
    let b = client.ceil(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [2.0, 3.0, -1.0, -2.0]);
}

#[test]
fn test_tensor_round() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.4f32, 1.5, 2.5, -1.5], &[4], &device).unwrap();
    let b = client.round(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    // Rust's round() rounds half away from zero: 2.5 -> 3.0, -1.5 -> -2.0
    assert_eq!(result, [1.0, 2.0, 3.0, -2.0]);
}

// ===== New Element-wise Binary Operations Tests =====

#[test]
fn test_tensor_pow() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 0.5], &[3], &device).unwrap();

    let c = client.pow(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [4.0, 9.0, 2.0]); // 2^2=4, 3^2=9, 4^0.5=2
}

#[test]
fn test_tensor_maximum() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 8.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 7.0], &[4], &device).unwrap();

    let c = client.maximum(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [2.0, 5.0, 6.0, 8.0]);
}

#[test]
fn test_tensor_minimum() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 8.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 4.0, 6.0, 7.0], &[4], &device).unwrap();

    let c = client.minimum(&a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 4.0, 3.0, 7.0]);
}

// ===== ScalarOps Tests =====

#[test]
fn test_scalar_add() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = ScalarOps::add_scalar(&client, &a, 10.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_scalar_mul() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = ScalarOps::mul_scalar(&client, &a, 2.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_scalar_pow() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = ScalarOps::pow_scalar(&client, &a, 2.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 4.0, 9.0, 16.0]);
}

#[test]
fn test_scalar_div() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let b = ScalarOps::div_scalar(&client, &a, 10.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
}

// ===== CompareOps Tests =====

#[test]
fn test_compare_eq() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 3.0, 5.0], &[4], &device).unwrap();

    let c = CompareOps::eq(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 0.0, 1.0, 0.0]); // 1=true, 0=false
}

#[test]
fn test_compare_lt() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device).unwrap();

    let c = CompareOps::lt(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 0.0, 0.0, 0.0]); // 1<2, 2<2?, 3<2?, 4<2?
}

#[test]
fn test_compare_gt() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device).unwrap();

    let c = CompareOps::gt(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [0.0, 0.0, 1.0, 1.0]); // 1>2?, 2>2?, 3>2, 4>2
}

#[test]
fn test_compare_le() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device).unwrap();

    let c = CompareOps::le(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 1.0, 0.0, 0.0]); // 1<=2, 2<=2, 3<=2?, 4<=2?
}

#[test]
fn test_compare_ge() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 2.0, 2.0, 2.0], &[4], &device).unwrap();

    let c = CompareOps::ge(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [0.0, 1.0, 1.0, 1.0]); // 1>=2?, 2>=2, 3>=2, 4>=2
}

#[test]
fn test_compare_ne() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 3.0, 3.0, 5.0], &[4], &device).unwrap();

    let c = CompareOps::ne(&client, &a, &b).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [0.0, 1.0, 0.0, 1.0]); // opposite of eq
}

#[test]
fn test_compare_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2i32, 2, 2, 2], &[4], &device).unwrap();

    let c = CompareOps::lt(&client, &a, &b).unwrap();

    let result: Vec<i32> = c.to_vec();
    assert_eq!(result, [1, 0, 0, 0]);
}

// ===== Broadcasting Tests =====

#[test]
fn test_broadcast_scalar_to_vector() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [4] + [1] -> [4]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[4]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_broadcast_vector_to_matrix_row() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 3] + [3] -> [2, 3] (broadcast along rows)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0], &[3], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_broadcast_vector_to_matrix_col() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 3] + [2, 1] -> [2, 3] (broadcast along columns)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 100.0], &[2, 1], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 12.0, 13.0, 104.0, 105.0, 106.0]);
}

#[test]
fn test_broadcast_both_directions() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [3, 1] + [1, 4] -> [3, 4]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device).unwrap();
    let b =
        Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 4], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3, 4]);
    let result: Vec<f32> = c.to_vec();
    // Row 0: 1 + [10, 20, 30, 40] = [11, 21, 31, 41]
    // Row 1: 2 + [10, 20, 30, 40] = [12, 22, 32, 42]
    // Row 2: 3 + [10, 20, 30, 40] = [13, 23, 33, 43]
    assert_eq!(
        result,
        [
            11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
        ]
    );
}

#[test]
fn test_broadcast_mul() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 3] * [1] -> [2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
}

#[test]
fn test_broadcast_sub() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 2] - [2] -> [2, 2]
    let a =
        Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

    let c = client.sub(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [9.0, 18.0, 29.0, 38.0]);
}

#[test]
fn test_broadcast_div() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [4] / [1] -> [4]
    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

    let c = client.div(&a, &b).unwrap();

    assert_eq!(c.shape(), &[4]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [5.0, 10.0, 15.0, 20.0]);
}

#[test]
fn test_broadcast_3d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 2, 3] + [3] -> [2, 2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        &[2, 2, 3],
        &device,
    )
    .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[100.0f32, 200.0, 300.0], &[3], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(
        result,
        [
            101.0, 202.0, 303.0, 104.0, 205.0, 306.0, 107.0, 208.0, 309.0, 110.0, 211.0, 312.0
        ]
    );
}

#[test]
fn test_broadcast_pow() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [3] ^ [1] -> [3]
    let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

    let c = client.pow(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [4.0, 9.0, 16.0]);
}

#[test]
fn test_broadcast_maximum() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // max([2, 3], [3]) -> [2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 2.0, 4.0, 0.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 3.0, 3.0], &[3], &device).unwrap();

    let c = client.maximum(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [3.0, 5.0, 3.0, 4.0, 3.0, 6.0]);
}

#[test]
fn test_broadcast_incompatible_shapes() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [3] + [4] -> Error (incompatible)
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.add(&a, &b);
    assert!(result.is_err());
}

#[test]
fn test_broadcast_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 2] + [2] -> [2, 2] (integer type)
    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[10i32, 20], &[2], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<i32> = c.to_vec();
    assert_eq!(result, [11, 22, 13, 24]);
}

// ========================================================================
// Comparison Broadcasting Tests
// ========================================================================

#[test]
fn test_broadcast_compare_scalar() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Compare [4] with scalar [1] -> [4]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.5f32], &[1], &device).unwrap();

    // a > 2.5: [false, false, true, true] -> [0, 0, 1, 1]
    let c = client.gt(&a, &b).unwrap();
    assert_eq!(c.shape(), &[4]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [0.0, 0.0, 1.0, 1.0]);

    // a <= 2.5: [true, true, false, false] -> [1, 1, 0, 0]
    let c = client.le(&a, &b).unwrap();
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn test_broadcast_compare_matrix_row() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [2, 3] compared with [3] -> [2, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device).unwrap();

    // a < b: row0: [1<2, 2<3, 3<4]=[T,T,T], row1: [4<2, 5<3, 6<4]=[F,F,F]
    let c = client.lt(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_broadcast_compare_eq() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // [3, 1] compared with [1, 3] -> [3, 3]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device).unwrap();

    // Result is identity matrix (1s on diagonal)
    let c = client.eq(&a, &b).unwrap();
    assert_eq!(c.shape(), &[3, 3]);
    let result: Vec<f32> = c.to_vec();
    // Row 0: [1==1, 1==2, 1==3] = [1, 0, 0]
    // Row 1: [2==1, 2==2, 2==3] = [0, 1, 0]
    // Row 2: [3==1, 3==2, 3==3] = [0, 0, 1]
    assert_eq!(result, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

// ========================================================================
// Half-Precision Tests (requires "f16" feature)
// ========================================================================

#[cfg(feature = "f16")]
#[test]
fn test_f16_tensor_add() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b_data: Vec<f16> = vec![0.5, 1.5, 2.5, 3.5]
        .into_iter()
        .map(f16::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[4]);
    let result: Vec<f16> = c.to_vec();
    let expected: Vec<f16> = vec![1.5, 3.5, 5.5, 7.5]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

#[cfg(feature = "f16")]
#[test]
fn test_f16_matmul() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b_data: Vec<f16> = vec![5.0, 6.0, 7.0, 8.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2, 2], &device).unwrap();

    let c = client.matmul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f16> = c.to_vec();
    let expected: Vec<f16> = vec![19.0, 22.0, 43.0, 50.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

#[cfg(feature = "f16")]
#[test]
fn test_bf16_tensor_mul() {
    use half::bf16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<bf16> = vec![2.0, 3.0, 4.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    let b_data: Vec<bf16> = vec![1.5, 2.5, 3.5]
        .into_iter()
        .map(bf16::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[3], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[3], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3]);
    let result: Vec<bf16> = c.to_vec();
    let expected: Vec<bf16> = vec![3.0, 7.5, 14.0]
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

#[cfg(feature = "f16")]
#[test]
fn test_f16_unary_ops() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data: Vec<f16> = vec![1.0, 4.0, 9.0, 16.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device).unwrap();

    let sqrt_result = client.sqrt(&a).unwrap();
    let sqrt_vec: Vec<f16> = sqrt_result.to_vec();
    let expected_sqrt: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    assert_eq!(sqrt_vec, expected_sqrt);

    let neg_result = client.neg(&a).unwrap();
    let neg_vec: Vec<f16> = neg_result.to_vec();
    let expected_neg: Vec<f16> = vec![-1.0, -4.0, -9.0, -16.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    assert_eq!(neg_vec, expected_neg);
}

#[cfg(feature = "f16")]
#[test]
fn test_f16_reduce() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();

    let sum = client.sum(&a, &[1], false).unwrap();
    assert_eq!(sum.shape(), &[2]);
    let sum_vec: Vec<f16> = sum.to_vec();
    let expected: Vec<f16> = vec![6.0, 15.0].into_iter().map(f16::from_f32).collect();
    assert_eq!(sum_vec, expected);
}

#[cfg(feature = "f16")]
#[test]
fn test_f16_broadcast() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<f16> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    let b_data: Vec<f16> = vec![10.0, 20.0].into_iter().map(f16::from_f32).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    let result: Vec<f16> = c.to_vec();
    let expected: Vec<f16> = vec![11.0, 22.0, 13.0, 24.0]
        .into_iter()
        .map(f16::from_f32)
        .collect();
    assert_eq!(result, expected);
}

// ===== FP8 Integration Tests =====

#[cfg(feature = "fp8")]
#[test]
fn test_fp8e4m3_tensor_creation() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();

    // Create FP8E4M3 tensor from slice
    let data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(FP8E4M3::from_f32)
        .collect();

    let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 2], &device).unwrap();

    assert_eq!(tensor.shape(), &[2, 2]);
    assert_eq!(tensor.dtype(), DType::FP8E4M3);
    assert_eq!(tensor.numel(), 4);

    // Verify data roundtrips
    let result: Vec<FP8E4M3> = tensor.to_vec();
    for (a, b) in data.iter().zip(result.iter()) {
        assert!((a.to_f32() - b.to_f32()).abs() < 0.1);
    }
}

#[cfg(feature = "fp8")]
#[test]
fn test_fp8e5m2_tensor_creation() {
    use numr::dtype::FP8E5M2;

    let device = CpuDevice::new();

    // Create FP8E5M2 tensor from slice
    let data: Vec<FP8E5M2> = vec![10.0, 20.0, 30.0, 40.0]
        .into_iter()
        .map(FP8E5M2::from_f32)
        .collect();

    let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[4], &device).unwrap();

    assert_eq!(tensor.shape(), &[4]);
    assert_eq!(tensor.dtype(), DType::FP8E5M2);

    // Verify data roundtrips (FP8E5M2 has less precision)
    let result: Vec<FP8E5M2> = tensor.to_vec();
    for (a, b) in data.iter().zip(result.iter()) {
        assert!((a.to_f32() - b.to_f32()).abs() < 5.0);
    }
}

#[cfg(feature = "fp8")]
#[test]
fn test_fp8e4m3_add() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(FP8E4M3::from_f32)
        .collect();
    let b_data: Vec<FP8E4M3> = vec![5.0, 6.0, 7.0, 8.0]
        .into_iter()
        .map(FP8E4M3::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[2, 2], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[2, 2], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.dtype(), DType::FP8E4M3);

    let result: Vec<FP8E4M3> = c.to_vec();
    let expected = [6.0, 8.0, 10.0, 12.0];
    for (val, exp) in result.iter().zip(expected.iter()) {
        // FP8E4M3 has ~20% tolerance
        assert!((val.to_f32() - exp).abs() < exp * 0.25 + 0.5);
    }
}

#[cfg(feature = "fp8")]
#[test]
fn test_fp8e4m3_mul() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a_data: Vec<FP8E4M3> = vec![1.0, 2.0, 3.0, 4.0]
        .into_iter()
        .map(FP8E4M3::from_f32)
        .collect();
    let b_data: Vec<FP8E4M3> = vec![2.0, 2.0, 2.0, 2.0]
        .into_iter()
        .map(FP8E4M3::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    let result: Vec<FP8E4M3> = c.to_vec();
    let expected = [2.0, 4.0, 6.0, 8.0];
    for (val, exp) in result.iter().zip(expected.iter()) {
        assert!((val.to_f32() - exp).abs() < exp * 0.25 + 0.5);
    }
}

#[cfg(feature = "fp8")]
#[test]
fn test_fp8e5m2_large_values() {
    use numr::dtype::FP8E5M2;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // FP8E5M2 has larger dynamic range (~[-57344, 57344])
    let a_data: Vec<FP8E5M2> = vec![100.0, 200.0, 500.0, 1000.0]
        .into_iter()
        .map(FP8E5M2::from_f32)
        .collect();
    let b_data: Vec<FP8E5M2> = vec![2.0, 2.0, 2.0, 2.0]
        .into_iter()
        .map(FP8E5M2::from_f32)
        .collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[4], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    let result: Vec<FP8E5M2> = c.to_vec();
    let expected = [200.0, 400.0, 1000.0, 2000.0];
    for (val, exp) in result.iter().zip(expected.iter()) {
        // FP8E5M2 has ~30% tolerance for larger values
        assert!((val.to_f32() - exp).abs() < exp * 0.35 + 10.0);
    }
}

#[cfg(feature = "fp8")]
#[test]
fn test_fp8_full_scalar_tensor() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();

    // Test Tensor::full_scalar with FP8E4M3
    let tensor = Tensor::<CpuRuntime>::full_scalar(&[2, 3], DType::FP8E4M3, 2.5, &device).unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.dtype(), DType::FP8E4M3);

    let result: Vec<FP8E4M3> = tensor.to_vec();
    for val in result {
        assert!((val.to_f32() - 2.5).abs() < 0.5);
    }
}

// ===== Cast Operation Tests =====

#[test]
fn test_cast_f32_to_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.5f32, 2.5, 3.5, 4.5], &[2, 2], &device).unwrap();
    let b = client.cast(&a, DType::F64).unwrap();

    assert_eq!(b.dtype(), DType::F64);
    assert_eq!(b.shape(), &[2, 2]);
    let result: Vec<f64> = b.to_vec();
    assert_eq!(result, [1.5, 2.5, 3.5, 4.5]);
}

#[test]
fn test_cast_f64_to_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.9f64, -2.1, 3.5, -4.9], &[4], &device).unwrap();
    let b = client.cast(&a, DType::I32).unwrap();

    assert_eq!(b.dtype(), DType::I32);
    let result: Vec<i32> = b.to_vec();
    // Truncation toward zero
    assert_eq!(result, [1, -2, 3, -4]);
}

#[test]
fn test_cast_i32_to_f32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, -2, 100, -50], &[4], &device).unwrap();
    let b = client.cast(&a, DType::F32).unwrap();

    assert_eq!(b.dtype(), DType::F32);
    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, -2.0, 100.0, -50.0]);
}

#[test]
fn test_cast_same_dtype_noop() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
    let b = client.cast(&a, DType::F32).unwrap();

    assert_eq!(b.dtype(), DType::F32);
    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0]);
}

#[test]
fn test_cast_f32_to_fp8e4m3() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 4.0, 8.0], &[4], &device).unwrap();
    let b = client.cast(&a, DType::FP8E4M3).unwrap();

    assert_eq!(b.dtype(), DType::FP8E4M3);
    let result: Vec<FP8E4M3> = b.to_vec();

    // FP8 has limited precision, so check with tolerance
    assert!((result[0].to_f32() - 1.0).abs() < 0.1);
    assert!((result[1].to_f32() - 2.0).abs() < 0.2);
    assert!((result[2].to_f32() - 4.0).abs() < 0.5);
    assert!((result[3].to_f32() - 8.0).abs() < 1.0);
}

#[test]
fn test_cast_fp8e4m3_to_f32() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Create FP8 tensor
    let fp8_data: Vec<FP8E4M3> = vec![
        FP8E4M3::from_f32(1.0),
        FP8E4M3::from_f32(2.0),
        FP8E4M3::from_f32(4.0),
        FP8E4M3::from_f32(8.0),
    ];
    let a = Tensor::<CpuRuntime>::from_slice(&fp8_data, &[4], &device).unwrap();
    let b = client.cast(&a, DType::F32).unwrap();

    assert_eq!(b.dtype(), DType::F32);
    let result: Vec<f32> = b.to_vec();

    // Check roundtrip - values should be close
    assert!((result[0] - 1.0).abs() < 0.1);
    assert!((result[1] - 2.0).abs() < 0.2);
    assert!((result[2] - 4.0).abs() < 0.5);
    assert!((result[3] - 8.0).abs() < 1.0);
}

#[test]
fn test_cast_f32_to_fp8e5m2() {
    use numr::dtype::FP8E5M2;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // E5M2 has larger range but less precision
    let a =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 100.0, 1000.0, 10000.0], &[4], &device).unwrap();
    let b = client.cast(&a, DType::FP8E5M2).unwrap();

    assert_eq!(b.dtype(), DType::FP8E5M2);
    let result: Vec<FP8E5M2> = b.to_vec();

    // E5M2 can represent larger values but with less precision
    assert!((result[0].to_f32() - 1.0).abs() < 0.5);
    assert!((result[1].to_f32() - 100.0).abs() < 50.0);
    assert!((result[2].to_f32() - 1000.0).abs() < 500.0);
    assert!((result[3].to_f32() - 10000.0).abs() < 5000.0);
}

#[test]
fn test_cast_fp8e4m3_to_fp8e5m2() {
    use numr::dtype::{FP8E4M3, FP8E5M2};

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Create FP8E4M3 tensor
    let fp8_data: Vec<FP8E4M3> = vec![
        FP8E4M3::from_f32(1.0),
        FP8E4M3::from_f32(2.0),
        FP8E4M3::from_f32(4.0),
    ];
    let a = Tensor::<CpuRuntime>::from_slice(&fp8_data, &[3], &device).unwrap();

    // Cast E4M3 -> E5M2
    let b = client.cast(&a, DType::FP8E5M2).unwrap();
    assert_eq!(b.dtype(), DType::FP8E5M2);

    let result: Vec<FP8E5M2> = b.to_vec();
    // Values should be preserved (approximately)
    assert!((result[0].to_f32() - 1.0).abs() < 0.5);
    assert!((result[1].to_f32() - 2.0).abs() < 1.0);
    assert!((result[2].to_f32() - 4.0).abs() < 2.0);
}

// ========================================================================
// New Operations Tests (sign, isnan, isinf, logical ops, where_cond)
// ========================================================================

#[test]
fn test_tensor_sign() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[-3.0f32, -0.5, 0.0, 0.5, 3.0], &[5], &device).unwrap();
    let b = client.sign(&a).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert_eq!(result, [-1.0, -1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn test_tensor_isnan() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, f32::NAN, 3.0, f32::NAN, 5.0], &[5], &device)
            .unwrap();
    let b = client.isnan(&a).unwrap();

    let result: Vec<u8> = b.to_vec();
    assert_eq!(result, [0, 1, 0, 1, 0]); // NaN at positions 1 and 3
}

#[test]
fn test_tensor_isinf() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, f32::INFINITY, 3.0, f32::NEG_INFINITY, 5.0],
        &[5],
        &device,
    )
    .unwrap();
    let b = client.isinf(&a).unwrap();

    let result: Vec<u8> = b.to_vec();
    assert_eq!(result, [0, 1, 0, 1, 0]); // Inf at positions 1 and 3
}

#[test]
fn test_tensor_logical_not() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0u8, 1, 0, 1, 1], &[5], &device).unwrap();
    let b = client.logical_not(&a).unwrap();

    let result: Vec<u8> = b.to_vec();
    assert_eq!(result, [1, 0, 1, 0, 0]); // Inverted
}

#[test]
fn test_tensor_logical_and() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 1, 1], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0u8, 1, 0, 1], &[4], &device).unwrap();
    let c = client.logical_and(&a, &b).unwrap();

    let result: Vec<u8> = c.to_vec();
    assert_eq!(result, [0, 0, 0, 1]); // AND truth table
}

#[test]
fn test_tensor_logical_or() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 1, 1], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0u8, 1, 0, 1], &[4], &device).unwrap();
    let c = client.logical_or(&a, &b).unwrap();

    let result: Vec<u8> = c.to_vec();
    assert_eq!(result, [0, 1, 1, 1]); // OR truth table
}

#[test]
fn test_tensor_logical_xor() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 1, 1], &[4], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[0u8, 1, 0, 1], &[4], &device).unwrap();
    let c = client.logical_xor(&a, &b).unwrap();

    let result: Vec<u8> = c.to_vec();
    assert_eq!(result, [0, 1, 1, 0]); // XOR truth table
}

#[test]
fn test_tensor_where_cond_same_shape() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // cond ? x : y
    let cond = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [10.0, 2.0, 30.0, 4.0]); // Select x where cond=1, y where cond=0
}

#[test]
fn test_tensor_where_cond_broadcast_cond() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Broadcast condition: [1] with x,y: [4]
    let cond = Tensor::<CpuRuntime>::from_slice(&[1u8], &[1], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [10.0, 20.0, 30.0, 40.0]); // All from x since cond=1
}

#[test]
fn test_tensor_where_cond_broadcast_xy() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Broadcast x,y: cond: [4], x: [1], y: [4]
    let cond = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 1, 0], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[100.0f32], &[1], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [100.0, 2.0, 100.0, 4.0]); // 100 where cond=1, y values where cond=0
}

#[test]
fn test_tensor_where_cond_2d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D where: cond: [2,2], x: [2,2], y: [2,2]
    let cond = Tensor::<CpuRuntime>::from_slice(&[1u8, 0, 0, 1], &[2, 2], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
    let y =
        Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    assert_eq!(result.shape(), &[2, 2]);
    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 20.0, 30.0, 4.0]);
}

// ========================================================================
// Generic Condition Dtype Tests (non-U8 conditions)
// ========================================================================

#[test]
fn test_where_cond_i32_condition() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // I32 condition: non-zero values are true, zero is false
    let cond = Tensor::<CpuRuntime>::from_slice(&[1i32, 0, -1, 42], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f32> = result.to_vec();
    // 1 (true) -> x[0], 0 (false) -> y[1], -1 (true) -> x[2], 42 (true) -> x[3]
    assert_eq!(data, [10.0, 2.0, 30.0, 40.0]);
}

#[test]
fn test_where_cond_i64_condition() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // I64 condition
    let cond = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 0, -999], &[4], &device).unwrap();
    let x =
        Tensor::<CpuRuntime>::from_slice(&[100.0f64, 200.0, 300.0, 400.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f64> = result.to_vec();
    assert_eq!(data, [1.0, 200.0, 3.0, 400.0]);
}

#[test]
fn test_where_cond_f32_condition() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // F32 condition: 0.0 is false, any non-zero (including negative) is true
    let cond =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, -0.5, 0.001], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f32> = result.to_vec();
    // 1.0 (true), 0.0 (false), -0.5 (true), 0.001 (true)
    assert_eq!(data, [10.0, 2.0, 30.0, 40.0]);
}

#[test]
fn test_where_cond_f64_condition() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // F64 condition
    let cond =
        Tensor::<CpuRuntime>::from_slice(&[0.0f64, 1e-10, 0.0, -1e100], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f32> = result.to_vec();
    // 0.0 (false), 1e-10 (true), 0.0 (false), -1e100 (true)
    assert_eq!(data, [1.0, 20.0, 3.0, 40.0]);
}

#[test]
fn test_where_cond_with_comparison_result() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // This is the primary use case: comparison ops return same dtype as input
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 7.0], &[4], &device).unwrap();
    let threshold =
        Tensor::<CpuRuntime>::from_slice(&[4.0f32, 4.0, 4.0, 4.0], &[4], &device).unwrap();

    // gt returns F32 (1.0 or 0.0), not U8
    let mask = client.gt(&a, &threshold).unwrap();
    assert_eq!(mask.dtype(), numr::dtype::DType::F32);

    let high_values = Tensor::<CpuRuntime>::from_slice(&[100.0f32; 4], &[4], &device).unwrap();
    let low_values = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 4], &[4], &device).unwrap();

    // where_cond should work directly with F32 mask
    let result = client.where_cond(&mask, &high_values, &low_values).unwrap();

    let data: Vec<f32> = result.to_vec();
    // a > threshold: [false, true, false, true]
    assert_eq!(data, [0.0, 100.0, 0.0, 100.0]);
}

#[test]
fn test_where_cond_with_lt_comparison() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let values =
        Tensor::<CpuRuntime>::from_slice(&[-2.0f64, -1.0, 0.0, 1.0, 2.0], &[5], &device).unwrap();
    let zero = Tensor::<CpuRuntime>::from_slice(&[0.0f64; 5], &[5], &device).unwrap();

    // Find negative values
    let is_negative = client.lt(&values, &zero).unwrap();
    assert_eq!(is_negative.dtype(), numr::dtype::DType::F64);

    // Select negated values where negative, original elsewhere (implements abs())
    let negated = client.mul_scalar(&values, -1.0).unwrap();
    let result = client.where_cond(&is_negative, &negated, &values).unwrap();

    let data: Vec<f64> = result.to_vec();
    // Manual abs: [-2, -1] become [2, 1], rest stay same
    assert_eq!(data, [2.0, 1.0, 0.0, 1.0, 2.0]);
}

#[test]
fn test_where_cond_generic_broadcast() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // I32 condition with broadcasting: cond [1], x [4], y [4]
    let cond = Tensor::<CpuRuntime>::from_slice(&[1i32], &[1], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    assert_eq!(result.shape(), &[4]);
    let data: Vec<f32> = result.to_vec();
    // All from x since cond = [1] broadcasts to all true
    assert_eq!(data, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_where_cond_generic_2d_broadcast() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // F32 condition [2, 1] broadcasts with x, y [2, 3]
    let cond = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2, 1], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &device,
    )
    .unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    assert_eq!(result.shape(), &[2, 3]);
    let data: Vec<f32> = result.to_vec();
    // Row 0: cond=1.0 (true) -> x values [1, 2, 3]
    // Row 1: cond=0.0 (false) -> y values [40, 50, 60]
    assert_eq!(data, [1.0, 2.0, 3.0, 40.0, 50.0, 60.0]);
}

#[test]
fn test_where_cond_u32_condition() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // U32 condition
    let cond = Tensor::<CpuRuntime>::from_slice(&[0u32, 1, 0, u32::MAX], &[4], &device).unwrap();
    let x = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();
    let y = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.where_cond(&cond, &x, &y).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, [1.0, 20.0, 3.0, 40.0]);
}

// ========================================================================
// Parametric Activation Tests (leaky_relu, elu)
// ========================================================================

#[test]
fn test_tensor_leaky_relu() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device).unwrap();

    // leaky_relu with negative_slope = 0.1
    let b = client.leaky_relu(&a, 0.1).unwrap();

    let result: Vec<f32> = b.to_vec();
    // For x > 0: output = x
    // For x <= 0: output = negative_slope * x
    assert!((result[0] - (-0.2)).abs() < 1e-6); // -2.0 * 0.1 = -0.2
    assert!((result[1] - (-0.1)).abs() < 1e-6); // -1.0 * 0.1 = -0.1
    assert!((result[2] - 0.0).abs() < 1e-6); // 0.0 * 0.1 = 0.0
    assert!((result[3] - 1.0).abs() < 1e-6); // 1.0 (positive, unchanged)
    assert!((result[4] - 2.0).abs() < 1e-6); // 2.0 (positive, unchanged)
}

#[test]
fn test_tensor_leaky_relu_default_slope() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[-10.0f32, 5.0], &[2], &device).unwrap();

    // leaky_relu with default negative_slope = 0.01
    let b = client.leaky_relu(&a, 0.01).unwrap();

    let result: Vec<f32> = b.to_vec();
    assert!((result[0] - (-0.1)).abs() < 1e-6); // -10.0 * 0.01 = -0.1
    assert!((result[1] - 5.0).abs() < 1e-6); // 5.0 (positive, unchanged)
}

#[test]
fn test_tensor_elu() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device).unwrap();

    // elu with alpha = 1.0
    let b = client.elu(&a, 1.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    // For x > 0: output = x
    // For x <= 0: output = alpha * (exp(x) - 1)
    let expected_neg2 = 1.0 * ((-2.0f32).exp() - 1.0); // ≈ -0.8647
    let expected_neg1 = 1.0 * ((-1.0f32).exp() - 1.0); // ≈ -0.6321

    assert!((result[0] - expected_neg2).abs() < 1e-5);
    assert!((result[1] - expected_neg1).abs() < 1e-5);
    assert!((result[2] - 0.0).abs() < 1e-6); // 0 is exactly 0 in ELU
    assert!((result[3] - 1.0).abs() < 1e-6); // 1.0 (positive, unchanged)
    assert!((result[4] - 2.0).abs() < 1e-6); // 2.0 (positive, unchanged)
}

#[test]
fn test_tensor_elu_custom_alpha() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 1.0], &[2], &device).unwrap();

    // elu with alpha = 2.0
    let b = client.elu(&a, 2.0).unwrap();

    let result: Vec<f32> = b.to_vec();
    let expected_neg1 = 2.0 * ((-1.0f32).exp() - 1.0); // ≈ -1.2642

    assert!((result[0] - expected_neg1).abs() < 1e-5);
    assert!((result[1] - 1.0).abs() < 1e-6); // 1.0 (positive, unchanged)
}

#[test]
fn test_tensor_leaky_relu_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[-2.0f64, 0.0, 2.0], &[3], &device).unwrap();

    let b = client.leaky_relu(&a, 0.2).unwrap();

    let result: Vec<f64> = b.to_vec();
    assert!((result[0] - (-0.4)).abs() < 1e-10);
    assert!((result[1] - 0.0).abs() < 1e-10);
    assert!((result[2] - 2.0).abs() < 1e-10);
}

#[test]
fn test_tensor_elu_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[-1.0f64, 0.0, 1.0], &[3], &device).unwrap();

    let b = client.elu(&a, 1.0).unwrap();

    let result: Vec<f64> = b.to_vec();
    let expected_neg1 = 1.0 * ((-1.0f64).exp() - 1.0);

    assert!((result[0] - expected_neg1).abs() < 1e-10);
    assert!((result[1] - 0.0).abs() < 1e-10);
    assert!((result[2] - 1.0).abs() < 1e-10);
}

// ===== Indexing Operation Tests =====

#[test]
fn test_tensor_index_select() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D tensor: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    let a = Tensor::<CpuRuntime>::from_slice(
        &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        &[3, 3],
        &device,
    )
    .unwrap();

    // Select rows 0 and 2
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 2], &[2], &device).unwrap();
    let out = client.index_select(&a, 0, &indices).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0, 7.0, 8.0, 9.0]);

    // Select columns 1 and 0
    let indices = Tensor::<CpuRuntime>::from_slice(&[1i64, 0], &[2], &device).unwrap();
    let out = client.index_select(&a, 1, &indices).unwrap();

    assert_eq!(out.shape(), &[3, 2]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [2.0, 1.0, 5.0, 4.0, 8.0, 7.0]);
}

#[test]
fn test_tensor_index_select_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 1D tensor
    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0], &[5], &device)
        .unwrap();

    // Select indices 4, 2, 0
    let indices = Tensor::<CpuRuntime>::from_slice(&[4i64, 2, 0], &[3], &device).unwrap();
    let out = client.index_select(&a, 0, &indices).unwrap();

    assert_eq!(out.shape(), &[3]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [50.0, 30.0, 10.0]);
}

#[test]
fn test_tensor_gather() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 2D tensor: [[1, 2], [3, 4], [5, 6]]
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
        .unwrap();

    // Gather along dim 0 using indices [[0, 1], [2, 0]]
    let indices = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2, 0], &[2, 2], &device).unwrap();
    let out = client.gather(&a, 0, &indices).unwrap();

    assert_eq!(out.shape(), &[2, 2]);
    let result: Vec<f32> = out.to_vec();
    // For position [0,0]: index[0,0]=0 -> a[0,0] = 1
    // For position [0,1]: index[0,1]=1 -> a[1,1] = 4
    // For position [1,0]: index[1,0]=2 -> a[2,0] = 5
    // For position [1,1]: index[1,1]=0 -> a[0,1] = 2
    assert_eq!(result, [1.0, 4.0, 5.0, 2.0]);
}

#[test]
fn test_tensor_gather_1d() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // 1D tensor
    let a = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();

    // Gather specific indices
    let indices = Tensor::<CpuRuntime>::from_slice(&[3i64, 0, 2, 1, 3], &[5], &device).unwrap();
    let out = client.gather(&a, 0, &indices).unwrap();

    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [40.0, 10.0, 30.0, 20.0, 40.0]);
}

#[test]
fn test_tensor_scatter() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Initialize destination tensor with zeros: 3x3
    let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 9], &[3, 3], &device).unwrap();

    // Source values to scatter: 2x3
    let src =
        Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
            .unwrap();

    // Indices for dim 0: [[0, 2, 1], [2, 1, 0]]
    let indices =
        Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 1, 2, 1, 0], &[2, 3], &device).unwrap();
    let out = client.scatter(&a, 0, &indices, &src).unwrap();

    assert_eq!(out.shape(), &[3, 3]);
    let result: Vec<f32> = out.to_vec();
    // Position [0,0] <- src[0,0]=1 (index 0)
    // Position [0,1] <- src[1,1]=5 (index 1)
    // Position [0,2] <- src[1,2]=6 (index 0)
    // Position [1,0] <- src[0,2]=3 (index 1) -- wait, need to check
    // Let me trace more carefully:
    // src[0,0]=1 -> indices[0,0]=0 -> a[0,0]
    // src[0,1]=2 -> indices[0,1]=2 -> a[2,1]
    // src[0,2]=3 -> indices[0,2]=1 -> a[1,2]
    // src[1,0]=4 -> indices[1,0]=2 -> a[2,0]
    // src[1,1]=5 -> indices[1,1]=1 -> a[1,1]
    // src[1,2]=6 -> indices[1,2]=0 -> a[0,2]
    // Result: [[1,0,6], [0,5,3], [4,2,0]]
    assert_eq!(result, [1.0, 0.0, 6.0, 0.0, 5.0, 3.0, 4.0, 2.0, 0.0]);
}

#[test]
fn test_tensor_masked_fill() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input tensor
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();

    // Mask: true where value > 3
    let mask = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 0, 1, 1, 1], &[2, 3], &device).unwrap();

    let out = client.masked_fill(&a, &mask, -1.0).unwrap();

    assert_eq!(out.shape(), &[2, 3]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0, -1.0, -1.0, -1.0]);
}

#[test]
fn test_tensor_masked_fill_all_false() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 0, 0], &[4], &device).unwrap();

    let out = client.masked_fill(&a, &mask, 999.0).unwrap();
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_masked_fill_all_true() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 1, 1, 1], &[4], &device).unwrap();

    let out = client.masked_fill(&a, &mask, 0.0).unwrap();
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_tensor_masked_select() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Input tensor
    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();

    // Select values where mask is true (values > 2)
    let mask = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 1, 1, 1, 1], &[2, 3], &device).unwrap();

    let out = client.masked_select(&a, &mask).unwrap();

    // Result should be 1D with only selected values
    assert_eq!(out.shape(), &[4]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_tensor_masked_select_none() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::from_slice(&[0u8, 0, 0], &[3], &device).unwrap();

    let out = client.masked_select(&a, &mask).unwrap();

    // Empty result - check shape and numel, don't call to_vec() on empty tensor
    assert_eq!(out.shape(), &[0]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn test_tensor_masked_select_all() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a =
        Tensor::<CpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[2, 2], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::from_slice(&[1u8, 1, 1, 1], &[2, 2], &device).unwrap();

    let out = client.masked_select(&a, &mask).unwrap();

    assert_eq!(out.shape(), &[4]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_tensor_index_select_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Test with i32 dtype
    let a = Tensor::<CpuRuntime>::from_slice(&[10i32, 20, 30, 40, 50], &[5], &device).unwrap();
    let indices = Tensor::<CpuRuntime>::from_slice(&[4i64, 0, 2], &[3], &device).unwrap();

    let out = client.index_select(&a, 0, &indices).unwrap();
    assert_eq!(out.dtype(), DType::I32);
    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [50, 10, 30]);
}

#[test]
fn test_tensor_gather_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[100i32, 200, 300, 400], &[4], &device).unwrap();
    let indices = Tensor::<CpuRuntime>::from_slice(&[2i64, 0, 3, 1], &[4], &device).unwrap();

    let out = client.gather(&a, 0, &indices).unwrap();
    assert_eq!(out.dtype(), DType::I32);
    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [300, 100, 400, 200]);
}

// ===== Utility Operations Tests (arange, linspace, eye) =====

#[test]
fn test_arange_f32_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(0, 5, 1) -> [0, 1, 2, 3, 4]
    let out = client.arange(0.0, 5.0, 1.0, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_arange_f32_step() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(0, 10, 2) -> [0, 2, 4, 6, 8]
    let out = client.arange(0.0, 10.0, 2.0, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [0.0, 2.0, 4.0, 6.0, 8.0]);
}

#[test]
fn test_arange_f32_negative_step() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(5, 0, -1) -> [5, 4, 3, 2, 1]
    let out = client.arange(5.0, 0.0, -1.0, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [5.0, 4.0, 3.0, 2.0, 1.0]);
}

#[test]
fn test_arange_f32_fractional() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(0, 2.5, 0.5) -> [0.0, 0.5, 1.0, 1.5, 2.0]
    let out = client.arange(0.0, 2.5, 0.5, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    for (i, &v) in result.iter().enumerate() {
        let expected = i as f32 * 0.5;
        assert!(
            (v - expected).abs() < 1e-6,
            "Expected {}, got {}",
            expected,
            v
        );
    }
}

#[test]
fn test_arange_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(0, 5, 1) -> [0, 1, 2, 3, 4]
    let out = client.arange(0.0, 5.0, 1.0, DType::I32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [0, 1, 2, 3, 4]);
}

#[test]
fn test_arange_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // arange(5, 5, 1) -> [] (empty)
    let out = client.arange(5.0, 5.0, 1.0, DType::F32).unwrap();
    assert_eq!(out.shape(), &[0]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn test_arange_zero_step_error() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // step=0 should error
    let result = client.arange(0.0, 5.0, 0.0, DType::F32);
    assert!(result.is_err());
}

#[test]
fn test_linspace_f32_basic() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // linspace(0, 4, 5) -> [0, 1, 2, 3, 4]
    let out = client.linspace(0.0, 4.0, 5, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [0.0, 1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_linspace_f32_fractional() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // linspace(0, 1, 5) -> [0.0, 0.25, 0.5, 0.75, 1.0]
    let out = client.linspace(0.0, 1.0, 5, DType::F32).unwrap();
    assert_eq!(out.shape(), &[5]);
    let result: Vec<f32> = out.to_vec();
    let expected = [0.0, 0.25, 0.5, 0.75, 1.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "linspace[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
}

#[test]
fn test_linspace_single_step() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // linspace(5, 5, 1) -> [5]
    let out = client.linspace(5.0, 5.0, 1, DType::F32).unwrap();
    assert_eq!(out.shape(), &[1]);
    let result: Vec<f32> = out.to_vec();
    assert_eq!(result, [5.0]);
}

#[test]
fn test_linspace_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // linspace(0, 1, 0) -> []
    let out = client.linspace(0.0, 1.0, 0, DType::F32).unwrap();
    assert_eq!(out.shape(), &[0]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn test_eye_square() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // eye(3) -> 3x3 identity
    let out = client.eye(3, None, DType::F32).unwrap();
    assert_eq!(out.shape(), &[3, 3]);
    let result: Vec<f32> = out.to_vec();
    #[rustfmt::skip]
    let expected = [
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_eye_rectangular_wide() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // eye(2, 4) -> 2x4 identity-like
    let out = client.eye(2, Some(4), DType::F32).unwrap();
    assert_eq!(out.shape(), &[2, 4]);
    let result: Vec<f32> = out.to_vec();
    #[rustfmt::skip]
    let expected = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_eye_rectangular_tall() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // eye(4, 2) -> 4x2 identity-like
    let out = client.eye(4, Some(2), DType::F32).unwrap();
    assert_eq!(out.shape(), &[4, 2]);
    let result: Vec<f32> = out.to_vec();
    #[rustfmt::skip]
    let expected = [
        1.0, 0.0,
        0.0, 1.0,
        0.0, 0.0,
        0.0, 0.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_eye_i32() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let out = client.eye(2, None, DType::I32).unwrap();
    assert_eq!(out.shape(), &[2, 2]);
    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [1, 0, 0, 1]);
}

#[test]
fn test_eye_empty() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // eye(0) -> empty 0x0
    let out = client.eye(0, None, DType::F32).unwrap();
    assert_eq!(out.shape(), &[0, 0]);
    assert_eq!(out.numel(), 0);
}
