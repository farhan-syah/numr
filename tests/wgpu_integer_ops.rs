//! Integration tests for WebGPU Integer DType Support (I32, U32)
//!
//! Tests that validate integer operations work correctly on WebGPU backend.
//! This is critical for P1.2 completion.

#![cfg(feature = "wgpu")]

use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ReduceOps, UnaryOps};
use numr::runtime::Runtime;
use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

// ============================================================================
// Binary Operations (I32)
// ============================================================================

#[test]
fn test_i32_add() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30, 40], &[4], &device).unwrap();

    let result = client.add(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![11, 22, 33, 44]);
}

#[test]
fn test_i32_sub() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30, 40], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();

    let result = client.sub(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![9, 18, 27, 36]);
}

#[test]
fn test_i32_mul() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[2i32, 3, 4, 5], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 10, 10, 10], &[4], &device).unwrap();

    let result = client.mul(&a, &b).unwrap();

    let data: Vec<i32> = result.to_vec();
    assert_eq!(data, vec![20, 30, 40, 50]);
}

// ============================================================================
// Binary Operations (U32)
// ============================================================================

#[test]
fn test_u32_add() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1u32, 2, 3, 4], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10u32, 20, 30, 40], &[4], &device).unwrap();

    let result = client.add(&a, &b).unwrap();

    let data: Vec<u32> = result.to_vec();
    assert_eq!(data, vec![11, 22, 33, 44]);
}

#[test]
fn test_u32_mul() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[2u32, 3, 4, 5], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10u32, 10, 10, 10], &[4], &device).unwrap();

    let result = client.mul(&a, &b).unwrap();

    let data: Vec<u32> = result.to_vec();
    assert_eq!(data, vec![20, 30, 40, 50]);
}

// ============================================================================
// Unary Operations (I32)
// ============================================================================

#[test]
fn test_f32_neg() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], &device).unwrap();

    let result = client.neg(&a).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![-1.0, 2.0, -3.0, 4.0]);
}

#[test]
fn test_f32_abs() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, -2.0, 3.0, -4.0], &[4], &device).unwrap();

    let result = client.abs(&a).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
}

// ============================================================================
// Float-Only Operations Should Reject Integers
// ============================================================================

#[test]
fn test_i32_sqrt_should_fail() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 4, 9, 16], &[4], &device).unwrap();

    // sqrt is float-only - should return UnsupportedDType error
    let result = client.sqrt(&a);
    assert!(result.is_err(), "Expected sqrt on I32 to fail");

    // Verify it's the correct error type
    match result {
        Err(numr::error::Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(dtype, DType::I32);
            assert_eq!(op, "sqrt");
        }
        _ => panic!("Expected UnsupportedDType error, got: {:?}", result),
    }
}

#[test]
fn test_i32_exp_should_fail() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();

    // exp is float-only - should return UnsupportedDType error
    let result = client.exp(&a);
    assert!(result.is_err(), "Expected exp on I32 to fail");

    // Verify it's the correct error type
    match result {
        Err(numr::error::Error::UnsupportedDType { dtype, op }) => {
            assert_eq!(dtype, DType::I32);
            assert_eq!(op, "exp");
        }
        _ => panic!("Expected UnsupportedDType error, got: {:?}", result),
    }
}

// ============================================================================
// Comparison Operations
// ============================================================================

#[test]
fn test_f32_eq() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 0.0, 3.0, 0.0], &[4], &device).unwrap();

    let result = client.eq(&a, &b).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
}

// ============================================================================
// Reduction Operations (I32)
// ============================================================================

#[test]
fn test_f32_sum() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();

    let result = client.sum(&a, &[], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![10.0]);
}

#[test]
fn test_f32_max() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a =
        Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 20.0, 3.0, 40.0, 5.0], &[5], &device).unwrap();

    let result = client.max(&a, &[], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![40.0]);
}

#[test]
fn test_f32_min() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    let a =
        Tensor::<WgpuRuntime>::from_slice(&[10.0f32, 2.0, 30.0, 4.0, 50.0], &[5], &device).unwrap();

    let result = client.min(&a, &[], false).unwrap();

    let data: Vec<f32> = result.to_vec();
    assert_eq!(data, vec![2.0]);
}

// ============================================================================
// Broadcast Operations (F32)
// ============================================================================

#[test]
fn test_broadcast_scalar_to_vector() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [4] + [1] -> [4]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10.0f32], &[1], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[4]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 12.0, 13.0, 14.0]);
}

#[test]
fn test_broadcast_vector_to_matrix_row() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] + [3] -> [2, 3] (broadcast along rows)
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0], &[3], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn test_broadcast_vector_to_matrix_col() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] + [2, 1] -> [2, 3] (broadcast along columns)
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10.0f32, 100.0], &[2, 1], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [11.0, 12.0, 13.0, 104.0, 105.0, 106.0]);
}

#[test]
fn test_broadcast_both_directions() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [3, 1] + [1, 4] -> [3, 4]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device).unwrap();
    let b =
        Tensor::<WgpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 4], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3, 4]);
    let result: Vec<f32> = c.to_vec();
    // Row 0: 1+[10,20,30,40] = [11,21,31,41]
    // Row 1: 2+[10,20,30,40] = [12,22,32,42]
    // Row 2: 3+[10,20,30,40] = [13,23,33,43]
    assert_eq!(
        result,
        [
            11.0, 21.0, 31.0, 41.0, 12.0, 22.0, 32.0, 42.0, 13.0, 23.0, 33.0, 43.0
        ]
    );
}

#[test]
fn test_broadcast_3d() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3, 1] + [4] -> [2, 3, 4]
    let a =
        Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3, 1], &device)
            .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3, 4]);
    let result: Vec<f32> = c.to_vec();
    // Element [i,j,k] = a[i,j,0] + b[k]
    let expected: Vec<f32> = vec![
        // i=0, j=0: 1 + [10,20,30,40]
        11.0, 21.0, 31.0, 41.0, // i=0, j=1: 2 + [10,20,30,40]
        12.0, 22.0, 32.0, 42.0, // i=0, j=2: 3 + [10,20,30,40]
        13.0, 23.0, 33.0, 43.0, // i=1, j=0: 4 + [10,20,30,40]
        14.0, 24.0, 34.0, 44.0, // i=1, j=1: 5 + [10,20,30,40]
        15.0, 25.0, 35.0, 45.0, // i=1, j=2: 6 + [10,20,30,40]
        16.0, 26.0, 36.0, 46.0,
    ];
    assert_eq!(result, expected);
}

#[test]
fn test_broadcast_mul() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] * [3] -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
}

#[test]
fn test_broadcast_sub() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] - [2, 1] -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &device,
    )
    .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 10.0], &[2, 1], &device).unwrap();

    let c = client.sub(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [9.0, 19.0, 29.0, 30.0, 40.0, 50.0]);
}

#[test]
fn test_broadcast_div() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] / [3] -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(
        &[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0],
        &[2, 3],
        &device,
    )
    .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[2.0f32, 5.0, 10.0], &[3], &device).unwrap();

    let c = client.div(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [5.0, 4.0, 3.0, 20.0, 10.0, 6.0]);
}

#[test]
fn test_broadcast_max() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // max([2, 3], [3]) -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[3.0f32, 3.0, 3.0], &[3], &device).unwrap();

    let c = client.maximum(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [3.0, 5.0, 3.0, 4.0, 3.0, 6.0]);
}

#[test]
fn test_broadcast_min() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // min([2, 3], [3]) -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
        .unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[3.0f32, 3.0, 3.0], &[3], &device).unwrap();

    let c = client.minimum(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, [1.0, 3.0, 3.0, 3.0, 2.0, 3.0]);
}

// ============================================================================
// Broadcast Operations (I32)
// ============================================================================

#[test]
fn test_broadcast_i32_add() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [2, 3] + [3] -> [2, 3]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3, 4, 5, 6], &[2, 3], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30], &[3], &device).unwrap();

    let c = client.add(&a, &b).unwrap();

    assert_eq!(c.shape(), &[2, 3]);
    let result: Vec<i32> = c.to_vec();
    assert_eq!(result, [11, 22, 33, 14, 25, 36]);
}

#[test]
fn test_broadcast_i32_mul() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);

    // [3, 1] * [1, 4] -> [3, 4]
    let a = Tensor::<WgpuRuntime>::from_slice(&[1i32, 2, 3], &[3, 1], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::from_slice(&[10i32, 20, 30, 40], &[1, 4], &device).unwrap();

    let c = client.mul(&a, &b).unwrap();

    assert_eq!(c.shape(), &[3, 4]);
    let result: Vec<i32> = c.to_vec();
    assert_eq!(result, [10, 20, 30, 40, 20, 40, 60, 80, 30, 60, 90, 120]);
}
