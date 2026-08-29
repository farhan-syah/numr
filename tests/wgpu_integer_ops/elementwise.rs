// Binary, unary, comparison, and reduction operations (I32/U32/F32).

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
