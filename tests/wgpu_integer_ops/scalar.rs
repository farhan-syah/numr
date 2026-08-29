// Scalar Operations (I32 / U32)
//
// Regression coverage for the WebGPU scalar-encoding bug: ScalarParams used
// to write the scalar as a bit-reinterpreted `f32` even for integer shaders,
// so `add_scalar(i32_tensor, 3.0)` added 1078530011 (the f32 `3.0` bit
// pattern read as i32) instead of 3. Every case here compares against the
// CPU backend, which is the reference for correct scalar conversion.

use numr::ops::ScalarOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::runtime::wgpu::{WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

#[test]
fn test_i32_add_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1i32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    // 3.0's f32 bit pattern is 1078530011 - if the scalar is bit-reinterpreted
    // instead of converted, this test fails loudly.
    let result = client.add_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.add_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![4, 5, 6, 7]);
}

#[test]
fn test_i32_sub_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [10i32, 20, 30, 40];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.sub_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.sub_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![7, 17, 27, 37]);
}

#[test]
fn test_i32_rsub_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1i32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.rsub_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.rsub_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![2, 1, 0, -1]);
}

#[test]
fn test_i32_mul_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1i32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.mul_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.mul_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![3, 6, 9, 12]);
}

#[test]
fn test_i32_div_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [9i32, 20, 33, 44];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.div_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.div_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![3, 6, 11, 14]);
}

#[test]
fn test_i32_add_scalar_negative() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1i32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    // Negative scalar pins the i32 conversion rule, not just its sign bit.
    let result = client.add_scalar(&a, -7.0).unwrap();
    let expected = cpu_client.add_scalar(&a_cpu, -7.0).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![-6, -5, -4, -3]);
}

#[test]
fn test_i32_add_scalar_fractional() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1i32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    // Fractional scalar pins the truncating `as i32` conversion rule shared
    // with CPU: 2.7 truncates to 2 before it is applied.
    let result = client.add_scalar(&a, 2.7).unwrap();
    let expected = cpu_client.add_scalar(&a_cpu, 2.7).unwrap();

    let data: Vec<i32> = result.to_vec();
    let expected: Vec<i32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![3, 4, 5, 6]);
}

#[test]
fn test_u32_add_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1u32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    // 3.0's f32 bit pattern is 1078530011 - if the scalar is bit-reinterpreted
    // instead of converted, this test fails loudly.
    let result = client.add_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.add_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<u32> = result.to_vec();
    let expected: Vec<u32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![4, 5, 6, 7]);
}

#[test]
fn test_u32_sub_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [10u32, 20, 30, 40];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.sub_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.sub_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<u32> = result.to_vec();
    let expected: Vec<u32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![7, 17, 27, 37]);
}

#[test]
fn test_u32_rsub_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1u32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    // rsub_scalar(a, 10) = 10 - a, stays within u32 range for this input.
    let result = client.rsub_scalar(&a, 10.0).unwrap();
    let expected = cpu_client.rsub_scalar(&a_cpu, 10.0).unwrap();

    let data: Vec<u32> = result.to_vec();
    let expected: Vec<u32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![9, 8, 7, 6]);
}

#[test]
fn test_u32_mul_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [1u32, 2, 3, 4];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.mul_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.mul_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<u32> = result.to_vec();
    let expected: Vec<u32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![3, 6, 9, 12]);
}

#[test]
fn test_u32_div_scalar() {
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("WebGPU not available, skipping");
        return;
    }

    let device = WgpuDevice::new(0);
    let client = WgpuRuntime::default_client(&device);
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let data = [9u32, 20, 33, 44];
    let a = Tensor::<WgpuRuntime>::from_slice(&data, &[4], &device).unwrap();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&data, &[4], &cpu_device).unwrap();

    let result = client.div_scalar(&a, 3.0).unwrap();
    let expected = cpu_client.div_scalar(&a_cpu, 3.0).unwrap();

    let data: Vec<u32> = result.to_vec();
    let expected: Vec<u32> = expected.to_vec();
    assert_eq!(data, expected);
    assert_eq!(data, vec![3, 6, 11, 14]);
}
