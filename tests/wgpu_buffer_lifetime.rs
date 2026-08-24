//! Tests for WebGPU buffer lifetime / device identity issues.
//!
//! Root cause: WgpuClient::new() used to create a separate wgpu::Device,
//! while Tensor::from_slice → Runtime::allocate used a cached client with
//! a different device. Buffers from Device A can't be used in bind groups
//! on Device B — wgpu-core panics with "Buffer[Id(...)] does not exist".
//!
//! These tests use WgpuClient::new() directly (like solvr does) to ensure
//! the cached-device fix works. Using WgpuRuntime::default_client() would
//! NOT catch this bug since it already goes through the cache.

#![cfg(feature = "wgpu")]

use numr::ops::{BinaryOps, ScalarOps};
use numr::runtime::wgpu::{WgpuClient, WgpuDevice, WgpuRuntime};
use numr::tensor::Tensor;

/// Create client via WgpuClient::new() — the path that triggered the bug.
fn setup() -> Option<(WgpuDevice, WgpuClient)> {
    if !numr::runtime::wgpu::is_wgpu_available() {
        return None;
    }
    let device = WgpuDevice::new(0);
    let client = WgpuClient::new(device.clone()).ok()?;
    Some((device, client))
}

/// Baseline: two from_slice + one binary op via WgpuClient::new().
/// Before the fix, even this would panic because from_slice allocates
/// on the cached device while client.sub() uses a different device.
#[test]
fn test_basic_binary_op_separate_client() {
    let Some((device, client)) = setup() else {
        return;
    };

    let a = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
    let b = Tensor::<WgpuRuntime>::try_from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).unwrap();
    let c = client.sub(&b, &a).unwrap();

    let result: Vec<f32> = c.to_vec();
    assert_eq!(result, vec![3.0, 3.0, 3.0]);
}

/// Reproduces the exact solvr::trapezoid crash pattern:
/// narrow → contiguous → sub (first binary op after creating views)
#[test]
fn test_narrow_contiguous_binary_op_separate_client() {
    let Some((device, client)) = setup() else {
        return;
    };

    let n = 101;
    let data: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    let x = Tensor::<WgpuRuntime>::try_from_slice(&data, &[n], &device).unwrap();

    let x_left = x.narrow(0, 0, n - 1).unwrap().contiguous().unwrap();
    let x_right = x.narrow(0, 1, n - 1).unwrap().contiguous().unwrap();
    let dx = client.sub(&x_right, &x_left).unwrap();

    let result: Vec<f32> = dx.to_vec();
    assert_eq!(result.len(), n - 1);
    let step = 1.0 / (n - 1) as f32;
    for &val in &result {
        assert!((val - step).abs() < 1e-5, "expected ~{step}, got {val}");
    }
}

/// Full solvr::trapezoid chain via WgpuClient::new():
/// narrow → contiguous → sub → add → mul_scalar → mul → sum
#[test]
fn test_trapezoid_chain_separate_client() {
    let Some((device, client)) = setup() else {
        return;
    };

    let n = 101;
    let x_data: Vec<f32> = (0..n).map(|i| i as f32 / (n - 1) as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&xi| xi * xi).collect();

    let x = Tensor::<WgpuRuntime>::try_from_slice(&x_data, &[n], &device).unwrap();
    let y = Tensor::<WgpuRuntime>::try_from_slice(&y_data, &[n], &device).unwrap();

    let x_left = x.narrow(0, 0, n - 1).unwrap().contiguous().unwrap();
    let x_right = x.narrow(0, 1, n - 1).unwrap().contiguous().unwrap();
    let dx = client.sub(&x_right, &x_left).unwrap();

    let y_left = y.narrow(0, 0, n - 1).unwrap().contiguous().unwrap();
    let y_right = y.narrow(0, 1, n - 1).unwrap().contiguous().unwrap();
    let y_sum = client.add(&y_left, &y_right).unwrap();

    let scaled = client.mul_scalar(&y_sum, 0.5).unwrap();
    let areas = client.mul(&dx, &scaled).unwrap();

    let result: Vec<f32> = areas.to_vec();
    assert_eq!(result.len(), n - 1);

    let total: f32 = result.iter().sum();
    assert!(
        (total - 1.0 / 3.0).abs() < 1e-3,
        "trapezoid integral of x^2 should be ~0.333, got {total}"
    );
}

/// Test that dropping the source tensor before using narrow+contiguous
/// results works correctly via WgpuClient::new().
#[test]
fn test_narrow_contiguous_source_dropped_separate_client() {
    let Some((device, client)) = setup() else {
        return;
    };

    let n = 50;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();

    let (left, right) = {
        let x = Tensor::<WgpuRuntime>::try_from_slice(&data, &[n], &device).unwrap();
        let l = x.narrow(0, 0, n - 1).unwrap().contiguous().unwrap();
        let r = x.narrow(0, 1, n - 1).unwrap().contiguous().unwrap();
        (l, r)
    };

    let diff = client.sub(&right, &left).unwrap();
    let result: Vec<f32> = diff.to_vec();

    for &val in &result {
        assert!((val - 1.0).abs() < 1e-5, "expected 1.0, got {val}");
    }
}
