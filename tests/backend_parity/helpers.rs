//! Shared helpers for backend parity tests: assertion utilities, backend locks, client creation.

#[cfg(feature = "cuda")]
use crate::common::create_cuda_client;
#[cfg(feature = "wgpu")]
use crate::common::create_wgpu_client;
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
static CUDA_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
#[cfg(feature = "wgpu")]
static WGPU_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

pub fn assert_parity_f32(a: &[f32], b: &[f32], op: &str) {
    let rtol = 1e-5f32;
    let atol = 1e-7f32;
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();

        if diff > tol {
            panic!(
                "parity_f32[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

#[allow(dead_code)]
pub fn assert_parity_f64(a: &[f64], b: &[f64], op: &str) {
    let rtol = 1e-12f64;
    let atol = 1e-14f64;
    assert_eq!(
        a.len(),
        b.len(),
        "parity_f64[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let tol = atol + rtol * y.abs();

        if diff > tol {
            panic!(
                "parity_f64[{}] at index {}: {} vs {} (diff={}, tol={})",
                op, i, x, y, diff, tol
            );
        }
    }
}

#[allow(dead_code)]
pub fn assert_parity_i32(a: &[i32], b: &[i32], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_i32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_i32[{}] at index {}: {} vs {}", op, i, x, y);
    }
}

pub fn assert_parity_u32(a: &[u32], b: &[u32], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_u32[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_u32[{}] at index {}: {} vs {}", op, i, x, y);
    }
}

#[allow(dead_code)]
pub fn assert_parity_bool(a: &[bool], b: &[bool], op: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "parity_bool[{}]: length mismatch: {} vs {}",
        op,
        a.len(),
        b.len()
    );

    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x, y, "parity_bool[{}] at index {}: {} vs {}", op, i, x, y);
    }
}

#[cfg(feature = "cuda")]
pub fn create_cuda_client_checked() -> Option<(
    numr::runtime::cuda::CudaClient,
    numr::runtime::cuda::CudaDevice,
)> {
    create_cuda_client()
}

#[cfg(feature = "wgpu")]
pub fn create_wgpu_client_checked() -> Option<(
    numr::runtime::wgpu::WgpuClient,
    numr::runtime::wgpu::WgpuDevice,
)> {
    create_wgpu_client()
}

#[cfg(feature = "cuda")]
pub fn with_cuda_backend<F>(mut f: F)
where
    F: FnMut(numr::runtime::cuda::CudaClient, numr::runtime::cuda::CudaDevice),
{
    use numr::runtime::RuntimeClient;
    let _guard = CUDA_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (client, device) = create_cuda_client_checked()
        .expect("CUDA feature is enabled but CUDA runtime is unavailable");
    // Sync before test to clear any pending errors from a prior panicked test
    client.synchronize();
    f(client, device);
}

#[cfg(feature = "wgpu")]
pub fn with_wgpu_backend<F>(mut f: F)
where
    F: FnMut(numr::runtime::wgpu::WgpuClient, numr::runtime::wgpu::WgpuDevice),
{
    let _guard = WGPU_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (client, device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    f(client, device);
}

/// Same lock as `with_wgpu_backend`, but skips instead of panicking when no
/// adapter is present. Use this for tests that must run cleanly on machines
/// without a WebGPU-capable GPU.
///
/// The lock itself is mandatory, not optional: concurrent WebGPU device use
/// loses the device (`Buffer ... is invalid` validation errors), and that
/// failure cascades into every other WebGPU test in the binary, not just the
/// one that raced. Every wgpu test must serialize through this lock or
/// `with_wgpu_backend`.
#[cfg(feature = "wgpu")]
pub fn with_wgpu_backend_or_skip<F>(mut f: F)
where
    F: FnMut(numr::runtime::wgpu::WgpuClient, numr::runtime::wgpu::WgpuDevice),
{
    if !numr::runtime::wgpu::is_wgpu_available() {
        println!("skipping: no WGPU adapter available");
        return;
    }
    let _guard = WGPU_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (client, device) = create_wgpu_client_checked()
        .expect("WGPU feature is enabled but WGPU runtime is unavailable");
    f(client, device);
}
