//! Serialized access to the CUDA and WebGPU backends.
//!
//! Every test binary that touches a GPU goes through these helpers. Two reasons:
//!
//! * The lock is mandatory. Concurrent WebGPU device use loses the device
//!   (`Buffer ... is invalid` validation errors), and that failure cascades into
//!   every other WebGPU test in the same binary, not just the one that raced.
//!   CUDA serializes for the same reason plus a sync that clears sticky errors
//!   left by a prior panicked test.
//! * An absent device is LOUD. With the feature enabled, a missing runtime is a
//!   broken machine, not a reason to report `ok` while asserting nothing.
//!   `with_wgpu_backend_or_skip` is the one exception, for tests that must run
//!   on a machine with no WebGPU adapter.

#[cfg(feature = "cuda")]
use super::create_cuda_client;
#[cfg(feature = "wgpu")]
use super::create_wgpu_client;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "cuda")]
static CUDA_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
#[cfg(feature = "wgpu")]
static WGPU_BACKEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

/// Run `f` against the CUDA backend, one test at a time.
///
/// Panics when the runtime is unavailable: the `cuda` feature is a claim that
/// this machine has a device.
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
    let (client, device) =
        create_cuda_client().expect("CUDA feature is enabled but CUDA runtime is unavailable");
    // Sync before test to clear any pending errors from a prior panicked test
    client.synchronize();
    f(client, device);
}

/// Run `f` against the WebGPU backend, one test at a time.
///
/// Panics when the runtime is unavailable.
#[cfg(feature = "wgpu")]
pub fn with_wgpu_backend<F>(mut f: F)
where
    F: FnMut(numr::runtime::wgpu::WgpuClient, numr::runtime::wgpu::WgpuDevice),
{
    let _guard = WGPU_BACKEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let (client, device) =
        create_wgpu_client().expect("WGPU feature is enabled but WGPU runtime is unavailable");
    f(client, device);
}

/// Same lock as [`with_wgpu_backend`], but skips instead of panicking when no
/// adapter is present. Use this for tests that must run cleanly on machines
/// without a WebGPU-capable GPU.
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
    let (client, device) =
        create_wgpu_client().expect("WGPU feature is enabled but WGPU runtime is unavailable");
    f(client, device);
}
