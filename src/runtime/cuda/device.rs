//! CUDA Device implementation
//!
//! Provides CUDA device abstraction using cudarc for direct GPU control.

use std::sync::OnceLock;

use crate::runtime::Device;
use crate::runtime::traits::profile::DeviceProfile;

/// Upper bound on distinct CUDA device indices this process will cache a
/// profile for. Multi-GPU nodes rarely exceed single digits; a fixed-size
/// array avoids a map lock on every kernel-selection lookup.
const MAX_CACHED_DEVICES: usize = 16;

/// Per-device-index profile cache. `profile()` is called on the kernel
/// launch path, so it must not re-query the driver every time — query once
/// per index, then serve the cached value.
static PROFILE_CACHE: [OnceLock<DeviceProfile>; MAX_CACHED_DEVICES] =
    [const { OnceLock::new() }; MAX_CACHED_DEVICES];

/// CUDA Device using cudarc
///
/// Represents a single GPU device and manages context for kernel launches.
/// Used by CudaClient for stream management.
#[derive(Clone, Debug)]
pub struct CudaDevice {
    /// Index of the GPU device (0, 1, 2, ...)
    pub(crate) index: usize,
}

/// Initialize the CUDA driver if it is not already.
///
/// Every device ATTRIBUTE query needs the driver up, but the driver is
/// otherwise only initialized as a side effect of `CudaContext::new` inside
/// `CudaClient`. Querying a device before any client exists therefore failed —
/// silently, in `profile()`, which swallows the error and reports an unknown
/// device forever after. `cuInit` is idempotent, so calling it here costs
/// nothing on the paths that already had a context.
fn ensure_driver_init() -> Result<(), CudaError> {
    cudarc::driver::result::init()
        .map_err(|e| CudaError::DeviceError(format!("Failed to initialize CUDA driver: {e:?}")))
}

impl CudaDevice {
    /// Create a new CUDA device
    pub fn new(index: usize) -> Self {
        Self { index }
    }

    /// Get the compute capability of this CUDA device
    ///
    /// Returns (major, minor) version numbers (e.g., (8, 6) for sm_86 / RTX 3090)
    ///
    /// # Examples
    /// - (7, 5): Turing (RTX 20xx, T4)
    /// - (8, 0): Ampere (A100)
    /// - (8, 6): Ampere (RTX 30xx, A6000)
    /// - (8, 9): Ada Lovelace (RTX 40xx, L4)
    /// - (9, 0): Hopper (H100)
    pub fn compute_capability(&self) -> Result<(u32, u32), CudaError> {
        ensure_driver_init()?;
        let device = cudarc::driver::result::device::get(self.index as i32).map_err(|e| {
            CudaError::DeviceError(format!("Failed to get CUDA device {}: {:?}", self.index, e))
        })?;

        let major = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
        }
        .map_err(|e| CudaError::DeviceError(format!("Failed to get compute capability major: {:?}", e)))? as u32;

        let minor = unsafe {
            cudarc::driver::result::device::get_attribute(
                device,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
        }
        .map_err(|e| CudaError::DeviceError(format!("Failed to get compute capability minor: {:?}", e)))? as u32;

        Ok((major, minor))
    }

    /// Synchronize all operations on this device
    ///
    /// This synchronizes the current CUDA context on this thread.
    /// For stream-specific synchronization, use `CudaClient::synchronize()` instead.
    pub fn sync(&self) -> Result<(), CudaError> {
        cudarc::driver::result::ctx::synchronize().map_err(|e| {
            CudaError::SyncError(format!(
                "Failed to synchronize CUDA context for device {}: {:?}",
                self.index, e
            ))
        })
    }

    /// Get memory information for this device
    ///
    /// Returns (free_bytes, total_bytes) for the device's global memory.
    pub fn memory_info(&self) -> Result<(u64, u64), CudaError> {
        let (free, total) = cudarc::driver::result::mem_get_info().map_err(|e| {
            CudaError::DeviceError(format!(
                "Failed to get memory info for device {}: {:?}",
                self.index, e
            ))
        })?;
        Ok((free as u64, total as u64))
    }

    /// Get available (free) GPU memory in bytes
    pub fn available_memory(&self) -> Result<u64, CudaError> {
        let (free, _) = self.memory_info()?;
        Ok(free)
    }

    /// Get total GPU memory in bytes
    pub fn total_memory(&self) -> Result<u64, CudaError> {
        let (_, total) = self.memory_info()?;
        Ok(total)
    }

    /// Query the driver for this device's real capabilities.
    ///
    /// Never returns `Err` to the caller — a query failure (unlikely, but
    /// possible on an unusual driver/device combination) degrades to
    /// `DeviceProfile::unknown("cuda")` rather than panicking or bubbling an
    /// error out of the infallible `Device::profile()` signature.
    fn query_profile(&self) -> DeviceProfile {
        let Ok((major, minor)) = self.compute_capability() else {
            return DeviceProfile::unknown("cuda");
        };

        if ensure_driver_init().is_err() {
            return DeviceProfile::unknown("cuda");
        }
        let Ok(device) = cudarc::driver::result::device::get(self.index as i32) else {
            return DeviceProfile::unknown("cuda");
        };

        let get_attr = |attr: cudarc::driver::sys::CUdevice_attribute| -> Option<u32> {
            unsafe { cudarc::driver::result::device::get_attribute(device, attr) }
                .ok()
                .and_then(|v| u32::try_from(v).ok())
        };

        let (Some(compute_units), Some(shared_mem_per_block), Some(shared_mem_per_unit), Some(max_threads_per_block), Some(lane_width)) = (
            get_attr(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT),
            get_attr(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK),
            get_attr(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
            ),
            get_attr(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK),
            get_attr(cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE),
        )
        else {
            return DeviceProfile::unknown("cuda");
        };

        let (arch, caps) = DeviceProfile::arch_and_caps_for_compute_capability(major, minor);

        DeviceProfile {
            backend: "cuda",
            arch,
            compute_units,
            shared_mem_per_block,
            shared_mem_per_unit,
            max_threads_per_block,
            lane_width,
            caps,
        }
    }
}

impl Device for CudaDevice {
    fn id(&self) -> usize {
        self.index
    }

    fn name(&self) -> String {
        format!("cuda:{}", self.index)
    }

    fn profile(&self) -> DeviceProfile {
        match PROFILE_CACHE.get(self.index) {
            Some(slot) => *slot.get_or_init(|| self.query_profile()),
            // Index outside the fixed cache: query directly rather than
            // panicking or silently truncating to a wrong device's profile.
            None => self.query_profile(),
        }
    }
}

impl Default for CudaDevice {
    fn default() -> Self {
        Self::new(0)
    }
}

/// CUDA-specific errors
#[derive(Debug, Clone)]
pub enum CudaError {
    /// Device initialization or query error
    DeviceError(String),
    /// Memory allocation error
    AllocationError(String),
    /// Memory copy error
    CopyError(String),
    /// Kernel launch error
    KernelError(String),
    /// Synchronization error
    SyncError(String),
    /// cuBLAS error
    CublasError(String),
    /// Context error
    ContextError(String),
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CudaError::DeviceError(msg) => write!(f, "CUDA device error: {}", msg),
            CudaError::AllocationError(msg) => write!(f, "CUDA allocation error: {}", msg),
            CudaError::CopyError(msg) => write!(f, "CUDA copy error: {}", msg),
            CudaError::KernelError(msg) => write!(f, "CUDA kernel error: {}", msg),
            CudaError::SyncError(msg) => write!(f, "CUDA sync error: {}", msg),
            CudaError::CublasError(msg) => write!(f, "cuBLAS error: {}", msg),
            CudaError::ContextError(msg) => write!(f, "CUDA context error: {}", msg),
        }
    }
}

impl std::error::Error for CudaError {}
