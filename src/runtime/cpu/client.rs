//! CPU client and allocator implementation

use super::device::CpuDevice;
use super::runtime::CpuRuntime;
use crate::runtime::{DefaultAllocator, RuntimeClient};
use std::alloc::{Layout as AllocLayout, alloc_zeroed, dealloc};
#[cfg(feature = "rayon")]
use std::sync::Arc;

/// CPU parallelism configuration.
///
/// This is intentionally runtime-local and opt-in:
/// - `None` keeps the existing default behavior (global Rayon pool)
/// - `Some(n)` uses a dedicated pool with `n` threads for this client
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParallelismConfig {
    /// Maximum number of worker threads for Rayon-backed kernels.
    pub max_threads: Option<usize>,
    /// Optional chunk size hint for future CPU kernels.
    pub chunk_size: Option<usize>,
}

impl ParallelismConfig {
    /// Create a new parallelism config.
    #[must_use]
    pub const fn new(max_threads: Option<usize>, chunk_size: Option<usize>) -> Self {
        Self {
            max_threads,
            chunk_size,
        }
    }
}

/// CPU client for operation dispatch
#[derive(Clone)]
pub struct CpuClient {
    pub(crate) device: CpuDevice,
    allocator: CpuAllocator,
    parallelism: ParallelismConfig,
    #[cfg(feature = "rayon")]
    thread_pool: Option<Arc<rayon::ThreadPool>>,
}

impl std::fmt::Debug for CpuClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuClient")
            .field("device", &self.device)
            .field("parallelism", &self.parallelism)
            .finish()
    }
}

impl CpuClient {
    /// Create a new CPU client
    pub fn new(device: CpuDevice) -> Self {
        let allocator = create_cpu_allocator(device.clone());
        Self {
            device,
            allocator,
            parallelism: ParallelismConfig::default(),
            #[cfg(feature = "rayon")]
            thread_pool: None,
        }
    }

    /// Return a cloned client with explicit CPU parallelism settings.
    ///
    /// This allows composition with external runtimes (e.g. solvers) without
    /// oversubscribing the global Rayon pool.
    #[must_use]
    pub fn with_parallelism(&self, config: ParallelismConfig) -> Self {
        Self {
            device: self.device.clone(),
            allocator: self.allocator.clone(),
            parallelism: config,
            #[cfg(feature = "rayon")]
            thread_pool: build_thread_pool(config.max_threads),
        }
    }

    /// Get the current parallelism settings for this client.
    #[must_use]
    pub const fn parallelism(&self) -> ParallelismConfig {
        self.parallelism
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn install_parallelism<F, T>(&self, f: F) -> T
    where
        F: FnOnce() -> T + Send,
        T: Send,
    {
        if let Some(pool) = &self.thread_pool {
            pool.install(f)
        } else {
            f()
        }
    }

    #[cfg(not(feature = "rayon"))]
    pub(crate) fn install_parallelism<F, T>(&self, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f()
    }

    #[inline]
    pub(crate) fn chunk_size_hint(&self) -> usize {
        self.parallelism.chunk_size.unwrap_or(1).max(1)
    }

    #[cfg(feature = "rayon")]
    #[inline]
    pub(crate) fn rayon_min_len(&self) -> usize {
        self.chunk_size_hint()
    }
}

impl RuntimeClient<CpuRuntime> for CpuClient {
    fn device(&self) -> &CpuDevice {
        &self.device
    }

    fn synchronize(&self) {
        // CPU operations are synchronous, nothing to do
    }

    fn allocator(&self) -> &CpuAllocator {
        &self.allocator
    }
}

/// CPU-specific allocator type alias
pub type CpuAllocator = DefaultAllocator<CpuDevice>;

#[cfg(feature = "rayon")]
fn build_thread_pool(max_threads: Option<usize>) -> Option<Arc<rayon::ThreadPool>> {
    let threads = max_threads?;
    if threads == 0 {
        return None;
    }

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .ok()
        .map(Arc::new)
}

/// Create a CPU allocator for the given device
fn create_cpu_allocator(device: CpuDevice) -> CpuAllocator {
    DefaultAllocator::new(
        device,
        |size, _dev| {
            // Non-null and aligned even at size 0: see `CpuRuntime::allocate`.
            if size == 0 {
                return Ok(64);
            }
            let align = 64; // AVX-512 alignment
            // Fails when `size` rounded up to `align` exceeds isize::MAX — an
            // allocation that can never succeed, so report it as OOM.
            let layout = AllocLayout::from_size_align(size, align)
                .map_err(|_| crate::error::Error::OutOfMemory { size })?;
            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                return Err(crate::error::Error::OutOfMemory { size });
            }
            Ok(ptr as u64)
        },
        |ptr, size, _dev| {
            if ptr == 0 || size == 0 {
                return;
            }
            let align = 64;
            // Unrepresentable layout: the alloc closure above rejected this size,
            // so no live allocation matches it. Nothing to free.
            let Ok(layout) = AllocLayout::from_size_align(size, align) else {
                return;
            };
            unsafe {
                dealloc(ptr as *mut u8, layout);
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Device;

    #[test]
    fn test_with_parallelism_preserves_device_and_updates_config() {
        let client = CpuClient::new(CpuDevice::new());
        let configured = client.with_parallelism(ParallelismConfig::new(Some(2), Some(512)));

        assert_eq!(configured.device.id(), client.device.id());
        assert_eq!(configured.parallelism().max_threads, Some(2));
        assert_eq!(configured.parallelism().chunk_size, Some(512));
    }

    #[cfg(feature = "rayon")]
    #[test]
    fn test_rayon_min_len_defaults_and_normalizes_zero() {
        let client = CpuClient::new(CpuDevice::new());
        assert_eq!(client.rayon_min_len(), 1);

        let configured = client.with_parallelism(ParallelismConfig::new(Some(2), Some(64)));
        assert_eq!(configured.rayon_min_len(), 64);

        let zero_chunk = client.with_parallelism(ParallelismConfig::new(Some(2), Some(0)));
        assert_eq!(zero_chunk.rayon_min_len(), 1);
    }

    /// A size that no `Layout` can represent must be reported as an allocation
    /// error, not a panic. `AllocLayout::from_size_align` rejects any size that
    /// exceeds `isize::MAX` once rounded up to the alignment.
    ///
    /// Sabotage check: restore
    /// `.expect("Invalid allocation layout")` in `create_cpu_allocator`'s alloc
    /// closure and this test aborts with
    /// `panicked at 'Invalid allocation layout: LayoutError'`.
    #[test]
    fn allocating_past_isize_max_returns_out_of_memory() {
        use crate::runtime::Allocator;

        let allocator = create_cpu_allocator(CpuDevice::new());
        let err = allocator
            .allocate(usize::MAX)
            .expect_err("an unrepresentable layout must not allocate");

        assert!(
            matches!(err, crate::error::Error::OutOfMemory { size } if size == usize::MAX),
            "expected OutOfMemory, got {err}"
        );
    }
}
