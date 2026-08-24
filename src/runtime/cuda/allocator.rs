//! CUDA stream-ordered allocator with a Rust-side free list.

use cudarc::driver::safe::CudaStream;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, Mutex};

use super::arena::CudaArena;
use crate::runtime::Allocator;

/// Maximum number of cached buffers per exact byte-size bucket.
///
/// 64 gives enough depth to absorb a full decode loop's worth of temporaries
/// (typically 8–32 per forward pass) while keeping the worst-case cached
/// footprint to 64× the largest recurring buffer — acceptable for VRAM.
const FREE_LIST_CAP: usize = 64;

/// Default global cap on the TOTAL bytes retained by the free list, across all
/// size buckets.
///
/// The per-bucket [`FREE_LIST_CAP`] bounds a single size, but the number of
/// distinct sizes is unbounded: variable-length / many-shape workloads (e.g.
/// packed varlen embedding ingest) free buffers of continuously varying sizes,
/// each spawning a new bucket. Without a global cap the free list retains
/// `Σ (per-size buffers × size)` device memory indefinitely — the driver sees
/// those cached pointers as live, so free VRAM falls monotonically until OOM.
///
/// This cap bounds the total cached footprint; when exceeded, the oldest cached
/// buffers are returned to the driver. Overridable via the
/// `NUMR_CUDA_FREE_LIST_CAP_MB` environment variable (value in MiB).
const DEFAULT_FREE_LIST_CAP_BYTES: usize = 1024 * 1024 * 1024;

/// Resolve the global free-list byte cap from the environment (MiB) or default.
fn resolve_free_list_cap_bytes() -> usize {
    super::env_config::env_mib_to_bytes(
        "NUMR_CUDA_FREE_LIST_CAP_MB",
        DEFAULT_FREE_LIST_CAP_BYTES as u64,
    ) as usize
}

/// Per-size free buffers plus a running total of cached bytes.
#[derive(Default)]
struct FreeList {
    /// size_bytes → cached device pointers of that exact size (FIFO per bucket).
    map: HashMap<u64, VecDeque<u64>>,
    /// Sum of `size × count` across every bucket — the live cached footprint.
    total_bytes: usize,
}

impl FreeList {
    /// Pop a cached buffer of exactly `size_bytes`, updating the byte total.
    fn pop(&mut self, size_bytes: u64) -> Option<u64> {
        let ptr = self.map.get_mut(&size_bytes).and_then(|b| b.pop_front())?;
        self.total_bytes -= size_bytes as usize;
        Some(ptr)
    }

    /// Push a cached buffer of `size_bytes`, updating the byte total.
    fn push(&mut self, size_bytes: u64, ptr: u64) {
        self.map.entry(size_bytes).or_default().push_back(ptr);
        self.total_bytes += size_bytes as usize;
    }

    /// Evict cached buffers (largest size first) until `total_bytes <= cap`.
    /// Returns the evicted pointers for the caller to free outside the lock.
    fn evict_to_cap(&mut self, cap: usize) -> Vec<u64> {
        let mut evicted = Vec::new();
        while self.total_bytes > cap {
            let largest = self
                .map
                .iter()
                .filter(|(_, b)| !b.is_empty())
                .map(|(&s, _)| s)
                .max();
            match largest {
                Some(size) => {
                    if let Some(ptr) = self.pop(size) {
                        evicted.push(ptr);
                    } else {
                        break;
                    }
                }
                None => break,
            }
        }
        evicted
    }
}

/// CUDA stream-ordered allocator with a Rust-side free list.
///
/// Combines a per-size-bucket Rust free list (fast path: no driver call) with
/// `cuMemAllocAsync`/`cuMemFreeAsync` for cold misses and pool management.
///
/// # Safety invariant — single canonical stream
///
/// Every pointer in the free list was allocated on `self.stream`. Every
/// deallocation is also issued on `self.stream`. Because all alloc/free ops
/// share the same stream, there is no cross-stream ordering hazard: a buffer
/// popped from the free list is guaranteed to have no pending use on any
/// other stream before it is returned to the caller.
///
/// This invariant holds because `CudaClient::new` always returns the cached
/// canonical client for a device (via `register_or_get_client`), and stream
/// creation only happens in `new_uncached`. The `copy_stream` is exclusively
/// for D2H copies and never used with the allocator.
///
/// # Pool Threshold
///
/// The default memory pool's release threshold is set to 512 MiB at context
/// creation (see `CudaClient` construction in `client.rs`): freed segments up
/// to that size stay warm for reuse in decode loops, while larger freed
/// segments are returned to the OS to avoid address-space fragmentation on
/// many-shape workloads. The threshold is overridable via the
/// `NUMR_CUDA_POOL_RELEASE_THRESHOLD_MB` environment variable (value in MiB).
/// The OOM-retry path additionally trims the pool to 0.
#[derive(Clone)]
pub struct CudaAllocator {
    stream: Arc<CudaStream>,
    /// Per-size free list with a running cached-byte total.
    ///
    /// VecDeque gives O(1) push_back / pop_front so the oldest entry is
    /// evicted first when the per-bucket cap is reached (FIFO within a bucket).
    /// A global byte cap ([`free_list_cap_bytes`](Self::free_list_cap_bytes))
    /// bounds the total across all buckets so many-shape workloads cannot retain
    /// device memory without bound.
    ///
    /// Every lock acquisition recovers from poisoning via `into_inner()`. The
    /// free list is a cache, not correctness-critical state: a panic mid-mutation
    /// can only desynchronise `total_bytes` from the bucket contents (over- or
    /// under-retention), never duplicate a pointer or publish an invalid one.
    /// Refusing to allocate for the rest of the process is strictly worse.
    free_list: Arc<Mutex<FreeList>>,
    /// Global cap on total bytes retained by `free_list` across all buckets.
    free_list_cap_bytes: usize,
    /// When frozen, alloc/free go directly to the driver to create proper
    /// CUDA graph alloc/free nodes — bypassing the Rust free list.
    frozen: Arc<std::sync::atomic::AtomicBool>,
    /// Pointers allocated during the current freeze window.
    ///
    /// Every pointer returned by `allocate()` while `frozen` is true is inserted
    /// here. At `unfreeze()` time we verify that none of these addresses ended up
    /// inside `free_list` (which would mean a frozen-allocated pointer was
    /// accidentally routed through the un-frozen `deallocate()` path and absorbed
    /// into the Rust cache — a silent graph-state corruption).
    ///
    /// The set is cleared at `unfreeze()`. In debug builds the check panics;
    /// in release builds the assertion compiles away but the set is still
    /// maintained and cleared so the bookkeeping remains coherent.
    captured_ptrs: Arc<Mutex<HashSet<u64>>>,
    /// Cached handle to the default memory pool for the device. Stored as
    /// `u64` (the raw pointer value) so the struct stays `Send + Sync`.
    /// Used only by the OOM retry path to call `cuMemPoolTrimTo`.
    /// Zero when the pool could not be obtained at construction.
    pool_handle: u64,
    /// Optional bump-pointer arena for graph-internal intermediates.
    ///
    /// When `Some`, the frozen `allocate()` path returns offsets into this
    /// pre-allocated device buffer instead of calling `cuMemAllocAsync`.
    /// This gives captured-kernel arguments stable device addresses that
    /// survive across `cuGraphLaunch` replays.
    ///
    /// Installed via `install_arena()` before `freeze()`.
    /// Cleared by `unfreeze()` via `clear_arena()`.
    arena: Arc<Mutex<Option<CudaArena>>>,
}

impl CudaAllocator {
    /// Construct an allocator bound to `stream`.
    ///
    /// `pool_handle` is the raw `CUmemoryPool` pointer value for the device's
    /// default pool (zero if it could not be obtained), used only by the OOM
    /// retry path.
    pub(super) fn new(stream: Arc<CudaStream>, pool_handle: u64) -> Self {
        Self {
            stream,
            free_list: Arc::new(Mutex::new(FreeList::default())),
            free_list_cap_bytes: resolve_free_list_cap_bytes(),
            frozen: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            captured_ptrs: Arc::new(Mutex::new(HashSet::new())),
            pool_handle,
            arena: Arc::new(Mutex::new(None)),
        }
    }

    /// Allocate directly from the driver (no free-list lookup).
    ///
    /// On failure, drains the Rust free list back to the driver pool so those
    /// segments become available for reuse, syncs the stream, then retries once.
    unsafe fn driver_alloc(&self, size_bytes: usize) -> crate::error::Result<u64> {
        let mut ptr: u64 = 0;
        let result = unsafe {
            cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream())
        };
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            return Ok(ptr);
        }

        // Drain free list: return cached segments to the driver pool so it can
        // reclaim VRAM that our Rust-side cache was holding "live" from the
        // driver's perspective.
        let drained: Vec<u64> = {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            fl.total_bytes = 0;
            fl.map
                .drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for p in drained {
            let _ = unsafe { cudarc::driver::sys::cuMemFreeAsync(p, self.stream.cu_stream()) };
        }

        // Sync so the pool can process the frees before the retry alloc.
        let _ = self.stream.synchronize();

        // Trim the pool to release cached segments back to the OS. The
        // pool retains freed allocations (per the release threshold) which
        // is fast for tight decode loops but fragments the address space
        // when a one-shot large allocation arrives after many small frees.
        // Calling trim only on the retry path keeps the steady-state cache
        // behaviour but recovers from fragmentation when it matters.
        if self.pool_handle != 0 {
            let pool = self.pool_handle as cudarc::driver::sys::CUmemoryPool;
            let _ = unsafe { cudarc::driver::sys::cuMemPoolTrimTo(pool, 0) };
        }

        let result = unsafe {
            cudarc::driver::sys::cuMemAllocAsync(&mut ptr, size_bytes, self.stream.cu_stream())
        };
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            Ok(ptr)
        } else {
            Err(crate::error::Error::OutOfMemory { size: size_bytes })
        }
    }

    /// Free directly to the driver (no free-list insertion).
    unsafe fn driver_free(&self, ptr: u64) {
        let _ = unsafe { cudarc::driver::sys::cuMemFreeAsync(ptr, self.stream.cu_stream()) };
    }

    /// Install a bump-pointer arena for the next freeze window.
    ///
    /// Must be called **before** [`Allocator::freeze`]. `base` is the device
    /// address of a pre-allocated buffer of `size` bytes; both must remain
    /// valid until [`clear_arena`](Self::clear_arena) is called (which happens
    /// inside `unfreeze()`).
    ///
    /// # Errors
    ///
    /// Returns [`Error::Internal`][crate::error::Error::Internal] if an arena is
    /// already installed. Graph capture on a single client is not re-entrant or
    /// thread-safe (the freeze flag and stream-capture state are shared), so a
    /// double install indicates two overlapping capture attempts — failing fast
    /// surfaces the misuse instead of silently overwriting the live arena.
    pub fn install_arena(&self, base: u64, size: usize) -> crate::error::Result<()> {
        let mut guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        if guard.is_some() {
            return Err(crate::error::Error::Internal(
                "CudaAllocator::install_arena: an arena is already installed; \
                 graph capture is not re-entrant on a single client"
                    .into(),
            ));
        }
        *guard = Some(CudaArena::new(base, size));
        Ok(())
    }

    /// Remove the arena.
    ///
    /// Called automatically by `unfreeze()`. May also be called on error paths
    /// to discard an installed arena without going through a full freeze cycle.
    /// Dropping the bookkeeping does **not** free the device buffer — that is
    /// owned by the `Tensor<CudaRuntime>` held in `CapturedGraph`.
    pub fn clear_arena(&self) {
        let mut guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        *guard = None;
    }

    /// Returns `true` if a bump-pointer arena is currently installed.
    pub fn has_arena(&self) -> bool {
        let guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
        guard.is_some()
    }
}

impl Allocator for CudaAllocator {
    fn allocate(&self, size_bytes: usize) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        // Graph-capture path: bypass the Rust free list.
        //
        // When an arena is installed: redirect allocations into the
        // bump-pointer arena so that the device addresses baked into graph
        // kernel-parameter blocks are stable across replays.
        //
        // When no arena is installed: fall through to driver_alloc, which
        // creates a proper graph alloc-node (existing behaviour).
        //
        // Either way, record the pointer in `captured_ptrs` so `unfreeze()`
        // can assert it never migrated into the free list.
        if self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            let ptr = {
                let mut arena_guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
                if let Some(ref mut arena) = *arena_guard {
                    arena.allocate(size_bytes)?
                } else {
                    // No arena: fall through to driver.
                    drop(arena_guard);
                    unsafe { self.driver_alloc(size_bytes) }?
                }
            };
            self.captured_ptrs
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .insert(ptr);
            return Ok(ptr);
        }

        // Fast path: pop from the free list if a cached buffer exists.
        {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(ptr) = fl.pop(size_bytes as u64) {
                return Ok(ptr);
            }
        }

        // Cold miss: ask the driver.
        unsafe { self.driver_alloc(size_bytes) }
    }

    fn deallocate(&self, ptr: u64, size_bytes: usize) {
        if ptr == 0 {
            return;
        }

        // Graph-capture path: handle differently depending on whether the
        // pointer came from the arena or from the driver.
        if self.frozen.load(std::sync::atomic::Ordering::Relaxed) {
            let mut arena_guard = self.arena.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(ref mut arena) = *arena_guard {
                // Pointer came from the arena: record the logical free
                // (bookkeeping only — the arena is strictly monotone and never
                // rewinds high_water). Do NOT call cuMemFreeAsync; the arena
                // buffer is not owned by the driver pool.
                arena.deallocate(ptr);
            } else {
                // No arena: pointer came from driver_alloc; issue a driver
                // free node so the graph records the deallocation.
                drop(arena_guard);
                unsafe { self.driver_free(ptr) };
            }
            return;
        }

        let mut evict: Vec<u64> = Vec::new();
        {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            let size = size_bytes as u64;
            match fl.map.get_mut(&size) {
                // Per-bucket cap exceeded: evict the oldest of this size and
                // keep the new one (byte total unchanged — same-size swap).
                Some(bucket) if bucket.len() >= FREE_LIST_CAP => {
                    if let Some(old) = bucket.pop_front() {
                        evict.push(old);
                    }
                    bucket.push_back(ptr);
                }
                _ => fl.push(size, ptr),
            }
            // Global byte cap: return oldest-largest cached buffers to the
            // driver until total cached bytes are back under the cap. Bounds
            // retention on many-shape workloads that spawn unbounded size
            // buckets.
            evict.extend(fl.evict_to_cap(self.free_list_cap_bytes));
        }

        // Free outside the lock so the lock is not held during driver calls.
        for old_ptr in evict {
            unsafe { self.driver_free(old_ptr) };
        }
    }

    fn is_frozen(&self) -> bool {
        self.frozen.load(std::sync::atomic::Ordering::Relaxed)
    }

    fn freeze(&self) -> bool {
        self.frozen
            .store(true, std::sync::atomic::Ordering::Relaxed);
        true
    }

    fn unfreeze(&self) {
        self.frozen
            .store(false, std::sync::atomic::Ordering::Relaxed);

        // Clear the arena bookkeeping.  The device buffer itself is NOT freed
        // here — it is owned by the Tensor<CudaRuntime> held in
        // CapturedGraph::arena, and will be freed when CapturedGraph is dropped
        // (after the CudaGraph handle is destroyed).
        self.clear_arena();

        // ---- Graph-capture bookkeeping (defense-in-depth) ----
        //
        // Drain the captured_ptrs set and verify that none of those addresses
        // are present in the free list. If a match is found it means a pointer
        // that was allocated during the freeze window was later passed to the
        // *un-frozen* `deallocate()` path, which cached it in `free_list`. On
        // the next allocation the Rust cache would hand that address back to a
        // non-graph caller while the CUDA graph still owns it — silent
        // graph-state corruption. Panic early so the bug is caught immediately.
        let captured: HashSet<u64> = {
            let mut set = self.captured_ptrs.lock().unwrap_or_else(|p| p.into_inner());
            std::mem::take(&mut *set)
        };

        if captured.is_empty() {
            return;
        }

        // Non-empty set means the closure did not explicitly free all of its
        // graph-internal scratch. With AUTO_FREE_ON_LAUNCH the driver will
        // handle the frees on next launch — this is expected for intermediate
        // buffers that live until graph replay. Log once so it is visible
        // without being fatal.
        #[cfg(debug_assertions)]
        eprintln!(
            "[numr::cuda] unfreeze: {} pointer(s) from the freeze window were \
             still live at unfreeze (will be freed by the driver on next graph \
             launch — expected for graph-internal scratch).",
            captured.len()
        );

        // CRITICAL assertion: none of the captured pointers must appear in
        // the free list. Finding one there means the Rust cache absorbed a
        // graph-internal address — a definite bug.
        #[cfg(debug_assertions)]
        {
            let fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            for bucket in fl.map.values() {
                for &cached_ptr in bucket {
                    debug_assert!(
                        !captured.contains(&cached_ptr),
                        "[numr::cuda] GRAPH CORRUPTION DETECTED: pointer 0x{:x} was \
                         allocated during a freeze window (graph-capture) but was \
                         subsequently absorbed into the Rust free list via the \
                         un-frozen deallocate() path. On next allocation this \
                         address would be handed to a non-graph caller while the \
                         CUDA graph still holds a reference to it.",
                        cached_ptr
                    );
                }
            }
        }
    }

    fn reset(&self) -> crate::error::Result<()> {
        // Drain the free list and return all cached buffers to the driver pool.
        // Callers must have dropped all live tensors (which call `deallocate`)
        // before calling `reset`, so every pointer here is idle on the stream.
        let drained: Vec<u64> = {
            let mut fl = self.free_list.lock().unwrap_or_else(|p| p.into_inner());
            fl.total_bytes = 0;
            fl.map
                .drain()
                .flat_map(|(_, bucket)| bucket.into_iter())
                .collect()
        };
        for ptr in drained {
            unsafe { self.driver_free(ptr) };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::client::CudaClient;
    use super::super::device::CudaDevice;
    use crate::runtime::Allocator;

    /// Verify that `unfreeze()` fires the corruption assertion when a pointer
    /// that was allocated during a freeze window is found in the free list.
    ///
    /// This test simulates the bug: a freeze-allocated pointer is injected
    /// directly into the free list (mimicking a future op that accidentally
    /// routes a graph-internal address through the un-frozen deallocate path).
    /// `unfreeze()` must detect the overlap and panic.
    ///
    /// The test is `#[ignore]` because `CudaAllocator::driver_alloc` requires a
    /// live CUDA context and GPU. Run with:
    ///
    /// ```text
    /// cargo test --features cuda captured_ptrs_unfreeze_detects_corruption -- --ignored
    /// ```
    ///
    /// In CI without a GPU the structural invariant can be verified by
    /// inspection: the debug_assert in `unfreeze()` iterates `free_list` and
    /// panics on any pointer that appears in `captured_ptrs`.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn captured_ptrs_unfreeze_detects_corruption() {
        // --- Setup: build a real CudaAllocator via CudaClient::new_uncached ---
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        // Allocate two pointers normally (pre-freeze, goes into free list on drop).
        let _p1 = alloc.allocate(256).expect("alloc p1");
        let _p2 = alloc.allocate(512).expect("alloc p2");

        // Transition into freeze mode (simulates start of CUDA graph capture).
        alloc.freeze();
        assert!(alloc.is_frozen(), "allocator should be frozen");

        // Allocate a third pointer — this goes directly to the driver and is
        // recorded in `captured_ptrs`.
        let p3 = alloc.allocate(128).expect("alloc p3 during freeze");
        assert_ne!(p3, 0, "frozen alloc must return a non-null pointer");

        // Verify p3 is tracked in captured_ptrs.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(
                set.contains(&p3),
                "p3 must be present in captured_ptrs after frozen allocate"
            );
        }

        // --- Inject the bug: manually push p3 into the free list ---
        //
        // This simulates a future op author calling `deallocate(p3, 128)` on
        // an un-frozen allocator (e.g., because they cached the raw pointer and
        // called deallocate after the freeze window closed).
        {
            let mut fl = alloc.free_list.lock().unwrap();
            fl.push(128, p3);
        }

        // `unfreeze()` must detect the overlap and panic (debug builds only).
        // In release builds the assert compiles away so we skip the
        // should_panic check to avoid a false-pass.
        #[cfg(debug_assertions)]
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                alloc.unfreeze();
            }));
            assert!(
                result.is_err(),
                "unfreeze() must panic when a captured pointer leaks into the free list"
            );
        }

        // In release builds: just call unfreeze and verify the set is cleared
        // (no assertion fires, but bookkeeping must still be coherent).
        #[cfg(not(debug_assertions))]
        {
            alloc.unfreeze();
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(
                set.is_empty(),
                "captured_ptrs must be empty after unfreeze (release build)"
            );
        }
    }

    /// Verify that `captured_ptrs` is properly cleared after a clean freeze
    /// window where the closure frees everything it allocated.
    ///
    /// This test exercises the non-buggy path: allocate during freeze, free
    /// during freeze (driver_free, does NOT touch free_list), then unfreeze.
    /// `captured_ptrs` should be non-empty at unfreeze time (the closure did
    /// not explicitly call deallocate — typical for graph-internal scratch whose
    /// lifetime the CUDA runtime manages), but no overlap with free_list exists,
    /// so no panic occurs.
    ///
    /// This test is also `#[ignore]` as it requires a live GPU.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn captured_ptrs_cleared_after_clean_unfreeze() {
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        alloc.freeze();
        let p = alloc.allocate(64).expect("alloc during freeze");
        assert_ne!(p, 0);

        // captured_ptrs must contain p.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(set.contains(&p), "p must be in captured_ptrs");
        }

        // Unfreeze without injecting p into the free list — clean path.
        // Should NOT panic.
        alloc.unfreeze();

        // captured_ptrs must be cleared after unfreeze.
        {
            let set = alloc.captured_ptrs.lock().unwrap();
            assert!(set.is_empty(), "captured_ptrs must be empty after unfreeze");
        }

        // frozen flag must be cleared.
        assert!(!alloc.is_frozen(), "allocator must be unfrozen");
    }

    /// A panic while the free-list mutex is held must not disable the allocator.
    ///
    /// Before the fix every `free_list.lock()` was `.unwrap()`, so one poisoned
    /// lock turned every subsequent allocation into a panic — the exact cascade
    /// an OOM panic could trigger mid-training. Recovery is sound because the
    /// free list is a cache: the worst a torn mutation leaves behind is a
    /// `total_bytes` that disagrees with the bucket contents.
    ///
    /// Sabotage check: restore `.unwrap()` on the `free_list` locks and this
    /// test fails with
    /// `called `Result::unwrap()` on an `Err` value: PoisonError { .. }`.
    ///
    /// `#[ignore]` for the same reason as the tests above — `driver_alloc`
    /// needs a live CUDA context.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore = "requires a live CUDA GPU"]
    fn allocation_survives_a_poisoned_free_list() {
        let device = CudaDevice { index: 0 };
        let client =
            CudaClient::new_uncached(device).expect("CudaClient creation requires a CUDA GPU");
        let alloc = &client.allocator;

        // Put a buffer in the free list so the poisoned state has real content.
        let p = alloc.allocate(1024).expect("alloc before poisoning");
        alloc.deallocate(p, 1024);

        // Poison the mutex: a thread panics while holding the guard.
        let poisoner = alloc.clone();
        let handle = std::thread::spawn(move || {
            let _guard = poisoner.free_list.lock().expect("lock is not yet poisoned");
            panic!("deliberate panic while holding the free-list lock");
        });
        assert!(handle.join().is_err(), "the poisoning thread must panic");
        assert!(
            alloc.free_list.is_poisoned(),
            "the free-list mutex must be poisoned"
        );

        // Every free-list path must still work.
        let p2 = alloc.allocate(1024).expect("alloc after poisoning");
        assert_ne!(p2, 0, "allocation must return a real pointer");
        alloc.deallocate(p2, 1024);
        alloc.reset().expect("reset after poisoning");
    }
}
