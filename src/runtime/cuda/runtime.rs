//! CUDA runtime implementation

use super::allocator::CudaAllocator;
use super::cache::{
    get_or_create_client, is_cuda_context_valid, log_cuda_memory_error, try_get_cached_client,
    try_get_cached_stream,
};
use super::client::CudaClient;
use super::device::CudaDevice;
use super::kernels;
use crate::runtime::Runtime;
use crate::runtime::common::Allocator;

/// CUDA Runtime adapter
///
/// Implements the generic Runtime trait for CUDA backend.
/// Uses cudarc for direct GPU control.
#[derive(Clone, Debug, Default)]
pub struct CudaRuntime;

impl Runtime for CudaRuntime {
    type Device = CudaDevice;
    type Client = CudaClient;
    type Allocator = CudaAllocator;
    type Graph = super::CudaGraph;
    type RawHandle = super::CudaRawHandle;
    type DType = crate::dtype::DType;

    fn name() -> &'static str {
        "cuda"
    }

    fn supports_graph_capture() -> bool {
        true // CUDA supports graph capture
    }

    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&crate::tensor::Tensor<Self>],
        outputs: &[&crate::tensor::Tensor<Self>],
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> crate::error::Result<()>,
    {
        use cudarc::driver::sys::CUstreamCaptureMode;

        // Clone each I/O tensor (cheap Arc bump on Storage).  Holding these
        // clones prevents the underlying device memory from being freed for
        // the lifetime of the resulting CapturedGraph.
        let owned_inputs: Vec<crate::tensor::Tensor<Self>> =
            inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<crate::tensor::Tensor<Self>> =
            outputs.iter().map(|t| (*t).clone()).collect();

        // Freeze the caching allocator: alloc/free calls go directly through
        // cuMemAllocAsync/cuMemFreeAsync, creating proper graph nodes.
        // (Same rationale as the allocator freeze in capture_graph_into.)
        client.allocator.freeze();

        // Begin stream capture — all ops on this stream are recorded.
        client
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;

        // Execute the closure — ops are recorded into the graph.
        let closure_result = f(client);

        // End capture — MUST happen even if the closure failed, otherwise the
        // stream is left in capture mode and all subsequent operations fail.
        //
        // AUTO_FREE_ON_LAUNCH: graph-managed memory allocated during capture is
        // freed on each launch. Intermediate tensors (scratch buffers) created
        // inside the closure are subject to this. The caller-supplied output
        // tensors are allocated OUTSIDE the closure (before capture begins) and
        // are NOT subject to auto-free — their lifetime is controlled by the
        // Arc-clones held in owned_outputs.
        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph_result = client.stream.end_capture(flags);

        // Restore caching allocator for normal (non-capture) operations.
        client.allocator.unfreeze();

        // Propagate closure error (after restoring stream/allocator state).
        closure_result?;

        // Propagate capture error.
        let graph_opt = graph_result?;

        let cudarc_graph = graph_opt.ok_or_else(|| {
            crate::error::Error::Backend(
                "CUDA graph capture produced no operations — closure recorded nothing".into(),
            )
        })?;

        Ok(crate::runtime::CapturedGraph::new(
            super::CudaGraph::new(cudarc_graph),
            owned_inputs,
            owned_outputs,
        ))
    }

    /// Allocate GPU memory.
    ///
    /// Routes through the client's caching allocator (free-list pool) to avoid
    /// cuMemAllocAsync driver round-trips for repeated same-size allocations.
    fn allocate(size_bytes: usize, device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        let client = get_or_create_client(device);
        client.allocator.allocate(size_bytes)
    }

    /// Deallocate GPU memory.
    ///
    /// Routes through the client's caching allocator — buffers are returned to
    /// the free-list for reuse instead of calling cuMemFreeAsync.
    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        if ptr == 0 {
            return;
        }

        // Try to use the client's caching allocator (returns to free-list)
        if let Some(client) = try_get_cached_client(device.index) {
            client.allocator.deallocate(ptr, size_bytes);
            return;
        }

        // Client not available (shutdown) — free directly
        unsafe {
            if !is_cuda_context_valid() {
                return;
            }

            let result = if let Some(stream) = try_get_cached_stream(device.index) {
                cudarc::driver::sys::cuMemFreeAsync(ptr, stream)
            } else {
                cudarc::driver::sys::cuMemFree_v2(ptr)
            };

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS
                && result != cudarc::driver::sys::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS
            {
                log_cuda_memory_error("cuMemFree", ptr, result);
            }
        }
    }

    /// Copy data from host to device.
    ///
    /// Returns an error if the CUDA copy operation fails.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::Error::Backend(format!(
                    "CUDA host-to-device copy failed: {} bytes ({:?})",
                    src.len(),
                    result
                )));
            }

            // No explicit sync needed: with pageable (non-pinned) host memory,
            // cuMemcpyHtoDAsync is synchronous w.r.t. the host buffer — the call
            // returns only after the copy is complete. An explicit stream.synchronize()
            // here would also drain ALL pending GPU work, destroying pipeline throughput.
        }
        Ok(())
    }

    /// Copy data from device to host.
    ///
    /// Returns an error if the CUDA copy operation fails.
    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                dst.len(),
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Device-to-host copy failed: {} bytes ({:?})",
                    dst.len(),
                    result
                )));
            }

            // With pageable host memory, cuMemcpyDtoHAsync blocks the host until
            // the copy completes. However, we still need to synchronize the stream
            // to ensure all prior GPU kernels have finished producing the data.
            let _ = client.stream.synchronize();
        }
        Ok(())
    }

    /// Record an event on the compute stream.
    fn record_compute_event(device: &Self::Device) -> crate::error::Result<u64> {
        let client = get_or_create_client(device);
        client
            .record_event_on_compute()
            .map_err(|e| crate::error::Error::Backend(format!("Event record failed: {}", e)))
    }

    /// Pipelined D2H copy: copy stream waits on the provided event, copies,
    /// and syncs only the copy stream. Compute stream continues concurrently.
    fn copy_from_device_pipelined(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
        event: u64,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device);

        unsafe {
            // 1. Copy stream waits for event (waits for argmax to finish)
            client.copy_stream_wait_event(event).map_err(|e| {
                client.destroy_event(event);
                crate::error::Error::Backend(format!("Stream wait event failed: {}", e))
            })?;

            // 2. Launch D2H copy on copy stream
            let result = cudarc::driver::sys::cuMemcpyDtoHAsync_v2(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                dst.len(),
                client.copy_stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                client.destroy_event(event);
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Pipelined D2H copy failed: {} bytes ({:?})",
                    dst.len(),
                    result
                )));
            }

            // 3. Sync ONLY the copy stream (compute stream keeps running)
            let _ = client.copy_stream.synchronize();

            client.destroy_event(event);
        }
        Ok(())
    }

    /// Copy data within device memory.
    ///
    /// Returns an error if the CUDA copy operation fails.
    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device);

        unsafe {
            let result = cudarc::driver::sys::cuMemcpyDtoDAsync_v2(
                dst,
                src,
                size_bytes,
                client.stream.cu_stream(),
            );

            if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Device-to-device copy failed: {} bytes ({:?})",
                    size_bytes, result
                )));
            }
        }
        Ok(())
    }

    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if src_handle == 0 || dst_handle == 0 || shape.is_empty() {
            return Ok(());
        }

        let numel: usize = shape.iter().product();
        if numel == 0 {
            return Ok(());
        }

        let ndim = shape.len();
        let client = get_or_create_client(device);

        // Shape and strides are passed as kernel arguments (by value), not device
        // memory pointers.  This is critical for CUDA graph capture compatibility:
        // H2D copies of temporary host data create graph memcpy nodes that re-read
        // from stale host addresses on replay, causing CUDA_ERROR_ILLEGAL_ADDRESS.
        unsafe {
            let kernel_result = kernels::launch_strided_copy(
                &client.context,
                &client.stream,
                device.index,
                src_handle,
                dst_handle,
                shape,
                strides,
                numel,
                ndim,
                elem_size,
                src_byte_offset,
            );

            if let Err(e) = kernel_result {
                return Err(crate::error::Error::Backend(format!(
                    "[numr::cuda] Strided copy kernel failed: {} bytes ({} elements × {} bytes/elem) from {} to {} on device {}: {:?}",
                    numel * elem_size,
                    numel,
                    elem_size,
                    src_handle,
                    dst_handle,
                    device.index,
                    e
                )));
            }
        }
        Ok(())
    }

    fn default_device() -> Self::Device {
        CudaDevice::new(0)
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        get_or_create_client(device)
    }

    fn raw_handle(client: &Self::Client) -> &Self::RawHandle {
        &client.raw_handle
    }
}

// ============================================================================
// Arena-enabled graph capture
// ============================================================================

impl CudaRuntime {
    /// Capture a CUDA graph with a dedicated bump-pointer arena for all
    /// graph-internal intermediate allocations.
    ///
    /// # Why this exists
    ///
    /// During normal graph capture (`capture_graph_into`), intermediate tensors
    /// allocated inside the closure via `cuMemAllocAsync` produce graph-managed
    /// memory that is subject to `AUTO_FREE_ON_LAUNCH`.  On the **first** replay
    /// `cuGraphLaunch` frees those addresses; on the **second** replay the
    /// graph's kernel nodes dereference the now-freed addresses and the driver
    /// raises `CUDA_ERROR_ILLEGAL_ADDRESS`.
    ///
    /// This variant pre-allocates an `arena_bytes`-sized device buffer
    /// **outside** the capture region and redirects every frozen `allocate()`
    /// call into a bump-pointer offset inside that buffer.  Because the buffer
    /// was not allocated during capture, it is NOT subject to
    /// `AUTO_FREE_ON_LAUNCH`; its address is stable across all replays for the
    /// lifetime of the returned `CapturedGraph`.
    ///
    /// # Arguments
    ///
    /// * `client` — the CUDA client to capture on.
    /// * `inputs` / `outputs` — tensors whose device addresses are passed to
    ///   the closure and encoded in the graph (same semantics as
    ///   `capture_graph_into`).
    /// * `arena_bytes` — size in bytes of the pre-allocated scratch arena.
    ///   Must be large enough to hold all intermediate tensors created inside
    ///   `f`.  Returns `Err(OutOfMemory)` inside the closure if the arena is
    ///   exhausted; the graph is not produced in that case.
    /// * `f` — the closure to capture.  Same contract as `capture_graph_into`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The arena device allocation fails (device OOM).
    /// - The closure returns an error (including arena OOM for intermediates).
    /// - The CUDA stream capture API fails.
    ///
    /// # Drop ordering
    ///
    /// The returned `CapturedGraph` holds the arena tensor in its `arena` field,
    /// which drops AFTER `graph`, `inputs`, and `outputs` (declaration order).
    /// This guarantees the arena buffer outlives the compiled graph handle.
    pub fn capture_graph_into_with_arena<F>(
        client: &CudaClient,
        inputs: &[&crate::tensor::Tensor<CudaRuntime>],
        outputs: &[&crate::tensor::Tensor<CudaRuntime>],
        arena_bytes: usize,
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<CudaRuntime>>
    where
        F: FnOnce(&CudaClient) -> crate::error::Result<()>,
    {
        use crate::dtype::DType;
        use crate::tensor::Tensor;
        use cudarc::driver::sys::CUstreamCaptureMode;

        // Allocate the arena buffer OUTSIDE the capture region so its address
        // is NOT baked into graph-managed memory and is NOT subject to
        // AUTO_FREE_ON_LAUNCH.  We use F32 storage (4 bytes/element) as a
        // neutral dtype; the arena is only accessed as raw bytes by the
        // bump-pointer logic.
        let arena_elems = arena_bytes.div_ceil(std::mem::size_of::<f32>());
        let arena_tensor =
            Tensor::<CudaRuntime>::try_empty(&[arena_elems], DType::F32, &client.device).map_err(
                |e| {
                    crate::error::Error::Backend(format!(
                        "capture_graph_into_with_arena: arena allocation failed \
                         ({arena_bytes} bytes): {e}"
                    ))
                },
            )?;
        let arena_ptr = arena_tensor.ptr();

        // Install the arena so freeze-time allocations go into it. Fails if a
        // capture is already in progress on this client (arena already present);
        // `arena_tensor` is dropped here, freeing its buffer.
        client.allocator.install_arena(arena_ptr, arena_bytes)?;

        // Clone each I/O tensor (cheap Arc bump).
        let owned_inputs: Vec<Tensor<CudaRuntime>> = inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<Tensor<CudaRuntime>> =
            outputs.iter().map(|t| (*t).clone()).collect();

        // Freeze: allocations now serve from the arena.
        client.allocator.freeze();

        // Begin stream capture.
        client
            .stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;

        // Execute the closure — ops are recorded into the graph.
        let closure_result = f(client);

        // End capture — MUST happen even if the closure failed.
        //
        // With the arena approach, ALL intermediate allocations go into the
        // pre-allocated bump-pointer buffer (no `cuMemAllocAsync` inside the
        // capture region), so there are normally NO graph-managed allocation
        // nodes for AUTO_FREE_ON_LAUNCH to act on. We deliberately pass it
        // anyway: cudarc's safe `end_capture` requires a flag from
        // `CUgraphInstantiate_flags` (which has no zero/NONE variant), and
        // AUTO_FREE_ON_LAUNCH is the correct, defensive choice — a no-op when
        // there are no alloc nodes, and the safe behavior (free on replay) if a
        // future op ever does allocate inside the capture region.
        let flags = cudarc::driver::sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
        let graph_result = client.stream.end_capture(flags);

        // Restore the allocator (clears the arena bookkeeping, resets frozen).
        client.allocator.unfreeze();

        // Propagate closure error after restoring stream/allocator state.
        closure_result?;

        // Propagate capture error.
        let graph_opt = graph_result?;
        let cudarc_graph = graph_opt.ok_or_else(|| {
            crate::error::Error::Backend(
                "CUDA graph capture (with_arena) produced no operations — \
                 closure recorded nothing"
                    .into(),
            )
        })?;

        Ok(crate::runtime::CapturedGraph::new_with_arena(
            super::CudaGraph::new(cudarc_graph),
            owned_inputs,
            owned_outputs,
            arena_tensor,
        ))
    }
}

/// Get the default CUDA device (device 0)
pub fn cuda_device() -> CudaDevice {
    CudaDevice::new(0)
}

/// Get a specific CUDA device by ID
pub fn cuda_device_id(device_id: usize) -> CudaDevice {
    CudaDevice::new(device_id)
}

/// Check if CUDA is available on this system
pub fn is_cuda_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = CudaDevice::new(0);
        let _client = get_or_create_client(&device);
    })
    .is_ok()
}

#[cfg(test)]
mod tests {
    use crate::dtype::DType;
    use crate::ops::BinaryOps;
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaDevice, CudaRuntime};
    use crate::runtime::traits::client::RuntimeClient;
    use crate::tensor::Tensor;

    /// Full `capture_graph_into` integration test using the destination-passing
    /// `add_into` primitive.
    ///
    /// Captures `c = a + b` into a graph where `a`, `b`, `c` are all allocated
    /// OUTSIDE the closure (so their device addresses are stable and not subject
    /// to `AUTO_FREE_ON_LAUNCH`), then verifies the captured graph replays
    /// correctly after the inputs are overwritten in place.
    #[ignore = "requires a live CUDA GPU"]
    #[test]
    fn test_capture_graph_into_add() {
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Inputs and output allocated OUTSIDE the closure.
        let a =
            Tensor::<CudaRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let b = Tensor::<CudaRuntime>::try_from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[4], &device)
            .unwrap();
        let c = Tensor::<CudaRuntime>::try_zeros(&[4], DType::F32, &device).unwrap();

        // Capture: c = a + b (destination-passing, no allocation inside).
        let captured = CudaRuntime::capture_graph_into(&client, &[&a, &b], &[&c], |cc| {
            cc.add_into(&c, &a, &b)
        })
        .expect("capture_graph_into failed");

        // First replay.
        captured.launch().expect("launch failed");
        client.synchronize();
        assert_eq!(c.to_vec::<f32>(), vec![11.0, 22.0, 33.0, 44.0]);

        // Overwrite `a` in place (stable address) and replay again — the graph
        // must pick up the new input values since it reads from the same buffer.
        CudaRuntime::copy_to_device(
            bytemuck::cast_slice(&[5.0f32, 6.0, 7.0, 8.0]),
            a.ptr(),
            &device,
        )
        .expect("H2D copy failed");
        captured.launch().expect("relaunch failed");
        client.synchronize();
        assert_eq!(c.to_vec::<f32>(), vec![15.0, 26.0, 37.0, 48.0]);

        // Drop the captured graph and confirm a subsequent normal allocation
        // does not hit CUDA_ERROR_ILLEGAL_ADDRESS (clean teardown).
        drop(captured);
        let fresh = Tensor::<CudaRuntime>::try_zeros(&[4], DType::F32, &device).unwrap();
        client.synchronize();
        assert_eq!(fresh.to_vec::<f32>(), vec![0.0, 0.0, 0.0, 0.0]);
    }
}
