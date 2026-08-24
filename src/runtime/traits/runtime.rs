//! Core trait for compute backends

/// Core trait for compute backends
///
/// `Runtime` abstracts over different compute devices (CPU, GPU, etc.).
/// It uses static dispatch via generics for zero-cost abstraction.
///
/// # Associated Types
///
/// - `Device`: Identifies a specific compute unit (e.g., GPU 0, GPU 1)
/// - `Client`: Handles operation dispatch and synchronization
/// - `Allocator`: Memory management with optional freeze support
/// - `Graph`: Captured computation sequence for replay (CUDA Graphs, etc.)
/// - `RawHandle`: Escape hatch for custom kernel launching
///
/// # Example
///
/// ```ignore
/// let device = CpuRuntime::default_device();
/// let ptr = CpuRuntime::allocate(1024, &device)?;
/// // ... use memory ...
/// CpuRuntime::deallocate(ptr, 1024, &device);
/// ```
pub trait Runtime: Clone + Send + Sync + 'static {
    /// Device identifier type
    type Device: super::Device;

    /// Client for dispatching operations
    type Client: super::RuntimeClient<Self>;

    /// Memory allocator type
    type Allocator: crate::runtime::Allocator;

    /// Captured computation graph for replay
    ///
    /// For CPU/WebGPU: `NoOpGraph` (operations execute eagerly, launch is no-op)
    /// For CUDA: `CudaGraph` wrapping cudarc's graph types
    type Graph: crate::runtime::Graph;

    /// Raw handle for custom kernel launching (escape hatch)
    ///
    /// For CPU: `()` (no raw handle needed)
    /// For CUDA: Access to CudaDevice/CudaStream
    /// For WGPU: Access to wgpu::Device/Queue
    type RawHandle: Send + Sync;

    /// Data type enum for tensor elements.
    ///
    /// numr runtimes use `numr::DType`. Downstream runtimes (e.g. boostr)
    /// can specify their own dtype enum with quantized variants.
    type DType: crate::dtype::DataType;

    /// Human-readable name of this runtime
    fn name() -> &'static str;

    /// Does this backend support graph capture (e.g., CUDA Graphs)?
    ///
    /// Check this BEFORE calling `capture_graph_into` to skip capture on
    /// non-capture backends.
    fn supports_graph_capture() -> bool {
        false
    }

    /// Destination-passing graph capture.
    ///
    /// Captures the operations performed by `f` into a replayable
    /// [`CapturedGraph`][crate::runtime::CapturedGraph], which bundles the
    /// compiled graph with `Arc` clones of the caller-supplied `inputs` and
    /// `outputs` tensors. The bundled tensors prevent their device memory from
    /// being freed while the graph is alive.
    ///
    /// # Contract
    ///
    /// - `outputs` must be allocated **before** calling this method (outside
    ///   the closure). The graph encodes their device addresses at capture
    ///   time; they must not move.
    /// - The closure `f` must write results into the output tensors in-place.
    ///   It must not allocate new output buffers (those would be subject to
    ///   `AUTO_FREE_ON_LAUNCH` and freed after the first replay).
    /// - `inputs` and `outputs` are passed as shared slices; the method clones
    ///   each tensor (cheap `Arc` bump) to hold in the `CapturedGraph`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::BackendLimitation`][crate::error::Error::BackendLimitation]
    /// on backends that do not support graph capture (CPU, WebGPU).
    ///
    /// # Example (CUDA)
    ///
    /// ```ignore
    /// let a = Tensor::<CudaRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], device).unwrap();
    /// let b = Tensor::<CudaRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], device).unwrap();
    /// let c = Tensor::<CudaRuntime>::try_zeros(&[2], DType::F32, device).unwrap();
    ///
    /// let captured = CudaRuntime::capture_graph_into(
    ///     client, &[&a, &b], &[&c],
    ///     |cc| { cc.copy_within_device(cc.add(&a, &b)?.ptr(), c.ptr(), ...) },
    /// )?;
    ///
    /// // First replay with original data
    /// captured.launch()?;
    /// // Update inputs, replay again
    /// CudaRuntime::copy_to_device(..., a.ptr(), device)?;
    /// captured.launch()?;
    /// ```
    fn capture_graph_into<F>(
        client: &Self::Client,
        inputs: &[&crate::tensor::Tensor<Self>],
        outputs: &[&crate::tensor::Tensor<Self>],
        f: F,
    ) -> crate::error::Result<crate::runtime::CapturedGraph<Self>>
    where
        F: FnOnce(&Self::Client) -> crate::error::Result<()>,
        Self: Sized,
    {
        let _ = (client, inputs, outputs, f);
        Err(crate::error::Error::BackendLimitation {
            backend: Self::name(),
            operation: "capture_graph_into",
            reason: "this backend does not support CUDA-style graph capture".into(),
        })
    }

    /// Allocate device memory
    ///
    /// Returns a device pointer (u64) that can be used for operations.
    /// Returns `Err(OutOfMemory)` if allocation fails.
    fn allocate(size_bytes: usize, device: &Self::Device) -> crate::error::Result<u64>;

    /// Deallocate device memory
    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device);

    /// Copy data from host to device
    ///
    /// Returns an error if the transfer fails.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> crate::error::Result<()>;

    /// Copy data from device to host
    ///
    /// Returns an error if the transfer fails.
    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
    ) -> crate::error::Result<()>;

    /// Copy data within device (device to device)
    ///
    /// Returns an error if the transfer fails.
    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()>;

    /// Copy strided data to a contiguous buffer
    ///
    /// This is the proper way to make a non-contiguous tensor contiguous.
    /// Each backend implements this according to its memory model:
    /// - CPU/CUDA: can use pointer arithmetic (src_handle + byte_offset)
    /// - WGPU: must use compute shader with buffer binding + offset uniform
    ///
    /// # Parameters
    /// - `src_handle`: Source buffer handle (pointer for CUDA, buffer ID for WGPU)
    /// - `src_byte_offset`: Byte offset into source buffer
    /// - `dst_handle`: Destination buffer handle
    /// - `shape`: Shape of the tensor
    /// - `strides`: Strides of the source tensor (in elements, not bytes)
    /// - `elem_size`: Size of each element in bytes
    fn copy_strided(
        src_handle: u64,
        src_byte_offset: usize,
        dst_handle: u64,
        shape: &[usize],
        strides: &[isize],
        elem_size: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()>;

    /// Record an event on the compute stream. Returns an opaque handle.
    /// On non-CUDA backends, returns 0 (no-op).
    fn record_compute_event(_device: &Self::Device) -> crate::error::Result<u64> {
        Ok(0)
    }

    /// Copy data from device to host using a dedicated copy stream,
    /// synchronized via a previously recorded event.
    ///
    /// On CUDA: copy stream waits on the event, performs D2H, syncs only copy stream.
    /// The compute stream continues running concurrently.
    ///
    /// Default: ignores event, falls back to `copy_from_device`.
    fn copy_from_device_pipelined(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
        event: u64,
    ) -> crate::error::Result<()> {
        let _ = event;
        Self::copy_from_device(src, dst, device)
    }

    /// Get the default device
    fn default_device() -> Self::Device;

    /// Get the default client for a device
    fn default_client(device: &Self::Device) -> Self::Client;

    /// Get the raw handle from a client (escape hatch for custom kernels)
    fn raw_handle(client: &Self::Client) -> &Self::RawHandle;
}
