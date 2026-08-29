//! WebGPU runtime implementation

use super::cache::get_or_create_client;

/// Convert WgpuError to numr Error for use in Runtime trait methods.
fn wgpu_err(e: super::device::WgpuError) -> crate::error::Error {
    crate::error::Error::Backend(format!("WebGPU client error: {e}"))
}
use super::client::WgpuClient;
use super::device::WgpuDevice;
use super::shaders;
use crate::runtime::{Allocator, NoOpGraph, Runtime, RuntimeClient};
use std::time::Duration;

/// WebGPU Runtime adapter
///
/// Implements the generic Runtime trait for WebGPU backend.
/// Provides cross-platform GPU acceleration.
#[derive(Clone, Debug, Default)]
pub struct WgpuRuntime;

impl Runtime for WgpuRuntime {
    type Device = WgpuDevice;
    type Client = WgpuClient;
    type Allocator = super::WgpuAllocator;
    type Graph = NoOpGraph;
    type RawHandle = super::WgpuRawHandle;
    type DType = crate::dtype::DType;

    fn name() -> &'static str {
        "wgpu"
    }

    fn supports_graph_capture() -> bool {
        false // WebGPU doesn't have CUDA-style graph capture
    }

    /// Allocate GPU memory (storage buffer).
    ///
    /// Returns `Err(OutOfMemory)` if buffer creation fails.
    fn allocate(size_bytes: usize, device: &Self::Device) -> crate::error::Result<u64> {
        if size_bytes == 0 {
            return Ok(0);
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;
        client.allocator.allocate(size_bytes)
    }

    fn deallocate(ptr: u64, size_bytes: usize, device: &Self::Device) {
        if ptr == 0 {
            return;
        }

        if let Ok(client) = get_or_create_client(device) {
            client.allocator.deallocate(ptr, size_bytes);
        }
    }

    /// Copy data from host to device.
    ///
    /// Returns an error if the buffer doesn't exist.
    fn copy_to_device(src: &[u8], dst: u64, device: &Self::Device) -> crate::error::Result<()> {
        if src.is_empty() || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        // Get the buffer from registry
        let buffer = super::client::get_buffer(dst).ok_or_else(|| {
            crate::error::Error::Backend("Buffer not found for copy_to_device".into())
        })?;

        // WebGPU copies whole 4-byte words, and `allocate` already rounded the
        // buffer up to that. A payload whose length is not a multiple of 4 — a
        // run of 18-byte GGUF blocks, a TCF payload with an odd super-block
        // count — is written as its aligned prefix plus one zero-padded tail
        // word, so it reaches the device intact instead of tripping wgpu's
        // copy-alignment validation.
        let tail = src.len() % 4;
        if tail == 0 {
            client.queue.write_buffer(&buffer, 0, src);
        } else {
            let head = src.len() - tail;
            if head > 0 {
                client.queue.write_buffer(&buffer, 0, &src[..head]);
            }
            let mut word = [0u8; 4];
            word[..tail].copy_from_slice(&src[head..]);
            client.queue.write_buffer(&buffer, head as u64, &word);
        }

        // Ensure write is complete
        client.synchronize();
        Ok(())
    }

    /// Copy data from device to host.
    ///
    /// Returns an error if the buffer doesn't exist or read fails.
    fn copy_from_device(
        src: u64,
        dst: &mut [u8],
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if dst.is_empty() || src == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        // Get the source buffer from registry
        let buffer = super::client::get_buffer(src).ok_or_else(|| {
            crate::error::Error::Backend("Buffer not found for copy_from_device".into())
        })?;

        // `copy_buffer_to_buffer` and `map_async` both reject a size that is
        // not a multiple of 4. A payload whose length is not a multiple of 4 —
        // a run of 18-byte GGUF blocks, a TCF payload with an odd super-block
        // count — is staged and mapped at its rounded-up length. `allocate`
        // rounded the source buffer up the same way, so the extra bytes are in
        // bounds, and only the first `dst.len()` bytes reach the caller.
        let aligned_len = dst.len().div_ceil(4) * 4;

        // Create a staging buffer for readback
        let staging = client.create_staging_buffer("copy_staging", aligned_len as u64);

        // Copy from storage to staging
        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_from_device"),
                });
        encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, aligned_len as u64);
        client.submit_and_wait(encoder);

        // Read from staging buffer
        let slice = staging.slice(..);

        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        client
            .wgpu_device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(60)),
            })
            .map_err(|e| {
                crate::error::Error::Backend(format!(
                    "GPU poll failed during copy_from_device: {e}"
                ))
            })?;

        // Check map_async result
        let map_result = receiver.recv().map_err(|_| {
            crate::error::Error::Backend(
                "map_async callback was not invoked during copy_from_device".into(),
            )
        })?;
        map_result.map_err(|e| {
            crate::error::Error::Backend(format!("map_async failed during copy_from_device: {e}"))
        })?;

        {
            let data = slice.get_mapped_range();
            dst.copy_from_slice(&data[..dst.len()]);
        }

        staging.unmap();
        Ok(())
    }

    /// Copy data within device memory.
    ///
    /// Returns an error if either buffer doesn't exist.
    fn copy_within_device(
        src: u64,
        dst: u64,
        size_bytes: usize,
        device: &Self::Device,
    ) -> crate::error::Result<()> {
        if size_bytes == 0 || src == 0 || dst == 0 {
            return Ok(());
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;

        let src_buffer = super::client::get_buffer(src)
            .ok_or_else(|| crate::error::Error::Backend("Source buffer not found".into()))?;
        let dst_buffer = super::client::get_buffer(dst)
            .ok_or_else(|| crate::error::Error::Backend("Destination buffer not found".into()))?;

        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("copy_within_device"),
                });
        encoder.copy_buffer_to_buffer(&src_buffer, 0, &dst_buffer, 0, size_bytes as u64);
        client.submit_and_wait(encoder);
        Ok(())
    }

    /// Copy strided data to a contiguous buffer using a compute shader.
    ///
    /// Unlike CPU/CUDA which can use pointer arithmetic, WGPU buffers are opaque
    /// handles that don't support arithmetic. This method uses a compute shader
    /// to read elements at strided offsets and write them contiguously.
    ///
    /// Returns an error if buffers don't exist or shader execution fails.
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

        // Validate rank <= 8 (shader uses fixed-size arrays)
        if shape.len() > 8 {
            return Err(crate::error::Error::BackendLimitation {
                backend: "wgpu",
                operation: "copy_strided",
                reason: format!("rank {} exceeds maximum of 8 dimensions", shape.len()),
            });
        }

        // Validate elem_size is a multiple of 4 bytes (shader uses array<u32>)
        if !elem_size.is_multiple_of(4) {
            return Err(crate::error::Error::BackendLimitation {
                backend: "wgpu",
                operation: "copy_strided",
                reason: format!(
                    "element size {} is not a multiple of 4 bytes; sub-4-byte dtypes (F16, Bool) require byte-level addressing not supported by the u32-based copy shader",
                    elem_size
                ),
            });
        }

        // Validate byte offset is 4-byte aligned
        if !src_byte_offset.is_multiple_of(4) {
            return Err(crate::error::Error::BackendLimitation {
                backend: "wgpu",
                operation: "copy_strided",
                reason: format!(
                    "source byte offset {} is not 4-byte aligned",
                    src_byte_offset
                ),
            });
        }

        let client = get_or_create_client(device).map_err(wgpu_err)?;
        let src_buffer = super::client::get_buffer(src_handle).ok_or_else(|| {
            crate::error::Error::Backend("Source buffer not found for copy_strided".into())
        })?;
        let dst_buffer = super::client::get_buffer(dst_handle).ok_or_else(|| {
            crate::error::Error::Backend("Destination buffer not found for copy_strided".into())
        })?;

        // Create shader module
        let module = client
            .wgpu_device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("strided_copy"),
                source: wgpu::ShaderSource::Wgsl(shaders::copy::STRIDED_COPY_SHADER.into()),
            });

        // Create bind group layout (all storage buffers to avoid alignment issues)
        let bind_group_layout =
            client
                .wgpu_device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("strided_copy_layout"),
                    entries: &[
                        // src (read-only input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // dst (output)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // params (read-only input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        // Create pipeline
        let pipeline_layout =
            client
                .wgpu_device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("strided_copy_pipeline_layout"),
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let pipeline =
            client
                .wgpu_device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("strided_copy_pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &module,
                    entry_point: Some("strided_copy"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        // Prepare parameters
        // Element size in 4-byte units (WGSL array<u32>)
        // Safe: validated elem_size % 4 == 0 above
        let elem_size_units = (elem_size / 4) as u32;
        // Source offset in 4-byte units
        // Safe: validated src_byte_offset % 4 == 0 above
        let src_offset_units = (src_byte_offset / 4) as u32;

        // Pad shape and strides to 8 elements
        let mut shape_arr = [0u32; 8];
        let mut strides_arr = [0i32; 8];
        for (i, &s) in shape.iter().enumerate().take(8) {
            shape_arr[i] = s as u32;
        }
        for (i, &s) in strides.iter().enumerate().take(8) {
            strides_arr[i] = s as i32;
        }

        // Create params buffer (numel, ndim, elem_size_units, src_offset_units, shape[8], strides[8])
        // Total: 4 + 4 + 4 + 4 + 32 + 32 = 80 bytes
        let mut params_data = Vec::with_capacity(80);
        params_data.extend_from_slice(&(numel as u32).to_le_bytes());
        params_data.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        params_data.extend_from_slice(&elem_size_units.to_le_bytes());
        params_data.extend_from_slice(&src_offset_units.to_le_bytes());
        for s in shape_arr {
            params_data.extend_from_slice(&s.to_le_bytes());
        }
        for s in strides_arr {
            params_data.extend_from_slice(&s.to_le_bytes());
        }

        let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("strided_copy_params"),
            size: params_data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        client.queue.write_buffer(&params_buffer, 0, &params_data);

        // Create bind group
        let bind_group = client
            .wgpu_device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("strided_copy_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: dst_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Dispatch compute shader
        let workgroups = shaders::workgroup_count(numel);
        let mut encoder =
            client
                .wgpu_device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("strided_copy"),
                });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("strided_copy_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        client.submit_and_wait(encoder);
        Ok(())
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
        f(client)?;
        let owned_inputs: Vec<_> = inputs.iter().map(|t| (*t).clone()).collect();
        let owned_outputs: Vec<_> = outputs.iter().map(|t| (*t).clone()).collect();
        Ok(crate::runtime::CapturedGraph::new(
            crate::runtime::NoOpGraph,
            owned_inputs,
            owned_outputs,
        ))
    }

    fn default_device() -> Self::Device {
        WgpuDevice::new(0)
    }

    fn default_client(device: &Self::Device) -> Self::Client {
        get_or_create_client(device).expect("Failed to create WebGPU client")
    }

    fn raw_handle(client: &Self::Client) -> &Self::RawHandle {
        &client.raw_handle
    }
}

/// Get the default WebGPU device (first adapter)
pub fn wgpu_device() -> WgpuDevice {
    WgpuDevice::new(0)
}

/// Get a specific WebGPU device by adapter index
pub fn wgpu_device_id(index: usize) -> WgpuDevice {
    WgpuDevice::new(index)
}

/// Check if WebGPU is available on this system
pub fn is_wgpu_available() -> bool {
    super::device::query_adapter_info_blocking(0).is_ok()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// A length that is not a multiple of 4 must survive a device round trip.
    ///
    /// 18 bytes is one GGUF `Q4_0` block, the realistic case.
    #[test]
    fn test_copy_round_trip_unaligned_length() {
        let device = WgpuRuntime::default_device();
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let src: Vec<u8> = (0..18u8)
            .map(|i| i.wrapping_mul(7).wrapping_add(3))
            .collect();
        let ptr = WgpuRuntime::allocate(src.len(), &device).expect("allocation should succeed");

        WgpuRuntime::copy_to_device(&src, ptr, &device).expect("host to device copy");

        let mut dst = vec![0u8; src.len()];
        WgpuRuntime::copy_from_device(ptr, &mut dst, &device).expect("device to host copy");

        WgpuRuntime::deallocate(ptr, src.len(), &device);

        assert_eq!(dst, src, "unaligned round trip must be byte-exact");
    }
}
