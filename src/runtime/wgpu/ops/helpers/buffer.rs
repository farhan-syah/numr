//! GPU buffer creation, tensor buffer lookup and readback.
//!
//! Every WebGPU dispatch needs its uniform and storage buffers built here, and
//! the few ops whose output length is decided on the GPU read a single u32 back
//! through `read_u32_from_buffer`.

use wgpu::BufferUsages;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Create a uniform buffer with the given data.
pub(crate) fn create_params_buffer<T: bytemuck::Pod>(
    client: &WgpuClient,
    data: &T,
) -> wgpu::Buffer {
    let buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&buffer, 0, bytemuck::bytes_of(data));
    buffer
}

/// Create a storage buffer with the given data.
pub(crate) fn create_storage_buffer<T: bytemuck::Pod>(
    client: &WgpuClient,
    data: &[T],
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    client
        .wgpu_device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage_buffer"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
}

/// Get the wgpu buffer from a tensor's storage pointer.
pub(crate) fn get_tensor_buffer(
    tensor: &Tensor<WgpuRuntime>,
) -> Result<std::sync::Arc<wgpu::Buffer>> {
    let ptr = tensor.ptr();
    get_buffer(ptr).ok_or_else(|| Error::Internal("Buffer not found in registry".to_string()))
}

/// Allocate output tensor with given shape and dtype.
pub(crate) fn alloc_output(
    client: &WgpuClient,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    Tensor::empty(shape, dtype, client.device())
}

/// Read a single u32 value from a GPU buffer (synchronous)
pub(crate) fn read_u32_from_buffer(client: &WgpuClient, buffer: &wgpu::Buffer) -> Result<u32> {
    let staging_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_u32"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 4);
    client.queue.submit(std::iter::once(encoder.finish()));

    // Block until GPU work is done
    let (tx, rx) = std::sync::mpsc::channel();
    staging_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    let _ = client.wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });
    rx.recv()
        .map_err(|_| Error::Internal("Failed to read from GPU buffer".to_string()))?
        .map_err(|e| Error::Internal(format!("Buffer map failed: {:?}", e)))?;

    let data = staging_buffer.slice(..).get_mapped_range();
    let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    staging_buffer.unmap();

    Ok(value)
}
