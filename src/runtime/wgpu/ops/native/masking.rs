//! Masking operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::index;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;
use wgpu::BufferUsages;

pub(crate) fn native_masked_fill(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    mask: &Tensor<WgpuRuntime>,
    value: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let numel = a.numel();

    // Mask must be U32 on WebGPU (no U8 support)
    if mask.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to match tensor shape (same as CPU behavior)
    let mask_broadcast = mask
        .broadcast_to(a.shape())
        .map_err(|_| Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: mask.shape().to_vec(),
        })?;

    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(&mask_broadcast)?;

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let mask_buf = get_tensor_buffer(&mask_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = MaskedFillParams {
        numel: numel as u32,
        fill_value: value as f32,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_masked_fill(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &mask_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_embedding_lookup(
    client: &WgpuClient,
    embeddings: &Tensor<WgpuRuntime>,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = embeddings.dtype();
    let emb_shape = embeddings.shape();

    // Validate embeddings is 2D
    if emb_shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: emb_shape.to_vec(),
        });
    }

    // Normalize indices dtype to I32 for WebGPU shaders.
    let indices_i32 = ensure_i32_indices(client, indices)?;

    // Only F32, I32, U32 are supported on WebGPU natively
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "embedding_lookup",
        });
    }

    let vocab_size = emb_shape[0];
    let embedding_dim = emb_shape[1];
    let num_indices = indices_i32.numel();

    // Output shape: indices.shape() + [embedding_dim]
    let mut out_shape = indices_i32.shape().to_vec();
    out_shape.push(embedding_dim);

    let emb_contig = ensure_contiguous(embeddings)?;
    let idx_contig = ensure_contiguous(&indices_i32)?;
    let out = alloc_output(client, &out_shape, dtype)?;

    let emb_buf = get_tensor_buffer(&emb_contig)?;
    let idx_buf = get_tensor_buffer(&idx_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = EmbeddingLookupParams {
        num_indices: num_indices as u32,
        vocab_size: vocab_size as u32,
        embedding_dim: embedding_dim as u32,
        _pad0: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_embedding_lookup(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &emb_buf,
        &idx_buf,
        &out_buf,
        &params_buf,
        num_indices,
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_masked_select(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    mask: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let numel = a.numel();

    // Mask must be U32 on WebGPU
    if mask.dtype() != DType::U32 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U32,
            rhs: mask.dtype(),
        });
    }

    // Broadcast mask to match tensor shape (same as CPU behavior)
    let mask_broadcast = mask
        .broadcast_to(a.shape())
        .map_err(|_| Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: mask.shape().to_vec(),
        })?;

    let a_contig = ensure_contiguous(a)?;
    let mask_contig = ensure_contiguous(&mask_broadcast)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let mask_buf = get_tensor_buffer(&mask_contig)?;

    // Phase 1: Count the number of selected elements
    // Need an atomic buffer for count result
    let count_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("masked_count_result"),
        size: 4,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // Initialize count to 0
    client.queue.write_buffer(&count_buffer, 0, &[0u8; 4]);

    let count_params = MaskedCountParams {
        numel: numel as u32,
    };
    let count_params_buf = create_params_buffer(client, &count_params);

    index::launch_masked_count(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &mask_buf,
        &count_buffer,
        &count_params_buf,
        numel,
        dtype,
    )?;

    // Read count back to CPU (need to synchronize)
    let staging_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("count_staging"),
        size: 4,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_count"),
        });
    encoder.copy_buffer_to_buffer(&count_buffer, 0, &staging_buffer, 0, 4);
    client.queue.submit(std::iter::once(encoder.finish()));

    // Wait for GPU and read the count
    let slice = staging_buffer.slice(..);
    let (sender, receiver) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).unwrap();
    });
    let _ = client.wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });
    receiver.recv().unwrap().unwrap();

    let count = {
        let data = slice.get_mapped_range();
        u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize
    };
    drop(staging_buffer);

    if count == 0 {
        // Return empty tensor
        return Tensor::try_empty(&[0], dtype, client.device());
    }

    // Phase 2: Compute prefix sum
    let prefix_sum_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prefix_sum"),
        size: (numel * 4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let prefix_params = MaskedCountParams {
        numel: numel as u32,
    };
    let prefix_params_buf = create_params_buffer(client, &prefix_params);

    index::launch_masked_prefix_sum(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &mask_buf,
        &prefix_sum_buffer,
        &prefix_params_buf,
        numel,
        dtype,
    )?;

    // Phase 3: Gather selected elements
    let out = alloc_output(client, &[count], dtype)?;
    let out_buf = get_tensor_buffer(&out)?;

    let select_params = MaskedSelectParams {
        numel: numel as u32,
    };
    let select_params_buf = create_params_buffer(client, &select_params);

    index::launch_masked_select(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &mask_buf,
        &prefix_sum_buffer,
        &out_buf,
        &select_params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}
