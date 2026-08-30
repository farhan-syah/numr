//! Gather whole slices along one dimension, addressed by an index vector.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::index;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;
use wgpu::BufferUsages;

pub(crate) fn native_index_select(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let indices_i32 = ensure_i32_indices(client, indices)?;
    let indices_contig = ensure_contiguous(&indices_i32)?;

    // Compute output shape
    let index_len = indices.numel();
    let mut out_shape = shape.to_vec();
    out_shape[dim] = index_len;

    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let total_output = outer_size * index_len * inner_size;

    // Nothing to select: the source or the index set is empty, so the output
    // carries a zero dimension and `get_tensor_buffer` has no buffer to return
    // for it. A non-empty index set against an empty source is NOT caught here
    // — it is out of bounds, and the validation dispatch below reports it.
    if total_output == 0 {
        return alloc_output(client, &out_shape, dtype);
    }

    let indices_buf = get_tensor_buffer(&indices_contig)?;

    // Validate indices on GPU (only costs copying 4 bytes back)
    if index_len > 0 {
        let error_count_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("validate_indices_error_count"),
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Initialize error count to 0
        client.queue.write_buffer(&error_count_buffer, 0, &[0u8; 4]);

        let validate_params = ValidateIndicesParams {
            index_len: index_len as u32,
            dim_size: dim_size as u32,
            _pad0: 0,
            _pad1: 0,
        };
        let validate_params_buf = create_params_buffer(client, &validate_params);

        index::launch_validate_indices(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &indices_buf,
            &error_count_buffer,
            &validate_params_buf,
            index_len,
        )?;

        // Read back error count (only 4 bytes)
        let error_count = read_u32_from_buffer(client, &error_count_buffer)?;
        if error_count > 0 {
            return Err(Error::IndexOutOfBounds {
                index: 0, // We don't know which specific index failed
                size: dim_size,
            });
        }
    }

    let out = alloc_output(client, &out_shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Never restore a `.max(1)` on these: the `total_output == 0` guard above
    // already rules a zero out, and a clamp would tell the shader about a row the
    // allocation does not contain.
    let params = IndexSelectParams {
        outer_size: outer_size as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size as u32,
        index_len: index_len as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_index_select(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        total_output,
        dtype,
    )?;

    Ok(out)
}
