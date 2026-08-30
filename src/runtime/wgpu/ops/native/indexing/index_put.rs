//! Write whole slices along one dimension, addressed by an index vector.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::index;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;
use wgpu::BufferUsages;

pub(crate) fn native_index_put(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    indices: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
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

    // Src dtype must match input
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let indices_i32 = ensure_i32_indices(client, indices)?;
    let indices_contig = ensure_contiguous(&indices_i32)?;
    let src_contig = ensure_contiguous(src)?;

    let index_len = indices.numel();
    let outer_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner_size: usize = shape[dim + 1..].iter().product();
    let total_src = outer_size * index_len * inner_size;

    // An empty destination has no buffer to bind and no element any index could
    // address, so the empty result is returned before anything is dispatched.
    // An empty index set is handled further down: the destination is still
    // copied to the output, only the put dispatch is skipped.
    if a.numel() == 0 {
        return alloc_output(client, shape, dtype);
    }

    // Allocate output and copy input first
    let out = alloc_output(client, shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // First copy input to output
    let copy_params = CopyParams {
        numel: a.numel() as u32,
    };
    let copy_params_buf = create_params_buffer(client, &copy_params);
    index::launch_copy(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &copy_params_buf,
        a.numel(),
        dtype,
    )?;

    // An empty index set writes nothing, so the copy above is already the whole
    // result. `src` is the zero-byte allocation here, and `get_tensor_buffer`
    // has no buffer to return for it, so the dispatch must not be reached.
    if total_src == 0 {
        return Ok(out);
    }

    // Validated after the copy: an empty index set returns above, and
    // `get_tensor_buffer` has no buffer to hand back for a zero-byte index
    // allocation.
    let indices_buf = get_tensor_buffer(&indices_contig)?;

    // Validate indices on GPU (only costs copying 4 bytes back)
    {
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

    let src_buf = get_tensor_buffer(&src_contig)?;

    // Then apply index_put
    // Never restore a `.max(1)` on these: the `total_src == 0` guard above already
    // rules a zero out, and a clamp would tell the shader about a row the
    // allocation does not contain.
    let params = IndexSelectParams {
        outer_size: outer_size as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size as u32,
        index_len: index_len as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_index_put(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &indices_buf,
        &src_buf,
        &out_buf,
        &params_buf,
        total_src,
        dtype,
    )?;

    Ok(out)
}
