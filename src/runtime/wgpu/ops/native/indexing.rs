//! Indexing operation implementations for WebGPU.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::{index, launch_slice_assign};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
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

    let params = IndexSelectParams {
        outer_size: outer_size.max(1) as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size.max(1) as u32,
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
        total_output.max(1),
        dtype,
    )?;

    Ok(out)
}

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

    // Allocate output and copy input first
    let out = alloc_output(client, shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let src_buf = get_tensor_buffer(&src_contig)?;
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

    // Then apply index_put
    let params = IndexSelectParams {
        outer_size: outer_size.max(1) as u32,
        dim_size: dim_size as u32,
        inner_size: inner_size.max(1) as u32,
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
        total_src.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_gather(
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

    if ndim > 4 {
        return Err(Error::Internal(
            "gather: WebGPU implementation supports max 4 dimensions".to_string(),
        ));
    }

    // Output shape is same as index shape
    let indices_i32 = ensure_i32_indices(client, indices)?;
    let out_shape = indices_i32.shape().to_vec();
    let total_elements = indices_i32.numel();

    let a_contig = ensure_contiguous(a)?;
    let indices_contig = ensure_contiguous(&indices_i32)?;

    let out = alloc_output(client, &out_shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Pack shape and strides into vec4<u32> format
    let input_strides = compute_contiguous_strides(shape);
    let output_strides = compute_contiguous_strides(&out_shape);

    let mut input_shape_arr = [1u32; 4];
    let mut input_strides_arr = [1u32; 4];
    let mut output_shape_arr = [1u32; 4];
    let mut output_strides_arr = [1u32; 4];

    for i in 0..ndim.min(4) {
        input_shape_arr[i] = shape[i] as u32;
        input_strides_arr[i] = input_strides[i] as u32;
    }
    for i in 0..out_shape.len().min(4) {
        output_shape_arr[i] = out_shape[i] as u32;
        output_strides_arr[i] = output_strides[i] as u32;
    }

    let params = GatherParams {
        ndim: ndim as u32,
        dim: dim as u32,
        total_elements: total_elements as u32,
        _padding: 0,
        input_shape: input_shape_arr,
        input_strides: input_strides_arr,
        output_shape: output_shape_arr,
        output_strides: output_strides_arr,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_gather(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        total_elements.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_scatter(
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

    if ndim > 4 {
        return Err(Error::Internal(
            "scatter: WebGPU implementation supports max 4 dimensions".to_string(),
        ));
    }

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

    let src_shape = src.shape();
    let src_total = src.numel();

    // Output is same shape as input
    let out = alloc_output(client, shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let src_buf = get_tensor_buffer(&src_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // First, copy input to output
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

    // Then scatter src values into output
    let output_strides = compute_contiguous_strides(shape);
    let src_strides = compute_contiguous_strides(src_shape);

    let mut output_shape_arr = [1u32; 4];
    let mut output_strides_arr = [1u32; 4];
    let mut src_shape_arr = [1u32; 4];
    let mut src_strides_arr = [1u32; 4];

    for i in 0..ndim.min(4) {
        output_shape_arr[i] = shape[i] as u32;
        output_strides_arr[i] = output_strides[i] as u32;
    }
    for i in 0..src_shape.len().min(4) {
        src_shape_arr[i] = src_shape[i] as u32;
        src_strides_arr[i] = src_strides[i] as u32;
    }

    let params = ScatterParams {
        ndim: ndim as u32,
        dim: dim as u32,
        src_total: src_total as u32,
        _padding: 0,
        output_shape: output_shape_arr,
        output_strides: output_strides_arr,
        src_shape: src_shape_arr,
        src_strides: src_strides_arr,
    };
    let params_buf = create_params_buffer(client, &params);

    index::launch_scatter(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &src_buf,
        &indices_buf,
        &out_buf,
        &params_buf,
        src_total.max(1),
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_slice_assign(
    client: &WgpuClient,
    dst: &Tensor<WgpuRuntime>,
    src: &Tensor<WgpuRuntime>,
    dim: usize,
    start: usize,
) -> Result<Tensor<WgpuRuntime>> {
    let ndim = dst.ndim();
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    if src.ndim() != ndim {
        return Err(Error::ShapeMismatch {
            expected: dst.shape().to_vec(),
            got: src.shape().to_vec(),
        });
    }
    for d in 0..ndim {
        if d != dim && src.shape()[d] != dst.shape()[d] {
            return Err(Error::ShapeMismatch {
                expected: dst.shape().to_vec(),
                got: src.shape().to_vec(),
            });
        }
    }

    let src_dim_size = src.shape()[dim];
    let dst_dim_size = dst.shape()[dim];
    if start + src_dim_size > dst_dim_size {
        return Err(Error::InvalidArgument {
            arg: "start",
            reason: format!(
                "start ({}) + src dim size ({}) exceeds dst dim size ({})",
                start, src_dim_size, dst_dim_size
            ),
        });
    }

    let dtype = dst.dtype();
    if src.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: src.dtype(),
        });
    }

    let outer_size: usize = dst.shape()[..dim].iter().product();
    let outer_size = outer_size.max(1);
    let inner_size: usize = dst.shape()[dim + 1..].iter().product();
    let inner_size = inner_size.max(1);
    let total_src = outer_size * src_dim_size * inner_size;

    let dst_contig = ensure_contiguous(dst)?;
    let src_contig = ensure_contiguous(src)?;

    let out = alloc_output(client, dst.shape(), dtype)?;

    let dst_buf = get_tensor_buffer(&dst_contig)?;
    let src_buf = get_tensor_buffer(&src_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // First copy dst → output
    let copy_params = CopyParams {
        numel: dst.numel() as u32,
    };
    let copy_params_buf = create_params_buffer(client, &copy_params);
    index::launch_copy(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &dst_buf,
        &out_buf,
        &copy_params_buf,
        dst.numel(),
        dtype,
    )?;

    // Then overwrite the slice with src
    let params = SliceAssignParams {
        outer_size: outer_size as u32,
        dst_dim_size: dst_dim_size as u32,
        src_dim_size: src_dim_size as u32,
        inner_size: inner_size as u32,
        start: start as u32,
        _pad0: 0,
        _pad1: 0,
        _pad2: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_slice_assign(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &src_buf,
        &out_buf,
        &params_buf,
        total_src.max(1),
        dtype,
    )?;

    Ok(out)
}
