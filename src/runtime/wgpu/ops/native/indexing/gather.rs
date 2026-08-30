//! Element-wise gather: one index per output element.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::index;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

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

    // Nothing to gather: the index set is empty, so the output is empty and
    // `get_tensor_buffer` has no buffer to return for it.
    if total_elements == 0 {
        return Ok(out);
    }

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
        // Never restore a `.max(1)` here: the `total_elements == 0` guard above
        // already rules a zero out.
        total_elements,
        dtype,
    )?;

    Ok(out)
}
