//! Element-wise scatter: one index per source element.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::index;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_contiguous_strides, ensure_contiguous};
use crate::tensor::Tensor;

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

    // An empty destination has no buffer to bind and no slot any index could
    // address, so the empty result is returned before anything is dispatched.
    if a.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
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

    // An empty source writes nothing, so the copy above is already the whole
    // result. `src` and `indices` are the zero-byte allocations here, and
    // `get_tensor_buffer` has no buffer to return for them.
    if src_total == 0 {
        return Ok(out);
    }

    let indices_buf = get_tensor_buffer(&indices_contig)?;
    let src_buf = get_tensor_buffer(&src_contig)?;

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
        // Never restore a `.max(1)` here: the `src_total == 0` guard above already
        // rules a zero out.
        src_total,
        dtype,
    )?;

    Ok(out)
}
