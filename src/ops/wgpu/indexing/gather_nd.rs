//! gather_nd for WebGPU.
//!
//! Reads N-dimensional coordinates out of an index tensor shaped
//! `[num_slices, index_depth]` and copies one contiguous slice per coordinate.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    GatherNdParams, alloc_output, create_params_buffer, ensure_i32_indices, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::launch_gather_nd;
use crate::tensor::Tensor;

pub(super) fn gather_nd(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = input.dtype();

    // Check supported dtypes
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "gather_nd",
        });
    }

    // Validate indices dtype
    if !matches!(indices.dtype(), DType::I32 | DType::I64) {
        return Err(Error::InvalidArgument {
            arg: "indices",
            reason: "gather_nd indices must be I32 or I64".to_string(),
        });
    }

    // Ensure contiguous
    let input = ensure_contiguous(input)?;
    let indices_i32 = ensure_i32_indices(client, indices)?;
    let indices = ensure_contiguous(&indices_i32)?;

    let input_shape = input.shape();
    let indices_shape = indices.shape();

    // indices has shape [..., index_depth]
    // where index_depth <= input_ndim
    let index_depth = *indices_shape.last().unwrap_or(&0);
    let num_slices: usize = indices_shape[..indices_shape.len() - 1].iter().product();

    if index_depth > input_shape.len() {
        return Err(Error::InvalidArgument {
            arg: "indices",
            reason: format!(
                "index depth {} exceeds input dimensions {}",
                index_depth,
                input_shape.len()
            ),
        });
    }

    // Compute output shape and slice size
    // Output shape = indices_shape[:-1] + input_shape[index_depth:]
    let slice_size: usize = input_shape[index_depth..].iter().product();
    let slice_size = if slice_size == 0 { 1 } else { slice_size };

    let mut output_shape: Vec<usize> = indices_shape[..indices_shape.len() - 1].to_vec();
    output_shape.extend_from_slice(&input_shape[index_depth..]);
    if output_shape.is_empty() {
        output_shape.push(1);
    }

    let total_output = num_slices * slice_size;

    // Allocate output
    let output = alloc_output(client, &output_shape, dtype)?;

    // Get buffers
    let input_buf = get_tensor_buffer(&input)?;
    let indices_buf = get_tensor_buffer(&indices)?;
    let output_buf = get_tensor_buffer(&output)?;

    // Compute strides
    let ndim = input_shape.len();
    let mut input_strides = [0u32; 8];
    let mut input_shape_arr = [0u32; 8];
    let mut stride = 1usize;
    for i in (0..ndim).rev() {
        if i < 8 {
            input_strides[i] = stride as u32;
            input_shape_arr[i] = input_shape[i] as u32;
        }
        stride *= input_shape[i];
    }

    // Create params
    let params = GatherNdParams {
        num_slices: num_slices as u32,
        slice_size: slice_size as u32,
        index_depth: index_depth as u32,
        ndim: ndim as u32,
        input_shape: input_shape_arr,
        input_strides,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_gather_nd(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        &indices_buf,
        &output_buf,
        &params_buf,
        total_output,
        dtype,
    )?;

    Ok(output)
}
