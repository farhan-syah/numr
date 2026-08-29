//! gather_2d for WebGPU.
//!
//! Reads one element per (row, col) pair from a 2D input, the coordinate pairs
//! arriving as two parallel index tensors.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    Gather2dParams, alloc_output, create_params_buffer, ensure_i32_indices, get_tensor_buffer,
};
use crate::runtime::wgpu::shaders::launch_gather_2d;
use crate::tensor::Tensor;

pub(super) fn gather_2d(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    rows: &Tensor<WgpuRuntime>,
    cols: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = input.dtype();
    let shape = input.shape();

    // Check supported dtypes (WebGPU doesn't support f64)
    if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "gather_2d",
        });
    }

    // Validate input is 2D
    if shape.len() != 2 {
        return Err(Error::ShapeMismatch {
            expected: vec![0, 0], // Indicates 2D expected
            got: shape.to_vec(),
        });
    }

    let nrows = shape[0];
    let ncols = shape[1];

    // Validate index dtypes (WebGPU prefers I32)
    if !matches!(rows.dtype(), DType::I32 | DType::I64) {
        return Err(Error::InvalidArgument {
            arg: "rows",
            reason: "gather_2d rows must be I32 or I64".to_string(),
        });
    }

    if !matches!(cols.dtype(), DType::I32 | DType::I64) {
        return Err(Error::InvalidArgument {
            arg: "cols",
            reason: "gather_2d cols must be I32 or I64".to_string(),
        });
    }

    // Validate rows and cols are 1D and have same length
    if rows.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![rows.numel()],
            got: rows.shape().to_vec(),
        });
    }

    if cols.ndim() != 1 {
        return Err(Error::ShapeMismatch {
            expected: vec![cols.numel()],
            got: cols.shape().to_vec(),
        });
    }

    let num_indices = rows.numel();
    if cols.numel() != num_indices {
        return Err(Error::ShapeMismatch {
            expected: vec![num_indices],
            got: cols.shape().to_vec(),
        });
    }

    // Make all inputs contiguous
    let input = ensure_contiguous(input)?;
    let rows_i32 = ensure_i32_indices(client, rows)?;
    let cols_i32 = ensure_i32_indices(client, cols)?;
    let rows = ensure_contiguous(&rows_i32)?;
    let cols = ensure_contiguous(&cols_i32)?;

    // Allocate output
    let output = alloc_output(client, &[num_indices], dtype)?;

    // Get buffers
    let input_buf = get_tensor_buffer(&input)?;
    let rows_buf = get_tensor_buffer(&rows)?;
    let cols_buf = get_tensor_buffer(&cols)?;
    let output_buf = get_tensor_buffer(&output)?;

    // Create params
    let params = Gather2dParams {
        nrows: nrows as u32,
        ncols: ncols as u32,
        num_indices: num_indices as u32,
        _pad: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_gather_2d(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        &rows_buf,
        &cols_buf,
        &output_buf,
        &params_buf,
        num_indices,
        dtype,
    )?;

    Ok(output)
}
