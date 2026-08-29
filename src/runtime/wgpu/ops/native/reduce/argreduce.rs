//! Index-returning reductions (argmax, argmin) for WebGPU.
//!
//! The output carries positions, so it is always allocated as I32 regardless of
//! the input dtype.

use super::super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::reduce::reduce_output_shape;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::reduce;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_argreduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dim: usize,
    keepdim: bool,
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

    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Reducing the only dimension of a 1-D tensor must give a scalar, not `[1]`.
    let out_shape = reduce_output_shape(shape, &[dim], keepdim);

    // Output indices as I32 (WebGPU doesn't support I64, shader uses u32)
    let out = alloc_output(client, &out_shape, DType::I32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ArgReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_argreduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out.max(1),
        dtype,
    )?;

    Ok(out)
}
