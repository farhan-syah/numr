//! Conditional operation implementations for WebGPU.

use super::helpers::*;
use crate::error::Result;
use crate::ops::broadcast_shape;
use crate::runtime::wgpu::shaders::{activation_launcher, where_launcher};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous};
use crate::tensor::Tensor;

pub(crate) fn native_clamp(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    min_val: f64,
    max_val: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ClampParams::new(numel as u32, min_val, max_val, dtype);
    let params_buf = create_params_buffer(client, &params);

    activation_launcher::launch_clamp_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_where_cond(
    client: &WgpuClient,
    cond: &Tensor<WgpuRuntime>,
    x: &Tensor<WgpuRuntime>,
    y: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let cond_dtype = cond.dtype();
    let out_dtype = x.dtype();

    // Validate x and y have same dtype
    if x.dtype() != y.dtype() {
        return Err(crate::error::Error::DTypeMismatch {
            lhs: x.dtype(),
            rhs: y.dtype(),
        });
    }

    // Compute broadcast shape for all three tensors
    let xy_shape = compute_broadcast_shape(x, y)?;
    let out_shape = broadcast_shape(cond.shape(), &xy_shape).ok_or_else(|| {
        crate::error::Error::BroadcastError {
            lhs: cond.shape().to_vec(),
            rhs: xy_shape.clone(),
        }
    })?;

    let numel: usize = out_shape.iter().product();

    // Same shape case - use element-wise kernel
    if cond.shape() == x.shape() && x.shape() == y.shape() {
        let cond_contig = ensure_contiguous(cond)?;
        let x_contig = ensure_contiguous(x)?;
        let y_contig = ensure_contiguous(y)?;

        let out = alloc_output(client, &out_shape, out_dtype)?;

        let cond_buf = get_tensor_buffer(&cond_contig)?;
        let x_buf = get_tensor_buffer(&x_contig)?;
        let y_buf = get_tensor_buffer(&y_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = WhereParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        where_launcher::launch_where_generic_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &cond_buf,
            &x_buf,
            &y_buf,
            &out_buf,
            &params_buf,
            numel,
            cond_dtype,
            out_dtype,
        )?;

        return Ok(out);
    }

    // Broadcasting case - use broadcast kernel
    let cond_contig = ensure_contiguous(cond)?;
    let x_contig = ensure_contiguous(x)?;
    let y_contig = ensure_contiguous(y)?;

    let cond_strides = compute_broadcast_strides(cond.shape(), &out_shape)?;
    let x_strides = compute_broadcast_strides(x.shape(), &out_shape)?;
    let y_strides = compute_broadcast_strides(y.shape(), &out_shape)?;
    let shape_u32: Vec<u32> = out_shape.iter().map(|&s| s as u32).collect();

    let out = alloc_output(client, &out_shape, out_dtype)?;

    let cond_buf = get_tensor_buffer(&cond_contig)?;
    let x_buf = get_tensor_buffer(&x_contig)?;
    let y_buf = get_tensor_buffer(&y_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Create storage buffers for strides and shape
    let cond_strides_buf = create_storage_buffer(client, &cond_strides);
    let x_strides_buf = create_storage_buffer(client, &x_strides);
    let y_strides_buf = create_storage_buffer(client, &y_strides);
    let shape_buf = create_storage_buffer(client, &shape_u32);

    let params = WhereBroadcastParams {
        numel: numel as u32,
        ndim: out_shape.len() as u32,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    where_launcher::launch_where_broadcast_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &cond_buf,
        &x_buf,
        &y_buf,
        &out_buf,
        &cond_strides_buf,
        &x_strides_buf,
        &y_strides_buf,
        &shape_buf,
        &params_buf,
        numel,
        cond_dtype,
        out_dtype,
    )?;

    Ok(out)
}
