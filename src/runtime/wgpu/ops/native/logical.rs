//! Native GPU implementation for logical operations on WebGPU.
//!
//! WebGPU uses U32 for boolean tensors (0 = false, non-zero = true).

use super::helpers::*;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::logical::{
    launch_logical_and, launch_logical_not, launch_logical_or, launch_logical_xor,
};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Parameters for binary logical operations.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LogicalBinaryParams {
    pub(crate) numel: u32,
}

/// Parameters for unary logical operations.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LogicalUnaryParams {
    pub(crate) numel: u32,
}

/// Native logical AND operation.
#[allow(dead_code)]
pub(crate) fn native_logical_and(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), DType::U32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LogicalBinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_logical_and(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        numel,
    )?;

    Ok(out)
}

/// Native logical OR operation.
#[allow(dead_code)]
pub(crate) fn native_logical_or(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), DType::U32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LogicalBinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_logical_or(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        numel,
    )?;

    Ok(out)
}

/// Native logical XOR operation.
#[allow(dead_code)]
pub(crate) fn native_logical_xor(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), DType::U32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LogicalBinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_logical_xor(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &out_buf,
        &params_buf,
        numel,
    )?;

    Ok(out)
}

/// Native logical NOT operation.
#[allow(dead_code)]
pub(crate) fn native_logical_not(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), DType::U32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LogicalUnaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    launch_logical_not(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
    )?;

    Ok(out)
}
