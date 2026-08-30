//! Plain normalization operations for WebGPU: RMSNorm, LayerNorm and GroupNorm.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::norm;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_rms_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "rms_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a)?;
    let weight_contig = ensure_contiguous(weight)?;

    let out = alloc_output(client, shape, dtype)?;

    // A zero-element input normalizes nothing, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Hand the empty result back.
    if out.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Unguarded by `.max(1)`: the guard above rules the zero out, and clamping
    // would tell the shader about a row the allocation does not contain.
    let params = RmsNormParams {
        batch_size: batch_size as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_rms_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &out_buf,
        &params_buf,
        batch_size,
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_layer_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "layer_norm requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let a_contig = ensure_contiguous(a)?;
    let weight_contig = ensure_contiguous(weight)?;
    let bias_contig = ensure_contiguous(bias)?;

    let out = alloc_output(client, shape, dtype)?;

    // A zero-element input normalizes nothing, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Hand the empty result back.
    if out.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Unguarded by `.max(1)`: the guard above rules the zero out, and clamping
    // would tell the shader about a row the allocation does not contain.
    let params = LayerNormParams {
        batch_size: batch_size as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_layer_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &weight_buf,
        &bias_buf,
        &out_buf,
        &params_buf,
        batch_size,
        dtype,
    )?;

    Ok(out)
}

pub(crate) fn native_group_norm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    num_groups: usize,
    eps: f32,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = input.dtype();
    let shape = input.shape();

    if shape.len() < 2 {
        return Err(Error::InvalidArgument {
            arg: "input",
            reason: "group_norm requires at least 2D input [batch, channels, ...]".into(),
        });
    }

    let batch = shape[0];
    let channels = shape[1];
    if !channels.is_multiple_of(num_groups) {
        return Err(Error::InvalidArgument {
            arg: "num_groups",
            reason: format!("channels {channels} not divisible by num_groups {num_groups}"),
        });
    }
    let channels_per_group = channels / num_groups;
    // Unclamped: a 2D input already products to 1 over the empty trailing slice,
    // and a zero spatial dim must stay 0 so the shader touches nothing.
    let spatial: usize = shape[2..].iter().product();

    if weight.shape() != [channels] || bias.shape() != [channels] {
        return Err(Error::ShapeMismatch {
            expected: vec![channels],
            got: if weight.shape() != [channels] {
                weight.shape().to_vec()
            } else {
                bias.shape().to_vec()
            },
        });
    }

    let input_contig = ensure_contiguous(input)?;
    let weight_contig = ensure_contiguous(weight)?;
    let bias_contig = ensure_contiguous(bias)?;
    let out = alloc_output(client, shape, dtype)?;

    // A zero-element input normalizes nothing, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Hand the empty result back.
    if out.numel() == 0 {
        return Ok(out);
    }

    let input_buf = get_tensor_buffer(&input_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = GroupNormParams {
        batch_size: batch as u32,
        channels: channels as u32,
        spatial: spatial as u32,
        num_groups: num_groups as u32,
        channels_per_group: channels_per_group as u32,
        eps,
        _pad0: 0,
        _pad1: 0,
    };
    let params_buf = create_params_buffer(client, &params);

    norm::launch_group_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        &weight_buf,
        &bias_buf,
        &out_buf,
        &params_buf,
        batch,
        num_groups,
        dtype,
    )?;

    Ok(out)
}
