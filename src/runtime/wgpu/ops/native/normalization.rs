//! Normalization operation implementations for WebGPU.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{fused_add_norm, norm};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ReduceSumParams {
    batch_size: u32,
    hidden_size: u32,
}

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

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = RmsNormParams {
        batch_size: batch_size.max(1) as u32,
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
        batch_size.max(1),
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

    let a_buf = get_tensor_buffer(&a_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = LayerNormParams {
        batch_size: batch_size.max(1) as u32,
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
        batch_size.max(1),
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
    let spatial: usize = shape[2..].iter().product::<usize>().max(1);

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

// ============================================================================
// Fused Add + Normalization Operations
// ============================================================================

pub(crate) fn native_fused_add_rms_norm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    residual: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    let dtype = input.dtype();
    let shape = input.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "fused_add_rms_norm requires at least 1D input".to_string(),
        ));
    }

    if shape != residual.shape() {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: residual.shape().to_vec(),
        });
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let input_contig = ensure_contiguous(input)?;
    let residual_contig = ensure_contiguous(residual)?;
    let weight_contig = ensure_contiguous(weight)?;

    let output = alloc_output(client, shape, dtype)?;
    let pre_norm = alloc_output(client, shape, dtype)?;

    let input_buf = get_tensor_buffer(&input_contig)?;
    let residual_buf = get_tensor_buffer(&residual_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let output_buf = get_tensor_buffer(&output)?;
    let pre_norm_buf = get_tensor_buffer(&pre_norm)?;

    let params = RmsNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    fused_add_norm::launch_fused_add_rms_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        &residual_buf,
        &weight_buf,
        &output_buf,
        &pre_norm_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok((output, pre_norm))
}

pub(crate) fn native_fused_add_layer_norm(
    client: &WgpuClient,
    input: &Tensor<WgpuRuntime>,
    residual: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    let dtype = input.dtype();
    let shape = input.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "fused_add_layer_norm requires at least 1D input".to_string(),
        ));
    }

    if shape != residual.shape() {
        return Err(Error::ShapeMismatch {
            expected: shape.to_vec(),
            got: residual.shape().to_vec(),
        });
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let input_contig = ensure_contiguous(input)?;
    let residual_contig = ensure_contiguous(residual)?;
    let weight_contig = ensure_contiguous(weight)?;
    let bias_contig = ensure_contiguous(bias)?;

    let output = alloc_output(client, shape, dtype)?;
    let pre_norm = alloc_output(client, shape, dtype)?;

    let input_buf = get_tensor_buffer(&input_contig)?;
    let residual_buf = get_tensor_buffer(&residual_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let output_buf = get_tensor_buffer(&output)?;
    let pre_norm_buf = get_tensor_buffer(&pre_norm)?;

    let params = LayerNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    fused_add_norm::launch_fused_add_layer_norm(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &input_buf,
        &residual_buf,
        &weight_buf,
        &bias_buf,
        &output_buf,
        &pre_norm_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    Ok((output, pre_norm))
}

pub(crate) fn native_fused_add_rms_norm_bwd(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    pre_norm: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    let dtype = grad.dtype();
    let shape = grad.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "fused_add_rms_norm_bwd requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let grad_contig = ensure_contiguous(grad)?;
    let pn_contig = ensure_contiguous(pre_norm)?;
    let weight_contig = ensure_contiguous(weight)?;

    let d_input_residual = alloc_output(client, shape, dtype)?;
    let d_weight_scratch = alloc_output(client, &[batch_size, hidden_size], dtype)?;
    let d_weight = alloc_output(client, &[hidden_size], dtype)?;

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let pn_buf = get_tensor_buffer(&pn_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let d_ir_buf = get_tensor_buffer(&d_input_residual)?;
    let dws_buf = get_tensor_buffer(&d_weight_scratch)?;
    let dw_buf = get_tensor_buffer(&d_weight)?;

    let params = RmsNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    fused_add_norm::launch_fused_add_rms_norm_bwd(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &grad_buf,
        &pn_buf,
        &weight_buf,
        &d_ir_buf,
        &dws_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    // Launch reduce_sum_rows to sum d_weight_scratch across batch
    let reduce_params = ReduceSumParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
    };
    let reduce_params_buf = create_params_buffer(client, &reduce_params);

    fused_add_norm::launch_reduce_sum_rows(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &dws_buf,
        &dw_buf,
        &reduce_params_buf,
        hidden_size,
        dtype,
    )?;

    Ok((d_input_residual, d_weight))
}

pub(crate) fn native_fused_add_layer_norm_bwd(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    pre_norm: &Tensor<WgpuRuntime>,
    weight: &Tensor<WgpuRuntime>,
    bias: &Tensor<WgpuRuntime>,
    eps: f32,
) -> Result<(
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
    Tensor<WgpuRuntime>,
)> {
    let dtype = grad.dtype();
    let shape = grad.shape();

    if shape.len() < 1 {
        return Err(Error::Internal(
            "fused_add_layer_norm_bwd requires at least 1D input".to_string(),
        ));
    }

    let hidden_size = shape[shape.len() - 1];
    let batch_size: usize = shape[..shape.len() - 1].iter().product();

    let grad_contig = ensure_contiguous(grad)?;
    let pn_contig = ensure_contiguous(pre_norm)?;
    let weight_contig = ensure_contiguous(weight)?;
    let bias_contig = ensure_contiguous(bias)?;

    let d_input_residual = alloc_output(client, shape, dtype)?;
    let d_weight_scratch = alloc_output(client, &[batch_size, hidden_size], dtype)?;
    let d_bias_scratch = alloc_output(client, &[batch_size, hidden_size], dtype)?;
    let d_weight = alloc_output(client, &[hidden_size], dtype)?;
    let d_bias = alloc_output(client, &[hidden_size], dtype)?;

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let pn_buf = get_tensor_buffer(&pn_contig)?;
    let weight_buf = get_tensor_buffer(&weight_contig)?;
    let bias_buf = get_tensor_buffer(&bias_contig)?;
    let d_ir_buf = get_tensor_buffer(&d_input_residual)?;
    let dws_buf = get_tensor_buffer(&d_weight_scratch)?;
    let dbs_buf = get_tensor_buffer(&d_bias_scratch)?;
    let dw_buf = get_tensor_buffer(&d_weight)?;
    let db_buf = get_tensor_buffer(&d_bias)?;

    let params = LayerNormParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
        eps,
    };
    let params_buf = create_params_buffer(client, &params);

    fused_add_norm::launch_fused_add_layer_norm_bwd(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &grad_buf,
        &pn_buf,
        &weight_buf,
        &bias_buf,
        &d_ir_buf,
        &dws_buf,
        &dbs_buf,
        &params_buf,
        batch_size.max(1),
        dtype,
    )?;

    // Launch reduce_sum_rows for d_weight_scratch
    let reduce_params = ReduceSumParams {
        batch_size: batch_size.max(1) as u32,
        hidden_size: hidden_size as u32,
    };
    let reduce_params_buf = create_params_buffer(client, &reduce_params);

    fused_add_norm::launch_reduce_sum_rows(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &dws_buf,
        &dw_buf,
        &reduce_params_buf,
        hidden_size,
        dtype,
    )?;

    // Launch reduce_sum_rows for d_bias_scratch
    let reduce_params_buf = create_params_buffer(client, &reduce_params);

    fused_add_norm::launch_reduce_sum_rows(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &dbs_buf,
        &db_buf,
        &reduce_params_buf,
        hidden_size,
        dtype,
    )?;

    Ok((d_input_residual, d_weight, d_bias))
}
