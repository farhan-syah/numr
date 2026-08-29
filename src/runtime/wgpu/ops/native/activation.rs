//! Parametric activation operation implementation for WebGPU.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::{activation_launcher, fused_activation_mul};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Native parametric activation operation (leaky_relu, elu).
///
/// These activations take an extra scalar parameter (negative_slope or alpha).
pub(crate) fn native_parametric_activation(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    param: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Uses ScalarParams to pass the parameter. leaky_relu/elu are F32-only,
    // so this always takes the f32 encoding path.
    let params = ScalarParams::new(numel as u32, param, dtype);
    let params_buf = create_params_buffer(client, &params);

    match op {
        "leaky_relu" => {
            activation_launcher::launch_leaky_relu(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &out_buf,
                &params_buf,
                numel,
                dtype,
            )?;
        }
        "elu" => {
            activation_launcher::launch_elu(
                client.pipeline_cache(),
                client.wgpu_queue(),
                &a_buf,
                &out_buf,
                &params_buf,
                numel,
                dtype,
            )?;
        }
        _ => {
            return Err(Error::Internal(format!(
                "Unknown parametric activation: {}",
                op
            )));
        }
    }

    Ok(out)
}

/// Native fused activation-mul forward: out = activation(a) * b. F32 only.
pub(crate) fn native_fused_activation_mul_fwd(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    match op {
        "silu_mul" => fused_activation_mul::launch_silu_mul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "gelu_mul" => fused_activation_mul::launch_gelu_mul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "relu_mul" => fused_activation_mul::launch_relu_mul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "sigmoid_mul" => fused_activation_mul::launch_sigmoid_mul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        _ => {
            return Err(Error::Internal(format!(
                "Unknown fused activation-mul op: {}",
                op
            )));
        }
    }

    Ok(out)
}

/// Native fused activation-mul backward: d_a = grad * b * act'(a), d_b = grad * act(a). F32 only.
pub(crate) fn native_fused_activation_mul_bwd(
    client: &WgpuClient,
    op: &'static str,
    grad: &Tensor<WgpuRuntime>,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    let dtype = a.dtype();
    let grad_contig = ensure_contiguous(grad)?;
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    let d_a = alloc_output(client, a.shape(), dtype)?;
    let d_b = alloc_output(client, b.shape(), dtype)?;

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let d_a_buf = get_tensor_buffer(&d_a)?;
    let d_b_buf = get_tensor_buffer(&d_b)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    match op {
        "silu_mul_bwd" => fused_activation_mul::launch_silu_mul_bwd(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &grad_buf,
            &a_buf,
            &b_buf,
            &d_a_buf,
            &d_b_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "gelu_mul_bwd" => fused_activation_mul::launch_gelu_mul_bwd(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &grad_buf,
            &a_buf,
            &b_buf,
            &d_a_buf,
            &d_b_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "relu_mul_bwd" => fused_activation_mul::launch_relu_mul_bwd(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &grad_buf,
            &a_buf,
            &b_buf,
            &d_a_buf,
            &d_b_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        "sigmoid_mul_bwd" => fused_activation_mul::launch_sigmoid_mul_bwd(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &grad_buf,
            &a_buf,
            &b_buf,
            &d_a_buf,
            &d_b_buf,
            &params_buf,
            numel,
            dtype,
        )?,
        _ => {
            return Err(Error::Internal(format!(
                "Unknown fused activation-mul bwd op: {}",
                op
            )));
        }
    }

    Ok((d_a, d_b))
}
