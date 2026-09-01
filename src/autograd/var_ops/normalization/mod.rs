//! Normalization operations (rms_norm, layer_norm)

use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{NormalizationOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// RMS Normalization: y = x / rms(x) * weight
///
/// Uses the fused `NormalizationOps::rms_norm` kernel for the forward pass
/// and tracks gradients for both input and weight.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_rms_norm<R, C>(input: &Var<R>, weight: &Var<R>, eps: f32, client: &C) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.rms_norm(input.tensor(), weight.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() {
        let grad_fn = RmsNormBackward::<R>::new(
            input.id(),
            weight.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Layer Normalization: y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
///
/// Uses the fused `NormalizationOps::layer_norm` kernel for the forward pass
/// and tracks gradients for input, weight, and bias.
///
/// # Arguments
///
/// * `input` - Input variable of shape `[..., hidden_size]`
/// * `weight` - Weight (gamma) variable of shape `[hidden_size]`
/// * `bias` - Bias (beta) variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_layer_norm<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.layer_norm(input.tensor(), weight.tensor(), bias.tensor(), eps)?;

    if input.requires_grad() || weight.requires_grad() || bias.requires_grad() {
        let grad_fn = LayerNormBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Group Normalization with autograd support.
///
/// Input: `[batch, channels, *spatial]`
/// Normalizes over groups of channels independently.
///
/// # Arguments
/// * `input` - Input variable `[batch, channels, *spatial]`
/// * `weight` - Gamma variable `[channels]`
/// * `bias` - Beta variable `[channels]`
/// * `num_groups` - Number of groups (must divide channels)
/// * `eps` - Numerical stability constant
/// * `client` - Runtime client
pub fn var_group_norm<R, C>(
    input: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    num_groups: usize,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let output = client.group_norm(
        input.tensor(),
        weight.tensor(),
        bias.tensor(),
        num_groups,
        eps,
    )?;

    if input.requires_grad() || weight.requires_grad() || bias.requires_grad() {
        let grad_fn = GroupNormBackward::<R>::new(
            input.id(),
            weight.id(),
            bias.id(),
            input.tensor().clone(),
            weight.tensor().clone(),
            num_groups,
            eps,
            input.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Fused Add + RMS Normalization: pre_norm = x + residual, output = rms_norm(pre_norm, weight, eps)
///
/// Returns a single output variable. Both `x` and `residual` receive the same gradient.
///
/// # Arguments
///
/// * `x` - Input variable of shape `[..., hidden_size]`
/// * `residual` - Residual variable of same shape as `x`
/// * `weight` - Weight variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_fused_add_rms_norm<R, C>(
    x: &Var<R>,
    residual: &Var<R>,
    weight: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let (output, pre_norm) =
        client.fused_add_rms_norm(x.tensor(), residual.tensor(), weight.tensor(), eps)?;

    if x.requires_grad() || residual.requires_grad() || weight.requires_grad() {
        let grad_fn = FusedAddRmsNormBackward::<R>::new(
            x.id(),
            residual.id(),
            weight.id(),
            pre_norm,
            weight.tensor().clone(),
            eps,
            x.grad_fn().cloned(),
            residual.grad_fn().cloned(),
            weight.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

/// Fused Add + Layer Normalization: pre_norm = x + residual, output = layer_norm(pre_norm, weight, bias, eps)
///
/// Returns a single output variable. Both `x` and `residual` receive the same gradient.
///
/// # Arguments
///
/// * `x` - Input variable of shape `[..., hidden_size]`
/// * `residual` - Residual variable of same shape as `x`
/// * `weight` - Weight (gamma) variable of shape `[hidden_size]`
/// * `bias` - Bias (beta) variable of shape `[hidden_size]`
/// * `eps` - Small constant for numerical stability
/// * `client` - Runtime client
pub fn var_fused_add_layer_norm<R, C>(
    x: &Var<R>,
    residual: &Var<R>,
    weight: &Var<R>,
    bias: &Var<R>,
    eps: f32,
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + NormalizationOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    let (output, pre_norm) = client.fused_add_layer_norm(
        x.tensor(),
        residual.tensor(),
        weight.tensor(),
        bias.tensor(),
        eps,
    )?;

    if x.requires_grad()
        || residual.requires_grad()
        || weight.requires_grad()
        || bias.requires_grad()
    {
        let grad_fn = FusedAddLayerNormBackward::<R>::new(
            x.id(),
            residual.id(),
            weight.id(),
            bias.id(),
            pre_norm,
            weight.tensor().clone(),
            bias.tensor().clone(),
            eps,
            x.grad_fn().cloned(),
            residual.grad_fn().cloned(),
            weight.grad_fn().cloned(),
            bias.grad_fn().cloned(),
        );
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests;
