//! Gradient reduction helpers for broadcast operands
//!
//! When an operand is broadcast during the forward pass, its gradient comes back
//! carrying the OUTPUT's shape. These helpers sum the gradient over every
//! broadcast dimension and reshape it back to the operand's original shape.

use crate::autograd::{Var, var_sum};
use crate::error::Result;
use crate::ops::{ReduceOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Reduce gradient to match target shape (for broadcasting)
///
/// When broadcasting occurs during forward, we need to sum over the
/// broadcast dimensions during backward.
pub(crate) fn reduce_grad_for_broadcast<R: Runtime>(
    grad: &Tensor<R>,
    target_shape: &[usize],
) -> Result<Tensor<R>>
where
    R::Client: TensorOps<R> + ReduceOps<R>,
{
    let grad_shape = grad.shape();

    // If shapes match, no reduction needed
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    let client = R::default_client(grad.device());

    // Find dimensions that need reduction
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    // Pad target shape with leading 1s if necessary
    let mut padded_target = vec![1usize; grad_ndim];
    let offset = grad_ndim.saturating_sub(target_ndim);
    for (i, &dim) in target_shape.iter().enumerate() {
        padded_target[offset + i] = dim;
    }

    // Collect dimensions to reduce
    let mut reduce_dims = Vec::new();
    for (i, (&grad_dim, &target_dim)) in grad_shape.iter().zip(padded_target.iter()).enumerate() {
        if target_dim == 1 && grad_dim > 1 {
            reduce_dims.push(i);
        }
    }

    // Reduce over broadcast dimensions
    let mut result = grad.clone();
    if !reduce_dims.is_empty() {
        result = client.sum(&result, &reduce_dims, true)?;
    }

    // Remove leading dimensions if target has fewer dims
    if target_ndim < grad_ndim {
        // Sum over the extra leading dimensions
        let extra_dims: Vec<usize> = (0..(grad_ndim - target_ndim)).collect();
        if !extra_dims.is_empty() {
            result = client.sum(&result, &extra_dims, false)?;
        }
    }

    // Reshape to target shape
    if result.shape() != target_shape {
        result = result.reshape(target_shape)?;
    }

    Ok(result)
}

/// Reduce Var gradient to match target shape (for broadcasting)
///
/// Like [`reduce_grad_for_broadcast`] but operates on Vars and uses var_sum
/// to maintain the computation graph for second-order differentiation.
pub(crate) fn reduce_var_for_broadcast<R, C>(
    var: &Var<R>,
    target_shape: &[usize],
    client: &C,
) -> Result<Var<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + TensorOps<R>,
    R::Client: TensorOps<R>,
{
    let var_shape = var.shape();

    // If shapes match, no reduction needed
    if var_shape == target_shape {
        return Ok(var.clone());
    }

    // Find dimensions that need reduction
    let var_ndim = var_shape.len();
    let target_ndim = target_shape.len();

    // Pad target shape with leading 1s if necessary
    let mut padded_target = vec![1usize; var_ndim];
    let offset = var_ndim.saturating_sub(target_ndim);
    for (i, &dim) in target_shape.iter().enumerate() {
        padded_target[offset + i] = dim;
    }

    // Collect dimensions to reduce
    let mut reduce_dims = Vec::new();
    for (i, (&var_dim, &target_dim)) in var_shape.iter().zip(padded_target.iter()).enumerate() {
        if target_dim == 1 && var_dim > 1 {
            reduce_dims.push(i);
        }
    }

    // Reduce over broadcast dimensions using var_sum (builds graph)
    let mut result = var.clone();
    if !reduce_dims.is_empty() {
        result = var_sum(&result, &reduce_dims, true, client)?;
    }

    // Remove leading dimensions if target has fewer dims
    if target_ndim < var_ndim {
        let extra_dims: Vec<usize> = (0..(var_ndim - target_ndim)).collect();
        if !extra_dims.is_empty() {
            result = var_sum(&result, &extra_dims, false, client)?;
        }
    }

    // Reshape to target shape if needed using var_reshape to maintain gradient chain
    if result.shape() != target_shape {
        result = super::shape::var_reshape(&result, target_shape)?;
    }

    Ok(result)
}
