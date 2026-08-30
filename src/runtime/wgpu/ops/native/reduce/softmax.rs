//! Softmax forward and backward passes for WebGPU.
//!
//! Both transpose a non-trailing reduction dimension to the end, then run the
//! last-dimension kernel that the WGSL shaders implement.

use super::super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::reduce;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_softmax(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let shape = a.shape();
    let ndim = shape.len();

    // Normalize dim
    let dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // For non-last dimension, use permute-based approach:
    // 1. Permute target dim to last position
    // 2. Make contiguous
    // 3. Run softmax on last dimension
    // 4. Permute back to original order
    if dim != ndim - 1 {
        // Build permutation: move dim to end
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(dim);
        perm.push(dim);

        // Permute to move target dimension to last position
        let permuted = a.permute(&perm)?;
        let permuted_contig = permuted.contiguous()?;

        // Softmax on last dimension (now the original target dimension)
        let result = native_softmax_last_dim(client, &permuted_contig)?;

        // Build inverse permutation to restore original order
        // Original was [0, 1, ..., dim-1, dim, dim+1, ..., ndim-1]
        // After perm: [0, 1, ..., dim-1, dim+1, ..., ndim-1, dim]
        // inv_perm[perm[i]] = i
        let mut inv_perm = vec![0; ndim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }

        return result.permute(&inv_perm);
    }

    native_softmax_last_dim(client, a)
}

/// Softmax on the last dimension (optimized GPU implementation)
fn native_softmax_last_dim(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    let a_contig = ensure_contiguous(a)?;
    let dim = ndim - 1;
    let batch_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];

    let out = alloc_output(client, shape, dtype)?;

    // A zero-element input normalizes nothing, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Never restore a `.max(1)` on
    // `batch_size`: it would make this guard unreachable and tell the shader about
    // a row the allocation does not contain.
    if out.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = SoftmaxParams {
        batch_size: batch_size as u32,
        dim_size: dim_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_softmax_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        batch_size,
        dtype,
    )?;

    Ok(out)
}

/// Softmax backward with dedicated GPU kernel.
///
/// d_input = output * (grad - sum(grad * output))
pub(crate) fn native_softmax_bwd(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let shape = grad.shape();
    let ndim = shape.len();

    let dim = if dim < 0 {
        (ndim as isize + dim) as usize
    } else {
        dim as usize
    };

    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    // For non-last dimension, permute to last, compute, permute back
    if dim != ndim - 1 {
        let mut perm: Vec<usize> = (0..ndim).collect();
        perm.remove(dim);
        perm.push(dim);

        let grad_p = grad.permute(&perm)?.contiguous()?;
        let output_p = output.permute(&perm)?.contiguous()?;
        let result = native_softmax_bwd_last_dim(client, &grad_p, &output_p)?;

        let mut inv_perm = vec![0; ndim];
        for (i, &p) in perm.iter().enumerate() {
            inv_perm[p] = i;
        }
        return result.permute(&inv_perm);
    }

    native_softmax_bwd_last_dim(client, grad, output)
}

fn native_softmax_bwd_last_dim(
    client: &WgpuClient,
    grad: &Tensor<WgpuRuntime>,
    output: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let shape = grad.shape();
    let ndim = shape.len();
    let dtype = grad.dtype();

    let grad_contig = ensure_contiguous(grad)?;
    let output_contig = ensure_contiguous(output)?;
    let dim = ndim - 1;
    let batch_size: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];

    let d_input = alloc_output(client, shape, dtype)?;

    // Nothing to differentiate, and `get_tensor_buffer` has no buffer to return for
    // a zero-byte allocation. Never restore a `.max(1)` on `batch_size`: it would
    // make this guard unreachable and tell the shader about a row the allocation
    // does not contain.
    if d_input.numel() == 0 {
        return Ok(d_input);
    }

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let output_buf = get_tensor_buffer(&output_contig)?;
    let d_input_buf = get_tensor_buffer(&d_input)?;

    let params = SoftmaxParams {
        batch_size: batch_size as u32,
        dim_size: dim_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_softmax_bwd_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &grad_buf,
        &output_buf,
        &d_input_buf,
        &params_buf,
        batch_size,
        dtype,
    )?;

    Ok(d_input)
}
