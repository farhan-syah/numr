//! Reduction operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::ops::reduce::reduce_output_shape;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::reduce;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_reduce_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let _dtype = a.dtype();
    let shape = a.shape();

    // Empty dims means reduce every dimension, matching CPU and CUDA. Normalizing
    // here rather than taking a separate path keeps `keepdim` honored throughout.
    let dims: Vec<usize> = if dims.is_empty() {
        (0..shape.len()).collect()
    } else {
        dims.to_vec()
    };

    // The output shape is the same regardless of which path computes the values.
    let out_shape = reduce_output_shape(shape, &dims, keepdim);

    // Correctness, not a performance choice: an integer sum, prod or mean must
    // accumulate the whole reduced set in ONE wide accumulator and narrow (for
    // mean, divide) exactly once. Chaining a reduction per dimension narrows
    // once per dimension, and the two-pass whole-tensor path stores its partials
    // in the element type, so both would saturate early and disagree with CPU's
    // i128 epilogue in runtime/cpu/kernels/reduce/int_acc.rs.
    if a.dtype().is_int() && matches!(op, "sum" | "prod" | "mean") {
        return native_int_acc_reduce(client, op, a, &dims, &out_shape);
    }

    // Reducing every dimension has a dedicated two-pass kernel that is cheaper
    // than chaining one reduction per dimension.
    if dims.len() == shape.len() && dims.len() > 1 {
        let result = native_full_reduce(client, op, a)?;
        return result.reshape(&out_shape);
    }

    // For multi-dim reduction, reduce one dimension at a time
    if dims.len() > 1 {
        let mut sorted_dims = dims.clone();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Sort in descending order

        let mut result = a.clone();
        for &dim in &sorted_dims {
            result = native_single_dim_reduce(client, op, &result, dim, true)?;
        }

        // Every reduced dim is still present as size 1, so drop them in one step.
        if !keepdim {
            result = result.reshape(&out_shape)?;
        }

        return Ok(result);
    }

    // Single dimension reduction
    native_single_dim_reduce(client, op, a, dims[0], keepdim)
}

/// Reduce `dims` with a single wide accumulation, for integer sum/prod/mean.
///
/// Moves every reduced dimension to the end, materializes that layout, and
/// folds them into one trailing dimension. The single-dim kernel then sees one
/// contiguous run per output element, which is what lets it accumulate in the
/// 64-bit accumulator and narrow once.
fn native_int_acc_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    out_shape: &[usize],
) -> Result<Tensor<WgpuRuntime>> {
    let shape = a.shape();

    // A 0-dim tensor has nothing to reduce: every one of these ops returns the
    // element itself.
    if shape.is_empty() || dims.is_empty() {
        return a.reshape(out_shape);
    }

    for &d in dims {
        if d >= shape.len() {
            return Err(Error::InvalidDimension {
                dim: d as isize,
                ndim: shape.len(),
            });
        }
    }

    let mut reduced = dims.to_vec();
    reduced.sort_unstable();
    reduced.dedup();

    let mut perm: Vec<usize> = (0..shape.len()).filter(|d| !reduced.contains(d)).collect();
    let kept: usize = perm.iter().map(|&d| shape[d]).product();
    perm.extend_from_slice(&reduced);
    let reduce_total: usize = reduced.iter().map(|&d| shape[d]).product();

    let collapsed = a
        .permute(&perm)?
        .contiguous()?
        .reshape(&[kept, reduce_total])?;
    let result = native_single_dim_reduce(client, op, &collapsed, 1, false)?;
    result.reshape(out_shape)
}

fn native_single_dim_reduce(
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

    // Compute parameters
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();
    let numel_out = outer_size * inner_size;

    // Reducing the only dimension of a 1-D tensor must give a scalar, not `[1]`.
    let out_shape = reduce_output_shape(shape, &[dim], keepdim);

    let out = alloc_output(client, &out_shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size.max(1) as u32,
        inner_size: inner_size.max(1) as u32,
        numel_out: numel_out.max(1) as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_reduce_op(
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

fn native_full_reduce(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    // For mean, we need to divide by numel at the end
    let is_mean = op == "mean";
    let reduce_op = if is_mean { "sum" } else { op };

    // Two-pass reduction for large arrays
    let workgroup_size = 256;
    let num_workgroups = (numel + workgroup_size - 1) / workgroup_size;

    if num_workgroups <= 1 {
        // Single pass
        let out = alloc_output(client, &[1], dtype)?;
        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = FullReduceParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        reduce::launch_full_reduce_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            reduce_op,
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        if is_mean {
            return client.div_scalar(&out, numel as f64);
        }
        return Ok(out);
    }

    // Multi-pass: first reduce to num_workgroups values, then reduce again
    let partial = alloc_output(client, &[num_workgroups], dtype)?;
    let a_buf = get_tensor_buffer(&a_contig)?;
    let partial_buf = get_tensor_buffer(&partial)?;

    let params = FullReduceParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &a_buf,
        &partial_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    // Second pass
    let out = alloc_output(client, &[1], dtype)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params2 = FullReduceParams {
        numel: num_workgroups as u32,
    };
    let params_buf2 = create_params_buffer(client, &params2);

    reduce::launch_full_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        reduce_op,
        &partial_buf,
        &out_buf,
        &params_buf2,
        num_workgroups,
        dtype,
    )?;

    if is_mean {
        return client.div_scalar(&out, numel as f64);
    }
    Ok(out)
}

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

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = SoftmaxParams {
        batch_size: batch_size.max(1) as u32,
        dim_size: dim_size as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_softmax_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        batch_size.max(1),
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

    let grad_buf = get_tensor_buffer(&grad_contig)?;
    let output_buf = get_tensor_buffer(&output_contig)?;
    let d_input_buf = get_tensor_buffer(&d_input)?;

    let params = SoftmaxParams {
        batch_size: batch_size.max(1) as u32,
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
        batch_size.max(1),
        dtype,
    )?;

    Ok(d_input)
}

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
