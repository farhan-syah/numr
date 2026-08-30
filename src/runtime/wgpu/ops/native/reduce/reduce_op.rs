//! Dimension and full-tensor reductions for WebGPU.
//!
//! Covers sum, prod, mean, max and min: the dispatch entry point, the wide
//! integer accumulation path, the single-dimension kernel and the full-tensor
//! kernel.

use super::super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::ops::reduce::{max_identity, min_identity, reduce_output_shape};
use crate::runtime::RuntimeClient;
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

/// Value one output element takes when a reduction folds over zero inputs.
///
/// `sum`/`mean`/`any` fold to 0, `prod`/`all` to 1, and `max`/`min` to the
/// dtype's own extreme — negative/positive infinity for floats, which is what
/// the CPU and CUDA kernels answer for the same shape.
fn empty_reduce_identity(op: &str, dtype: DType) -> f64 {
    match op {
        "prod" | "all" => 1.0,
        "max" => max_identity(dtype),
        "min" => min_identity(dtype),
        _ => 0.0,
    }
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

    // A zero-element output has nothing to hold, and `get_tensor_buffer` has no
    // buffer to return for its zero-byte allocation. Never restore a `.max(1)` on
    // `outer_size`, `inner_size` or `numel_out`: it would make this guard
    // unreachable and bind a buffer the output does not have.
    if out.numel() == 0 {
        return Ok(out);
    }

    // A zero-length reduce dimension over a NON-empty output: every output
    // element folds over no input, so the value is the reduction's identity and
    // no dispatch can produce it — the input is the zero-byte allocation. `sum`,
    // `mean` and `any` fold to the additive identity, `prod` and `all` to the
    // multiplicative one, and `max`/`min` to the dtype's own extreme (-/+inf for
    // floats). These must match what CPU and CUDA answer for the same shape.
    if a.numel() == 0 {
        return Tensor::<WgpuRuntime>::full_scalar(
            &out_shape,
            dtype,
            empty_reduce_identity(op, dtype),
            client.device(),
        );
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ReduceParams {
        reduce_size: reduce_size as u32,
        outer_size: outer_size as u32,
        inner_size: inner_size as u32,
        numel_out: numel_out as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    reduce::launch_reduce_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel_out,
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

    // A zero-element input has no buffer to bind, so the value is produced here
    // rather than by a dispatch: each op folds to its own identity. The caller
    // reshapes this to the reduction's output shape, so the `[1]` result keeps
    // that contract.
    if numel == 0 {
        return Tensor::<WgpuRuntime>::full_scalar(
            &[1],
            dtype,
            empty_reduce_identity(op, dtype),
            client.device(),
        );
    }

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
