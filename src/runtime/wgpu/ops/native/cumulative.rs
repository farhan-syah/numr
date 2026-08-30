//! Cumulative operation implementations for WebGPU.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::cumulative;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Native cumulative sum along a dimension.
///
/// Computes the cumulative sum of elements along the specified dimension.
/// Output has the same shape as input.
pub(crate) fn native_cumsum(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    // Normalize dimension
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

    let a_contig = ensure_contiguous(a)?;

    // Compute parameters. Never clamp these with `.max(1)`: the `numel() == 0`
    // guard below already rules a zero out, and a clamp would tell the shader
    // about a row the allocation does not contain.
    let scan_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output has same shape as input
    let out = alloc_output(client, shape, dtype)?;

    // A zero-element tensor has nothing to scan, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Hand the empty result back.
    if a.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case: operating on last dim or only dim
        let params = CumsumParams {
            scan_size: scan_size as u32,
            outer_size: outer_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_cumsum(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size,
            dtype,
        )?;
    } else {
        // Strided case: need inner_size
        let params = CumsumStridedParams {
            scan_size: scan_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size * inner_size;
        cumulative::launch_cumsum_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}

/// Native cumulative product along a dimension.
///
/// Computes the cumulative product of elements along the specified dimension.
/// Output has the same shape as input.
pub(crate) fn native_cumprod(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: isize,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        return Err(Error::InvalidDimension { dim, ndim });
    }

    // Normalize dimension
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

    let a_contig = ensure_contiguous(a)?;

    // Compute parameters. Never clamp these with `.max(1)`: the `numel() == 0`
    // guard below already rules a zero out, and a clamp would tell the shader
    // about a row the allocation does not contain.
    let scan_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output has same shape as input
    let out = alloc_output(client, shape, dtype)?;

    // A zero-element tensor has nothing to scan, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation. Hand the empty result back.
    if a.numel() == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case: operating on last dim or only dim
        let params = CumprodParams {
            scan_size: scan_size as u32,
            outer_size: outer_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_cumprod(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size,
            dtype,
        )?;
    } else {
        // Strided case: need inner_size
        let params = CumprodStridedParams {
            scan_size: scan_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size * inner_size;
        cumulative::launch_cumprod_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}

/// Native log-sum-exp reduction.
///
/// Computes log(sum(exp(x))) along specified dimensions in a numerically stable way.
/// Uses the identity: logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
///
/// Only supports floating-point dtypes (F32 on WebGPU, F16 with extension).
pub(crate) fn native_logsumexp(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dims: &[usize],
    keepdim: bool,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Logsumexp only makes sense for floating-point types
    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "logsumexp",
        });
    }

    // Empty dims means no reduction - return a copy (matches CPU behavior)
    if dims.is_empty() {
        return Ok(a.clone());
    }

    // For multi-dim reduction, reduce one dimension at a time (from highest to lowest)
    if dims.len() > 1 {
        let mut sorted_dims = dims.to_vec();
        sorted_dims.sort_by(|a, b| b.cmp(a)); // Descending order

        let mut result = a.clone();
        for &dim in &sorted_dims {
            result = native_logsumexp_single_dim(client, &result, dim, true)?;
        }

        // Remove dims if !keepdim
        if !keepdim {
            let mut out_shape: Vec<usize> = shape.to_vec();
            for &dim in &sorted_dims {
                out_shape.remove(dim);
            }
            if out_shape.is_empty() {
                out_shape.push(1);
            }
            result = result.reshape(&out_shape)?;
        }

        return Ok(result);
    }

    // Single dimension reduction
    let dim = dims[0];
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    native_logsumexp_single_dim(client, a, dim, keepdim)
}

/// Logsumexp along a single dimension.
fn native_logsumexp_single_dim(
    client: &WgpuClient,
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

    // Never clamp these with `.max(1)`: the `numel() == 0` guard below already
    // rules a zero out, and a clamp would tell the shader about a row the
    // allocation does not contain.
    let reduce_size = shape[dim];
    let outer_size: usize = shape[..dim].iter().product();
    let inner_size: usize = shape[dim + 1..].iter().product();

    // Output shape
    let out_shape: Vec<usize> = if keepdim {
        let mut s = shape.to_vec();
        s[dim] = 1;
        s
    } else {
        let mut s: Vec<usize> = shape[..dim].to_vec();
        s.extend_from_slice(&shape[dim + 1..]);
        if s.is_empty() {
            s.push(1);
        }
        s
    };

    let out = alloc_output(client, &out_shape, dtype)?;

    // A zero-element output has nothing to hold, and `get_tensor_buffer` has no
    // buffer to return for a zero-byte allocation.
    if out.numel() == 0 {
        return Ok(out);
    }

    // A zero-length reduce dimension over a NON-empty output: every element
    // folds over no input, so `log(sum of nothing)` is `log(0)` = -inf, and no
    // dispatch can produce it — the input is the zero-byte allocation.
    if a.numel() == 0 {
        return Tensor::<WgpuRuntime>::full_scalar(
            &out_shape,
            dtype,
            f64::NEG_INFINITY,
            RuntimeClient::device(client),
        );
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    // Use strided kernel if not operating on last dimension
    if dim == ndim - 1 || inner_size == 1 {
        // Contiguous case
        let params = LogsumexpParams {
            reduce_size: reduce_size as u32,
            outer_size: outer_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        cumulative::launch_logsumexp(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            outer_size,
            dtype,
        )?;
    } else {
        // Strided case
        let params = LogsumexpStridedParams {
            reduce_size: reduce_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        let total_inner = outer_size * inner_size;
        cumulative::launch_logsumexp_strided(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &out_buf,
            &params_buf,
            total_inner,
            dtype,
        )?;
    }

    Ok(out)
}
