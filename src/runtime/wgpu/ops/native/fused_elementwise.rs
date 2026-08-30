//! Fused elementwise native GPU operations for WebGPU.

use super::helpers::*;
use crate::error::{Error, Result};
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::fused_elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Native fused_mul_add: out = a * b + c. F32, I32, U32.
pub(crate) fn native_fused_mul_add(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    c: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype || c.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: if b.dtype() != dtype {
                b.dtype()
            } else {
                c.dtype()
            },
        });
    }
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: if a.shape() != b.shape() {
                b.shape().to_vec()
            } else {
                c.shape().to_vec()
            },
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let c_contig = ensure_contiguous(c)?;
    let numel = a.numel();
    let out = alloc_output(client, a.shape(), dtype)?;

    // A zero-element tensor has nothing to compute, and `get_tensor_buffer` has
    // no buffer to return for a zero-byte allocation.
    if numel == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let c_buf = get_tensor_buffer(&c_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    fused_elementwise::launch_fused_mul_add(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &c_buf,
        &out_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

/// Native fused_add_mul: out = (a + b) * c. F32, I32, U32.
pub(crate) fn native_fused_add_mul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    c: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype || c.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: if b.dtype() != dtype {
                b.dtype()
            } else {
                c.dtype()
            },
        });
    }
    if a.shape() != b.shape() || a.shape() != c.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: if a.shape() != b.shape() {
                b.shape().to_vec()
            } else {
                c.shape().to_vec()
            },
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let c_contig = ensure_contiguous(c)?;
    let numel = a.numel();
    let out = alloc_output(client, a.shape(), dtype)?;

    // A zero-element tensor has nothing to compute, and `get_tensor_buffer` has
    // no buffer to return for a zero-byte allocation.
    if numel == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let c_buf = get_tensor_buffer(&c_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    fused_elementwise::launch_fused_add_mul(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &b_buf,
        &c_buf,
        &out_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}

/// Native fused_mul_add_scalar: out = a * scale + bias. F32, I32, U32.
pub(crate) fn native_fused_mul_add_scalar(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    scale: f64,
    bias: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();
    let out = alloc_output(client, a.shape(), dtype)?;

    // A zero-element tensor has nothing to compute, and `get_tensor_buffer` has
    // no buffer to return for a zero-byte allocation.
    if numel == 0 {
        return Ok(out);
    }

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    fused_elementwise::launch_fused_mul_add_scalar(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        numel,
        dtype,
        scale,
        bias,
    )?;

    Ok(out)
}
