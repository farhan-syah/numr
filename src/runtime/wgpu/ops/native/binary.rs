//! Binary and scalar operation implementations for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

pub(crate) fn native_binary_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Use shared helpers for validation (same as CPU and CUDA backends)
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    let out = alloc_output(client, &out_shape, dtype)?;
    run_binary(client, op, a, b, &out, &out_shape, dtype)?;

    Ok(out)
}

/// Destination-passing binary op: writes `op(a, b)` into the caller-owned `out`
/// tensor instead of allocating. Required for destination-passing workflows
/// where the output buffer must have a stable identity. `out` must be
/// contiguous, share the inputs' dtype, and have shape `broadcast(a, b)`.
pub(crate) fn native_binary_op_into(
    client: &WgpuClient,
    op: &'static str,
    out: &Tensor<WgpuRuntime>,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<()> {
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    if out.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: out.dtype(),
        });
    }
    if out.shape() != out_shape.as_slice() {
        return Err(Error::ShapeMismatch {
            expected: out_shape,
            got: out.shape().to_vec(),
        });
    }
    if !out.is_contiguous() {
        return Err(Error::Backend(
            "native_binary_op_into: destination tensor must be contiguous".into(),
        ));
    }

    run_binary(client, op, a, b, out, &out_shape, dtype)
}

/// Shared dispatch for both the allocating and destination-passing paths.
/// `out` must already have shape `out_shape` and the validated dtype.
fn run_binary(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    out: &Tensor<WgpuRuntime>,
    out_shape: &[usize],
    dtype: DType,
) -> Result<()> {
    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;

    let numel: usize = out_shape.iter().product();

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(out)?;

    // Use broadcast kernel if shapes differ, element-wise kernel otherwise
    if a.shape() != b.shape() {
        launch_broadcast(
            client,
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            a.shape(),
            b.shape(),
            out_shape,
            numel,
            dtype,
        )?;
    } else {
        let params = BinaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        elementwise::launch_binary_op(
            client.pipeline_cache(),
            client.wgpu_queue(),
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;
    }

    Ok(())
}

pub(crate) fn native_scalar_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    scalar: f64,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    let out = alloc_output(client, a.shape(), dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = ScalarParams::new(numel as u32, scalar, dtype);
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_scalar_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        op,
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        dtype,
    )?;

    Ok(out)
}
