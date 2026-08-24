//! Native WebGPU semiring matrix multiplication implementation.

use super::helpers::*;
use super::matmul_broadcast::flatten_batched_operands;
use crate::error::{Error, Result};
use crate::ops::matmul_output_shape;
use crate::ops::semiring::SemiringOp;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::semiring_matmul;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

pub(crate) fn native_semiring_matmul(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
    op: SemiringOp,
) -> Result<Tensor<WgpuRuntime>> {
    let dtype = a.dtype();

    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    if !op.validate_dtype(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "semiring_matmul",
        });
    }

    let out_shape = matmul_output_shape(a.shape(), b.shape())
        .ok_or_else(|| Error::shape_mismatch(a.shape(), b.shape()))?;

    // Operands the batched kernel cannot index directly are flattened to 3D first.
    if let Some((a3, b3)) = flatten_batched_operands(a, b, &out_shape)? {
        return native_semiring_matmul(client, &a3, &b3, op)?.reshape(&out_shape);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    // Handle 2D case
    if a_shape.len() == 2 && b_shape.len() == 2 {
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: 1,
        };
        let params_buf = create_params_buffer(client, &params);

        semiring_matmul::launch_semiring_matmul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            op,
            dtype,
        )?;

        return Ok(out);
    }

    // Handle batched (3D) case
    if a_shape.len() == 3 && b_shape.len() == 3 {
        let batch_size = a_shape[0];
        let m = a_shape[1];
        let k = a_shape[2];
        let n = b_shape[2];

        if b_shape[0] != batch_size {
            return Err(Error::ShapeMismatch {
                expected: vec![batch_size, m, k],
                got: b_shape.to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;

        let out = alloc_output(client, &out_shape, dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let b_buf = get_tensor_buffer(&b_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = MatmulParams {
            m: m as u32,
            k: k as u32,
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let params_buf = create_params_buffer(client, &params);

        semiring_matmul::launch_batched_semiring_matmul(
            client.pipeline_cache(),
            client.wgpu_queue(),
            &a_buf,
            &b_buf,
            &out_buf,
            &params_buf,
            m,
            n,
            batch_size,
            op,
            dtype,
        )?;

        return Ok(out);
    }

    Err(Error::BackendLimitation {
        backend: "WebGPU",
        operation: "semiring_matmul",
        reason: format!(
            "only supports 2D and 3D tensors, got shapes {:?} and {:?}",
            a.shape(),
            b.shape()
        ),
    })
}
