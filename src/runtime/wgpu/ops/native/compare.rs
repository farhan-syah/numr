//! Compare operation implementation for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

pub(crate) fn native_compare_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Use shared helpers for validation (same as CPU and CUDA backends)
    let dtype = validate_binary_dtypes(a, b)?;

    // Broadcasting not yet implemented natively
    if a.shape() != b.shape() {
        return crate::runtime::fallback::compare_op_fallback(
            a,
            b,
            match op {
                "eq" => crate::ops::CompareOp::Eq,
                "ne" => crate::ops::CompareOp::Ne,
                "lt" => crate::ops::CompareOp::Lt,
                "le" => crate::ops::CompareOp::Le,
                "gt" => crate::ops::CompareOp::Gt,
                "ge" => crate::ops::CompareOp::Ge,
                _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
            },
            &client.device_id,
            op,
        );
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel = a.numel();

    // Output is F32 (comparison results: 1.0 = true, 0.0 = false)
    // This allows using results directly in arithmetic ops (e.g., mask * tensor)
    let out = alloc_output(client, a.shape(), DType::F32)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = BinaryParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_compare_op(
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

    Ok(out)
}
