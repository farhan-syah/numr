//! Compare operation implementation for WebGPU.

use super::helpers::*;
use crate::error::Result;
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{compute_broadcast_shape, ensure_contiguous, validate_binary_dtypes};
use crate::tensor::Tensor;

pub(crate) fn native_compare_op(
    client: &WgpuClient,
    op: &'static str,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    // Use shared helpers for validation (same as CPU and CUDA backends)
    let dtype = validate_binary_dtypes(a, b)?;
    let out_shape = compute_broadcast_shape(a, b)?;

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let numel: usize = out_shape.iter().product();

    // The mask carries the input dtype, holding 1/0, matching CPU and CUDA.
    let out = alloc_output(client, &out_shape, dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let b_buf = get_tensor_buffer(&b_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    if a.shape() != b.shape() {
        launch_broadcast(
            client,
            op,
            &a_buf,
            &b_buf,
            &out_buf,
            a.shape(),
            b.shape(),
            &out_shape,
            numel,
            dtype,
        )?;
    } else {
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
    }

    Ok(out)
}
