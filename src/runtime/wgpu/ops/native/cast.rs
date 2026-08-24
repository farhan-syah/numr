//! Cast operation implementation for WebGPU.

use super::helpers::*;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::shaders::elementwise;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Native cast operation using WGSL compute shader.
///
/// Supports F32 ↔ I32 ↔ U32 conversions on GPU.
pub(crate) fn native_cast_op(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dst_dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    let src_dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let numel = a.numel();

    // Allocate output with target dtype
    let out = alloc_output(client, a.shape(), dst_dtype)?;

    let a_buf = get_tensor_buffer(&a_contig)?;
    let out_buf = get_tensor_buffer(&out)?;

    let params = CastParams {
        numel: numel as u32,
    };
    let params_buf = create_params_buffer(client, &params);

    elementwise::launch_cast_op(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &a_buf,
        &out_buf,
        &params_buf,
        numel,
        src_dtype,
        dst_dtype,
    )?;

    Ok(out)
}
