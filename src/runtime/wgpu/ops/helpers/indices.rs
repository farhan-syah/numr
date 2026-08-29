//! Index tensor dtype coercion.
//!
//! WGSL has no 64-bit integer, so an I64 index tensor is cast to I32 on the GPU
//! before any indexing kernel sees it.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

/// Cast indices tensor to I32 for WebGPU shaders.
/// WebGPU natively supports I32 indices; I64 indices are cast on GPU.
/// Returns an error for unsupported index dtypes.
pub(crate) fn ensure_i32_indices(
    client: &WgpuClient,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    use crate::ops::TypeConversionOps;
    match indices.dtype() {
        DType::I32 => Ok(indices.clone()),
        DType::I64 => client.cast(indices, DType::I32),
        other => Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: other,
        }),
    }
}
