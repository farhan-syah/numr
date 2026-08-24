//! CPU implementation of type conversion operations.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// TypeConversionOps implementation for CPU runtime.
impl TypeConversionOps<CpuRuntime> for CpuClient {
    fn cast(&self, a: &Tensor<CpuRuntime>, target_dtype: DType) -> Result<Tensor<CpuRuntime>> {
        let src_dtype = a.dtype();

        // No-op if types match
        if src_dtype == target_dtype {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::try_empty(shape, target_dtype, &self.device)?;

        let src_ptr = a_contig.ptr() as *const u8;
        let dst_ptr = out.ptr() as *mut u8;

        unsafe {
            kernels::cast_kernel(src_ptr, dst_ptr, numel, src_dtype, target_dtype)?;
        }

        Ok(out)
    }
}
