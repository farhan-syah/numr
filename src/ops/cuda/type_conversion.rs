//! Type conversion operations for CUDA runtime
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::cuda::kernels::launch_cast;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl TypeConversionOps<CudaRuntime> for CudaClient {
    fn cast(&self, a: &Tensor<CudaRuntime>, target_dtype: DType) -> Result<Tensor<CudaRuntime>> {
        let src_dtype = a.dtype();

        // No-op if types match
        if src_dtype == target_dtype {
            return Ok(a.clone());
        }

        let shape = a.shape();
        let numel = a.numel();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::try_empty(shape, target_dtype, &self.device)?;

        unsafe {
            launch_cast(
                &self.context,
                &self.stream,
                self.device.index,
                src_dtype,
                target_dtype,
                a_contig.ptr(),
                out.ptr(),
                numel,
            )?;
        }

        Ok(out)
    }
}
