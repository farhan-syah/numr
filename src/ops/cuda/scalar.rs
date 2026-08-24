//! CUDA implementation of scalar operations.

use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::cuda::kernels::launch_fused_mul_add_scalar;
use crate::runtime::cuda::ops::helpers::native_scalar_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ScalarOps<CudaRuntime> for CudaClient {
    fn add_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "add_scalar", scalar)
    }

    fn sub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "sub_scalar", scalar)
    }

    fn mul_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "mul_scalar", scalar)
    }

    fn div_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "div_scalar", scalar)
    }

    fn pow_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "pow_scalar", scalar)
    }

    fn rsub_scalar(&self, a: &Tensor<CudaRuntime>, scalar: f64) -> Result<Tensor<CudaRuntime>> {
        native_scalar_op(self, a, "rsub_scalar", scalar)
    }

    fn fused_mul_add_scalar(
        &self,
        a: &Tensor<CudaRuntime>,
        scale: f64,
        bias: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_fused_mul_add_scalar(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
                scale,
                bias,
            )?;
        }

        Ok(out)
    }
}
