//! CUDA implementation of scalar operations.

use crate::error::Result;
use crate::ops::{ScalarOps, TypeConversionOps};
use crate::runtime::cuda::kernels::launch_fused_mul_add_scalar;
use crate::runtime::cuda::ops::helpers::native_scalar_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{ensure_contiguous, pow_scalar_output_dtype};
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
        let out_dtype = pow_scalar_output_dtype(a.dtype(), scalar);
        if out_dtype != a.dtype() {
            // An integer raised to a negative or fractional power is a
            // non-integer real, so the result promotes to F64. The cast plus the
            // existing F64 kernel matches what CPU does, and adds no new kernel
            // surface.
            let promoted = self.cast(a, out_dtype)?;
            return native_scalar_op(self, &promoted, "pow_scalar", scalar);
        }
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
