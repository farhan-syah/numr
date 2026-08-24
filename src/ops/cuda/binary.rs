//! Binary operations for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::BinaryOps;
use crate::runtime::cuda::kernels::{launch_fused_add_mul, launch_fused_mul_add};
use crate::runtime::cuda::ops::helpers::{native_binary_op, native_binary_op_into};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl BinaryOps<CudaRuntime> for CudaClient {
    fn add(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "add")
    }

    fn sub(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "sub")
    }

    fn mul(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "mul")
    }

    fn div(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "div")
    }

    fn pow(&self, a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "pow")
    }

    fn maximum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "max")
    }

    fn minimum(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, a, b, "min")
    }

    fn atan2(
        &self,
        y: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        native_binary_op(self, y, x, "atan2")
    }

    fn fused_mul_add(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype || c.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if b.dtype() != dtype {
                    b.dtype()
                } else {
                    c.dtype()
                },
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let c_contig = ensure_contiguous(c)?;
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_fused_mul_add(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                c_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn fused_add_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        c: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype || c.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if b.dtype() != dtype {
                    b.dtype()
                } else {
                    c.dtype()
                },
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let c_contig = ensure_contiguous(c)?;
        let out = Tensor::<CudaRuntime>::try_empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_fused_add_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                c_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn add_into(
        &self,
        out: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<()> {
        native_binary_op_into(self, out, a, b, "add")
    }
}
