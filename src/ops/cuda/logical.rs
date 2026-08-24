//! CUDA implementation of logical operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::LogicalOps;
use crate::runtime::cuda::kernels::{
    launch_logical_and_op, launch_logical_not_op, launch_logical_or_op, launch_logical_xor_op,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Validate inputs for binary logical operations (U8 dtype, matching shapes).
fn validate_logical_inputs(a: &Tensor<CudaRuntime>, b: &Tensor<CudaRuntime>) -> Result<()> {
    if a.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: a.dtype(),
        });
    }
    if b.dtype() != DType::U8 {
        return Err(Error::DTypeMismatch {
            lhs: DType::U8,
            rhs: b.dtype(),
        });
    }
    if a.shape() != b.shape() {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }
    Ok(())
}

impl LogicalOps<CudaRuntime> for CudaClient {
    fn logical_and(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_logical_inputs(a, b)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_logical_and_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_or(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_logical_inputs(a, b)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_logical_or_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_xor(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        validate_logical_inputs(a, b)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_logical_xor_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn logical_not(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_logical_not_op(
                &self.context,
                &self.stream,
                self.device.index,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }
}
