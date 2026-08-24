//! CPU implementation of logical operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::LogicalOps;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

impl LogicalOps<CpuRuntime> for CpuClient {
    fn logical_and(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
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

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr() as *const u8;
        let b_ptr = b_contig.ptr() as *const u8;
        let out_ptr = out.ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_and_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_or(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
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

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr() as *const u8;
        let b_ptr = b_contig.ptr() as *const u8;
        let out_ptr = out.ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_or_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_xor(
        &self,
        a: &Tensor<CpuRuntime>,
        b: &Tensor<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        // Validate both tensors are U8 (boolean)
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

        // Validate same shape
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: a.shape().to_vec(),
                got: b.shape().to_vec(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr() as *const u8;
        let b_ptr = b_contig.ptr() as *const u8;
        let out_ptr = out.ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_xor_kernel(a_ptr, b_ptr, out_ptr, numel);
        }

        Ok(out)
    }

    fn logical_not(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        // Validate tensor is U8 (boolean)
        if a.dtype() != DType::U8 {
            return Err(Error::DTypeMismatch {
                lhs: DType::U8,
                rhs: a.dtype(),
            });
        }

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr() as *const u8;
        let out_ptr = out.ptr() as *mut u8;
        let numel = a.numel();

        unsafe {
            kernels::logical_not_kernel(a_ptr, out_ptr, numel);
        }

        Ok(out)
    }
}
