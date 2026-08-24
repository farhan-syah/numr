//! Sparse matrix-vector and sparse matrix-matrix multiplication

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::{
    launch_csr_spmm, launch_csr_spmv, launch_csr_spmv_warp, should_use_warp_kernel,
};
use crate::tensor::Tensor;

impl CudaClient {
    pub(crate) fn spmv_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let [nrows, ncols] = shape;
        let device = values.device();
        let dtype = values.dtype();

        // Validate input shapes
        if x.numel() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x.numel()],
            });
        }

        // Allocate output on GPU
        let y = Tensor::<CudaRuntime>::try_zeros(&[nrows], dtype, device)?;

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.ptr();
        let col_indices_ptr = col_indices.ptr();
        let values_ptr = values.ptr();
        let x_ptr = x.ptr();
        let y_ptr = y.ptr();

        // Choose optimal kernel based on sparsity
        let nnz = values.numel();
        let avg_nnz_per_row = nnz as f32 / nrows.max(1) as f32;

        // Dispatch based on dtype (only F32/F64/F16/BF16 supported on CUDA)
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f32>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f32>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            DType::F64 => unsafe {
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f64>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f64>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                use half::f16;
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<f16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<f16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                use half::bf16;
                if should_use_warp_kernel(avg_nnz_per_row) {
                    launch_csr_spmv_warp::<bf16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                } else {
                    launch_csr_spmv::<bf16>(
                        &self.context,
                        &self.stream,
                        self.device.index,
                        row_ptrs_ptr,
                        col_indices_ptr,
                        values_ptr,
                        x_ptr,
                        y_ptr,
                        nrows,
                    )?;
                }
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CUDA sparse SpMV: {:?}",
                    dtype
                )));
            }
        }

        // Return GPU tensor (no transfer back!)
        Ok(y)
    }

    pub(crate) fn spmm_csr_impl<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        let [m, k] = shape;
        let device = values.device();
        let dtype = values.dtype();

        // Validate B is 2D
        if b.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for SpMM, got {}D",
                b.ndim()
            )));
        }

        let b_shape = b.shape();
        let b_k = b_shape[0];
        let n = b_shape[1];

        // Validate dimensions match
        if b_k != k {
            return Err(Error::ShapeMismatch {
                expected: vec![k],
                got: vec![b_k],
            });
        }

        // Allocate output on GPU
        let c = Tensor::<CudaRuntime>::try_zeros(&[m, n], dtype, device)?;

        // Get device pointers (no data transfer!)
        let row_ptrs_ptr = row_ptrs.ptr();
        let col_indices_ptr = col_indices.ptr();
        let values_ptr = values.ptr();
        let b_ptr = b.ptr();
        let c_ptr = c.ptr();

        // Dispatch based on dtype (only F32/F64/F16/BF16 supported on CUDA)
        use crate::dtype::DType;
        match dtype {
            DType::F32 => unsafe {
                launch_csr_spmm::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            DType::F64 => unsafe {
                launch_csr_spmm::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                use half::f16;
                launch_csr_spmm::<f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                use half::bf16;
                launch_csr_spmm::<bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    b_ptr,
                    c_ptr,
                    m,
                    n,
                )?;
            },
            _ => {
                return Err(Error::Internal(format!(
                    "Unsupported dtype for CUDA sparse SpMM: {:?}",
                    dtype
                )));
            }
        }

        // Return GPU tensor (no transfer back!)
        Ok(c)
    }
}
