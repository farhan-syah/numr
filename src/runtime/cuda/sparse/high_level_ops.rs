//! High-level sparse operations for the CUDA runtime.
//!
//! All operations use GPU kernels with zero CPU transfers.

use super::super::{CudaClient, CudaRuntime};
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::ReduceOps;
use crate::sparse::SparseOps;
use crate::tensor::Tensor;

// ============================================================================
// Private Helper Methods
// ============================================================================

impl CudaClient {
    // Private helper methods moved to algorithm-specific files (dsmm.rs, esc_spgemm.rs)
}

// ============================================================================
// SparseOps Trait Implementation
// ============================================================================

impl SparseOps<CudaRuntime> for CudaClient {
    // =========================================================================
    // Low-Level Format-Specific Operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        self.spmv_csr_impl::<T>(row_ptrs, col_indices, values, x, shape)
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        self.spmm_csr_impl::<T>(row_ptrs, col_indices, values, b, shape)
    }

    fn add_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.add_csr_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn sub_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.sub_csr_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn mul_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.mul_csr_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn div_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_ptrs: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.div_csr_impl::<T>(
            a_row_ptrs,
            a_col_indices,
            a_values,
            b_row_ptrs,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn add_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.add_csc_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
        )
    }

    fn sub_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.sub_csc_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
        )
    }

    fn mul_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.mul_csc_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
        )
    }

    fn div_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<CudaRuntime>,
        a_row_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_col_ptrs: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.div_csc_impl::<T>(
            a_col_ptrs,
            a_row_indices,
            a_values,
            b_col_ptrs,
            b_row_indices,
            b_values,
            shape,
        )
    }

    fn add_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.add_coo_impl::<T>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn sub_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.sub_coo_impl::<T>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn mul_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.mul_coo_impl::<T>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn div_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<CudaRuntime>,
        a_col_indices: &Tensor<CudaRuntime>,
        a_values: &Tensor<CudaRuntime>,
        b_row_indices: &Tensor<CudaRuntime>,
        b_col_indices: &Tensor<CudaRuntime>,
        b_values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.div_coo_impl::<T>(
            a_row_indices,
            a_col_indices,
            a_values,
            b_row_indices,
            b_col_indices,
            b_values,
            shape,
        )
    }

    fn coo_to_csr<T: Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.coo_to_csr_impl::<T>(row_indices, col_indices, values, shape)
    }

    fn coo_to_csc<T: Element>(
        &self,
        row_indices: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.coo_to_csc_impl::<T>(row_indices, col_indices, values, shape)
    }

    fn csr_to_coo<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.csr_to_coo_impl::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_coo<T: Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.csc_to_coo_impl::<T>(col_ptrs, row_indices, values, shape)
    }

    fn csr_to_csc<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.csr_to_csc_impl::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_csr<T: Element>(
        &self,
        col_ptrs: &Tensor<CudaRuntime>,
        row_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        self.csc_to_csr_impl::<T>(col_ptrs, row_indices, values, shape)
    }

    fn extract_diagonal_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<CudaRuntime>,
        col_indices: &Tensor<CudaRuntime>,
        values: &Tensor<CudaRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::dtype::DType;

        let [nrows, ncols] = shape;
        let n = nrows.min(ncols);
        let device = values.device();
        let dtype = values.dtype();

        if n == 0 {
            return Tensor::try_empty(&[0], dtype, device);
        }

        let out = Tensor::<CudaRuntime>::try_zeros(&[n], dtype, device)?;

        let row_ptrs_ptr = row_ptrs.ptr();
        let col_indices_ptr = col_indices.ptr();
        let values_ptr = values.ptr();
        let out_ptr = out.ptr();

        match dtype {
            DType::F32 => unsafe {
                super::super::kernels::launch_csr_extract_diagonal::<f32>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    out_ptr,
                    n,
                )?;
            },
            DType::F64 => unsafe {
                super::super::kernels::launch_csr_extract_diagonal::<f64>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    out_ptr,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::F16 => unsafe {
                super::super::kernels::launch_csr_extract_diagonal::<half::f16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    out_ptr,
                    n,
                )?;
            },
            #[cfg(feature = "f16")]
            DType::BF16 => unsafe {
                super::super::kernels::launch_csr_extract_diagonal::<half::bf16>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    row_ptrs_ptr,
                    col_indices_ptr,
                    values_ptr,
                    out_ptr,
                    n,
                )?;
            },
            dt => {
                return Err(Error::UnsupportedDType {
                    dtype: dt,
                    op: "extract_diagonal_csr",
                });
            }
        }

        Ok(out)
    }

    fn sparse_transpose(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        self.sparse_transpose_impl(a)
    }

    fn spmv(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        x: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for SpMV)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();

        // Dispatch to dtype-specific low-level implementation
        crate::dispatch_dtype!(dtype, T => {
                self.spmv_csr::<T>(
                    &csr.row_ptrs,
                    &csr.col_indices,
                    &csr.values,
                    x,
                    shape,
                )
            }, "spmv")
    }

    fn spmm(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for SpMM)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();

        // Dispatch to dtype-specific low-level implementation
        crate::dispatch_dtype!(dtype, T => {
                self.spmm_csr::<T>(
                    &csr.row_ptrs,
                    &csr.col_indices,
                    &csr.values,
                    b,
                    shape,
                )
            }, "spmm")
    }

    fn dsmm(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
        use crate::sparse::SparseTensor;

        // Validate dtypes match
        validate_dtype_match(a.dtype(), b.dtype())?;

        // Convert B to CSC format (optimal for dense × sparse)
        let csc_b = match b {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Csr(data) => data.to_csc()?,
            SparseTensor::Coo(data) => data.to_csc()?,
        };

        // Delegate to algorithm trait (backend-consistent implementation)
        self.column_parallel_dsmm(a, &csc_b)
    }

    fn sparse_add(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes and dtypes
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform addition and return as CSR
        let result = csr_a.add(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_sub(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes and dtypes
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform subtraction and return as CSR
        let result = csr_a.sub(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_matmul(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
        use crate::sparse::SparseTensor;

        // Validate dtypes match
        validate_dtype_match(a.dtype(), b.dtype())?;

        // Convert both to CSR format (optimal for ESC+Hash algorithm)
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Delegate to ESC SpGEMM algorithm (backend-consistent implementation)
        let result_csr = self.esc_spgemm_csr(&csr_a, &csr_b)?;

        Ok(SparseTensor::Csr(result_csr))
    }

    fn sparse_mul(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        b: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Validate shapes and dtypes
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format
        let csr_a = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let csr_b = match b {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Perform element-wise multiplication and return as CSR
        let result = csr_a.mul(&csr_b)?;
        Ok(SparseTensor::Csr(result))
    }

    fn sparse_scale(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
        scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::ops::ScalarOps;
        use crate::sparse::SparseTensor;

        // Handle empty tensors without calling mul_scalar
        if a.nnz() == 0 {
            return Ok(a.clone());
        }

        // Scale values tensor directly on GPU using ScalarOps
        match a {
            SparseTensor::Csr(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                Ok(SparseTensor::Csr(crate::sparse::CsrData {
                    row_ptrs: data.row_ptrs.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                }))
            }
            SparseTensor::Csc(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                Ok(SparseTensor::Csc(crate::sparse::CscData {
                    col_ptrs: data.col_ptrs.clone(),
                    row_indices: data.row_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                }))
            }
            SparseTensor::Coo(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                Ok(SparseTensor::Coo(crate::sparse::CooData {
                    row_indices: data.row_indices.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                    sorted: data.sorted,
                }))
            }
        }
    }

    fn sparse_add_scalar(
        &self,
        _a: &crate::sparse::SparseTensor<CudaRuntime>,
        _scalar: f64,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        // Adding a scalar to a sparse matrix would make it dense
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    fn sparse_sum(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::sparse::SparseTensor;

        // Sum all non-zero values using GPU reduce kernel (no CPU transfer)
        let values = match a {
            SparseTensor::Csr(data) => &data.values,
            SparseTensor::Csc(data) => &data.values,
            SparseTensor::Coo(data) => &data.values,
        };

        // Reduce sum over all dimensions
        self.sum(values, &[0], false)
    }

    fn sparse_sum_rows(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::csr_sum_rows_gpu;
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, _] = csr.shape;
        let dtype = csr.values.dtype();
        let device = csr.values.device();

        // Sum each row using GPU kernel
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csr_sum_rows_gpu::<T>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    &csr.row_ptrs,
                    &csr.values,
                    nrows,
                )
            }, "sparse_sum_rows")
        }
    }

    fn sparse_sum_cols(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::csc_sum_cols_gpu;
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [_, ncols] = csc.shape;
        let dtype = csc.values.dtype();
        let device = csc.values.device();

        // Sum each column using GPU kernel
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csc_sum_cols_gpu::<T>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    &csc.col_ptrs,
                    &csc.values,
                    ncols,
                )
            }, "sparse_sum_cols")
        }
    }

    fn sparse_nnz_per_row(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::csr_nnz_per_row_gpu;
        use crate::sparse::SparseTensor;

        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, _] = csr.shape;
        let device = csr.values.device();

        // Count non-zeros per row using GPU kernel
        unsafe {
            csr_nnz_per_row_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                device,
                &csr.row_ptrs,
                nrows,
            )
        }
    }

    fn sparse_nnz_per_col(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::csc_nnz_per_col_gpu;
        use crate::sparse::SparseTensor;

        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [_, ncols] = csc.shape;
        let device = csc.values.device();

        // Count non-zeros per column using GPU kernel
        unsafe {
            csc_nnz_per_col_gpu(
                &self.context,
                &self.stream,
                self.device.index,
                device,
                &csc.col_ptrs,
                ncols,
            )
        }
    }

    fn sparse_to_dense(
        &self,
        a: &crate::sparse::SparseTensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::csr_to_dense_gpu;
        use crate::sparse::SparseTensor;

        // Convert to CSR format for efficient conversion
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();
        let device = csr.values.device();

        // Expand CSR to dense using GPU kernel
        unsafe {
            crate::dispatch_dtype!(dtype, T => {
                csr_to_dense_gpu::<T>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    &csr.row_ptrs,
                    &csr.col_indices,
                    &csr.values,
                    shape,
                )
            }, "sparse_to_dense")
        }
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<CudaRuntime>,
        threshold: f64,
    ) -> Result<crate::sparse::SparseTensor<CudaRuntime>> {
        use crate::runtime::cuda::kernels::dense_to_coo_gpu;
        use crate::sparse::SparseTensor;

        // Validate input is 2D
        if a.ndim() != 2 {
            return Err(Error::Internal(format!(
                "Expected 2D tensor for dense_to_sparse, got {}D",
                a.ndim()
            )));
        }

        let shape_vec = a.shape();
        let nrows = shape_vec[0];
        let ncols = shape_vec[1];
        let dtype = a.dtype();
        let device = a.device();

        // Convert dense to COO using GPU kernel
        let (row_indices, col_indices, values) = unsafe {
            crate::dispatch_dtype!(dtype, T => {
                dense_to_coo_gpu::<T>(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    device,
                    dtype,
                    a,
                    T::from_f64(threshold),
                )
            }, "dense_to_sparse")?
        };

        // Create COO sparse tensor
        let coo = crate::sparse::CooData::new(row_indices, col_indices, values, [nrows, ncols])?;

        Ok(SparseTensor::Coo(coo))
    }
}
