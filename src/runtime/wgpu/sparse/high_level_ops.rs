//! High-level sparse operations for the WebGPU runtime.
//!
//! Implements SparseOps trait for WgpuClient.
//! Supports SpMV, SpMM, element-wise operations, and format conversions.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::sparse::{SparseAlgorithms, validate_dtype_match};
use crate::dtype::DType;
use crate::dtype::Element;
use crate::error::{Error, Result};
use crate::ops::{ReduceOps, ScalarOps, TypeConversionOps};
use crate::sparse::{CooData, CscData, CsrData, SparseOps, SparseTensor};
use crate::tensor::Tensor;

impl SparseOps<WgpuRuntime> for WgpuClient {
    // =========================================================================
    // Low-Level Format-Specific Operations
    // =========================================================================

    fn spmv_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        self.spmv_csr_impl::<T>(row_ptrs, col_indices, values, x, shape)
    }

    fn spmm_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        self.spmm_csr_impl::<T>(row_ptrs, col_indices, values, b, shape)
    }

    // =========================================================================
    // CSR Element-wise Operations
    // =========================================================================

    fn add_csr<T: Element>(
        &self,
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_ptrs: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_ptrs: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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

    // =========================================================================
    // CSC Element-wise Operations
    // =========================================================================

    fn add_csc<T: Element>(
        &self,
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_col_ptrs: &Tensor<WgpuRuntime>,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_col_ptrs: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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

    // =========================================================================
    // COO Element-wise Operations (require format conversion)
    // =========================================================================

    fn add_coo<T: Element>(
        &self,
        a_row_indices: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_indices: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_indices: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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
        a_row_indices: &Tensor<WgpuRuntime>,
        a_col_indices: &Tensor<WgpuRuntime>,
        a_values: &Tensor<WgpuRuntime>,
        b_row_indices: &Tensor<WgpuRuntime>,
        b_col_indices: &Tensor<WgpuRuntime>,
        b_values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
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

    // =========================================================================
    // High-Level Format-Agnostic Operations
    // =========================================================================

    fn spmv(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        x: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
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
        a: &SparseTensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
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
        a: &Tensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_dtype_match(a.dtype(), b.dtype())?;

        // WGPU sparse kernels are F32-only today. For F64, use a loss-limited
        // compatibility path: cast -> run kernel -> cast back.
        if a.dtype() == DType::F64 {
            let a_f32 = self.cast(a, DType::F32)?;
            let b_f32 = match b {
                SparseTensor::Csr(data) => SparseTensor::Csr(CsrData {
                    row_ptrs: data.row_ptrs.clone(),
                    col_indices: data.col_indices.clone(),
                    values: self.cast(&data.values, DType::F32)?,
                    shape: data.shape,
                }),
                SparseTensor::Csc(data) => SparseTensor::Csc(CscData {
                    col_ptrs: data.col_ptrs.clone(),
                    row_indices: data.row_indices.clone(),
                    values: self.cast(&data.values, DType::F32)?,
                    shape: data.shape,
                }),
                SparseTensor::Coo(data) => SparseTensor::Coo(CooData {
                    row_indices: data.row_indices.clone(),
                    col_indices: data.col_indices.clone(),
                    values: self.cast(&data.values, DType::F32)?,
                    shape: data.shape,
                    sorted: data.sorted,
                }),
            };

            let out_f32 = self.dsmm(&a_f32, &b_f32)?;
            return self.cast(&out_f32, DType::F64);
        }

        // Convert B to CSC format (optimal for dense × sparse)
        let csc_b = match b {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Csr(data) => data.to_csc()?,
            SparseTensor::Coo(data) => data.to_csc()?,
        };

        // Delegate to algorithm trait (backend-consistent implementation)
        self.column_parallel_dsmm(a, &csc_b)
    }

    // =========================================================================
    // Sparse-Sparse Operations
    // =========================================================================

    fn sparse_add(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient addition
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
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient subtraction
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
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        validate_dtype_match(a.dtype(), b.dtype())?;

        // Convert both to CSR format (optimal for ESC SpGEMM)
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

        if csr_a.values.dtype() != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype: csr_a.values.dtype(),
                op: "wgpu sparse_matmul",
            });
        }

        // Delegate to ESC SpGEMM algorithm on WGPU.
        let result_csr = self.esc_spgemm_csr(&csr_a, &csr_b)?;

        Ok(SparseTensor::Csr(result_csr))
    }

    fn sparse_mul(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        b: &SparseTensor<WgpuRuntime>,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Validate shapes match
        if a.shape() != b.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![a.shape()[0], a.shape()[1]],
                got: vec![b.shape()[0], b.shape()[1]],
            });
        }

        // Validate dtypes match
        if a.dtype() != b.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: a.dtype(),
                rhs: b.dtype(),
            });
        }

        // Convert both to CSR format for efficient element-wise multiplication
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

    // =========================================================================
    // Sparse-Scalar Operations
    // =========================================================================

    fn sparse_scale(
        &self,
        a: &SparseTensor<WgpuRuntime>,
        scalar: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Handle empty tensors without calling mul_scalar
        if a.nnz() == 0 {
            return Ok(a.clone());
        }

        // mul_scalar already handles dtype dispatch
        match a {
            SparseTensor::Csr(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = CsrData {
                    row_ptrs: data.row_ptrs.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                };
                Ok(SparseTensor::Csr(result))
            }
            SparseTensor::Csc(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = CscData {
                    col_ptrs: data.col_ptrs.clone(),
                    row_indices: data.row_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                };
                Ok(SparseTensor::Csc(result))
            }
            SparseTensor::Coo(data) => {
                let scaled_values = self.mul_scalar(&data.values, scalar)?;
                let result = CooData {
                    row_indices: data.row_indices.clone(),
                    col_indices: data.col_indices.clone(),
                    values: scaled_values,
                    shape: data.shape,
                    sorted: data.sorted,
                };
                Ok(SparseTensor::Coo(result))
            }
        }
    }

    fn sparse_add_scalar(
        &self,
        _a: &SparseTensor<WgpuRuntime>,
        _scalar: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        // Adding a scalar to a sparse matrix would make it dense
        // (all implicit zeros become non-zero)
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }

    // =========================================================================
    // Reductions
    // =========================================================================

    fn sparse_sum(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // Get the values tensor (all formats store values the same way)
        let values = match a {
            SparseTensor::Csr(data) => &data.values,
            SparseTensor::Csc(data) => &data.values,
            SparseTensor::Coo(data) => &data.values,
        };

        // Sum all values using the existing reduce operation
        // Reduce over all dimensions (dim 0 for 1D tensor)
        self.sum(values, &[0], false)
    }

    fn sparse_sum_rows(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        // Use SpMV with a vector of ones to compute row sums efficiently
        // row_sums = A * ones_vector
        let ones = Tensor::try_ones(&[csr.shape[1]], csr.values.dtype(), csr.values.device())?;

        let dtype = csr.values.dtype();
        crate::dispatch_dtype!(dtype, T => {
            self.spmv_csr::<T>(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                &ones,
                csr.shape,
            )
        }, "sparse_sum_rows")
    }

    fn sparse_sum_cols(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        // For column sums, we use the transposed interpretation:
        // If A is CSC with shape [nrows, ncols], then A^T is CSR with shape [ncols, nrows]
        // col_sums(A) = row_sums(A^T) = A^T * ones_nrows
        //
        // In CSC format: col_ptrs -> row_ptrs, row_indices -> col_indices
        // So we can compute as if it were CSR for the transpose
        let ones = Tensor::try_ones(&[csc.shape[0]], csc.values.dtype(), csc.values.device())?;

        let dtype = csc.values.dtype();
        // Shape for transpose: [ncols, nrows]
        let transpose_shape = [csc.shape[1], csc.shape[0]];

        crate::dispatch_dtype!(dtype, T => {
            self.spmv_csr::<T>(
                &csc.col_ptrs,
                &csc.row_indices,
                &csc.values,
                &ones,
                transpose_shape,
            )
        }, "sparse_sum_cols")
    }

    fn sparse_nnz_per_row(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::ops::BinaryOps;

        // Convert to CSR format (optimal for row operations)
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let [nrows, _] = csr.shape;

        // nnz per row = row_ptrs[i+1] - row_ptrs[i]
        // Compute this using array slicing and subtraction
        let row_ptrs_start = csr.row_ptrs.narrow(0, 0, nrows)?;
        let row_ptrs_end = csr.row_ptrs.narrow(0, 1, nrows)?;

        self.sub(&row_ptrs_end, &row_ptrs_start)
    }

    fn sparse_nnz_per_col(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        use crate::ops::BinaryOps;

        // Convert to CSC format (optimal for column operations)
        let csc = match a {
            SparseTensor::Csc(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csc()?,
            SparseTensor::Csr(data) => data.to_csc()?,
        };

        let [_, ncols] = csc.shape;

        // nnz per col = col_ptrs[i+1] - col_ptrs[i]
        // Compute this using array slicing and subtraction
        let col_ptrs_start = csc.col_ptrs.narrow(0, 0, ncols)?;
        let col_ptrs_end = csc.col_ptrs.narrow(0, 1, ncols)?;

        self.sub(&col_ptrs_end, &col_ptrs_start)
    }

    // =========================================================================
    // Conversion
    // =========================================================================

    fn sparse_to_dense(&self, a: &SparseTensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // Convert to CSR format for efficient conversion
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let dtype = csr.values.dtype();

        // Dispatch to dtype-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            self.sparse_to_dense_impl::<T>(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                csr.shape,
            )
        }, "sparse_to_dense")
    }

    fn dense_to_sparse(
        &self,
        a: &Tensor<WgpuRuntime>,
        threshold: f64,
    ) -> Result<SparseTensor<WgpuRuntime>> {
        self.dense_to_coo_impl(a, threshold)
    }

    // =========================================================================
    // Format Conversions (Low-Level)
    // =========================================================================

    fn coo_to_csr<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.coo_to_csr_impl::<T>(row_indices, col_indices, values, shape)
    }

    fn coo_to_csc<T: Element>(
        &self,
        row_indices: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.coo_to_csc_impl::<T>(row_indices, col_indices, values, shape)
    }

    fn csr_to_coo<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_to_coo_impl::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_coo<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_to_coo_impl::<T>(col_ptrs, row_indices, values, shape)
    }

    fn csr_to_csc<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csr_to_csc_impl::<T>(row_ptrs, col_indices, values, shape)
    }

    fn csc_to_csr<T: Element>(
        &self,
        col_ptrs: &Tensor<WgpuRuntime>,
        row_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<(
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
        Tensor<WgpuRuntime>,
    )> {
        self.csc_to_csr_impl::<T>(col_ptrs, row_indices, values, shape)
    }

    fn extract_diagonal_csr<T: Element>(
        &self,
        row_ptrs: &Tensor<WgpuRuntime>,
        col_indices: &Tensor<WgpuRuntime>,
        values: &Tensor<WgpuRuntime>,
        shape: [usize; 2],
    ) -> Result<Tensor<WgpuRuntime>> {
        use super::super::ops::helpers::get_tensor_buffer;
        use super::super::shaders::launch_csr_extract_diagonal;
        use super::common::validate_wgpu_dtype;

        let [nrows, ncols] = shape;
        let n = nrows.min(ncols);
        let dtype = values.dtype();

        validate_wgpu_dtype(dtype, "extract_diagonal_csr")?;

        if n == 0 {
            return Tensor::try_empty(&[0], dtype, &self.device_id);
        }

        let diag = Tensor::<WgpuRuntime>::try_zeros(&[n], dtype, &self.device_id)?;

        // Create params buffer
        let params_buffer = self.create_uniform_buffer("diag_params", 16);
        self.write_buffer(&params_buffer, &[n as u32, 0u32, 0u32, 0u32]);

        let row_ptrs_buffer = get_tensor_buffer(row_ptrs)?;
        let col_indices_buffer = get_tensor_buffer(col_indices)?;
        let values_buffer = get_tensor_buffer(values)?;
        let diag_buffer = get_tensor_buffer(&diag)?;

        launch_csr_extract_diagonal(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &row_ptrs_buffer,
            &col_indices_buffer,
            &values_buffer,
            &diag_buffer,
            &params_buffer,
            n,
            dtype,
        )?;

        Ok(diag)
    }

    // =========================================================================
    // Transpose
    // =========================================================================

    fn sparse_transpose(&self, a: &SparseTensor<WgpuRuntime>) -> Result<SparseTensor<WgpuRuntime>> {
        use crate::sparse::{CooData, CscData, CsrData};

        match a {
            SparseTensor::Csr(data) => {
                // CSR transpose → CSC with swapped dimensions
                let [nrows, ncols] = data.shape;
                let dtype = data.values.dtype();
                let (col_ptrs, row_indices, values) = crate::dispatch_dtype!(dtype, T => {
                    self.csr_to_csc_impl::<T>(
                        &data.row_ptrs,
                        &data.col_indices,
                        &data.values,
                        data.shape,
                    )
                }, "sparse_transpose")?;

                Ok(SparseTensor::Csc(CscData {
                    col_ptrs,
                    row_indices,
                    values,
                    shape: [ncols, nrows], // Transpose swaps dimensions
                }))
            }
            SparseTensor::Csc(data) => {
                // CSC transpose → CSR with swapped dimensions
                let [nrows, ncols] = data.shape;
                let dtype = data.values.dtype();
                let (row_ptrs, col_indices, values) = crate::dispatch_dtype!(dtype, T => {
                    self.csc_to_csr_impl::<T>(
                        &data.col_ptrs,
                        &data.row_indices,
                        &data.values,
                        data.shape,
                    )
                }, "sparse_transpose")?;

                Ok(SparseTensor::Csr(CsrData {
                    row_ptrs,
                    col_indices,
                    values,
                    shape: [ncols, nrows], // Transpose swaps dimensions
                }))
            }
            SparseTensor::Coo(data) => {
                // COO transpose: just swap row and column indices
                let [nrows, ncols] = data.shape;
                Ok(SparseTensor::Coo(CooData {
                    row_indices: data.col_indices.clone(),
                    col_indices: data.row_indices.clone(),
                    values: data.values.clone(),
                    shape: [ncols, nrows],
                    sorted: false, // Sorting order is invalidated by transpose
                }))
            }
        }
    }
}
