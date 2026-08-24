//! CSR element-wise operations: add, sub, mul, div, scalar_mul, scalar_add
//!
//! All element-wise operations dispatch to runtime-specific implementations
//! via the SparseOps trait, enabling GPU acceleration when available.

use super::CsrData;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};

impl<R: Runtime<DType = DType>> CsrData<R> {
    /// Element-wise addition: C = A + B
    ///
    /// Computes the sum of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSR matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSR matrix containing the element-wise sum
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Row-by-row merge of sorted column indices using union semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-row merge
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A + B:
    /// // [1, 0]      [0, 2]      [1, 2]
    /// // [0, 3]  +   [4, 0]  =   [4, 3]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[1.0f32, 3.0], [2, 2], &device)?.to_csr()?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[1, 0], &[2.0f32, 4.0], [2, 2], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(a) = a_sp { if let numr::sparse::SparseTensor::Csr(b) = b_sp {
    /// let c = a.add(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        let dtype = self.dtype();
        let device = self.values.device();

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_ptrs, out_col_indices, out_values) = client.add_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                &other.row_ptrs,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_ptrs: out_row_ptrs,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csr_add")
    }

    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSR matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSR matrix containing the element-wise difference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Row-by-row merge of sorted column indices using union semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-row merge
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 1]      [3, -1]
    /// // [0, 4]  -   [0, 3]  =   [0,  1]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[5.0f32, 4.0], [2, 2], &device)?.to_csr()?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[2.0f32, 1.0, 3.0], [2, 2], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(a) = a_sp { if let numr::sparse::SparseTensor::Csr(b) = b_sp {
    /// let c = a.sub(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        let dtype = self.dtype();
        let device = self.values.device();

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_ptrs, out_col_indices, out_values) = client.sub_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                &other.row_ptrs,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_ptrs: out_row_ptrs,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csr_sub")
    }

    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSR matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSR matrix containing the element-wise product
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Row-by-row intersection of sorted column indices using intersection semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-row merge
    /// - Result has at most min(nnz_a, nnz_b) non-zeros
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A .* B:
    /// // [2, 3]      [4, 0]      [8, 0]
    /// // [0, 5]  .*  [6, 7]  =   [0, 35]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[2.0f32, 3.0, 5.0], [2, 2], &device)?.to_csr()?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[4.0f32, 7.0], [2, 2], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(a) = a_sp { if let numr::sparse::SparseTensor::Csr(b) = b_sp {
    /// let c = a.mul(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        let dtype = self.dtype();
        let device = self.values.device();

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_ptrs, out_col_indices, out_values) = client.mul_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                &other.row_ptrs,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_ptrs: out_row_ptrs,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csr_mul")
    }

    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse matrices.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result (same as mul for sparsity).
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSR matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSR matrix containing the element-wise quotient
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    /// - Division by zero occurs (no special handling, produces inf/nan)
    ///
    /// # Note
    ///
    /// Division by zero in the result will produce inf or nan according to
    /// IEEE 754 floating point rules.
    pub fn div(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape != other.shape {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape[0], self.shape[1]],
                got: vec![other.shape[0], other.shape[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        let dtype = self.dtype();
        let device = self.values.device();

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_ptrs, out_col_indices, out_values) = client.div_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                &other.row_ptrs,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_ptrs: out_row_ptrs,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csr_div")
    }

    /// Scalar multiplication: C = A * s
    ///
    /// Multiplies all non-zero values by a scalar.
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to multiply by
    ///
    /// # Returns
    ///
    /// A new CSR matrix with all values scaled
    ///
    /// # Performance
    ///
    /// O(nnz) - simply scales the values tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          C = A * 2:
    /// // [1, 2]      [2, 4]
    /// // [3, 0]      [6, 0]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 0], &[1.0f32, 2.0, 3.0], [2, 2], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(a) = a_sp {
    /// let c = a.scalar_mul(2.0)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn scalar_mul<T: Element>(&self, scalar: T) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        let device = self.values.device();
        let client = R::default_client(device);

        // Convert scalar to f64 for ScalarOps trait
        let scalar_f64 = scalar.to_f64();
        let scaled_values = client.mul_scalar(&self.values, scalar_f64)?;

        Ok(Self {
            row_ptrs: self.row_ptrs.clone(),
            col_indices: self.col_indices.clone(),
            values: scaled_values,
            shape: self.shape,
        })
    }

    /// Scalar addition: C = A + s
    ///
    /// Adds a scalar to all elements (including implicit zeros).
    ///
    /// # Warning
    ///
    /// This operation converts the sparse matrix to dense since adding to
    /// implicit zeros creates non-zero values everywhere.
    ///
    /// # Arguments
    ///
    /// * `scalar` - Scalar value to add
    ///
    /// # Returns
    ///
    /// Error indicating the operation would create a dense result
    pub fn scalar_add<T: Element>(&self, _scalar: T) -> Result<Self> {
        Err(Error::Internal(
            "Scalar addition to sparse matrix creates dense result - convert to dense first"
                .to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::{Runtime, cpu::CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_csr_add() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create first matrix (2x2):
        // [1.0, 0.0]
        // [0.0, 2.0]
        let row_ptrs_a = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let col_indices_a = Tensor::<CpuRuntime>::from_slice(&[0i64, 1], &[2], &device).unwrap();
        let values_a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();
        let a = CsrData::new(row_ptrs_a, col_indices_a, values_a, [2, 2]).unwrap();

        // Create second matrix (2x2):
        // [0.0, 3.0]
        // [4.0, 0.0]
        let row_ptrs_b = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 2], &[3], &device).unwrap();
        let col_indices_b = Tensor::<CpuRuntime>::from_slice(&[1i64, 0], &[2], &device).unwrap();
        let values_b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device).unwrap();
        let b = CsrData::new(row_ptrs_b, col_indices_b, values_b, [2, 2]).unwrap();

        // Add: should get [1, 3], [4, 2]
        let c = a.add(&b).unwrap();

        assert_eq!(c.nnz(), 4);
        let values: Vec<f32> = c.values.to_vec();
        assert_eq!(values.len(), 4);
        // Values should be [1, 3, 4, 2] based on column order
    }

    #[test]
    fn test_csr_mul() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create first matrix (2x2):
        // [2.0, 3.0]
        // [0.0, 5.0]
        let row_ptrs_a = Tensor::<CpuRuntime>::from_slice(&[0i64, 2, 3], &[3], &device).unwrap();
        let col_indices_a = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 1], &[3], &device).unwrap();
        let values_a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 5.0], &[3], &device).unwrap();
        let a = CsrData::new(row_ptrs_a, col_indices_a, values_a, [2, 2]).unwrap();

        // Create second matrix (2x2):
        // [4.0, 0.0]
        // [6.0, 7.0]
        let row_ptrs_b = Tensor::<CpuRuntime>::from_slice(&[0i64, 1, 3], &[3], &device).unwrap();
        let col_indices_b = Tensor::<CpuRuntime>::from_slice(&[0i64, 0, 1], &[3], &device).unwrap();
        let values_b =
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 6.0, 7.0], &[3], &device).unwrap();
        let b = CsrData::new(row_ptrs_b, col_indices_b, values_b, [2, 2]).unwrap();

        // Mul (Hadamard): only where both non-zero
        // Row 0: [2*4, 3*0] = [8, 0] -> only col 0
        // Row 1: [0*6, 5*7] = [0, 35] -> only col 1
        let c = a.mul(&b).unwrap();

        assert_eq!(c.nnz(), 2); // Only 2 overlapping positions
        let values: Vec<f32> = c.values.to_vec();
        assert!((values[0] - 8.0).abs() < 1e-5); // First value should be 8
        assert!((values[1] - 35.0).abs() < 1e-5); // Second value should be 35
    }
}
