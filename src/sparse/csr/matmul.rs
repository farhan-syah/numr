//! CSR matrix multiplication: spmv, spmm

use super::CsrData;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{CscData, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CsrData<R> {
    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// Computes the product of this sparse matrix with a dense vector.
    ///
    /// # Arguments
    ///
    /// * `x` - Dense vector of length `ncols` (or shape `` `[ncols]` `` or `` `[ncols, 1]` ``)
    ///
    /// # Returns
    ///
    /// Dense vector of length `` `nrows` ``
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `x` length doesn't match matrix ncols
    /// - dtype mismatch between matrix and vector
    ///
    /// # Algorithm
    ///
    /// For each row i:
    /// ```text
    /// `` `y[i] = sum(values[j] * x[col_indices[j]]) for j in row_ptrs[i]..row_ptrs[i+1]` ``
    /// ```
    ///
    /// # Performance
    ///
    /// - O(nnz) time complexity
    /// - CSR format provides optimal memory access pattern for SpMV
    /// - Each row's non-zeros are contiguous in memory
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 0], &[1.0f32, 2.0, 3.0], [2, 2], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(csr) = sp {
    /// let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
    /// let y = csr.spmv(&x)?;  // y = [1*1 + 2*2, 3*1] = [5, 3]
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn spmv(&self, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        use crate::sparse::SparseOps;

        let [nrows, ncols] = self.shape;
        let dtype = self.dtype();
        let device = self.values.device();

        // Validate vector length
        let x_len = x.numel();
        if x_len != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![x_len],
            });
        }

        // Validate dtype match
        if x.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: x.dtype(),
            });
        }

        // Handle empty matrix case
        if self.is_empty() {
            crate::dispatch_dtype!(dtype, T => {
                let zeros: Vec<T> = vec![T::zero(); nrows];
                return Tensor::try_from_slice(&zeros, &[nrows], device);
            }, "spmv empty");
        }

        // Get runtime client to dispatch to backend-specific implementation
        let client = R::default_client(device);

        // Dispatch on dtype to call backend spmv_csr
        crate::dispatch_dtype!(dtype, T => {
            return client.spmv_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                x,
                self.shape,
            );
        }, "spmv");
    }

    /// Sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// Computes the product of this sparse matrix with a dense matrix.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix of shape `` `[K, N]` `` where K == ncols of sparse matrix
    ///
    /// # Returns
    ///
    /// Dense matrix of shape `` `[M, N]` `` where M == nrows of sparse matrix
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `b` first dimension doesn't match matrix ncols
    /// - `b` is not 2D
    /// - dtype mismatch between matrix and input
    ///
    /// # Algorithm
    ///
    /// For each row i of A and each column n of B:
    /// ```text
    /// `` `C[i, n] = sum(A.values[j] * B[A.col_indices[j], n])` ``
    ///           for j in `` `row_ptrs[i]..row_ptrs[i+1]` ``
    /// ```
    ///
    /// # Performance
    ///
    /// - `O(nnz * N)` time complexity
    /// - CSR format provides good memory access for row-wise traversal
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A: `[2, 3]` sparse, B: `[3, 2]` dense -> C: `[2, 2]` dense
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 2], &[1.0f32, 2.0, 3.0], [2, 3], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(csr) = sp {
    /// # let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0f32], &[3, 2], &device);
    /// let c = csr.spmm(&b)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn spmm(&self, b: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        use crate::sparse::SparseOps;

        let [m, k] = self.shape;
        let dtype = self.dtype();
        let device = self.values.device();

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

        // Validate dtype match
        if b.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: b.dtype(),
            });
        }

        // Handle empty matrix case
        if self.is_empty() {
            crate::dispatch_dtype!(dtype, T => {
                let zeros: Vec<T> = vec![T::zero(); m * n];
                return Tensor::try_from_slice(&zeros, &[m, n], device);
            }, "spmm empty");
        }

        // Get runtime client to dispatch to backend-specific implementation
        let client = R::default_client(device);

        // Dispatch on dtype to call backend spmm_csr
        crate::dispatch_dtype!(dtype, T => {
            return client.spmm_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                b,
                self.shape,
            );
        }, "spmm");
    }

    /// Transpose the sparse matrix: B = A^T
    ///
    /// Returns the transpose as a CSC matrix. This is an `O(1)` operation
    /// that reinterprets the CSR structure as CSC:
    /// - `row_ptrs` become `col_ptrs`
    /// - `col_indices` become `row_indices`
    /// - `values` remain the same
    /// - `shape` is swapped
    ///
    /// # Returns
    ///
    /// CSC matrix representing the transpose
    ///
    /// # Performance
    ///
    /// `O(1)` - structural reinterpretation, no data copying beyond cloning tensors.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A `[2, 3]` in CSR:
    /// // `[1, 0, 2]`
    /// // `[0, 3, 0]`
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], [2, 3], &device)?.to_csr()?;
    /// # if let numr::sparse::SparseTensor::Csr(a) = sp {
    /// let a_t = a.transpose();
    /// // A^T `[3, 2]` in CSC (same underlying data)
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn transpose(&self) -> CscData<R> {
        let [nrows, ncols] = self.shape;
        // CSR row_ptrs -> CSC col_ptrs
        // CSR col_indices -> CSC row_indices
        // Shape [nrows, ncols] -> [ncols, nrows]
        CscData {
            col_ptrs: self.row_ptrs.clone(),
            row_indices: self.col_indices.clone(),
            values: self.values.clone(),
            shape: [ncols, nrows],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::{SparseFormat, SparseStorage};
    use crate::tensor::Tensor;

    // =========================================================================
    // SpMV tests
    // =========================================================================

    #[test]
    fn test_spmv_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // x = [1, 2, 3]
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // y = A * x
        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*3 = 9
        // y[2] = 4*1 + 5*2 = 14
        let y = csr.spmv(&x).unwrap();

        assert_eq!(y.shape(), &[3]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_spmv_empty_matrix() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csr = CsrData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let y = csr.spmv(&x).unwrap();

        assert_eq!(y.shape(), &[3]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_spmv_identity() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Identity matrix:
        // [1, 0, 0]
        // [0, 1, 0]
        // [0, 0, 1]
        let row_ptrs = vec![0i64, 1, 2, 3];
        let col_indices = vec![0i64, 1, 2];
        let values = vec![1.0f32, 1.0, 1.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&[7.0f32, 8.0, 9.0], &[3], &device);
        let y = csr.spmv(&x).unwrap();

        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_spmv_non_square() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 4]:
        // [1, 2, 0, 3]
        // [0, 4, 5, 0]
        let row_ptrs = vec![0i64, 3, 5];
        let col_indices = vec![0i64, 1, 3, 1, 2];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 4], &device)
                .unwrap();

        // x = [1, 2, 3, 4]
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device);

        // y = A * x
        // y[0] = 1*1 + 2*2 + 3*4 = 17
        // y[1] = 4*2 + 5*3 = 23
        let y = csr.spmv(&x).unwrap();

        assert_eq!(y.shape(), &[2]);
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![17.0, 23.0]);
    }

    #[test]
    fn test_spmv_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // Wrong vector length (2 instead of 3)
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);

        let result = csr.spmv(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmv_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0]; // F32

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // F64 vector
        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device);

        let result = csr.spmv(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmv_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 2]
        // [3, 4]
        let row_ptrs = vec![0i64, 2, 4];
        let col_indices = vec![0i64, 1, 0, 1];
        let values = vec![1.0f64, 2.0, 3.0, 4.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 2], &device)
                .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[2], &device);

        // y = A * x
        // y[0] = 1 + 2 = 3
        // y[1] = 3 + 4 = 7
        let y = csr.spmv(&x).unwrap();

        assert_eq!(y.dtype(), DType::F64);
        let y_data: Vec<f64> = y.to_vec();
        assert_eq!(y_data, vec![3.0, 7.0]);
    }

    #[test]
    fn test_spmv_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Single element at (1, 2) with value 5
        let row_ptrs = vec![0i64, 0, 1, 1];
        let col_indices = vec![2i64];
        let values = vec![5.0f32];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        let x = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        // y = A * x
        // y[0] = 0
        // y[1] = 5 * 3 = 15
        // y[2] = 0
        let y = csr.spmv(&x).unwrap();

        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![0.0, 15.0, 0.0]);
    }

    // =========================================================================
    // SpMM tests
    // =========================================================================

    #[test]
    fn test_spmm_basic() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Sparse A [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();

        // Dense B [3, 2]:
        // [1, 2]
        // [3, 4]
        // [5, 6]
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        // C = A * B [2, 2]:
        // C[0,0] = 1*1 + 2*5 = 11
        // C[0,1] = 1*2 + 2*6 = 14
        // C[1,0] = 3*3 = 9
        // C[1,1] = 3*4 = 12
        let c = csr.spmm(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_spmm_empty_matrix() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csr = CsrData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        let c = csr.spmm(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_spmm_identity() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Identity matrix [3, 3]
        let row_ptrs = vec![0i64, 1, 2, 3];
        let col_indices = vec![0i64, 1, 2];
        let values = vec![1.0f32, 1.0, 1.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // B [3, 2]
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        // I * B = B
        let c = csr.spmm(&b).unwrap();

        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_spmm_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A [2, 3]
        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();

        // B [2, 2] - wrong dimension (should be [3, ...])
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device);

        let result = csr.spmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmm_not_2d() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();

        // 1D tensor instead of 2D
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);

        let result = csr.spmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmm_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0]; // F32

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();

        // F64 matrix
        let b =
            Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device);

        let result = csr.spmm(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_spmm_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A [2, 2]
        let row_ptrs = vec![0i64, 2, 4];
        let col_indices = vec![0i64, 1, 0, 1];
        let values = vec![1.0f64, 2.0, 3.0, 4.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 2], &device)
                .unwrap();

        // B [2, 2]
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2], &device);

        // C = A * I = A
        let c = csr.spmm(&b).unwrap();

        assert_eq!(c.dtype(), DType::F64);
        let c_data: Vec<f64> = c.to_vec();
        assert_eq!(c_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_spmm_single_column() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A [3, 3]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // B [3, 1] - single column (like a vector reshaped)
        let b = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3, 1], &device);

        // Should match spmv result
        let c = csr.spmm(&b).unwrap();

        assert_eq!(c.shape(), &[3, 1]);
        let c_data: Vec<f32> = c.to_vec();
        // Same as spmv: [7, 9, 14]
        assert_eq!(c_data, vec![7.0, 9.0, 14.0]);
    }

    // =========================================================================
    // Transpose tests
    // =========================================================================

    #[test]
    fn test_csr_transpose() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();
        let csc = csr.transpose();

        // Transposed [3, 2] as CSC
        assert_eq!(csc.shape(), [3, 2]);
        assert_eq!(csc.nnz(), 3);
        assert_eq!(csc.format(), SparseFormat::Csc);

        // CSR row_ptrs become CSC col_ptrs
        let col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let row_indices: Vec<i64> = csc.row_indices().to_vec();
        let t_values: Vec<f32> = csc.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 3]); // Same as original row_ptrs
        assert_eq!(row_indices, vec![0, 2, 1]); // Same as original col_indices
        assert_eq!(t_values, vec![1.0, 2.0, 3.0]); // Values unchanged
    }

    #[test]
    fn test_csr_transpose_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let csr = CsrData::<CpuRuntime>::empty([3, 5], DType::F32, &device).unwrap();
        let csc = csr.transpose();

        assert_eq!(csc.shape(), [5, 3]);
        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.format(), SparseFormat::Csc);
    }

    #[test]
    fn test_csr_transpose_to_dense_matches() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let row_ptrs = vec![0i64, 2, 3];
        let col_indices = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [2, 3], &device)
                .unwrap();

        // Convert to dense, then transpose CSC to dense
        let csc = csr.transpose();

        // Convert CSC transpose to CSR to use to_dense via COO
        let csr_t = csc.to_csr().unwrap();
        let coo_t = csr_t.to_coo().unwrap();

        // Build dense from COO
        let t_rows: Vec<i64> = coo_t.row_indices().to_vec();
        let t_cols: Vec<i64> = coo_t.col_indices().to_vec();
        let t_vals: Vec<f32> = coo_t.values().to_vec();

        // Transposed [3, 2]:
        // [1, 0]
        // [0, 3]
        // [2, 0]
        // Check that values are in correct positions
        let mut dense_t = vec![0.0f32; 6];
        for i in 0..t_vals.len() {
            let r = t_rows[i] as usize;
            let c = t_cols[i] as usize;
            dense_t[r * 2 + c] = t_vals[i];
        }
        assert_eq!(dense_t, vec![1.0, 0.0, 0.0, 3.0, 2.0, 0.0]);
    }
}
