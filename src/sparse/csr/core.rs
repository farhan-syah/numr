//! Core CSR implementation: struct, creation, getters

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::format::{SparseFormat, SparseStorage};

/// CSR (Compressed Sparse Row) sparse matrix data
#[derive(Debug, Clone)]
pub struct CsrData<R: Runtime> {
    pub(crate) row_ptrs: Tensor<R>,
    pub(crate) col_indices: Tensor<R>,
    pub(crate) values: Tensor<R>,
    pub(crate) shape: [usize; 2],
}

impl<R: Runtime<DType = DType>> CsrData<R> {
    /// Create a new CSR matrix from components
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - Row pointers (length: nrows + 1)
    /// * `col_indices` - Column indices for each non-zero
    /// * `values` - Values at each position
    /// * `shape` - Matrix shape [nrows, ncols]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - row_ptrs length != nrows + 1
    /// - col_indices and values have different lengths
    /// - Index tensors are not I64
    pub fn new(
        row_ptrs: Tensor<R>,
        col_indices: Tensor<R>,
        values: Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Self> {
        let [nrows, _ncols] = shape;
        let nnz = values.numel();

        // Validate row_ptrs length
        if row_ptrs.numel() != nrows + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![nrows + 1],
                got: vec![row_ptrs.numel()],
            });
        }

        // Validate col_indices length
        if col_indices.numel() != nnz {
            return Err(Error::ShapeMismatch {
                expected: vec![nnz],
                got: vec![col_indices.numel()],
            });
        }

        // Validate dtypes
        if row_ptrs.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: row_ptrs.dtype(),
            });
        }
        if col_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: col_indices.dtype(),
            });
        }

        // Validate 1D
        if row_ptrs.ndim() != 1 || col_indices.ndim() != 1 || values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensors, got row_ptrs: {}D, col_indices: {}D, values: {}D",
                row_ptrs.ndim(),
                col_indices.ndim(),
                values.ndim()
            )));
        }

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSR matrix
    pub fn empty(shape: [usize; 2], dtype: DType, device: &R::Device) -> Result<Self> {
        let [nrows, _ncols] = shape;
        // Row pointers are all zeros for empty matrix
        let row_ptrs_data: Vec<i64> = vec![0; nrows + 1];

        Ok(Self {
            row_ptrs: Tensor::from_slice(&row_ptrs_data, &[nrows + 1], device)?,
            col_indices: Tensor::empty(&[0], DType::I64, device)?,
            values: Tensor::empty(&[0], dtype, device)?,
            shape,
        })
    }

    /// Returns the row pointers tensor
    pub fn row_ptrs(&self) -> &Tensor<R> {
        &self.row_ptrs
    }

    /// Returns the column indices tensor
    pub fn col_indices(&self) -> &Tensor<R> {
        &self.col_indices
    }

    /// Returns the values tensor
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }

    /// Returns the number of non-zeros in a specific row
    ///
    /// # Arguments
    ///
    /// * `row` - Row index (0-based)
    ///
    /// # Panics
    ///
    /// Panics if row >= nrows (only in debug mode)
    ///
    /// # Note
    ///
    /// For GPU tensors, this method narrows to a 2-element slice and transfers it
    /// to host memory. While minimal, this still involves a GPU-to-CPU transfer.
    /// For batch queries or hot paths, consider using the row_ptrs tensor directly
    /// with GPU operations to avoid synchronization overhead.
    pub fn row_nnz(&self, row: usize) -> usize {
        debug_assert!(row < self.nrows());
        // Narrow to the 2 values needed: row_ptrs[row] and row_ptrs[row+1]
        // This minimizes transfer size but still requires GPU→CPU sync
        let slice = self
            .row_ptrs
            .narrow(0, row, 2)
            .expect("row_nnz: invalid row index");
        let slice = slice
            .contiguous()
            .expect("row_nnz: contiguous allocation failed");
        let ptrs: Vec<i64> = slice.to_vec();
        (ptrs[1] - ptrs[0]) as usize
    }

    /// Update the values tensor in-place while preserving the sparsity pattern.
    ///
    /// This method allows efficient numeric refactorization by reusing the same
    /// sparsity structure (row_ptrs, col_indices) with new numerical values.
    ///
    /// # Arguments
    ///
    /// * `new_values` - Tensor with the same number of elements and dtype as current values
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `new_values.numel() != self.nnz()`
    /// - `new_values.dtype() != self.dtype()`
    /// - `new_values` is not 1D
    pub fn update_values(&mut self, new_values: Tensor<R>) -> Result<()> {
        if new_values.numel() != self.values.numel() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.values.numel()],
                got: vec![new_values.numel()],
            });
        }
        if new_values.dtype() != self.values.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.values.dtype(),
                rhs: new_values.dtype(),
            });
        }
        if new_values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensor for values, got {}D",
                new_values.ndim()
            )));
        }
        self.values = new_values;
        Ok(())
    }

    /// Extract the diagonal elements as a 1D tensor.
    ///
    /// Returns a tensor of length `min(nrows, ncols)` containing the diagonal
    /// entries. Missing diagonal entries are returned as zeros.
    ///
    /// # Note
    ///
    /// This method transfers data to CPU for extraction, then creates the result
    /// tensor on the original device. For large matrices on GPU, consider using
    /// a client-based approach with `index_select` for better performance.
    pub fn diagonal<T: Element + Default + Copy>(&self) -> Result<Tensor<R>> {
        let n = self.nrows().min(self.ncols());
        let device = self.values.device();

        if n == 0 {
            return Tensor::empty(&[0], self.dtype(), device);
        }

        // Validate dtype matches T
        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer data to CPU for scanning
        let row_ptrs: Vec<i64> = self.row_ptrs.to_vec();
        let col_indices: Vec<i64> = self.col_indices.to_vec();
        let values: Vec<T> = self.values.to_vec();

        // Extract diagonal values
        let mut diag_values = vec![T::default(); n];

        for row in 0..n {
            let start = row_ptrs[row] as usize;
            let end = row_ptrs[row + 1] as usize;

            // Search for col == row in this row
            for pos in start..end {
                if col_indices[pos] as usize == row {
                    diag_values[row] = values[pos];
                    break;
                }
            }
        }

        // Create result tensor on original device
        Tensor::from_slice(&diag_values, &[n], device)
    }

    /// Extract diagonal elements using a `SparseOps` client (on-device).
    ///
    /// This method delegates to the client's `extract_diagonal_csr` kernel,
    /// keeping all computation on the compute device. Preferred over
    /// `diagonal()` for GPU tensors.
    pub fn diagonal_with_client<C: super::super::SparseOps<R>>(
        &self,
        client: &C,
    ) -> Result<Tensor<R>> {
        let dtype = self.dtype();
        crate::dispatch_dtype!(dtype, T => {
            client.extract_diagonal_csr::<T>(
                &self.row_ptrs,
                &self.col_indices,
                &self.values,
                self.shape,
            )
        }, "diagonal_with_client")
    }

    /// Check if the matrix has a structural nonzero on every diagonal position.
    ///
    /// For rectangular matrices, checks positions `0..min(nrows, ncols)`.
    /// Returns `true` if every diagonal position has a structural entry (even if zero-valued).
    pub fn has_full_diagonal(&self) -> bool {
        let n = self.nrows().min(self.ncols());
        if n == 0 {
            return true;
        }

        let row_ptrs: Vec<i64> = self.row_ptrs.to_vec();
        let col_indices: Vec<i64> = self.col_indices.to_vec();

        for row in 0..n {
            let start = row_ptrs[row] as usize;
            let end = row_ptrs[row + 1] as usize;

            let mut found = false;
            for pos in start..end {
                if col_indices[pos] as usize == row {
                    found = true;
                    break;
                }
            }
            if !found {
                return false;
            }
        }
        true
    }
}

impl<R: Runtime<DType = DType>> SparseStorage for CsrData<R> {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csr
    }

    fn shape(&self) -> [usize; 2] {
        self.shape
    }

    fn nnz(&self) -> usize {
        self.values.numel()
    }

    fn dtype(&self) -> DType {
        self.values.dtype()
    }

    fn memory_usage(&self) -> usize {
        // row_ptrs (I64) + col_indices (I64) + values
        let ptr_size = (self.nrows() + 1) * std::mem::size_of::<i64>();
        let index_size = self.nnz() * std::mem::size_of::<i64>();
        let value_size = self.nnz() * self.dtype().size_in_bytes();
        ptr_size + index_size + value_size
    }
}

/// Create CSR data from host arrays
impl<R: Runtime<DType = DType>> CsrData<R> {
    /// Create CSR matrix from host slices
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - Row pointers (length: nrows + 1)
    /// * `col_indices` - Column indices
    /// * `values` - Non-zero values
    /// * `shape` - Matrix shape [nrows, ncols]
    /// * `device` - Target device
    pub fn from_slices<T: Element>(
        row_ptrs: &[i64],
        col_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let [nrows, ncols] = shape;

        // Validate row_ptrs length
        if row_ptrs.len() != nrows + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![nrows + 1],
                got: vec![row_ptrs.len()],
            });
        }

        // Validate lengths match
        if col_indices.len() != values.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![col_indices.len()],
            });
        }

        // Validate row_ptrs is monotonic and matches nnz
        let nnz = values.len();
        if row_ptrs[0] != 0 || row_ptrs[nrows] as usize != nnz {
            return Err(Error::Internal(format!(
                "Invalid row_ptrs: expected [0]=0 and [{}]={}, got [0]={} and [{}]={}",
                nrows, nnz, row_ptrs[0], nrows, row_ptrs[nrows]
            )));
        }

        // Validate column indices
        for &c in col_indices {
            if c < 0 {
                return Err(Error::Internal(format!("Negative column index: {}", c)));
            }
            if c as usize >= ncols {
                return Err(Error::IndexOutOfBounds {
                    index: c as usize,
                    size: ncols,
                });
            }
        }

        let row_ptrs_tensor = Tensor::from_slice(row_ptrs, &[row_ptrs.len()], device)?;
        let col_indices_tensor = Tensor::from_slice(col_indices, &[col_indices.len()], device)?;
        let values_tensor = Tensor::from_slice(values, &[values.len()], device)?;

        Self::new(row_ptrs_tensor, col_indices_tensor, values_tensor, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_csr_creation() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device);
        assert!(csr.is_ok());

        let csr = csr.unwrap();
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), [3, 3]);
        assert_eq!(csr.nrows(), 3);
        assert_eq!(csr.ncols(), 3);
        assert_eq!(csr.dtype(), DType::F32);
    }

    #[test]
    fn test_csr_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csr = CsrData::<CpuRuntime>::empty([100, 200], DType::F64, &device).unwrap();

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), [100, 200]);
        assert!(csr.is_empty());
        assert_eq!(csr.row_ptrs().numel(), 101); // nrows + 1
    }

    #[test]
    fn test_csr_memory_usage() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // 4 row_ptrs * 8 bytes + 5 col_indices * 8 bytes + 5 values * 4 bytes
        // = 32 + 40 + 20 = 92 bytes
        assert_eq!(csr.memory_usage(), 92);
    }

    #[test]
    fn test_csr_invalid_row_ptrs() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let row_ptrs = vec![0i64, 2, 3]; // Wrong length (should be 4 for 3 rows)
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let result =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device);
        assert!(result.is_err());
    }

    #[test]
    fn test_csr_update_values() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let mut csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // Update values - double them
        let new_values = vec![2.0f32, 4.0, 6.0, 8.0, 10.0];
        let new_values_tensor = Tensor::from_slice(&new_values, &[5], &device).unwrap();
        csr.update_values(new_values_tensor).unwrap();

        // Verify values changed but structure unchanged
        let updated: Vec<f32> = csr.values().to_vec();
        assert_eq!(updated, vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        // Structure should be unchanged
        let ptrs: Vec<i64> = csr.row_ptrs().to_vec();
        let indices: Vec<i64> = csr.col_indices().to_vec();
        assert_eq!(ptrs, row_ptrs);
        assert_eq!(indices, col_indices);
    }

    #[test]
    fn test_csr_update_values_wrong_size() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let mut csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // Try to update with wrong size
        let wrong_size = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        assert!(csr.update_values(wrong_size).is_err());
    }

    #[test]
    fn test_csr_update_values_wrong_dtype() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let mut csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        // Try to update with wrong dtype (f64 instead of f32)
        let wrong_dtype = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device).unwrap();
        assert!(csr.update_values(wrong_dtype).is_err());
    }

    #[test]
    fn test_csr_diagonal() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix with full diagonal:
        // [1, 2, 3]
        // [4, 5, 6]
        // [7, 8, 9]
        // CSR row 0: cols 0,1,2 values 1,2,3
        // CSR row 1: cols 0,1,2 values 4,5,6
        // CSR row 2: cols 0,1,2 values 7,8,9
        let row_ptrs = vec![0i64, 3, 6, 9];
        let col_indices = vec![0i64, 1, 2, 0, 1, 2, 0, 1, 2];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        let diag: Vec<f32> = csr.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 5.0, 9.0]);
        assert!(csr.has_full_diagonal());
    }

    #[test]
    fn test_csr_diagonal_missing_entries() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix with missing diagonal entry at (1,1):
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 6]
        let row_ptrs = vec![0i64, 2, 3, 6];
        let col_indices = vec![0i64, 2, 2, 0, 1, 2];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();

        let diag: Vec<f32> = csr.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 0.0, 6.0]); // Missing (1,1) is 0
        assert!(!csr.has_full_diagonal());
    }

    #[test]
    fn test_csr_diagonal_rectangular() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // 3x2 matrix:
        // [1, 2]
        // [3, 4]
        // [5, 6]
        let row_ptrs = vec![0i64, 2, 4, 6];
        let col_indices = vec![0i64, 1, 0, 1, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 2], &device)
                .unwrap();

        // Diagonal length is min(3, 2) = 2
        let diag: Vec<f32> = csr.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 4.0]);
    }
}
