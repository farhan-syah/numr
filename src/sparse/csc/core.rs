//! Core CSC implementation: struct, creation, getters

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::format::{SparseFormat, SparseStorage};
use super::super::ops::{NormType, SparseScaling};

/// CSC (Compressed Sparse Column) sparse matrix data
#[derive(Debug, Clone)]
pub struct CscData<R: Runtime> {
    pub(crate) col_ptrs: Tensor<R>,
    pub(crate) row_indices: Tensor<R>,
    pub(crate) values: Tensor<R>,
    pub(crate) shape: [usize; 2],
}

impl<R: Runtime<DType = DType>> CscData<R> {
    /// Create a new CSC matrix from components
    pub fn new(
        col_ptrs: Tensor<R>,
        row_indices: Tensor<R>,
        values: Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Self> {
        let [_nrows, ncols] = shape;
        let nnz = values.numel();

        // Validate col_ptrs length
        if col_ptrs.numel() != ncols + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols + 1],
                got: vec![col_ptrs.numel()],
            });
        }

        // Validate row_indices length
        if row_indices.numel() != nnz {
            return Err(Error::ShapeMismatch {
                expected: vec![nnz],
                got: vec![row_indices.numel()],
            });
        }

        // Validate dtypes
        if col_ptrs.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: col_ptrs.dtype(),
            });
        }
        if row_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: row_indices.dtype(),
            });
        }

        // Validate 1D
        if col_ptrs.ndim() != 1 || row_indices.ndim() != 1 || values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensors, got col_ptrs: {}D, row_indices: {}D, values: {}D",
                col_ptrs.ndim(),
                row_indices.ndim(),
                values.ndim()
            )));
        }

        Ok(Self {
            col_ptrs,
            row_indices,
            values,
            shape,
        })
    }

    /// Create an empty CSC matrix
    pub fn empty(shape: [usize; 2], dtype: DType, device: &R::Device) -> Result<Self> {
        let [_nrows, ncols] = shape;
        let col_ptrs_data: Vec<i64> = vec![0; ncols + 1];

        Ok(Self {
            col_ptrs: Tensor::try_from_slice(&col_ptrs_data, &[ncols + 1], device)?,
            row_indices: Tensor::try_empty(&[0], DType::I64, device)?,
            values: Tensor::try_empty(&[0], dtype, device)?,
            shape,
        })
    }

    /// Returns the column pointers tensor
    pub fn col_ptrs(&self) -> &Tensor<R> {
        &self.col_ptrs
    }

    /// Returns the row indices tensor
    pub fn row_indices(&self) -> &Tensor<R> {
        &self.row_indices
    }

    /// Returns the values tensor
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }

    /// Update the values tensor in-place while preserving the sparsity pattern.
    ///
    /// This method allows efficient numeric refactorization by reusing the same
    /// sparsity structure (col_ptrs, row_indices) with new numerical values.
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
            return Tensor::try_empty(&[0], self.dtype(), device);
        }

        // Validate dtype matches T
        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer data to CPU for scanning
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();
        let values: Vec<T> = self.values.to_vec();

        // Extract diagonal values
        let mut diag_values = vec![T::default(); n];

        for col in 0..n {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;

            // Search for row == col in this column
            for pos in start..end {
                if row_indices[pos] as usize == col {
                    diag_values[col] = values[pos];
                    break;
                }
            }
        }

        // Create result tensor on original device
        Tensor::try_from_slice(&diag_values, &[n], device)
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

        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();

        for col in 0..n {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;

            let mut found = false;
            for pos in start..end {
                if row_indices[pos] as usize == col {
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

impl<R: Runtime<DType = DType>> SparseStorage for CscData<R> {
    fn format(&self) -> SparseFormat {
        SparseFormat::Csc
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
        let ptr_size = (self.ncols() + 1) * std::mem::size_of::<i64>();
        let index_size = self.nnz() * std::mem::size_of::<i64>();
        let value_size = self.nnz() * self.dtype().size_in_bytes();
        ptr_size + index_size + value_size
    }
}

impl<R: Runtime<DType = DType>> CscData<R> {
    /// Create CSC matrix from host slices
    pub fn from_slices<T: Element>(
        col_ptrs: &[i64],
        row_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let [nrows, ncols] = shape;

        if col_ptrs.len() != ncols + 1 {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols + 1],
                got: vec![col_ptrs.len()],
            });
        }

        if row_indices.len() != values.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![row_indices.len()],
            });
        }

        let nnz = values.len();
        if col_ptrs[0] != 0 || col_ptrs[ncols] as usize != nnz {
            return Err(Error::Internal(format!(
                "Invalid col_ptrs: expected [0]=0 and [{}]={}, got [0]={} and [{}]={}",
                ncols, nnz, col_ptrs[0], ncols, col_ptrs[ncols]
            )));
        }

        for &r in row_indices {
            if r < 0 {
                return Err(Error::Internal(format!("Negative row index: {}", r)));
            }
            if r as usize >= nrows {
                return Err(Error::IndexOutOfBounds {
                    index: r as usize,
                    size: nrows,
                });
            }
        }

        let col_ptrs_tensor = Tensor::try_from_slice(col_ptrs, &[col_ptrs.len()], device)?;
        let row_indices_tensor = Tensor::try_from_slice(row_indices, &[row_indices.len()], device)?;
        let values_tensor = Tensor::try_from_slice(values, &[values.len()], device)?;

        Self::new(col_ptrs_tensor, row_indices_tensor, values_tensor, shape)
    }
}

// ============================================================================
// SparseScaling Implementation for CscData
// ============================================================================

impl<R: Runtime<DType = DType>> SparseScaling<R> for CscData<R> {
    fn row_norms<T: Element + Default + Copy>(&self, norm: NormType) -> Result<Tensor<R>> {
        let [nrows, ncols] = self.shape;
        let device = self.values.device();

        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer to CPU for computation
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();
        let values: Vec<f64> = match self.dtype() {
            DType::F32 => self
                .values
                .to_vec::<f32>()
                .iter()
                .map(|&x| x as f64)
                .collect(),
            DType::F64 => self.values.to_vec(),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: self.dtype(),
                    op: "row_norms",
                });
            }
        };

        // Compute norms row by row
        let mut norms = vec![0.0f64; nrows];

        for col in 0..ncols {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;

            for idx in start..end {
                let row = row_indices[idx] as usize;
                let val = values[idx];
                match norm {
                    NormType::L1 => norms[row] += val.abs(),
                    NormType::L2 => norms[row] += val * val,
                    NormType::Linf => norms[row] = norms[row].max(val.abs()),
                }
            }
        }

        // Post-process L2 norm
        if norm == NormType::L2 {
            for n in &mut norms {
                *n = n.sqrt();
            }
        }

        // Convert back to requested type
        match T::DTYPE {
            DType::F32 => {
                let norms_f32: Vec<f32> = norms.iter().map(|&x| x as f32).collect();
                Tensor::try_from_slice(&norms_f32, &[nrows], device)
            }
            DType::F64 => Tensor::try_from_slice(&norms, &[nrows], device),
            _ => Err(Error::UnsupportedDType {
                dtype: T::DTYPE,
                op: "row_norms",
            }),
        }
    }

    fn col_norms<T: Element + Default + Copy>(&self, norm: NormType) -> Result<Tensor<R>> {
        let [_nrows, ncols] = self.shape;
        let device = self.values.device();

        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer to CPU for computation
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let values: Vec<f64> = match self.dtype() {
            DType::F32 => self
                .values
                .to_vec::<f32>()
                .iter()
                .map(|&x| x as f64)
                .collect(),
            DType::F64 => self.values.to_vec(),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: self.dtype(),
                    op: "col_norms",
                });
            }
        };

        // Compute norms column by column (efficient for CSC)
        let mut norms = vec![0.0f64; ncols];

        for col in 0..ncols {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;

            for idx in start..end {
                let val = values[idx];
                match norm {
                    NormType::L1 => norms[col] += val.abs(),
                    NormType::L2 => norms[col] += val * val,
                    NormType::Linf => norms[col] = norms[col].max(val.abs()),
                }
            }
        }

        // Post-process L2 norm
        if norm == NormType::L2 {
            for n in &mut norms {
                *n = n.sqrt();
            }
        }

        // Convert back to requested type
        match T::DTYPE {
            DType::F32 => {
                let norms_f32: Vec<f32> = norms.iter().map(|&x| x as f32).collect();
                Tensor::try_from_slice(&norms_f32, &[ncols], device)
            }
            DType::F64 => Tensor::try_from_slice(&norms, &[ncols], device),
            _ => Err(Error::UnsupportedDType {
                dtype: T::DTYPE,
                op: "col_norms",
            }),
        }
    }

    fn scale_rows<T: Element + Default + Copy + std::ops::Mul<Output = T>>(
        &self,
        scales: &[T],
    ) -> Result<Self> {
        let [nrows, _ncols] = self.shape;
        let device = self.values.device();

        if scales.len() != nrows {
            return Err(Error::ShapeMismatch {
                expected: vec![nrows],
                got: vec![scales.len()],
            });
        }

        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer data
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();
        let values: Vec<T> = self.values.to_vec();

        // Scale values by row
        let scaled_values: Vec<T> = values
            .iter()
            .zip(row_indices.iter())
            .map(|(&v, &row)| v * scales[row as usize])
            .collect();

        Self::from_slices(&col_ptrs, &row_indices, &scaled_values, self.shape, device)
    }

    fn scale_cols<T: Element + Default + Copy + std::ops::Mul<Output = T>>(
        &self,
        scales: &[T],
    ) -> Result<Self> {
        let [_nrows, ncols] = self.shape;
        let device = self.values.device();

        if scales.len() != ncols {
            return Err(Error::ShapeMismatch {
                expected: vec![ncols],
                got: vec![scales.len()],
            });
        }

        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Transfer data
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();
        let values: Vec<T> = self.values.to_vec();

        // Scale values by column (efficient for CSC - each column is contiguous)
        let mut scaled_values = values;
        for col in 0..ncols {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;
            let scale = scales[col];

            for idx in start..end {
                scaled_values[idx] = scaled_values[idx] * scale;
            }
        }

        Self::from_slices(&col_ptrs, &row_indices, &scaled_values, self.shape, device)
    }

    fn equilibrate<T: Element + Default + Copy + num_traits::Float>(
        &self,
    ) -> Result<(Self, Vec<T>, Vec<T>)> {
        let [_nrows, _ncols] = self.shape;

        if T::DTYPE != self.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: T::DTYPE,
                rhs: self.dtype(),
            });
        }

        // Compute row and column infinity norms
        let row_norms_tensor = self.row_norms::<T>(NormType::Linf)?;
        let col_norms_tensor = self.col_norms::<T>(NormType::Linf)?;

        let row_norms: Vec<T> = row_norms_tensor.to_vec();
        let col_norms: Vec<T> = col_norms_tensor.to_vec();

        // Compute scales as 1/norm, handling zeros
        let one: T = num_traits::one();
        let zero: T = num_traits::zero();
        let epsilon = T::from(1e-15).unwrap_or(zero);

        let row_scales: Vec<T> = row_norms
            .iter()
            .map(|&n| if n > epsilon { one / n } else { one })
            .collect();

        let col_scales: Vec<T> = col_norms
            .iter()
            .map(|&n| if n > epsilon { one / n } else { one })
            .collect();

        // Apply row scaling first, then column scaling
        let scaled = self.scale_rows(&row_scales)?;
        let scaled = scaled.scale_cols(&col_scales)?;

        Ok((scaled, row_scales, col_scales))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseFormat;

    #[test]
    fn test_csc_creation() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix (same as CSR test, but column-oriented):
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device);
        assert!(csc.is_ok());

        let csc = csc.unwrap();
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), [3, 3]);
        assert_eq!(csc.format(), SparseFormat::Csc);
    }

    #[test]
    fn test_csc_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csc = CscData::<CpuRuntime>::empty([100, 200], DType::F64, &device).unwrap();

        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.shape(), [100, 200]);
        assert!(csc.is_empty());
        assert_eq!(csc.col_ptrs().numel(), 201); // ncols + 1
    }

    #[test]
    fn test_csc_update_values() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let mut csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Update values - double them
        let new_values = vec![2.0f32, 8.0, 10.0, 4.0, 6.0];
        let new_values_tensor = Tensor::from_slice(&new_values, &[5], &device);
        csc.update_values(new_values_tensor).unwrap();

        // Verify values changed but structure unchanged
        let updated: Vec<f32> = csc.values().to_vec();
        assert_eq!(updated, vec![2.0, 8.0, 10.0, 4.0, 6.0]);

        // Structure should be unchanged
        let ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let indices: Vec<i64> = csc.row_indices().to_vec();
        assert_eq!(ptrs, col_ptrs);
        assert_eq!(indices, row_indices);
    }

    #[test]
    fn test_csc_update_values_wrong_size() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let mut csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Try to update with wrong size
        let wrong_size = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        assert!(csc.update_values(wrong_size).is_err());
    }

    #[test]
    fn test_csc_update_values_wrong_dtype() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let mut csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Try to update with wrong dtype (f64 instead of f32)
        let wrong_dtype = Tensor::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0], &[5], &device);
        assert!(csc.update_values(wrong_dtype).is_err());
    }

    #[test]
    fn test_csc_diagonal() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix with full diagonal:
        // [1, 0, 2]
        // [0, 5, 3]
        // [4, 0, 6]
        // CSC col 0: rows 0,2 values 1,4
        // CSC col 1: row 1 value 5
        // CSC col 2: rows 0,1,2 values 2,3,6
        let col_ptrs = vec![0i64, 2, 3, 6];
        let row_indices = vec![0i64, 2, 1, 0, 1, 2];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0, 6.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        let diag: Vec<f32> = csc.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 5.0, 6.0]);
        assert!(csc.has_full_diagonal());
    }

    #[test]
    fn test_csc_diagonal_missing_entries() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix with missing diagonal entry at (1,1):
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 6]
        // CSC col 0: rows 0,2 values 1,4
        // CSC col 1: row 2 value 5
        // CSC col 2: rows 0,1,2 values 2,3,6
        let col_ptrs = vec![0i64, 2, 3, 6];
        let row_indices = vec![0i64, 2, 2, 0, 1, 2];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0, 6.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        let diag: Vec<f32> = csc.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 0.0, 6.0]); // Missing (1,1) is 0
        assert!(!csc.has_full_diagonal());
    }

    #[test]
    fn test_csc_diagonal_rectangular() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // 2x3 matrix:
        // [1, 2, 3]
        // [4, 5, 6]
        // CSC col 0: rows 0,1 values 1,4
        // CSC col 1: rows 0,1 values 2,5
        // CSC col 2: rows 0,1 values 3,6
        let col_ptrs = vec![0i64, 2, 4, 6];
        let row_indices = vec![0i64, 1, 0, 1, 0, 1];
        let values = vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [2, 3], &device)
                .unwrap();

        // Diagonal length is min(2, 3) = 2
        let diag: Vec<f32> = csc.diagonal::<f32>().unwrap().to_vec();
        assert_eq!(diag, vec![1.0, 5.0]);
    }

    #[test]
    fn test_csc_row_norms() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]  -> row 0: L1=3, Linf=2
        // [0, 0, 3]  -> row 1: L1=3, Linf=3
        // [4, 5, 0]  -> row 2: L1=9, Linf=5
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f64, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        let l1_norms: Vec<f64> = csc.row_norms::<f64>(NormType::L1).unwrap().to_vec();
        assert_eq!(l1_norms, vec![3.0, 3.0, 9.0]);

        let linf_norms: Vec<f64> = csc.row_norms::<f64>(NormType::Linf).unwrap().to_vec();
        assert_eq!(linf_norms, vec![2.0, 3.0, 5.0]);
    }

    #[test]
    fn test_csc_col_norms() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]  col 0: L1=5, col 1: L1=5, col 2: L1=5
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f64, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        let l1_norms: Vec<f64> = csc.col_norms::<f64>(NormType::L1).unwrap().to_vec();
        assert_eq!(l1_norms, vec![5.0, 5.0, 5.0]);

        let linf_norms: Vec<f64> = csc.col_norms::<f64>(NormType::Linf).unwrap().to_vec();
        assert_eq!(linf_norms, vec![4.0, 5.0, 3.0]);
    }

    #[test]
    fn test_csc_scale_rows() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f64, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Scale rows by [2, 3, 0.5]
        let scales = vec![2.0f64, 3.0, 0.5];
        let scaled = csc.scale_rows(&scales).unwrap();

        // Expected:
        // [2, 0, 4]   (row 0 * 2)
        // [0, 0, 9]   (row 1 * 3)
        // [2, 2.5, 0] (row 2 * 0.5)
        // CSC values order: col0:[row0,row2], col1:[row2], col2:[row0,row1]
        // = [1*2, 4*0.5, 5*0.5, 2*2, 3*3] = [2, 2, 2.5, 4, 9]
        let scaled_values: Vec<f64> = scaled.values().to_vec();
        assert_eq!(scaled_values, vec![2.0, 2.0, 2.5, 4.0, 9.0]);
    }

    #[test]
    fn test_csc_scale_cols() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f64, 4.0, 5.0, 2.0, 3.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Scale cols by [2, 3, 0.5]
        let scales = vec![2.0f64, 3.0, 0.5];
        let scaled = csc.scale_cols(&scales).unwrap();

        // Expected:
        // [2, 0, 1]   (col 0 * 2, col 2 * 0.5)
        // [0, 0, 1.5] (col 2 * 0.5)
        // [8, 15, 0]  (col 0 * 2, col 1 * 3)
        // CSC values: col0*2, col1*3, col2*0.5
        // = [1*2, 4*2, 5*3, 2*0.5, 3*0.5] = [2, 8, 15, 1, 1.5]
        let scaled_values: Vec<f64> = scaled.values().to_vec();
        assert_eq!(scaled_values, vec![2.0, 8.0, 15.0, 1.0, 1.5]);
    }

    #[test]
    fn test_csc_equilibrate() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Moderately scaled matrix that equilibration can handle in one pass:
        // [4, 0, 2]
        // [0, 3, 0]
        // [1, 0, 5]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 1, 0, 2];
        let values = vec![4.0f64, 1.0, 3.0, 2.0, 5.0];

        let csc =
            CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [3, 3], &device)
                .unwrap();

        // Get original norms for comparison
        let orig_row_norms: Vec<f64> = csc.row_norms::<f64>(NormType::Linf).unwrap().to_vec();
        let _orig_col_norms: Vec<f64> = csc.col_norms::<f64>(NormType::Linf).unwrap().to_vec();

        let (scaled, row_scales, col_scales) = csc.equilibrate::<f64>().unwrap();

        // After equilibration, norms should be improved (closer to 1)
        let row_norms: Vec<f64> = scaled.row_norms::<f64>(NormType::Linf).unwrap().to_vec();
        let _col_norms: Vec<f64> = scaled.col_norms::<f64>(NormType::Linf).unwrap().to_vec();

        // Verify that scaling reduces the spread of norms
        // (Original spread was 3-5, after scaling should be tighter)
        let orig_row_spread = orig_row_norms.iter().cloned().fold(0.0_f64, f64::max)
            / orig_row_norms.iter().cloned().fold(f64::MAX, f64::min);
        let new_row_spread = row_norms.iter().cloned().fold(0.0_f64, f64::max)
            / row_norms.iter().cloned().fold(f64::MAX, f64::min);

        // The spread should be reduced or equal
        assert!(
            new_row_spread <= orig_row_spread * 1.5,
            "Row spread should be reduced: orig={}, new={}",
            orig_row_spread,
            new_row_spread
        );

        // Verify scales are returned and are positive
        assert_eq!(row_scales.len(), 3);
        assert_eq!(col_scales.len(), 3);
        for &s in &row_scales {
            assert!(s > 0.0, "Row scale should be positive");
        }
        for &s in &col_scales {
            assert!(s > 0.0, "Col scale should be positive");
        }
    }
}
