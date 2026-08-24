//! Core COO implementation: struct, creation, getters

use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::format::{SparseFormat, SparseStorage};

/// COO (Coordinate) sparse matrix data
#[derive(Debug, Clone)]
pub struct CooData<R: Runtime> {
    pub(crate) row_indices: Tensor<R>,
    pub(crate) col_indices: Tensor<R>,
    pub(crate) values: Tensor<R>,
    pub(crate) shape: [usize; 2],
    pub(crate) sorted: bool,
}

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Create a new COO matrix from components
    ///
    /// # Arguments
    ///
    /// * `row_indices` - 1D tensor of row indices (I64)
    /// * `col_indices` - 1D tensor of column indices (I64)
    /// * `values` - 1D tensor of values
    /// * `shape` - Matrix shape [nrows, ncols]
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Arrays have different lengths
    /// - Indices are out of bounds
    /// - Index tensors are not I64
    pub fn new(
        row_indices: Tensor<R>,
        col_indices: Tensor<R>,
        values: Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Self> {
        // Validate shapes
        let nnz = values.numel();
        if row_indices.numel() != nnz || col_indices.numel() != nnz {
            return Err(Error::ShapeMismatch {
                expected: vec![nnz],
                got: vec![row_indices.numel()],
            });
        }

        // Validate dtypes
        if row_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: row_indices.dtype(),
            });
        }
        if col_indices.dtype() != DType::I64 {
            return Err(Error::DTypeMismatch {
                lhs: DType::I64,
                rhs: col_indices.dtype(),
            });
        }

        // Validate 1D
        if row_indices.ndim() != 1 || col_indices.ndim() != 1 || values.ndim() != 1 {
            return Err(Error::Internal(format!(
                "Expected 1D tensors, got row: {}D, col: {}D, values: {}D",
                row_indices.ndim(),
                col_indices.ndim(),
                values.ndim()
            )));
        }

        Ok(Self {
            row_indices,
            col_indices,
            values,
            shape,
            sorted: false,
        })
    }

    /// Create an empty COO matrix
    pub fn empty(shape: [usize; 2], dtype: DType, device: &R::Device) -> Result<Self> {
        Ok(Self {
            row_indices: Tensor::empty(&[0], DType::I64, device)?,
            col_indices: Tensor::empty(&[0], DType::I64, device)?,
            values: Tensor::empty(&[0], dtype, device)?,
            shape,
            sorted: true,
        })
    }

    /// Returns the row indices tensor
    pub fn row_indices(&self) -> &Tensor<R> {
        &self.row_indices
    }

    /// Returns the column indices tensor
    pub fn col_indices(&self) -> &Tensor<R> {
        &self.col_indices
    }

    /// Returns the values tensor
    pub fn values(&self) -> &Tensor<R> {
        &self.values
    }

    /// Returns whether entries are sorted in row-major order
    pub fn is_sorted(&self) -> bool {
        self.sorted
    }

    /// Mark the COO data as sorted (caller must ensure this is true)
    ///
    /// # Safety
    ///
    /// Caller must ensure entries are actually sorted in row-major order
    /// (sorted by row, then by column within each row).
    pub unsafe fn set_sorted(&mut self, sorted: bool) {
        self.sorted = sorted;
    }
}
impl<R: Runtime<DType = DType>> SparseStorage for CooData<R> {
    fn format(&self) -> SparseFormat {
        SparseFormat::Coo
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
        // row_indices (I64) + col_indices (I64) + values
        let index_size = self.nnz() * std::mem::size_of::<i64>() * 2;
        let value_size = self.nnz() * self.dtype().size_in_bytes();
        index_size + value_size
    }
}

/// Create COO data from host arrays (CPU)
impl<R: Runtime<DType = DType>> CooData<R> {
    /// Create COO matrix from host slices
    ///
    /// # Arguments
    ///
    /// * `rows` - Row indices
    /// * `cols` - Column indices
    /// * `values` - Non-zero values
    /// * `shape` - Matrix shape [nrows, ncols]
    /// * `device` - Target device
    pub fn from_slices<T: Element>(
        rows: &[i64],
        cols: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        if rows.len() != values.len() || cols.len() != values.len() {
            return Err(Error::ShapeMismatch {
                expected: vec![values.len()],
                got: vec![rows.len()],
            });
        }

        // Validate indices
        for (&r, &c) in rows.iter().zip(cols.iter()) {
            if r < 0 {
                return Err(Error::Internal(format!("Negative row index: {}", r)));
            }
            if r as usize >= shape[0] {
                return Err(Error::IndexOutOfBounds {
                    index: r as usize,
                    size: shape[0],
                });
            }
            if c < 0 {
                return Err(Error::Internal(format!("Negative column index: {}", c)));
            }
            if c as usize >= shape[1] {
                return Err(Error::IndexOutOfBounds {
                    index: c as usize,
                    size: shape[1],
                });
            }
        }

        let row_indices = Tensor::from_slice(rows, &[rows.len()], device)?;
        let col_indices = Tensor::from_slice(cols, &[cols.len()], device)?;
        let values_tensor = Tensor::from_slice(values, &[values.len()], device)?;

        Self::new(row_indices, col_indices, values_tensor, shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_creation() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let rows = vec![0i64, 1, 2];
        let cols = vec![1i64, 0, 2];
        let values = vec![1.0f32, 2.0, 3.0];

        let coo = CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device);
        assert!(coo.is_ok());

        let coo = coo.unwrap();
        assert_eq!(coo.nnz(), 3);
        assert_eq!(coo.shape(), [3, 3]);
        assert_eq!(coo.dtype(), DType::F32);
        assert!(!coo.is_sorted());
    }

    #[test]
    fn test_coo_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let coo = CooData::<CpuRuntime>::empty([100, 100], DType::F32, &device).unwrap();

        assert_eq!(coo.nnz(), 0);
        assert_eq!(coo.shape(), [100, 100]);
        assert!(coo.is_empty());
        assert!(coo.is_sorted());
    }

    #[test]
    fn test_coo_sparsity() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let rows = vec![0i64, 1];
        let cols = vec![0i64, 1];
        let values = vec![1.0f32, 2.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [10, 10], &device).unwrap();

        // 2 non-zeros out of 100 elements = 2% density = 98% sparsity
        assert!((coo.density() - 0.02).abs() < 1e-10);
        assert!((coo.sparsity() - 0.98).abs() < 1e-10);
    }

    #[test]
    fn test_coo_invalid_indices() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let rows = vec![0i64, 5]; // 5 is out of bounds for 3x3
        let cols = vec![0i64, 0];
        let values = vec![1.0f32, 2.0];

        let result = CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device);
        assert!(result.is_err());
    }
}
