//! COO matrix multiplication: spmv, spmm, transpose

use super::CooData;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// Converts to CSR format (optimal for SpMV) and performs the multiplication.
    ///
    /// # Arguments
    ///
    /// * `x` - Dense vector of length `ncols`
    ///
    /// # Returns
    ///
    /// Dense vector of length `nrows`
    ///
    /// # Performance
    ///
    /// This method converts to CSR first, adding O(nnz log nnz) overhead.
    /// For repeated SpMV, convert to CSR once and reuse.
    pub fn spmv(&self, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        let csr = self.to_csr()?;
        csr.spmv(x)
    }

    /// Sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// Converts to CSR format and performs the multiplication.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix of shape [K, N] where K == ncols of sparse matrix
    ///
    /// # Returns
    ///
    /// Dense matrix of shape [M, N]
    ///
    /// # Performance
    ///
    /// This method converts to CSR first, adding O(nnz log nnz) overhead.
    /// For repeated SpMM, convert to CSR once and reuse.
    pub fn spmm(&self, b: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        let csr = self.to_csr()?;
        csr.spmm(b)
    }

    /// Transpose the sparse matrix: B = A^T
    ///
    /// Swaps row and column indices, producing a new COO matrix
    /// with transposed shape [ncols, nrows].
    ///
    /// # Returns
    ///
    /// New COO matrix representing the transpose
    ///
    /// # Performance
    ///
    /// O(1) - just swaps index tensors and shape, no data copying.
    /// The resulting COO may not be sorted even if the original was.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A [2, 3]:
    /// // [1, 0, 2]
    /// // [0, 3, 0]
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], [2, 3], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(a) = sp {
    /// let a_t = a.transpose();
    /// // A^T [3, 2]:
    /// // [1, 0]
    /// // [0, 3]
    /// // [2, 0]
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn transpose(&self) -> Self {
        let [nrows, ncols] = self.shape;
        Self {
            // Swap row and column indices
            row_indices: self.col_indices.clone(),
            col_indices: self.row_indices.clone(),
            values: self.values.clone(),
            shape: [ncols, nrows],
            // Transpose breaks row-major sorting
            sorted: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseStorage;

    // =========================================================================
    // Transpose tests
    // =========================================================================

    #[test]
    fn test_coo_transpose() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let rows = vec![0i64, 0, 1];
        let cols = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [2, 3], &device).unwrap();
        let coo_t = coo.transpose();

        // Transposed [3, 2]:
        // [1, 0]
        // [0, 3]
        // [2, 0]
        assert_eq!(coo_t.shape(), [3, 2]);
        assert_eq!(coo_t.nnz(), 3);
        assert!(!coo_t.is_sorted()); // Transpose breaks sorting

        // Indices should be swapped
        let t_rows: Vec<i64> = coo_t.row_indices().to_vec();
        let t_cols: Vec<i64> = coo_t.col_indices().to_vec();
        let t_values: Vec<f32> = coo_t.values().to_vec();

        // Original: (0,0,1), (0,2,2), (1,1,3)
        // Transposed: (0,0,1), (2,0,2), (1,1,3)
        assert_eq!(t_rows, vec![0, 2, 1]); // Original cols become rows
        assert_eq!(t_cols, vec![0, 0, 1]); // Original rows become cols
        assert_eq!(t_values, vec![1.0, 2.0, 3.0]); // Values unchanged
    }

    #[test]
    fn test_coo_transpose_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::empty([3, 5], DType::F32, &device).unwrap();
        let coo_t = coo.transpose();

        assert_eq!(coo_t.shape(), [5, 3]);
        assert_eq!(coo_t.nnz(), 0);
    }

    #[test]
    fn test_coo_transpose_square() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Square matrix [3, 3]
        let rows = vec![0i64, 1, 2];
        let cols = vec![1i64, 2, 0];
        let values = vec![1.0f32, 2.0, 3.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device).unwrap();
        let coo_t = coo.transpose();

        assert_eq!(coo_t.shape(), [3, 3]);
        assert_eq!(coo_t.nnz(), 3);

        let t_rows: Vec<i64> = coo_t.row_indices().to_vec();
        let t_cols: Vec<i64> = coo_t.col_indices().to_vec();

        assert_eq!(t_rows, vec![1, 2, 0]);
        assert_eq!(t_cols, vec![0, 1, 2]);
    }

    #[test]
    fn test_coo_transpose_double() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // (A^T)^T = A
        let rows = vec![0i64, 0, 1];
        let cols = vec![0i64, 2, 1];
        let values = vec![1.0f32, 2.0, 3.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [2, 3], &device).unwrap();
        let coo_tt = coo.transpose().transpose();

        assert_eq!(coo_tt.shape(), [2, 3]);

        let tt_rows: Vec<i64> = coo_tt.row_indices().to_vec();
        let tt_cols: Vec<i64> = coo_tt.col_indices().to_vec();

        assert_eq!(tt_rows, rows);
        assert_eq!(tt_cols, cols);
    }
}
