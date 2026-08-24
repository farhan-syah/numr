//! CSC format conversion: to_coo, to_csr

use super::CscData;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::{CooData, CsrData, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CscData<R> {
    /// Convert to COO format
    ///
    /// Expands the compressed column pointers into explicit column indices.
    /// The resulting COO is sorted in column-major order.
    pub fn to_coo(&self) -> Result<CooData<R>> {
        let [_nrows, ncols] = self.shape;
        let nnz = self.nnz();

        // Handle empty case
        if nnz == 0 {
            return CooData::empty(self.shape, self.dtype(), self.values.device());
        }

        // Read CSC data to host
        let col_ptrs: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices: Vec<i64> = self.row_indices.to_vec();

        // Expand column pointers into explicit column indices
        let mut col_indices: Vec<i64> = Vec::with_capacity(nnz);
        for col in 0..ncols {
            let start = col_ptrs[col] as usize;
            let end = col_ptrs[col + 1] as usize;
            for _ in start..end {
                col_indices.push(col as i64);
            }
        }

        // Dispatch on dtype to copy values
        let device = self.values.device();
        crate::dispatch_dtype!(self.dtype(), T => {
            let values: Vec<T> = self.values.to_vec();

            let row_indices_tensor =
                Tensor::try_from_slice(&row_indices, &[row_indices.len()], device)?;
            let col_indices_tensor =
                Tensor::try_from_slice(&col_indices, &[col_indices.len()], device)?;
            let values_tensor = Tensor::try_from_slice(&values, &[values.len()], device)?;

            // COO from CSC is sorted by column, not row, so don't mark as sorted
            return CooData::new(row_indices_tensor, col_indices_tensor, values_tensor, self.shape);
        }, "CSC to COO conversion");
    }

    /// Convert to CSR format
    ///
    /// Converts via COO format, then sorts by row.
    pub fn to_csr(&self) -> Result<CsrData<R>> {
        // Convert to COO first, then to CSR
        let coo = self.to_coo()?;
        coo.to_csr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_csc_to_coo_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csc = super::CscData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let coo = csc.to_coo().unwrap();

        assert_eq!(coo.nnz(), 0);
        assert_eq!(coo.shape(), [3, 3]);
    }

    #[test]
    fn test_csc_to_coo_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Single element at row 1, col 2
        let col_ptrs = vec![0i64, 0, 0, 1];
        let row_indices = vec![1i64];
        let values = vec![42.0f32];

        let csc = super::CscData::<CpuRuntime>::from_slices(
            &col_ptrs,
            &row_indices,
            &values,
            [3, 3],
            &device,
        )
        .unwrap();
        let coo = csc.to_coo().unwrap();

        assert_eq!(coo.nnz(), 1);

        let coo_rows: Vec<i64> = coo.row_indices().to_vec();
        let coo_cols: Vec<i64> = coo.col_indices().to_vec();
        let coo_values: Vec<f32> = coo.values().to_vec();

        assert_eq!(coo_rows, vec![1]);
        assert_eq!(coo_cols, vec![2]);
        assert_eq!(coo_values, vec![42.0]);
    }

    #[test]
    fn test_csc_to_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        //
        // CSC:
        // col_ptrs: [0, 2, 3, 5]
        // row_indices: [0, 2, 2, 0, 1]
        // values: [1, 4, 5, 2, 3]
        let col_ptrs = vec![0i64, 2, 3, 5];
        let row_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 4.0, 5.0, 2.0, 3.0];

        let csc = super::CscData::<CpuRuntime>::from_slices(
            &col_ptrs,
            &row_indices,
            &values,
            [3, 3],
            &device,
        )
        .unwrap();
        let csr = csc.to_csr().unwrap();

        // Expected CSR (row-major order):
        // row_ptrs: [0, 2, 3, 5]
        // col_indices: [0, 2, 2, 0, 1]
        // values: [1, 2, 3, 4, 5]
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), [3, 3]);

        let csr_row_ptrs: Vec<i64> = csr.row_ptrs().to_vec();
        let csr_col_indices: Vec<i64> = csr.col_indices().to_vec();
        let csr_values: Vec<f32> = csr.values().to_vec();

        assert_eq!(csr_row_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(csr_col_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(csr_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_csc_to_csr_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csc = super::CscData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let csr = csc.to_csr().unwrap();

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), [3, 3]);
    }
}
