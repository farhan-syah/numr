//! CSR format conversion: to_coo, to_csc

use super::CsrData;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::{CooData, CscData, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CsrData<R> {
    /// Convert to COO format
    ///
    /// Expands the compressed row pointers into explicit row indices.
    /// The resulting COO is sorted in row-major order.
    pub fn to_coo(&self) -> Result<CooData<R>> {
        let [nrows, _ncols] = self.shape;
        let nnz = self.nnz();

        // Handle empty case
        if nnz == 0 {
            return CooData::empty(self.shape, self.dtype(), self.values.device());
        }

        // Read CSR data to host
        let row_ptrs: Vec<i64> = self.row_ptrs.to_vec();
        let col_indices: Vec<i64> = self.col_indices.to_vec();

        // Expand row pointers into explicit row indices
        let mut row_indices: Vec<i64> = Vec::with_capacity(nnz);
        for row in 0..nrows {
            let start = row_ptrs[row] as usize;
            let end = row_ptrs[row + 1] as usize;
            for _ in start..end {
                row_indices.push(row as i64);
            }
        }

        // Dispatch on dtype to copy values
        let device = self.values.device();
        crate::dispatch_dtype!(self.dtype(), T => {
            let values: Vec<T> = self.values.to_vec();

            let row_indices_tensor = Tensor::from_slice(&row_indices, &[row_indices.len()], device);
            let col_indices_tensor = Tensor::from_slice(&col_indices, &[col_indices.len()], device);
            let values_tensor = Tensor::from_slice(&values, &[values.len()], device);

            let mut coo = CooData::new(row_indices_tensor, col_indices_tensor, values_tensor, self.shape)?;
            // CSR is already sorted by row, so mark COO as sorted
            unsafe { coo.set_sorted(true); }
            return Ok(coo);
        }, "CSR to COO conversion");
    }

    /// Convert to CSC format
    ///
    /// Converts via COO format, then sorts by column.
    pub fn to_csc(&self) -> Result<CscData<R>> {
        // Convert to COO first, then to CSC
        let coo = self.to_coo()?;
        coo.to_csc()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_csr_to_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        //
        // CSR:
        // row_ptrs: [0, 2, 3, 5]
        // col_indices: [0, 2, 2, 0, 1]
        // values: [1, 2, 3, 4, 5]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();
        let coo = csr.to_coo().unwrap();

        // Expected COO:
        // row_indices: [0, 0, 1, 2, 2]
        // col_indices: [0, 2, 2, 0, 1]
        // values: [1, 2, 3, 4, 5]
        assert_eq!(coo.nnz(), 5);
        assert_eq!(coo.shape(), [3, 3]);
        assert!(coo.is_sorted());

        let coo_rows: Vec<i64> = coo.row_indices().to_vec();
        let coo_cols: Vec<i64> = coo.col_indices().to_vec();
        let coo_values: Vec<f32> = coo.values().to_vec();

        assert_eq!(coo_rows, vec![0, 0, 1, 2, 2]);
        assert_eq!(coo_cols, vec![0, 2, 2, 0, 1]);
        assert_eq!(coo_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_csr_to_coo_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csr = CsrData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let coo = csr.to_coo().unwrap();

        assert_eq!(coo.nnz(), 0);
        assert_eq!(coo.shape(), [3, 3]);
        assert!(coo.is_sorted());
    }

    #[test]
    fn test_csr_to_coo_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Single element at row 1, col 2
        let row_ptrs = vec![0i64, 0, 1, 1];
        let col_indices = vec![2i64];
        let values = vec![42.0f32];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();
        let coo = csr.to_coo().unwrap();

        assert_eq!(coo.nnz(), 1);

        let coo_rows: Vec<i64> = coo.row_indices().to_vec();
        let coo_cols: Vec<i64> = coo.col_indices().to_vec();
        let coo_values: Vec<f32> = coo.values().to_vec();

        assert_eq!(coo_rows, vec![1]);
        assert_eq!(coo_cols, vec![2]);
        assert_eq!(coo_values, vec![42.0]);
    }

    #[test]
    fn test_csr_to_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        //
        // CSR:
        // row_ptrs: [0, 2, 3, 5]
        // col_indices: [0, 2, 2, 0, 1]
        // values: [1, 2, 3, 4, 5]
        let row_ptrs = vec![0i64, 2, 3, 5];
        let col_indices = vec![0i64, 2, 2, 0, 1];
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];

        let csr =
            CsrData::<CpuRuntime>::from_slices(&row_ptrs, &col_indices, &values, [3, 3], &device)
                .unwrap();
        let csc = csr.to_csc().unwrap();

        // Expected CSC (column-major order):
        // col_ptrs: [0, 2, 3, 5]
        // row_indices: [0, 2, 2, 0, 1]
        // values: [1, 4, 5, 2, 3]
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), [3, 3]);

        let csc_col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let csc_row_indices: Vec<i64> = csc.row_indices().to_vec();
        let csc_values: Vec<f32> = csc.values().to_vec();

        assert_eq!(csc_col_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(csc_row_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(csc_values, vec![1.0, 4.0, 5.0, 2.0, 3.0]);
    }

    #[test]
    fn test_csr_to_csc_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let csr = CsrData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let csc = csr.to_csc().unwrap();

        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.shape(), [3, 3]);
    }
}
