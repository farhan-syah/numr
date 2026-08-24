//! COO format conversion: to_csr, to_csc

use super::CooData;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::{CscData, CsrData, SparseStorage};
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Convert to CSR format
    ///
    /// This is an efficient conversion that:
    /// 1. Sorts entries by row (if not already sorted)
    /// 2. Computes row pointers
    /// 3. Sums duplicate entries
    pub fn to_csr(&self) -> Result<CsrData<R>> {
        let [nrows, _ncols] = self.shape;
        let nnz = self.nnz();

        // Handle empty case
        if nnz == 0 {
            return CsrData::empty(self.shape, self.dtype(), self.values.device());
        }

        // Read COO data to host
        let rows: Vec<i64> = self.row_indices.to_vec();
        let cols: Vec<i64> = self.col_indices.to_vec();

        // Create permutation for sorting by (row, col)
        let mut perm: Vec<usize> = (0..nnz).collect();
        perm.sort_by(|&a, &b| {
            let row_cmp = rows[a].cmp(&rows[b]);
            if row_cmp == std::cmp::Ordering::Equal {
                cols[a].cmp(&cols[b])
            } else {
                row_cmp
            }
        });

        // Apply permutation to get sorted col_indices
        let sorted_cols: Vec<i64> = perm.iter().map(|&i| cols[i]).collect();

        // Compute row_ptrs from sorted rows
        let mut row_ptrs: Vec<i64> = vec![0; nrows + 1];
        for &i in &perm {
            let row = rows[i] as usize;
            row_ptrs[row + 1] += 1;
        }
        // Cumulative sum
        for i in 1..=nrows {
            row_ptrs[i] += row_ptrs[i - 1];
        }

        // Dispatch on dtype to handle value permutation
        let device = self.values.device();
        crate::dispatch_dtype!(self.dtype(), T => {
            let values: Vec<T> = self.values.to_vec();
            let sorted_values: Vec<T> = perm.iter().map(|&i| values[i]).collect();

            let row_ptrs_tensor = Tensor::try_from_slice(&row_ptrs, &[row_ptrs.len()], device)?;
            let col_indices_tensor =
                Tensor::try_from_slice(&sorted_cols, &[sorted_cols.len()], device)?;
            let values_tensor = Tensor::try_from_slice(&sorted_values, &[sorted_values.len()], device)?;

            return CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, self.shape);
        }, "COO to CSR conversion");
    }

    /// Convert to CSC format
    ///
    /// This is an efficient conversion that:
    /// 1. Sorts entries by column (then by row within each column)
    /// 2. Computes column pointers
    /// 3. Sums duplicate entries
    pub fn to_csc(&self) -> Result<CscData<R>> {
        let [_nrows, ncols] = self.shape;
        let nnz = self.nnz();

        // Handle empty case
        if nnz == 0 {
            return CscData::empty(self.shape, self.dtype(), self.values.device());
        }

        // Read COO data to host
        let rows: Vec<i64> = self.row_indices.to_vec();
        let cols: Vec<i64> = self.col_indices.to_vec();

        // Create permutation for sorting by (col, row)
        let mut perm: Vec<usize> = (0..nnz).collect();
        perm.sort_by(|&a, &b| {
            let col_cmp = cols[a].cmp(&cols[b]);
            if col_cmp == std::cmp::Ordering::Equal {
                rows[a].cmp(&rows[b])
            } else {
                col_cmp
            }
        });

        // Apply permutation to get sorted row_indices
        let sorted_rows: Vec<i64> = perm.iter().map(|&i| rows[i]).collect();

        // Compute col_ptrs from sorted cols
        let mut col_ptrs: Vec<i64> = vec![0; ncols + 1];
        for &i in &perm {
            let col = cols[i] as usize;
            col_ptrs[col + 1] += 1;
        }
        // Cumulative sum
        for i in 1..=ncols {
            col_ptrs[i] += col_ptrs[i - 1];
        }

        // Dispatch on dtype to handle value permutation
        let device = self.values.device();
        crate::dispatch_dtype!(self.dtype(), T => {
            let values: Vec<T> = self.values.to_vec();
            let sorted_values: Vec<T> = perm.iter().map(|&i| values[i]).collect();

            let col_ptrs_tensor = Tensor::try_from_slice(&col_ptrs, &[col_ptrs.len()], device)?;
            let row_indices_tensor =
                Tensor::try_from_slice(&sorted_rows, &[sorted_rows.len()], device)?;
            let values_tensor = Tensor::try_from_slice(&sorted_values, &[sorted_values.len()], device)?;

            return CscData::new(col_ptrs_tensor, row_indices_tensor, values_tensor, self.shape);
        }, "COO to CSC conversion");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_to_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        //
        // COO (unsorted):
        // row: [2, 0, 1, 0, 2]
        // col: [1, 0, 2, 2, 0]
        // val: [5, 1, 3, 2, 4]
        let rows = vec![2i64, 0, 1, 0, 2];
        let cols = vec![1i64, 0, 2, 2, 0];
        let values = vec![5.0f32, 1.0, 3.0, 2.0, 4.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device).unwrap();
        let csr = coo.to_csr().unwrap();

        // Expected CSR:
        // row_ptrs: [0, 2, 3, 5]
        // col_indices: [0, 2, 2, 0, 1]  (sorted by row, then col)
        // values: [1, 2, 3, 4, 5]
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), [3, 3]);

        let row_ptrs: Vec<i64> = csr.row_ptrs().to_vec();
        let col_indices: Vec<i64> = csr.col_indices().to_vec();
        let csr_values: Vec<f32> = csr.values().to_vec();

        assert_eq!(row_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(col_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(csr_values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_coo_to_csr_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let coo = CooData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let csr = coo.to_csr().unwrap();

        assert_eq!(csr.nnz(), 0);
        assert_eq!(csr.shape(), [3, 3]);

        let row_ptrs: Vec<i64> = csr.row_ptrs().to_vec();
        assert_eq!(row_ptrs, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_coo_to_csr_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let rows = vec![1i64];
        let cols = vec![2i64];
        let values = vec![42.0f32];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device).unwrap();
        let csr = coo.to_csr().unwrap();

        assert_eq!(csr.nnz(), 1);

        let row_ptrs: Vec<i64> = csr.row_ptrs().to_vec();
        let col_indices: Vec<i64> = csr.col_indices().to_vec();
        let csr_values: Vec<f32> = csr.values().to_vec();

        assert_eq!(row_ptrs, vec![0, 0, 1, 1]);
        assert_eq!(col_indices, vec![2]);
        assert_eq!(csr_values, vec![42.0]);
    }

    #[test]
    fn test_coo_to_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        //
        // COO (unsorted):
        // row: [2, 0, 1, 0, 2]
        // col: [1, 0, 2, 2, 0]
        // val: [5, 1, 3, 2, 4]
        let rows = vec![2i64, 0, 1, 0, 2];
        let cols = vec![1i64, 0, 2, 2, 0];
        let values = vec![5.0f32, 1.0, 3.0, 2.0, 4.0];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device).unwrap();
        let csc = coo.to_csc().unwrap();

        // Expected CSC (sorted by col, then row):
        // col 0: rows [0, 2], values [1, 4]
        // col 1: rows [2],    values [5]
        // col 2: rows [0, 1], values [2, 3]
        //
        // col_ptrs: [0, 2, 3, 5]
        // row_indices: [0, 2, 2, 0, 1]
        // values: [1, 4, 5, 2, 3]
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), [3, 3]);

        let col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let row_indices: Vec<i64> = csc.row_indices().to_vec();
        let csc_values: Vec<f32> = csc.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(row_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(csc_values, vec![1.0, 4.0, 5.0, 2.0, 3.0]);
    }

    #[test]
    fn test_coo_to_csc_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let coo = CooData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();
        let csc = coo.to_csc().unwrap();

        assert_eq!(csc.nnz(), 0);
        assert_eq!(csc.shape(), [3, 3]);

        let col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        assert_eq!(col_ptrs, vec![0, 0, 0, 0]);
    }

    #[test]
    fn test_coo_to_csc_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let rows = vec![1i64];
        let cols = vec![2i64];
        let values = vec![42.0f32];

        let coo =
            CooData::<CpuRuntime>::from_slices(&rows, &cols, &values, [3, 3], &device).unwrap();
        let csc = coo.to_csc().unwrap();

        assert_eq!(csc.nnz(), 1);

        let col_ptrs: Vec<i64> = csc.col_ptrs().to_vec();
        let row_indices: Vec<i64> = csc.row_indices().to_vec();
        let csc_values: Vec<f32> = csc.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 0, 0, 1]);
        assert_eq!(row_indices, vec![1]);
        assert_eq!(csc_values, vec![42.0]);
    }
}
