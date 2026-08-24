//! Element-wise division for CSC matrices

use super::super::CscData;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::SparseStorage;
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CscData<R> {
    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise quotient
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column intersection of sorted row indices, dividing values
    /// at matching positions.
    ///
    /// # Performance
    ///
    /// O(nnz_a + nnz_b) - linear merge since rows are sorted within columns.
    /// Result has at most min(nnz_a, nnz_b) non-zeros.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A ./ B:
    /// // [8, 3]      [4, 0]      [2, 0]
    /// // [0, 10]  ./ [6, 2]  =   [0, 5]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[8.0f32, 3.0, 10.0], [2, 2], &device)?.to_csc()?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1, 1], &[0, 0, 1], &[4.0f32, 6.0, 2.0], [2, 2], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(a) = a_sp { if let numr::sparse::SparseTensor::Csc(b) = b_sp {
    /// let c = a.div(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn div(&self, other: &Self) -> Result<Self> {
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
        let [_nrows, ncols] = self.shape;

        // Handle empty cases - if either is empty, result is empty
        if self.nnz() == 0 || other.nnz() == 0 {
            return Self::empty(self.shape, dtype, device);
        }

        // Read CSC data from both matrices
        let col_ptrs_a: Vec<i64> = self.col_ptrs.to_vec();
        let row_indices_a: Vec<i64> = self.row_indices.to_vec();
        let col_ptrs_b: Vec<i64> = other.col_ptrs.to_vec();
        let row_indices_b: Vec<i64> = other.row_indices.to_vec();

        // Dispatch on dtype to merge and divide values
        crate::dispatch_dtype!(dtype, T => {
            let vals_a: Vec<T> = self.values.to_vec();
            let vals_b: Vec<T> = other.values.to_vec();

            // Build result arrays column by column
            let mut result_col_ptrs: Vec<i64> = Vec::with_capacity(ncols + 1);
            let mut result_rows: Vec<i64> = Vec::new();
            let mut result_vals: Vec<T> = Vec::new();

            result_col_ptrs.push(0);

            for col in 0..ncols {
                let start_a = col_ptrs_a[col] as usize;
                let end_a = col_ptrs_a[col + 1] as usize;
                let start_b = col_ptrs_b[col] as usize;
                let end_b = col_ptrs_b[col + 1] as usize;

                // Intersect sorted rows for this column
                let mut i = start_a;
                let mut j = start_b;

                while i < end_a && j < end_b {
                    let row_a = row_indices_a[i];
                    let row_b = row_indices_b[j];

                    if row_a < row_b {
                        // A only - skip (result is 0)
                        i += 1;
                    } else if row_b < row_a {
                        // B only - skip (result is 0)
                        j += 1;
                    } else {
                        // Same row - divide values
                        let quotient = vals_a[i].to_f64() / vals_b[j].to_f64();
                        result_rows.push(row_a);
                        result_vals.push(T::from_f64(quotient));
                        i += 1;
                        j += 1;
                    }
                }

                result_col_ptrs.push(result_rows.len() as i64);
            }

            let col_ptrs_tensor =
                Tensor::try_from_slice(&result_col_ptrs, &[result_col_ptrs.len()], device)?;
            let row_indices_tensor =
                Tensor::try_from_slice(&result_rows, &[result_rows.len()], device)?;
            let values_tensor = Tensor::try_from_slice(&result_vals, &[result_vals.len()], device)?;

            return Self::new(col_ptrs_tensor, row_indices_tensor, values_tensor, self.shape);
        }, "CSC element-wise div");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_csc_div_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[8,35]
        // CSC for B: col_ptrs=[0,2,4], row_indices=[0,1,0,1], values=[2,7,5,7]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 7.0, 5.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only positions where both have values: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 2]);
        assert_eq!(row_indices, vec![0, 1]);
        assert_eq!(vals, vec![4.0, 5.0]);
    }

    #[test]
    fn test_csc_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[1,3]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[1,0], values=[4,2]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[4.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_csc_div_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();

        // Divide with empty matrix gives empty result
        let c = a.div(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.div(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_csc_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device).unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_div_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        // Matrix layout:
        // [10, 0]    [2, 0]
        // [18, 28]   [6, 7]
        // CSC for A: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[10,18,28]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[2,6,7]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[10.0f32, 18.0, 28.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 3);

        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![5.0, 3.0, 4.0]); // 10/2, 18/6, 28/7
    }
}
