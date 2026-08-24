//! Element-wise division for COO matrices

use super::super::CooData;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::SparseStorage;
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise quotient (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// 1. Sort both matrices by (row, col)
    /// 2. Linear merge to find matching positions
    /// 3. Divide values at matching positions (A / B)
    ///
    /// # Performance
    ///
    /// O(nnz_a log nnz_a + nnz_b log nnz_b) for sorting, O(nnz_a + nnz_b) for merge.
    /// Result has at most min(nnz_a, nnz_b) non-zeros.
    ///
    /// # Note
    ///
    /// Division by zero can occur if B stores an explicit zero value. The result
    /// will be infinity or NaN depending on the numerator.
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
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[8.0f32, 3.0, 10.0], [2, 2], &device)?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1, 1], &[0, 0, 1], &[4.0f32, 6.0, 2.0], [2, 2], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(a) = a_sp { if let numr::sparse::SparseTensor::Coo(b) = b_sp {
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

        // Handle empty cases - if either is empty, result is empty
        if self.nnz() == 0 || other.nnz() == 0 {
            return Self::empty(self.shape, dtype, device);
        }

        // Read indices to host
        let rows_a: Vec<i64> = self.row_indices.to_vec();
        let cols_a: Vec<i64> = self.col_indices.to_vec();
        let rows_b: Vec<i64> = other.row_indices.to_vec();
        let cols_b: Vec<i64> = other.col_indices.to_vec();

        // Sort both matrices by (row, col)
        let mut perm_a: Vec<usize> = (0..self.nnz()).collect();
        perm_a.sort_by(|&i, &j| {
            let row_cmp = rows_a[i].cmp(&rows_a[j]);
            if row_cmp != std::cmp::Ordering::Equal {
                row_cmp
            } else {
                cols_a[i].cmp(&cols_a[j])
            }
        });

        let mut perm_b: Vec<usize> = (0..other.nnz()).collect();
        perm_b.sort_by(|&i, &j| {
            let row_cmp = rows_b[i].cmp(&rows_b[j]);
            if row_cmp != std::cmp::Ordering::Equal {
                row_cmp
            } else {
                cols_b[i].cmp(&cols_b[j])
            }
        });

        // Dispatch on dtype to merge and divide values
        crate::dispatch_dtype!(dtype, T => {
            let vals_a: Vec<T> = self.values.to_vec();
            let vals_b: Vec<T> = other.values.to_vec();

            // Linear merge to find matching positions
            let mut result_rows: Vec<i64> = Vec::new();
            let mut result_cols: Vec<i64> = Vec::new();
            let mut result_vals: Vec<T> = Vec::new();

            let mut i = 0;
            let mut j = 0;

            while i < perm_a.len() && j < perm_b.len() {
                let idx_a = perm_a[i];
                let idx_b = perm_b[j];

                let row_a = rows_a[idx_a];
                let col_a = cols_a[idx_a];
                let row_b = rows_b[idx_b];
                let col_b = cols_b[idx_b];

                if (row_a, col_a) < (row_b, col_b) {
                    // A only - skip (result is 0)
                    i += 1;
                } else if (row_a, col_a) > (row_b, col_b) {
                    // B only - skip (result is 0)
                    j += 1;
                } else {
                    // Same position - divide values
                    let quotient = vals_a[idx_a].to_f64() / vals_b[idx_b].to_f64();
                    result_rows.push(row_a);
                    result_cols.push(col_a);
                    result_vals.push(T::from_f64(quotient));
                    i += 1;
                    j += 1;
                }
            }

            // Handle empty result
            if result_rows.is_empty() {
                return Self::empty(self.shape, dtype, device);
            }

            let row_tensor = Tensor::from_slice(&result_rows, &[result_rows.len()], device)?;
            let col_tensor = Tensor::from_slice(&result_cols, &[result_cols.len()], device)?;
            let val_tensor = Tensor::from_slice(&result_vals, &[result_vals.len()], device)?;

            return Ok(Self {
                row_indices: row_tensor,
                col_indices: col_tensor,
                values: val_tensor,
                shape: self.shape,
                sorted: true,
            });
        }, "COO element-wise div");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_div_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        // Overlapping at (0,0) and (1,1)
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only positions where both have values: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 1]);
        assert_eq!(cols, vec![0, 1]);
        assert_eq!(vals, vec![4.0, 5.0]);
    }

    #[test]
    fn test_coo_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // Completely disjoint - no overlapping positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_coo_div_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();

        // Divide with empty matrix gives empty result
        let c = a.div(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.div(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_coo_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device).unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_div_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device).unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_div_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[10.0f32, 18.0, 28.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
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
