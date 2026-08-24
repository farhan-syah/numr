//! Element-wise multiplication (Hadamard product) for COO matrices

use super::super::CooData;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse matrices with the same shape.
    /// Only positions where BOTH matrices have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise product (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Sorts both matrices, then performs linear merge to find matching positions.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a log nnz_a + nnz_b log nnz_b + nnz_a + nnz_b)
    /// - GPU: Parallel sort-merge on device
    /// - Result has at most min(nnz_a, nnz_b) non-zeros
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A .* B:
    /// // [2, 3]      [4, 0]      [8, 0]
    /// // [0, 5]  .*  [6, 7]  =   [0, 35]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[2.0f32, 3.0, 5.0], [2, 2], &device)?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[4.0f32, 7.0], [2, 2], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(a) = a_sp { if let numr::sparse::SparseTensor::Coo(b) = b_sp {
    /// let c = a.mul(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn mul(&self, other: &Self) -> Result<Self>
    where
        R::Client: SparseOps<R>,
    {
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

        // Get client for runtime dispatch
        let client = R::default_client(device);

        // Dispatch to runtime-specific implementation
        crate::dispatch_dtype!(dtype, T => {
            let (out_row_indices, out_col_indices, out_values) = client.mul_coo::<T>(
                &self.row_indices,
                &self.col_indices,
                &self.values,
                &other.row_indices,
                &other.col_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                row_indices: out_row_indices,
                col_indices: out_col_indices,
                values: out_values,
                shape: self.shape,
                sorted: true,  // Backend guarantees sorted output
            })
        }, "coo_mul")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_mul_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // C = A .* B:
        // [8, 0]    (2*4=8 at (0,0), 3*0=0, 0*6=0)
        // [0, 35]   (0*6=0, 5*7=35)
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 2); // Only positions where both have values

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 1]);
        assert_eq!(cols, vec![0, 1]);
        assert_eq!(vals, vec![8.0, 35.0]);
    }

    #[test]
    fn test_coo_mul_disjoint() {
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

        let c = a.mul(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_coo_mul_empty() {
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

        // Multiply with empty matrix gives empty result
        let c = a.mul(&b).unwrap();
        assert_eq!(c.nnz(), 0);

        let c2 = b.mul(&a).unwrap();
        assert_eq!(c2.nnz(), 0);
    }

    #[test]
    fn test_coo_mul_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device).unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_mul_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device).unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_mul_same_positions() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Both have values at exactly the same positions
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[5.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert_eq!(c.nnz(), 3);

        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![10.0, 18.0, 28.0]); // 2*5, 3*6, 4*7
    }
}
