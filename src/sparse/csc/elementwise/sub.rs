//! Element-wise subtraction for CSC matrices

use super::super::CscData;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};

impl<R: Runtime<DType = DType>> CscData<R> {
    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another CSC matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new CSC matrix containing the element-wise difference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Column-by-column merge of sorted row indices using union semantics.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O(nnz_a + nnz_b) sequential merge
    /// - GPU: O(nnz_a + nnz_b) parallel per-column merge
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A - B:
    /// // [5, 0]      [2, 3]      [3, -3]
    /// // [0, 8]  -   [4, 0]  =   [-4, 8]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[5.0f32, 8.0], [2, 2], &device)?.to_csc()?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 0], &[2.0f32, 3.0, 4.0], [2, 2], &device)?.to_csc()?;
    /// # if let numr::sparse::SparseTensor::Csc(a) = a_sp { if let numr::sparse::SparseTensor::Csc(b) = b_sp {
    /// let c = a.sub(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn sub(&self, other: &Self) -> Result<Self>
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
            let (out_col_ptrs, out_row_indices, out_values) = client.sub_csc::<T>(
                &self.col_ptrs,
                &self.row_indices,
                &self.values,
                &other.col_ptrs,
                &other.row_indices,
                &other.values,
                self.shape,
            )?;

            Ok(Self {
                col_ptrs: out_col_ptrs,
                row_indices: out_row_indices,
                values: out_values,
                shape: self.shape,
            })
        }, "csc_sub")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_csc_sub_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,1], values=[5,8]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,0], values=[2,4,3]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 4.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        // C = A - B:
        // [3, -3]
        // [-4, 8]
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 4);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 4]);
        assert_eq!(row_indices, vec![0, 1, 0, 1]);
        assert_eq!(vals, vec![3.0, -4.0, -3.0, 8.0]);
    }

    #[test]
    fn test_csc_sub_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 2]     [3, 0]
        // [0, 0]     [0, 4]
        // CSC for A: col_ptrs=[0,1,2], row_indices=[0,0], values=[5,2]
        // CSC for B: col_ptrs=[0,1,2], row_indices=[0,1], values=[3,4]
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 0],
            &[5.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        // C = A - B:
        // [2, 2]   (5-3=2 at (0,0))
        // [0, -4]
        assert_eq!(c.nnz(), 3);

        let col_ptrs: Vec<i64> = c.col_ptrs().to_vec();
        let row_indices: Vec<i64> = c.row_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 1, 3]);
        assert_eq!(row_indices, vec![0, 0, 1]);
        assert_eq!(vals, vec![2.0, 2.0, -4.0]);
    }

    #[test]
    fn test_csc_sub_empty_a() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();
        let b = CscData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();
        assert_eq!(c.nnz(), 2);

        // Result should be -B
        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![-1.0, -2.0]);
    }

    #[test]
    fn test_csc_sub_empty_b() {
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

        let c = a.sub(&b).unwrap();
        assert_eq!(c.nnz(), 2);

        // Result should be A
        let vals: Vec<f32> = c.values().to_vec();
        assert_eq!(vals, vec![1.0, 2.0]);
    }

    #[test]
    fn test_csc_sub_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CscData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b = CscData::<CpuRuntime>::empty([3, 2], DType::F32, &device).unwrap();

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_csc_sub_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A - A should be all zeros (empty sparse matrix)
        let a = CscData::<CpuRuntime>::from_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 2.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&a).unwrap();
        // Sparse matrices don't store explicit zeros - result should be empty
        assert_eq!(c.nnz(), 0);
    }
}
