//! Element-wise addition for COO matrices

use super::super::CooData;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseStorage};

impl<R: Runtime<DType = DType>> CooData<R> {
    /// Element-wise addition: C = A + B
    ///
    /// Computes the sum of two sparse matrices with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another COO matrix with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new COO matrix containing the element-wise sum (sorted by row, then column)
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Algorithm
    ///
    /// Concatenates triplets from both matrices, sorts, and merges duplicates.
    /// GPU-accelerated when CUDA runtime is used.
    ///
    /// # Performance
    ///
    /// - CPU: O((nnz_a + nnz_b) log(nnz_a + nnz_b)) for sorting
    /// - GPU: O((nnz_a + nnz_b) log(nnz_a + nnz_b)) parallel sort-merge
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A:          B:          C = A + B:
    /// // [1, 0]      [0, 2]      [1, 2]
    /// // [0, 3]  +   [4, 0]  =   [4, 3]
    /// # let a_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[1.0f32, 3.0], [2, 2], &device)?;
    /// # let b_sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[1, 0], &[2.0f32, 4.0], [2, 2], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(a) = a_sp { if let numr::sparse::SparseTensor::Coo(b) = b_sp {
    /// let c = a.add(&b)?;
    /// # } }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn add(&self, other: &Self) -> Result<Self>
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
            let (out_row_indices, out_col_indices, out_values) = client.add_coo::<T>(
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
        }, "coo_add")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;

    #[test]
    fn test_coo_add_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
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

        let c = a.add(&b).unwrap();

        // C = A + B:
        // [1, 2]
        // [4, 3]
        assert_eq!(c.shape(), [2, 2]);
        assert_eq!(c.nnz(), 4);
        assert!(c.is_sorted());

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 0, 1, 1]);
        assert_eq!(cols, vec![0, 1, 0, 1]);
        assert_eq!(vals, vec![1.0, 2.0, 4.0, 3.0]);
    }

    #[test]
    fn test_coo_add_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 2]     [3, 0]
        // [0, 0]     [0, 4]
        let a = CooData::<CpuRuntime>::from_slices(
            &[0i64, 0],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        // C = A + B:
        // [4, 2]   (1+3=4 at (0,0))
        // [0, 4]
        assert_eq!(c.nnz(), 3);

        let rows: Vec<i64> = c.row_indices().to_vec();
        let cols: Vec<i64> = c.col_indices().to_vec();
        let vals: Vec<f32> = c.values().to_vec();

        assert_eq!(rows, vec![0, 0, 1]);
        assert_eq!(cols, vec![0, 1, 1]);
        assert_eq!(vals, vec![4.0, 2.0, 4.0]);
    }

    #[test]
    fn test_coo_add_empty_a() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.nnz(), 2);
    }

    #[test]
    fn test_coo_add_empty_b() {
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

        let c = a.add(&b).unwrap();

        assert_eq!(c.nnz(), 2);
    }

    #[test]
    fn test_coo_add_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 3], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([3, 2], DType::F32, &device).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_add_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::empty([2, 2], DType::F32, &device).unwrap();
        let b = CooData::<CpuRuntime>::empty([2, 2], DType::F64, &device).unwrap();

        let result = a.add(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_coo_add_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = CooData::<CpuRuntime>::from_slices(&[0i64], &[0i64], &[1.5f64], [2, 2], &device)
            .unwrap();

        let b = CooData::<CpuRuntime>::from_slices(&[0i64], &[0i64], &[2.5f64], [2, 2], &device)
            .unwrap();

        let c = a.add(&b).unwrap();

        assert_eq!(c.dtype(), DType::F64);
        let vals: Vec<f64> = c.values().to_vec();
        assert_eq!(vals, vec![4.0]);
    }
}
