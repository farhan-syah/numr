//! Element-wise division operation for sparse tensors

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseTensor};

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    /// Element-wise division: C = A ./ B
    ///
    /// Computes the element-wise quotient of two sparse tensors with the same shape.
    /// Only positions where BOTH tensors have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype (divisor)
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise quotient
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native div implementation
    /// - Different formats: Converts to COO, divides, returns COO
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
    /// # let a = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[8.0f32, 3.0, 10.0], [2, 2], &device)?;
    /// # let b = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1, 1], &[0, 0, 1], &[4.0f32, 6.0, 2.0], [2, 2], &device)?;
    /// let c = a.div(&b)?;
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn div(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R>,
    {
        // Validate shapes match
        if self.shape() != other.shape() {
            return Err(Error::ShapeMismatch {
                expected: vec![self.shape()[0], self.shape()[1]],
                got: vec![other.shape()[0], other.shape()[1]],
            });
        }

        // Validate dtypes match
        if self.dtype() != other.dtype() {
            return Err(Error::DTypeMismatch {
                lhs: self.dtype(),
                rhs: other.dtype(),
            });
        }

        // If same format, use native div
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.div(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.div(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.div(b)?)),
            // Different formats: convert to COO and divide
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.div(coo_b_data)?))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_div_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [8, 0]     [2, 5]
        // [0, 35]    [7, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        // Only overlapping positions: (0,0) and (1,1)
        // 8/2=4, 35/7=5
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // CSC format: col_ptrs point to column starts in row_indices
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 4],
            &[0i64, 1, 0, 1],
            &[2.0f32, 7.0, 5.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // CSR + COO -> COO
        let csr = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[8.0f32, 35.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let coo = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1, 1],
            &[0i64, 1, 0, 1],
            &[2.0f32, 5.0, 7.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = csr.div(&coo).unwrap();

        assert!(matches!(c, SparseTensor::Coo(_)));
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![4.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // No overlapping positions
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&b).unwrap();

        assert_eq!(c.nnz(), 0);
    }

    #[test]
    fn test_div_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [2, 3],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [3, 2],
            &device,
        )
        .unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f32],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64],
            &[0i64],
            &[1.0f64],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = a.div(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_div_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // A = [10, 0; 0, 20]
        // B = [2, 0; 5, 4]
        let dense_a =
            Tensor::<CpuRuntime>::from_slice(&[10.0f32, 0.0, 0.0, 20.0], &[2, 2], &device).unwrap();
        let dense_b =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 0.0, 5.0, 4.0], &[2, 2], &device).unwrap();

        let sparse_a = SparseTensor::from_dense(&client, &dense_a, 1e-10).unwrap();
        let sparse_b = SparseTensor::from_dense(&client, &dense_b, 1e-10).unwrap();

        let c = sparse_a.div(&sparse_b).unwrap();

        // Only (0,0) overlaps: 10/2 = 5
        // (1,1) in A but (1,0) and (1,1) in B - only (1,1) overlaps: 20/4 = 5
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 5.0]);
    }

    #[test]
    fn test_div_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        // A / A = all ones at non-zero positions
        let c = a.div(&a).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_div_by_ones() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A / 1 = A (where ones has same sparsity pattern)
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let ones = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[1.0f32, 1.0, 1.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.div(&ones).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 3.0, 0.0, 5.0]);
    }
}
