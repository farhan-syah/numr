//! Element-wise multiplication operation for sparse tensors

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseTensor};

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Computes the element-wise product of two sparse tensors with the same shape.
    /// Only positions where BOTH tensors have non-zero values will be non-zero
    /// in the result.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise product
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native mul implementation
    /// - Different formats: Converts to COO, multiplies, returns COO
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
    /// # let a = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 1], &[2.0f32, 3.0, 5.0], [2, 2], &device)?;
    /// # let b = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[4.0f32, 7.0], [2, 2], &device)?;
    /// let c = a.mul(&b)?;
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn mul(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
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

        // If same format, use native mul
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.mul(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.mul(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.mul(b)?)),
            // Different formats: convert to COO and multiply
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.mul(coo_b_data)?))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::{CpuClient, CpuRuntime};
    use crate::sparse::SparseFormat;
    use crate::tensor::Tensor;

    #[test]
    fn test_mul_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_coo());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_csr());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        // CSC for A: col_ptrs=[0,1,3], row_indices=[0,0,1], values=[2,3,5]
        // CSC for B: col_ptrs=[0,2,3], row_indices=[0,1,1], values=[4,6,7]
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        assert!(c.is_csc());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A (COO):   B (CSR):
        // [2, 3]     [4, 0]
        // [0, 5]     [6, 7]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 3],
            &[0i64, 0, 1],
            &[4.0f32, 6.0, 7.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert!(a.is_coo());
        assert!(b.is_csr());

        let c = a.mul(&b).unwrap();

        // Mixed formats convert to COO
        assert!(c.is_coo());
        assert_eq!(c.nnz(), 2);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_disjoint() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [1, 0]     [0, 2]
        // [0, 3]     [4, 0]
        // Completely disjoint positions
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[1.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[1i64, 0],
            &[2.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&b).unwrap();

        // Result is empty since no positions overlap
        assert_eq!(c.nnz(), 0);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_mul_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device)
            .unwrap();
        let b = SparseTensor::<CpuRuntime>::empty([3, 2], DType::F32, SparseFormat::Csr, &device)
            .unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F32, SparseFormat::Csr, &device)
            .unwrap();
        let b = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F64, SparseFormat::Csr, &device)
            .unwrap();

        let result = a.mul(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_mul_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create sparse matrices from dense
        let dense_a =
            Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 0.0, 5.0], &[2, 2], &device).unwrap();
        let dense_b =
            Tensor::<CpuRuntime>::from_slice(&[4.0f32, 0.0, 6.0, 7.0], &[2, 2], &device).unwrap();

        let a = SparseTensor::from_dense(&client, &dense_a, 1e-10).unwrap();
        let b = SparseTensor::from_dense(&client, &dense_b, 1e-10).unwrap();

        let c = a.mul(&b).unwrap();

        let dense_c = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_c.to_vec();
        assert_eq!(data, vec![8.0, 0.0, 0.0, 35.0]);
    }

    #[test]
    fn test_mul_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A .* A = A^2 (element-wise)
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[3.0f32, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.mul(&a).unwrap();

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![9.0, 0.0, 0.0, 16.0]);
    }

    #[test]
    fn test_mul_identity_sparse() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Multiplying by a sparse "all ones" matrix at same positions = same matrix
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

        let c = a.mul(&ones).unwrap();

        assert_eq!(c.nnz(), 3);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 3.0, 0.0, 5.0]);
    }
}
