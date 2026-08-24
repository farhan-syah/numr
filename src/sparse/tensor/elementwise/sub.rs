//! Element-wise subtraction operation for sparse tensors

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::{SparseOps, SparseTensor};

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    /// Element-wise subtraction: C = A - B
    ///
    /// Computes the difference of two sparse tensors with the same shape.
    ///
    /// # Arguments
    ///
    /// * `other` - Another sparse tensor with the same shape and dtype
    ///
    /// # Returns
    ///
    /// A new sparse tensor containing the element-wise difference
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - Shapes don't match
    /// - Dtypes don't match
    ///
    /// # Format Handling
    ///
    /// - Same format: Uses native sub implementation
    /// - Different formats: Converts to COO, subtracts, returns COO
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
    /// # let a = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[5.0f32, 8.0], [2, 2], &device)?;
    /// # let b = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 1, 0], &[2.0f32, 3.0, 4.0], [2, 2], &device)?;
    /// let c = a.sub(&b)?;
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn sub(&self, other: &SparseTensor<R>) -> Result<SparseTensor<R>>
    where
        R::Client: SparseOps<R> + ScalarOps<R>,
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

        // If same format, use native sub
        match (self, other) {
            (SparseTensor::Coo(a), SparseTensor::Coo(b)) => Ok(SparseTensor::Coo(a.sub(b)?)),
            (SparseTensor::Csr(a), SparseTensor::Csr(b)) => Ok(SparseTensor::Csr(a.sub(b)?)),
            (SparseTensor::Csc(a), SparseTensor::Csc(b)) => Ok(SparseTensor::Csc(a.sub(b)?)),
            // Different formats: convert to COO and subtract
            _ => {
                let coo_a = self.to_coo()?;
                let coo_b = other.to_coo()?;
                let coo_a_data = coo_a.as_coo().unwrap();
                let coo_b_data = coo_b.as_coo().unwrap();
                Ok(SparseTensor::Coo(coo_a_data.sub(coo_b_data)?))
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
    fn test_sub_coo_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_csr_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_csr());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_csc_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 4.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&b).unwrap();

        assert!(c.is_csc());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_mixed_formats() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A (COO):   B (CSR):
        // [5, 0]     [2, 3]
        // [0, 8]     [4, 0]
        let a = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        assert!(a.is_coo());
        assert!(b.is_csr());

        let c = a.sub(&b).unwrap();

        // Mixed formats convert to COO
        assert!(c.is_coo());
        assert_eq!(c.nnz(), 4);

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_overlapping() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A:         B:
        // [5, 2]     [3, 0]
        // [0, 0]     [0, 4]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 2],
            &[0i64, 1],
            &[5.0f32, 2.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
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

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 2.0, 0.0, -4.0]);
    }

    #[test]
    fn test_sub_shape_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device)
            .unwrap();
        let b = SparseTensor::<CpuRuntime>::empty([3, 2], DType::F32, SparseFormat::Csr, &device)
            .unwrap();

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_dtype_mismatch() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let a = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F32, SparseFormat::Csr, &device)
            .unwrap();
        let b = SparseTensor::<CpuRuntime>::empty([2, 2], DType::F64, SparseFormat::Csr, &device)
            .unwrap();

        let result = a.sub(&b);
        assert!(result.is_err());
    }

    #[test]
    fn test_sub_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create sparse matrices from dense
        let dense_a =
            Tensor::<CpuRuntime>::try_from_slice(&[5.0f32, 0.0, 0.0, 8.0], &[2, 2], &device)
                .unwrap();
        let dense_b =
            Tensor::<CpuRuntime>::try_from_slice(&[2.0f32, 3.0, 4.0, 0.0], &[2, 2], &device)
                .unwrap();

        let a = SparseTensor::from_dense(&client, &dense_a, 1e-10).unwrap();
        let b = SparseTensor::from_dense(&client, &dense_b, 1e-10).unwrap();

        let c = a.sub(&b).unwrap();

        let dense_c = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_c.to_vec();
        assert_eq!(data, vec![3.0, -3.0, -4.0, 8.0]);
    }

    #[test]
    fn test_sub_self() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // A - A should be all zeros
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 2.0, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let c = a.sub(&a).unwrap();

        let dense = c.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_add_then_sub() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // (A + B) - B should equal A
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 1, 2],
            &[0i64, 1],
            &[5.0f32, 8.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let b = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 1, 0],
            &[2.0f32, 3.0, 4.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let sum = a.add(&b).unwrap();
        let result = sum.sub(&b).unwrap();

        let dense_result = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense_result.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 8.0]);
    }
}
