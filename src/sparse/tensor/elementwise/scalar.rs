//! Scalar operations for sparse tensors

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::Runtime;
use crate::sparse::SparseTensor;

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    /// Scalar multiplication: C = A * scalar
    ///
    /// Multiplies all non-zero values by a scalar constant.
    /// Preserves the sparsity pattern and format.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply with
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?;
    /// let result = sparse.scalar_mul(2.0)?;  // Multiply all values by 2
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Coo(d.scalar_mul(scalar)?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csr(d.scalar_mul(scalar)?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csc(d.scalar_mul(scalar)?)),
        }
    }

    /// Scalar addition: C = A + scalar (on non-zeros only)
    ///
    /// **Warning**: This adds the scalar ONLY to non-zero elements, preserving sparsity.
    /// This is NOT standard mathematical scalar addition (which would densify the matrix).
    ///
    /// Standard scalar addition would add to ALL positions (including implicit zeros),
    /// making the result dense. This operation is a practical sparse-preserving variant.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to add
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?;
    /// let result = sparse.scalar_add(1.0)?;  // Add 1 to all non-zero values
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn scalar_add(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Coo(d.scalar_add(scalar)?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csr(d.scalar_add(scalar)?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csc(d.scalar_add(scalar)?)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseFormat;

    #[test]
    fn test_sparse_tensor_scalar_mul_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert!(result.is_coo());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 10.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 14.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(3.0).unwrap();

        assert_eq!(result.nnz(), 5);
        assert!(result.is_csr());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, 0.0, 6.0, 0.0, 0.0, 9.0, 12.0, 15.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 5);
        assert!(result.is_csc());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![2.0, 0.0, 4.0, 0.0, 0.0, 6.0, 8.0, 10.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_add_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 1],
            &[0i64, 0, 1],
            &[2.0f32, 3.0, 5.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_add(1.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert!(result.is_coo());

        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![3.0, 0.0, 4.0, 6.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_mul_zero() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(0.0).unwrap();

        assert_eq!(result.nnz(), 2); // Still has 2 entries, but values are 0
        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_sparse_tensor_scalar_add_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse =
            SparseTensor::<CpuRuntime>::empty([3, 3], DType::F32, SparseFormat::Coo, &device)
                .unwrap();

        let result = sparse.scalar_add(5.0).unwrap();

        assert_eq!(result.nnz(), 0); // Empty stays empty
    }

    #[test]
    fn test_sparse_tensor_scalar_ops_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[2.5f64, 3.5],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = sparse.scalar_mul(2.0).unwrap();

        assert_eq!(result.dtype(), DType::F64);
        let dense = result.to_dense(&device).unwrap();
        let data: Vec<f64> = dense.to_vec();
        assert_eq!(data, vec![5.0, 0.0, 0.0, 7.0]);
    }
}
