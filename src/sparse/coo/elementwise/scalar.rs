//! Scalar operations for COO matrices (scalar multiplication and addition)

use super::super::CooData;
use crate::error::Result;
use crate::ops::ScalarOps;
use crate::runtime::Runtime;

impl<R: Runtime> CooData<R> {
    /// Scalar multiplication: C = A * scalar
    ///
    /// Multiplies all non-zero values by a scalar constant.
    /// Preserves the sparsity pattern.
    ///
    /// # Arguments
    ///
    /// * `scalar` - The scalar value to multiply with
    ///
    /// # Performance
    ///
    /// O(nnz) - simply scales the values tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(coo) = sp {
    /// let result = coo.scalar_mul(2.0)?;  // Multiply all values by 2
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn scalar_mul(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        let device = self.values.device();
        let client = R::default_client(device);

        let scaled_values = client.mul_scalar(&self.values, scalar)?;

        Ok(Self {
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
            values: scaled_values,
            shape: self.shape,
            sorted: self.sorted,
        })
    }

    /// Add scalar to non-zero elements only (sparsity-preserving)
    ///
    /// # ⚠️ Important: This is NOT Standard Scalar Addition!
    ///
    /// Standard mathematical scalar addition (`A + s`) adds `s` to **ALL** elements
    /// including implicit zeros, creating a dense matrix. This operation only adds
    /// to existing non-zero values, preserving the sparse structure.
    ///
    /// Use this when you want to shift all non-zero values by a constant without
    /// densifying the matrix. For true scalar addition, convert to dense first.
    ///
    /// # Mathematical Behavior
    ///
    /// - **Non-zero elements**: `A[i,j] + s`
    /// - **Zero elements**: remain `0` (NOT `s`!)
    ///
    /// # Performance
    ///
    /// O(nnz) - simply adds to the values tensor
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // Sparse matrix:        After scalar_add(10):
    /// // [1, 0, 2]             [11,  0, 12]
    /// // [0, 3, 0]             [ 0, 13,  0]
    /// //
    /// // Note: Zeros stay 0, not 10!
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], [2, 3], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(coo) = sp {
    /// let result = coo.scalar_add(10.0)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    ///
    /// # See Also
    ///
    /// - [`add_to_nonzeros()`](Self::add_to_nonzeros) - Clearer alias for this method
    pub fn scalar_add(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        // Handle empty tensor case (no values to add to)
        if self.values.numel() == 0 {
            return Ok(Self {
                row_indices: self.row_indices.clone(),
                col_indices: self.col_indices.clone(),
                values: self.values.clone(),
                shape: self.shape,
                sorted: self.sorted,
            });
        }

        let device = self.values.device();
        let client = R::default_client(device);

        let shifted_values = client.add_scalar(&self.values, scalar)?;

        Ok(Self {
            row_indices: self.row_indices.clone(),
            col_indices: self.col_indices.clone(),
            values: shifted_values,
            shape: self.shape,
            sorted: self.sorted,
        })
    }

    /// Alias for [`scalar_add()`](Self::scalar_add) with clearer naming
    ///
    /// Adds a scalar value to all non-zero elements, preserving sparsity.
    /// This name makes it explicit that only non-zero elements are modified.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // Clearer intent than scalar_add
    /// # let sp = SparseTensor::<CpuRuntime>::from_coo_slices(&[0], &[0], &[1.0f32], [1, 1], &device)?;
    /// # if let numr::sparse::SparseTensor::Coo(coo) = sp {
    /// let result = coo.add_to_nonzeros(5.0)?;
    /// # }
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    #[inline]
    pub fn add_to_nonzeros(&self, scalar: f64) -> Result<Self>
    where
        R::Client: ScalarOps<R>,
    {
        self.scalar_add(scalar)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::CpuRuntime;
    use crate::sparse::SparseStorage;

    #[test]
    fn test_coo_scalar_mul() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(2.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert_eq!(result.shape(), [3, 3]);

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![10.0, 6.0, 14.0]);
    }

    #[test]
    fn test_coo_scalar_mul_zero() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(0.0).unwrap();

        assert_eq!(result.nnz(), 2);
        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![0.0, 0.0]);
    }

    #[test]
    fn test_coo_scalar_mul_negative() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, -3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(-2.0).unwrap();

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![-10.0, 6.0]);
    }

    #[test]
    fn test_coo_scalar_add() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let result = coo.scalar_add(1.0).unwrap();

        assert_eq!(result.nnz(), 3);
        assert_eq!(result.shape(), [3, 3]);

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![6.0, 4.0, 8.0]);
    }

    #[test]
    fn test_coo_scalar_add_negative() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[5.0f32, 3.0],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_add(-2.0).unwrap();

        let vals: Vec<f32> = result.values().to_vec();
        assert_eq!(vals, vec![3.0, 1.0]);
    }

    #[test]
    fn test_coo_scalar_add_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::empty([3, 3], DType::F32, &device).unwrap();

        let result = coo.scalar_add(5.0).unwrap();

        assert_eq!(result.nnz(), 0); // Empty stays empty
    }

    #[test]
    fn test_coo_scalar_mul_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let coo = CooData::<CpuRuntime>::from_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[2.5f64, 3.5],
            [2, 2],
            &device,
        )
        .unwrap();

        let result = coo.scalar_mul(2.0).unwrap();

        let vals: Vec<f64> = result.values().to_vec();
        assert_eq!(vals, vec![5.0, 7.0]);
    }
}
