//! SparseTensor format conversion: to_coo, to_csr, to_csc

use super::SparseTensor;
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::SparseFormat;
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    // =========================================================================
    // Format Conversion
    // =========================================================================

    /// Convert to COO format
    pub fn to_coo(&self) -> Result<SparseTensor<R>> {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Coo(d.clone())),
            SparseTensor::Csr(d) => Ok(SparseTensor::Coo(d.to_coo()?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Coo(d.to_coo()?)),
        }
    }

    /// Convert to CSR format
    pub fn to_csr(&self) -> Result<SparseTensor<R>> {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Csr(d.to_csr()?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csr(d.clone())),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csr(d.to_csr()?)),
        }
    }

    /// Convert to CSC format
    pub fn to_csc(&self) -> Result<SparseTensor<R>> {
        match self {
            SparseTensor::Coo(d) => Ok(SparseTensor::Csc(d.to_csc()?)),
            SparseTensor::Csr(d) => Ok(SparseTensor::Csc(d.to_csc()?)),
            SparseTensor::Csc(d) => Ok(SparseTensor::Csc(d.clone())),
        }
    }

    /// Convert to specified format
    pub fn to_format(&self, format: SparseFormat) -> Result<SparseTensor<R>> {
        match format {
            SparseFormat::Coo => self.to_coo(),
            SparseFormat::Csr => self.to_csr(),
            SparseFormat::Csc => self.to_csc(),
        }
    }

    // =========================================================================
    // Dense Conversion
    // =========================================================================

    /// Convert to dense tensor
    ///
    /// Creates a dense 2D tensor from the sparse representation.
    /// Non-zero elements are placed at their respective positions,
    /// and all other positions are filled with zeros.
    ///
    /// # Warning
    ///
    /// This may use significant memory for large sparse matrices.
    /// A matrix of shape [m, n] will allocate m*n elements.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
    ///     &[0, 1], &[1, 0], &[5.0f32, 3.0], [2, 2], &device
    /// )?;
    /// let dense = sparse.to_dense(&device)?;
    /// // dense is [[0, 5], [3, 0]]
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn to_dense(&self, device: &R::Device) -> Result<Tensor<R>> {
        let [nrows, ncols] = self.shape();
        let dtype = self.dtype();
        let numel = nrows * ncols;

        // Handle empty case - return zero tensor
        if self.is_empty() {
            crate::dispatch_dtype!(dtype, T => {
                let zeros: Vec<T> = vec![T::zero(); numel];
                return Ok(Tensor::from_slice(&zeros, &[nrows, ncols], device));
            }, "sparse to dense empty");
        }

        // Convert to COO for uniform access to indices and values
        let coo = match self {
            SparseTensor::Coo(d) => d.clone(),
            SparseTensor::Csr(d) => d.to_coo()?,
            SparseTensor::Csc(d) => d.to_coo()?,
        };

        // Read indices to host
        let row_indices: Vec<i64> = coo.row_indices().to_vec();
        let col_indices: Vec<i64> = coo.col_indices().to_vec();

        // Dispatch on dtype to create dense tensor
        crate::dispatch_dtype!(dtype, T => {
            let values: Vec<T> = coo.values().to_vec();

            // Create zero-initialized dense data
            let mut dense_data: Vec<T> = vec![T::zero(); numel];

            // Scatter values into dense tensor
            for (i, (val, (row, col))) in values.iter()
                .zip(row_indices.iter().zip(col_indices.iter()))
                .enumerate()
            {
                let r = *row as usize;
                let c = *col as usize;

                if r >= nrows || c >= ncols {
                    return Err(Error::IndexOutOfBounds {
                        index: i,
                        size: numel,
                    });
                }

                let idx = r * ncols + c;
                dense_data[idx] = *val;
            }

            return Ok(Tensor::from_slice(&dense_data, &[nrows, ncols], device));
        }, "sparse to dense conversion");
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

    // =========================================================================
    // to_dense tests
    // =========================================================================

    #[test]
    fn test_to_dense_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Sparse matrix:
        // [0, 5, 0]
        // [3, 0, 0]
        // [0, 0, 7]
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[5.0f32, 3.0, 7.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let dense = sparse.to_dense(&device).unwrap();

        assert_eq!(dense.shape(), &[3, 3]);
        assert_eq!(dense.dtype(), DType::F32);

        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 5.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 7.0]);
    }

    #[test]
    fn test_to_dense_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_csr());

        let dense = sparse.to_dense(&device).unwrap();

        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn test_to_dense_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_csc());

        let dense = sparse.to_dense(&device).unwrap();

        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn test_to_dense_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse =
            SparseTensor::<CpuRuntime>::empty([3, 3], DType::F32, SparseFormat::Coo, &device)
                .unwrap();

        let dense = sparse.to_dense(&device).unwrap();

        assert_eq!(dense.shape(), &[3, 3]);
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0; 9]);
    }

    #[test]
    fn test_to_dense_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Single element at (1, 1)
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[1i64],
            &[1i64],
            &[42.0f32],
            [2, 2],
            &device,
        )
        .unwrap();

        let dense = sparse.to_dense(&device).unwrap();

        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![0.0, 0.0, 0.0, 42.0]);
    }

    #[test]
    fn test_to_dense_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.5f64, 2.5],
            [2, 2],
            &device,
        )
        .unwrap();

        let dense = sparse.to_dense(&device).unwrap();

        assert_eq!(dense.dtype(), DType::F64);
        let data: Vec<f64> = dense.to_vec();
        assert_eq!(data, vec![1.5, 0.0, 0.0, 2.5]);
    }

    #[test]
    fn test_dense_sparse_roundtrip() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Original dense matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let original_data = vec![1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0];
        let original = Tensor::<CpuRuntime>::from_slice(&original_data, &[3, 3], &device);

        // Dense -> Sparse -> Dense
        let sparse = SparseTensor::from_dense(&client, &original, 1e-10).unwrap();
        let recovered = sparse.to_dense(&device).unwrap();

        let recovered_data: Vec<f32> = recovered.to_vec();
        assert_eq!(recovered_data, original_data);
    }

    #[test]
    fn test_csr_to_dense_to_sparse_roundtrip() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Start with CSR
        let sparse_csr = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        // CSR -> Dense -> COO
        let dense = sparse_csr.to_dense(&device).unwrap();
        let sparse_coo = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        assert!(sparse_coo.is_coo());
        assert_eq!(sparse_coo.nnz(), 5);

        // Verify same dense representation
        let recovered = sparse_coo.to_dense(&device).unwrap();
        let data: Vec<f32> = recovered.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0]);
    }
}
