//! SparseTensor matrix multiplication: spmv, spmm

use super::SparseTensor;
use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    /// Sparse matrix-vector multiplication: y = A * x
    ///
    /// Computes the product of this sparse matrix with a dense vector.
    ///
    /// # Arguments
    ///
    /// * `x` - Dense vector of length `ncols` (or shape `` `[ncols]` `` or `` `[ncols, 1]` ``)
    ///
    /// # Returns
    ///
    /// Dense vector of length `` `nrows` ``
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `x` length doesn't match matrix ncols
    /// - dtype mismatch between matrix and vector
    ///
    /// # Performance
    ///
    /// CSR format is optimal for SpMV. Other formats will be converted
    /// internally, adding overhead. For repeated SpMV, convert to CSR first.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[1.0f32, 2.0], [2, 3], &device)?;
    /// let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
    /// let y = sparse.spmv(&x)?;  // y = A * x
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn spmv(&self, x: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => d.spmv(x),
            SparseTensor::Csr(d) => d.spmv(x),
            SparseTensor::Csc(d) => d.spmv(x),
        }
    }

    /// Sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// Computes the product of this sparse matrix with a dense matrix.
    ///
    /// # Arguments
    ///
    /// * `b` - Dense matrix of shape `` `[K, N]` `` where K == ncols of sparse matrix
    ///
    /// # Returns
    ///
    /// Dense matrix of shape `` `[M, N]` `` where M == nrows of sparse matrix
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - `b` first dimension doesn't match matrix ncols
    /// - `b` is not 2D
    /// - dtype mismatch between matrix and input
    ///
    /// # Performance
    ///
    /// CSR format is optimal for SpMM. Other formats will be converted
    /// internally, adding overhead. For repeated SpMM, convert to CSR first.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A: [3, 4] sparse, B: [4, 2] dense -> C: [3, 2] dense
    /// # let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[0, 1], &[1.0f32, 2.0], [3, 4], &device)?;
    /// # let b = Tensor::try_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32], &[4, 2], &device)?;
    /// let c = sparse.spmm(&b)?;
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn spmm(&self, b: &Tensor<R>) -> Result<Tensor<R>>
    where
        R::Client: crate::sparse::SparseOps<R>,
    {
        match self {
            SparseTensor::Coo(d) => d.spmm(b),
            SparseTensor::Csr(d) => d.spmm(b),
            SparseTensor::Csc(d) => d.spmm(b),
        }
    }

    /// Transpose the sparse matrix: B = A^T
    ///
    /// Returns the transpose of the sparse matrix with swapped dimensions.
    ///
    /// # Returns
    ///
    /// - COO → COO (swapped indices)
    /// - CSR → CSC (O(1) reinterpretation)
    /// - CSC → CSR (O(1) reinterpretation)
    ///
    /// # Performance
    ///
    /// - COO: O(1) - swaps index tensors
    /// - CSR/CSC: O(1) - structural reinterpretation
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// // A [2, 3]:
    /// // [1, 0, 2]
    /// // [0, 3, 0]
    /// # let a = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 0, 1], &[0, 2, 1], &[1.0f32, 2.0, 3.0], [2, 3], &device)?;
    /// let a_t = a.transpose();
    /// // A^T [3, 2]:
    /// // [1, 0]
    /// // [0, 3]
    /// // [2, 0]
    /// assert_eq!(a_t.shape(), [3, 2]);
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn transpose(&self) -> SparseTensor<R> {
        match self {
            SparseTensor::Coo(d) => SparseTensor::Coo(d.transpose()),
            SparseTensor::Csr(d) => SparseTensor::Csc(d.transpose()),
            SparseTensor::Csc(d) => SparseTensor::Csr(d.transpose()),
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

    // =========================================================================
    // SpMV tests
    // =========================================================================

    #[test]
    fn test_spmv_csr() {
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

        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = sparse.spmv(&x).unwrap();

        // y[0] = 1*1 + 2*3 = 7
        // y[1] = 3*3 = 9
        // y[2] = 4*1 + 5*2 = 14
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_spmv_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Same matrix via COO
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1, 2, 2],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = sparse.spmv(&x).unwrap();

        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_spmv_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Same matrix via CSC
        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 4.0, 5.0, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = sparse.spmv(&x).unwrap();

        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    #[test]
    fn test_spmv_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create sparse from dense
        let dense_data = vec![1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&dense_data, &[3, 3], &device).unwrap();
        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let y = sparse.spmv(&x).unwrap();

        // Compare with manual dense matmul result
        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![7.0, 9.0, 14.0]);
    }

    // =========================================================================
    // SpMM tests
    // =========================================================================

    #[test]
    fn test_spmm_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Sparse A [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        // Dense B [3, 2]
        let b = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )
        .unwrap();

        let c = sparse.spmm(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_spmm_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Same matrix via COO
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        let b = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )
        .unwrap();

        let c = sparse.spmm(&b).unwrap();

        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_spmm_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Same matrix via CSC
        // A [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        // col 0: row 0, val 1
        // col 1: row 1, val 3
        // col 2: row 0, val 2
        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 3.0, 2.0],
            [2, 3],
            &device,
        )
        .unwrap();

        let b = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )
        .unwrap();

        let c = sparse.spmm(&b).unwrap();

        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![11.0, 14.0, 9.0, 12.0]);
    }

    #[test]
    fn test_spmm_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Create sparse from dense
        let dense_a = vec![1.0f32, 0.0, 2.0, 0.0, 3.0, 0.0];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&dense_a, &[2, 3], &device).unwrap();
        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        let b = Tensor::<CpuRuntime>::try_from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[3, 2],
            &device,
        )
        .unwrap();

        let c = sparse.spmm(&b).unwrap();

        // C[0,0] = 1*1 + 2*5 = 11
        // C[0,1] = 1*2 + 2*6 = 14
        // C[1,0] = 3*3 = 9
        // C[1,1] = 3*4 = 12
        let c_data: Vec<f32> = c.to_vec();
        assert_eq!(c_data, vec![11.0, 14.0, 9.0, 12.0]);
    }

    // =========================================================================
    // Transpose tests
    // =========================================================================

    #[test]
    fn test_transpose_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_coo());

        let transposed = sparse.transpose();

        // Transposed [3, 2]:
        // [1, 0]
        // [0, 3]
        // [2, 0]
        assert!(transposed.is_coo());
        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.nnz(), 3);

        // Verify via to_dense
        let dense = transposed.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_transpose_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_csr());

        let transposed = sparse.transpose();

        // CSR transpose -> CSC
        assert!(transposed.is_csc());
        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.nnz(), 3);

        // Verify via to_dense
        let dense = transposed.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_transpose_csc() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        // CSC representation:
        // col 0: row 0, val 1
        // col 1: row 1, val 3
        // col 2: row 0, val 2
        let sparse = SparseTensor::<CpuRuntime>::from_csc_slices(
            &[0i64, 1, 2, 3],
            &[0i64, 1, 0],
            &[1.0f32, 3.0, 2.0],
            [2, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_csc());

        let transposed = sparse.transpose();

        // CSC transpose -> CSR
        assert!(transposed.is_csr());
        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.nnz(), 3);

        // Verify via to_dense
        let dense = transposed.to_dense(&device).unwrap();
        let data: Vec<f32> = dense.to_vec();
        assert_eq!(data, vec![1.0, 0.0, 0.0, 3.0, 2.0, 0.0]);
    }

    #[test]
    fn test_transpose_double() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let original = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 0, 1],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        let orig_dense = original.to_dense(&device).unwrap();

        // Double transpose should recover original
        let double_transposed = original.transpose().transpose();

        assert_eq!(double_transposed.shape(), [2, 3]);
        assert_eq!(double_transposed.nnz(), 3);

        let recovered_dense = double_transposed.to_dense(&device).unwrap();

        let orig_data: Vec<f32> = orig_dense.to_vec();
        let recovered_data: Vec<f32> = recovered_dense.to_vec();
        assert_eq!(orig_data, recovered_data);
    }

    #[test]
    fn test_transpose_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();

        let empty =
            SparseTensor::<CpuRuntime>::empty([2, 3], DType::F32, SparseFormat::Csr, &device)
                .unwrap();

        let transposed = empty.transpose();

        assert!(transposed.is_csc());
        assert_eq!(transposed.shape(), [3, 2]);
        assert_eq!(transposed.nnz(), 0);
    }

    #[test]
    fn test_transpose_spmv() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Matrix A [2, 3]:
        // [1, 0, 2]
        // [0, 3, 0]
        let a = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3],
            &[0i64, 2, 1],
            &[1.0f32, 2.0, 3.0],
            [2, 3],
            &device,
        )
        .unwrap();

        // A^T [3, 2]:
        // [1, 0]
        // [0, 3]
        // [2, 0]
        let a_t = a.transpose();

        // x = [1, 2]
        let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap();

        // y = A^T * x = [1*1 + 0*2, 0*1 + 3*2, 2*1 + 0*2] = [1, 6, 2]
        let y = a_t.spmv(&x).unwrap();

        let y_data: Vec<f32> = y.to_vec();
        assert_eq!(y_data, vec![1.0, 6.0, 2.0]);
    }
}
