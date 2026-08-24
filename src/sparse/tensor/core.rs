//! Core SparseTensor implementation: enum, creation, format queries

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::super::coo::CooData;
use super::super::csc::CscData;
use super::super::csr::CsrData;
use super::super::format::{SparseFormat, SparseStorage};

/// Sparse tensor with runtime-selected storage format
///
/// `SparseTensor` is the high-level sparse tensor type that wraps different
/// sparse formats (COO, CSR, CSC). It provides a unified interface for
/// sparse operations regardless of the underlying storage.
///
/// # Format Selection
///
/// - **COO**: Best for construction and random insertion
/// - **CSR**: Best for row operations and SpMV (most common)
/// - **CSC**: Best for column operations
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # #[cfg(feature = "sparse")]
/// # {
/// # use numr::sparse::SparseTensor;
/// # let device = CpuDevice::new();
/// // Create from COO triplets
/// let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
///     &[0, 1, 2],      // rows
///     &[1, 0, 2],      // cols
///     &[1.0f32, 2.0, 3.0],  // values
///     [3, 3],           // shape
///     &device,
/// )?;
///
/// // Convert to CSR for efficient SpMV
/// let csr = sparse.to_csr()?;
/// # }
/// # Ok::<(), numr::error::Error>(())
/// ```
#[derive(Debug, Clone)]
pub enum SparseTensor<R: Runtime> {
    /// COO (Coordinate) format - best for construction, incremental updates, and format conversion.
    ///
    /// Stores (row, col, value) triplets. Flexible for building sparse matrices but slower
    /// for arithmetic operations. Use `to_csr()` or `to_csc()` for efficient computation.
    Coo(CooData<R>),

    /// CSR (Compressed Sparse Row) format - best for row-wise operations and SpMV (y = Ax).
    ///
    /// Compresses row indices using pointer array. Efficient for row slicing, matrix-vector
    /// products, and operations that iterate over rows.
    Csr(CsrData<R>),

    /// CSC (Compressed Sparse Column) format - best for column-wise operations and SpMV (y = A^T x).
    ///
    /// Compresses column indices using pointer array. Efficient for column slicing and
    /// operations that iterate over columns. Transpose of CSC is CSR.
    Csc(CscData<R>),
}

impl<R: Runtime<DType = DType>> SparseTensor<R> {
    // =========================================================================
    // Constructors
    // =========================================================================

    /// Create a sparse tensor from COO data
    pub fn from_coo(data: CooData<R>) -> Self {
        SparseTensor::Coo(data)
    }

    /// Create a sparse tensor from CSR data
    pub fn from_csr(data: CsrData<R>) -> Self {
        SparseTensor::Csr(data)
    }

    /// Create a sparse tensor from CSC data
    pub fn from_csc(data: CscData<R>) -> Self {
        SparseTensor::Csc(data)
    }

    /// Create an empty sparse tensor
    pub fn empty(
        shape: [usize; 2],
        dtype: DType,
        format: SparseFormat,
        device: &R::Device,
    ) -> Result<Self> {
        Ok(match format {
            SparseFormat::Coo => SparseTensor::Coo(CooData::empty(shape, dtype, device)?),
            SparseFormat::Csr => SparseTensor::Csr(CsrData::empty(shape, dtype, device)?),
            SparseFormat::Csc => SparseTensor::Csc(CscData::empty(shape, dtype, device)?),
        })
    }

    /// Create sparse tensor from COO triplet slices
    pub fn from_coo_slices<T: crate::dtype::Element>(
        rows: &[i64],
        cols: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let coo = CooData::from_slices(rows, cols, values, shape, device)?;
        Ok(SparseTensor::Coo(coo))
    }

    /// Create sparse tensor from CSR component slices
    pub fn from_csr_slices<T: crate::dtype::Element>(
        row_ptrs: &[i64],
        col_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let csr = CsrData::from_slices(row_ptrs, col_indices, values, shape, device)?;
        Ok(SparseTensor::Csr(csr))
    }

    /// Create sparse tensor from CSC component slices
    pub fn from_csc_slices<T: crate::dtype::Element>(
        col_ptrs: &[i64],
        row_indices: &[i64],
        values: &[T],
        shape: [usize; 2],
        device: &R::Device,
    ) -> Result<Self> {
        let csc = CscData::from_slices(col_ptrs, row_indices, values, shape, device)?;
        Ok(SparseTensor::Csc(csc))
    }

    /// Create sparse tensor from dense tensor
    ///
    /// Elements with absolute value below `threshold` are treated as zero.
    /// The resulting sparse tensor is in COO format.
    ///
    /// This method delegates to the backend's `SparseOps::dense_to_sparse()`
    /// implementation, which uses GPU-native kernels on CUDA/WebGPU backends.
    ///
    /// # Arguments
    ///
    /// * `client` - Runtime client that implements `SparseOps`
    /// * `tensor` - Dense 2D tensor to convert
    /// * `threshold` - Values with |value| < threshold become zero
    ///
    /// # Errors
    ///
    /// Returns error if tensor is not 2D or if the backend doesn't support
    /// dense-to-sparse conversion.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::SparseTensor;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let dense = Tensor::<CpuRuntime>::try_from_slice(&[1.0, 0.0, 2.0, 0.0], &[2, 2], &device)?;
    /// let sparse = SparseTensor::from_dense(&client, &dense, 1e-10)?;
    /// assert_eq!(sparse.nnz(), 2);  // Only non-zero elements
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    pub fn from_dense<C>(client: &C, tensor: &Tensor<R>, threshold: f64) -> Result<Self>
    where
        C: crate::sparse::SparseOps<R>,
    {
        client.dense_to_sparse(tensor, threshold)
    }

    // =========================================================================
    // Properties
    // =========================================================================

    /// Returns the sparse format
    pub fn format(&self) -> SparseFormat {
        match self {
            SparseTensor::Coo(d) => d.format(),
            SparseTensor::Csr(d) => d.format(),
            SparseTensor::Csc(d) => d.format(),
        }
    }

    /// Returns the matrix shape [nrows, ncols]
    pub fn shape(&self) -> [usize; 2] {
        match self {
            SparseTensor::Coo(d) => d.shape(),
            SparseTensor::Csr(d) => d.shape(),
            SparseTensor::Csc(d) => d.shape(),
        }
    }

    /// Returns the number of rows
    pub fn nrows(&self) -> usize {
        self.shape()[0]
    }

    /// Returns the number of columns
    pub fn ncols(&self) -> usize {
        self.shape()[1]
    }

    /// Returns the number of non-zero elements
    pub fn nnz(&self) -> usize {
        match self {
            SparseTensor::Coo(d) => d.nnz(),
            SparseTensor::Csr(d) => d.nnz(),
            SparseTensor::Csc(d) => d.nnz(),
        }
    }

    /// Returns the data type
    pub fn dtype(&self) -> DType {
        match self {
            SparseTensor::Coo(d) => d.dtype(),
            SparseTensor::Csr(d) => d.dtype(),
            SparseTensor::Csc(d) => d.dtype(),
        }
    }

    /// Returns the sparsity ratio (fraction of zeros)
    pub fn sparsity(&self) -> f64 {
        match self {
            SparseTensor::Coo(d) => d.sparsity(),
            SparseTensor::Csr(d) => d.sparsity(),
            SparseTensor::Csc(d) => d.sparsity(),
        }
    }

    /// Returns the density ratio (fraction of non-zeros)
    pub fn density(&self) -> f64 {
        match self {
            SparseTensor::Coo(d) => d.density(),
            SparseTensor::Csr(d) => d.density(),
            SparseTensor::Csc(d) => d.density(),
        }
    }

    /// Returns true if the tensor has no non-zero elements
    pub fn is_empty(&self) -> bool {
        self.nnz() == 0
    }

    /// Returns the memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match self {
            SparseTensor::Coo(d) => d.memory_usage(),
            SparseTensor::Csr(d) => d.memory_usage(),
            SparseTensor::Csc(d) => d.memory_usage(),
        }
    }

    /// Returns true if stored in COO format
    pub fn is_coo(&self) -> bool {
        matches!(self, SparseTensor::Coo(_))
    }

    /// Returns true if stored in CSR format
    pub fn is_csr(&self) -> bool {
        matches!(self, SparseTensor::Csr(_))
    }

    /// Returns true if stored in CSC format
    pub fn is_csc(&self) -> bool {
        matches!(self, SparseTensor::Csc(_))
    }

    // =========================================================================
    // Format Access
    // =========================================================================

    /// Returns reference to COO data if in COO format
    pub fn as_coo(&self) -> Option<&CooData<R>> {
        match self {
            SparseTensor::Coo(d) => Some(d),
            _ => None,
        }
    }

    /// Returns reference to CSR data if in CSR format
    pub fn as_csr(&self) -> Option<&CsrData<R>> {
        match self {
            SparseTensor::Csr(d) => Some(d),
            _ => None,
        }
    }

    /// Returns reference to CSC data if in CSC format
    pub fn as_csc(&self) -> Option<&CscData<R>> {
        match self {
            SparseTensor::Csc(d) => Some(d),
            _ => None,
        }
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Display for SparseTensor<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SparseTensor({:?}, nnz={}, format={}, dtype={}, sparsity={:.1}%)",
            self.shape(),
            self.nnz(),
            self.format(),
            self.dtype(),
            self.sparsity() * 100.0
        )
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
    fn test_sparse_tensor_coo() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1, 2],
            &[1i64, 0, 2],
            &[1.0f32, 2.0, 3.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_coo());
        assert_eq!(sparse.format(), SparseFormat::Coo);
        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.shape(), [3, 3]);
    }

    #[test]
    fn test_sparse_tensor_csr() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let sparse = SparseTensor::<CpuRuntime>::from_csr_slices(
            &[0i64, 2, 3, 5],
            &[0i64, 2, 2, 0, 1],
            &[1.0f32, 2.0, 3.0, 4.0, 5.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(sparse.is_csr());
        assert_eq!(sparse.format(), SparseFormat::Csr);
        assert_eq!(sparse.nnz(), 5);
    }

    #[test]
    fn test_sparse_tensor_coo_to_csr_conversion() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create COO tensor
        let coo = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[2i64, 0, 1, 0, 2], // unsorted rows
            &[1i64, 0, 2, 2, 0],
            &[5.0f32, 1.0, 3.0, 2.0, 4.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(coo.is_coo());
        assert_eq!(coo.nnz(), 5);

        // Convert to CSR
        let csr = coo.to_csr().unwrap();

        assert!(csr.is_csr());
        assert_eq!(csr.format(), SparseFormat::Csr);
        assert_eq!(csr.nnz(), 5);
        assert_eq!(csr.shape(), [3, 3]);

        // Verify CSR data
        let csr_data = csr.as_csr().unwrap();
        let row_ptrs: Vec<i64> = csr_data.row_ptrs().to_vec();
        let col_indices: Vec<i64> = csr_data.col_indices().to_vec();
        let values: Vec<f32> = csr_data.values().to_vec();

        assert_eq!(row_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(col_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_sparse_tensor_coo_to_csc_conversion() {
        let device = <CpuRuntime as Runtime>::Device::default();

        // Create COO tensor
        let coo = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[2i64, 0, 1, 0, 2], // unsorted rows
            &[1i64, 0, 2, 2, 0],
            &[5.0f32, 1.0, 3.0, 2.0, 4.0],
            [3, 3],
            &device,
        )
        .unwrap();

        assert!(coo.is_coo());

        // Convert to CSC
        let csc = coo.to_csc().unwrap();

        assert!(csc.is_csc());
        assert_eq!(csc.format(), SparseFormat::Csc);
        assert_eq!(csc.nnz(), 5);
        assert_eq!(csc.shape(), [3, 3]);

        // Verify CSC data
        let csc_data = csc.as_csc().unwrap();
        let col_ptrs: Vec<i64> = csc_data.col_ptrs().to_vec();
        let row_indices: Vec<i64> = csc_data.row_indices().to_vec();
        let values: Vec<f32> = csc_data.values().to_vec();

        assert_eq!(col_ptrs, vec![0, 2, 3, 5]);
        assert_eq!(row_indices, vec![0, 2, 2, 0, 1]);
        assert_eq!(values, vec![1.0, 4.0, 5.0, 2.0, 3.0]);
    }

    #[test]
    fn test_sparse_tensor_display() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let sparse = SparseTensor::<CpuRuntime>::from_coo_slices(
            &[0i64, 1],
            &[0i64, 1],
            &[1.0f32, 2.0],
            [10, 10],
            &device,
        )
        .unwrap();

        let display = format!("{}", sparse);
        assert!(display.contains("SparseTensor"));
        assert!(display.contains("nnz=2"));
        assert!(display.contains("COO"));
    }

    #[test]
    fn test_from_dense() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Dense matrix:
        // [1, 0, 2]
        // [0, 0, 3]
        // [4, 5, 0]
        let data = vec![1.0f32, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[3, 3], &device).unwrap();

        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        assert!(sparse.is_coo());
        assert_eq!(sparse.nnz(), 5);
        assert_eq!(sparse.shape(), [3, 3]);

        // Verify COO data (row-major order)
        let coo = sparse.as_coo().unwrap();
        let rows: Vec<i64> = coo.row_indices().to_vec();
        let cols: Vec<i64> = coo.col_indices().to_vec();
        let values: Vec<f32> = coo.values().to_vec();

        assert_eq!(rows, vec![0, 0, 1, 2, 2]);
        assert_eq!(cols, vec![0, 2, 2, 0, 1]);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(coo.is_sorted());
    }

    #[test]
    fn test_from_dense_empty() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // All zeros
        let data = vec![0.0f32; 9];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[3, 3], &device).unwrap();

        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        assert_eq!(sparse.nnz(), 0);
        assert_eq!(sparse.shape(), [3, 3]);
    }

    #[test]
    fn test_from_dense_with_threshold() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        // Dense matrix with small values:
        // [1.0, 0.001, 2.0]
        // [0.0, 0.0,   0.002]
        let data = vec![1.0f32, 0.001, 2.0, 0.0, 0.0, 0.002];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 3], &device).unwrap();

        // With threshold 0.01, values below should be treated as zero
        let sparse = SparseTensor::from_dense(&client, &dense, 0.01).unwrap();

        assert_eq!(sparse.nnz(), 2); // Only 1.0 and 2.0

        let coo = sparse.as_coo().unwrap();
        let values: Vec<f32> = coo.values().to_vec();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn test_from_dense_single_element() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        let data = vec![0.0f32, 0.0, 0.0, 42.0];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 2], &device).unwrap();

        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        assert_eq!(sparse.nnz(), 1);

        let coo = sparse.as_coo().unwrap();
        let rows: Vec<i64> = coo.row_indices().to_vec();
        let cols: Vec<i64> = coo.col_indices().to_vec();
        let values: Vec<f32> = coo.values().to_vec();

        assert_eq!(rows, vec![1]);
        assert_eq!(cols, vec![1]);
        assert_eq!(values, vec![42.0]);
    }

    #[test]
    fn test_from_dense_f64() {
        let device = <CpuRuntime as Runtime>::Device::default();
        let client = CpuClient::new(device.clone());

        let data = vec![1.0f64, 0.0, 2.0, 3.0];
        let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 2], &device).unwrap();

        let sparse = SparseTensor::from_dense(&client, &dense, 1e-10).unwrap();

        assert_eq!(sparse.nnz(), 3);
        assert_eq!(sparse.dtype(), DType::F64);
    }
}
