//! Sparse operations trait
//!
//! Defines the interface for sparse tensor operations that backends implement.

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::tensor::Tensor;

use super::{CscData, CsrData, SparseTensor};

/// Trait for sparse tensor operations
///
/// This trait defines sparse matrix operations that are implemented by each
/// backend (CPU, CUDA, WebGPU). Operations follow scipy.sparse conventions.
///
/// # Architecture
///
/// This trait has two levels of operations:
/// 1. **Low-level format-specific operations** (spmv_csr, add_csr, etc.)
///    - Work directly with raw CSR/COO/CSC data (row_ptrs, col_indices, values)
///    - Backend-specific implementations (no .to_vec() calls)
///    - Stay on device (GPU ops stay on GPU)
///
/// 2. **High-level format-agnostic operations** (spmv, sparse_add, etc.)
///    - Work with SparseTensor wrapper
///    - Dispatch to format-specific low-level methods
///    - Convenience API for users
///
/// # Implementation Notes
///
/// - All operations preserve sparsity when possible
/// - SpMV/SpMM are the most performance-critical operations
/// - Format conversion may be needed for optimal performance
/// - Backends implement low-level operations; high-level ones auto-delegate
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # #[cfg(feature = "sparse")]
/// # {
/// # use numr::sparse::{SparseOps, SparseTensor};
/// # let device = CpuDevice::new();
/// # let client = CpuRuntime::default_client(&device);
/// // High-level API
/// # let sparse_a = SparseTensor::<CpuRuntime>::from_coo_slices(&[0, 1], &[1, 0], &[1.0f32, 2.0], [2, 2], &device)?;
/// # let dense_x = Tensor::try_from_slice(&[1.0, 2.0], &[2], &device)?;
/// let y = client.spmv(&sparse_a, &dense_x)?;  // y = A * x
///
/// // Low-level API (format-specific)
/// # let csr = sparse_a.to_csr()?;
/// # let csr_data = csr.as_csr().unwrap();
/// # let shape = [2, 2];
/// # let row_ptrs = csr_data.row_ptrs();
/// # let col_indices = csr_data.col_indices();
/// # let values = csr_data.values();
/// let y = client.spmv_csr::<f32>(&row_ptrs, &col_indices, &values, &dense_x, shape)?;
/// # }
/// # Ok::<(), numr::error::Error>(())
/// ```
pub trait SparseOps<R: Runtime<DType = DType>>: Sized {
    // =========================================================================
    // Low-Level Format-Specific Operations (Backend Implementation Required)
    // =========================================================================
    //
    // These methods work directly with raw CSR/COO/CSC data and must be
    // implemented by each backend. They operate on device-native tensors
    // without data transfer (GPU tensors stay on GPU).

    /// CSR sparse matrix-vector multiplication: y = A * x
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers `` `[nrows + 1]` `` (dtype I64)
    /// * `col_indices` - CSR column indices `` `[nnz]` `` (dtype I64)
    /// * `values` - CSR values `` `[nnz]` `` (dtype T)
    /// * `x` - Dense vector `` `[ncols]` ``
    /// * `shape` - Matrix shape `` `[nrows, ncols]` ``
    ///
    /// # Returns
    ///
    /// Dense vector `` `[nrows]` ``
    fn spmv_csr<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        x: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Tensor<R>>;

    /// CSR sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers `` `[nrows + 1]` ``
    /// * `col_indices` - CSR column indices `` `[nnz]` ``
    /// * `values` - CSR values `` `[nnz]` ``
    /// * `b` - Dense matrix `` `[ncols, n]` ``
    /// * `shape` - Sparse matrix shape `` `[nrows, ncols]` ``
    ///
    /// # Returns
    ///
    /// Dense matrix `` `[nrows, n]` ``
    fn spmm_csr<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        b: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Tensor<R>>;

    /// CSR element-wise addition: C = A + B
    ///
    /// Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (row_ptrs, col_indices, values) for result CSR matrix
    fn add_csr<T: crate::dtype::Element>(
        &self,
        a_row_ptrs: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_ptrs: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSR element-wise subtraction: C = A - B
    fn sub_csr<T: crate::dtype::Element>(
        &self,
        a_row_ptrs: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_ptrs: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSR element-wise multiplication (Hadamard product): C = A .* B
    fn mul_csr<T: crate::dtype::Element>(
        &self,
        a_row_ptrs: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_ptrs: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSR element-wise division: C = A ./ B
    ///
    /// Computes element-wise division of two sparse matrices using intersection semantics.
    /// Only positions where BOTH matrices have non-zero values are included in the result.
    ///
    /// # Arguments
    ///
    /// * `a_row_ptrs`, `a_col_indices`, `a_values` - CSR format for numerator matrix A
    /// * `b_row_ptrs`, `b_col_indices`, `b_values` - CSR format for denominator matrix B
    /// * `shape` - Matrix dimensions [rows, cols]. Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (row_ptrs, col_indices, values) for result matrix C in CSR format
    ///
    /// # Semantics
    ///
    /// - **Intersection strategy**: Result contains only positions where both A and B have values
    /// - **Mathematical**: `` `C[i,j]` `` = `` `A[i,j]` `` / `` `B[i,j]` `` when both exist
    /// - **Division by zero**: Implicit zeros in B where A has values are skipped (no division by zero)
    /// - **Example**: `` `A=[1,0,3]` ``, `` `B=[2,4,0]` `` → `` `C=[0.5]` `` (only position 0 has both values)
    ///
    /// # Errors
    ///
    /// Returns error if matrix shapes don't match
    fn div_csr<T: crate::dtype::Element>(
        &self,
        a_row_ptrs: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_ptrs: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    // -------------------------------------------------------------------------
    // CSC Operations
    // -------------------------------------------------------------------------

    /// CSC element-wise addition: C = A + B
    ///
    /// Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (col_ptrs, row_indices, values) for result CSC matrix
    fn add_csc<T: crate::dtype::Element>(
        &self,
        a_col_ptrs: &Tensor<R>,
        a_row_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_col_ptrs: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSC element-wise subtraction: C = A - B
    fn sub_csc<T: crate::dtype::Element>(
        &self,
        a_col_ptrs: &Tensor<R>,
        a_row_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_col_ptrs: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSC element-wise multiplication (Hadamard product): C = A .* B
    fn mul_csc<T: crate::dtype::Element>(
        &self,
        a_col_ptrs: &Tensor<R>,
        a_row_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_col_ptrs: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSC element-wise division: C = A ./ B
    ///
    /// Computes element-wise division of two sparse matrices using intersection semantics.
    /// Only positions where BOTH matrices have non-zero values are included in the result.
    ///
    /// # Arguments
    ///
    /// * `a_col_ptrs`, `a_row_indices`, `a_values` - CSC format for numerator matrix A
    /// * `b_col_ptrs`, `b_row_indices`, `b_values` - CSC format for denominator matrix B
    /// * `shape` - Matrix dimensions [rows, cols]. Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (col_ptrs, row_indices, values) for result matrix C in CSC format
    ///
    /// # Semantics
    ///
    /// - **Intersection strategy**: Result contains only positions where both A and B have values
    /// - **Mathematical**: `` `C[i,j]` `` = `` `A[i,j]` `` / `` `B[i,j]` `` when both exist
    /// - **Division by zero**: Implicit zeros in B where A has values are skipped (no division by zero)
    /// - **Example**: `` `A=[1,0,3]` ``, `` `B=[2,4,0]` `` → `` `C=[0.5]` `` (only position 0 has both values)
    ///
    /// # Errors
    ///
    /// Returns error if matrix shapes don't match
    fn div_csc<T: crate::dtype::Element>(
        &self,
        a_col_ptrs: &Tensor<R>,
        a_row_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_col_ptrs: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    // -------------------------------------------------------------------------
    // COO Operations
    // -------------------------------------------------------------------------

    /// COO element-wise addition: C = A + B
    ///
    /// Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (row_indices, col_indices, values) for result COO matrix
    fn add_coo<T: crate::dtype::Element>(
        &self,
        a_row_indices: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// COO element-wise subtraction: C = A - B
    fn sub_coo<T: crate::dtype::Element>(
        &self,
        a_row_indices: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// COO element-wise multiplication (Hadamard product): C = A .* B
    fn mul_coo<T: crate::dtype::Element>(
        &self,
        a_row_indices: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// COO element-wise division: C = A ./ B
    ///
    /// Computes element-wise division of two sparse matrices using intersection semantics.
    /// Only positions where BOTH matrices have non-zero values are included in the result.
    ///
    /// # Arguments
    ///
    /// * `a_row_indices`, `a_col_indices`, `a_values` - COO format for numerator matrix A
    /// * `b_row_indices`, `b_col_indices`, `b_values` - COO format for denominator matrix B
    /// * `shape` - Matrix dimensions [rows, cols]. Both matrices must have the same shape.
    ///
    /// # Returns
    ///
    /// Tuple of (row_indices, col_indices, values) for result matrix C in COO format
    ///
    /// # Semantics
    ///
    /// - **Intersection strategy**: Result contains only positions where both A and B have values
    /// - **Mathematical**: `` `C[i,j]` `` = `` `A[i,j]` `` / `` `B[i,j]` `` when both exist
    /// - **Division by zero**: Implicit zeros in B where A has values are skipped (no division by zero)
    /// - **Example**: `` `A=[1,0,3]` ``, `` `B=[2,4,0]` `` → `` `C=[0.5]` `` (only position 0 has both values)
    ///
    /// # Errors
    ///
    /// Returns error if matrix shapes don't match
    fn div_coo<T: crate::dtype::Element>(
        &self,
        a_row_indices: &Tensor<R>,
        a_col_indices: &Tensor<R>,
        a_values: &Tensor<R>,
        b_row_indices: &Tensor<R>,
        b_col_indices: &Tensor<R>,
        b_values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    // =========================================================================
    // High-Level Format-Agnostic Operations
    // =========================================================================
    //
    // These methods work with SparseTensor and will dispatch to format-specific
    // low-level methods in their implementations.

    /// Sparse matrix-vector multiplication: y = A * x (format-agnostic)
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse matrix `` `[M, K]` ``
    /// * `x` - Dense vector `` `[K]` `` or `` `[K, 1]` ``
    ///
    /// # Returns
    ///
    /// Dense vector `` `[M]` `` or `` `[M, 1]` ``
    ///
    /// # Performance
    ///
    /// - CSR format is optimal for this operation
    /// - Automatically converts from other formats if needed
    fn spmv(&self, a: &SparseTensor<R>, x: &Tensor<R>) -> Result<Tensor<R>>;

    /// Sparse matrix-dense matrix multiplication: C = A * B
    ///
    /// # Arguments
    ///
    /// * `a` - Sparse matrix `` `[M, K]` ``
    /// * `b` - Dense matrix `` `[K, N]` ``
    ///
    /// # Returns
    ///
    /// Dense matrix `` `[M, N]` ``
    fn spmm(&self, a: &SparseTensor<R>, b: &Tensor<R>) -> Result<Tensor<R>>;

    /// Dense matrix-sparse matrix multiplication: C = A * B
    ///
    /// # Arguments
    ///
    /// * `a` - Dense matrix `` `[M, K]` ``
    /// * `b` - Sparse matrix `` `[K, N]` ``
    ///
    /// # Returns
    ///
    /// Dense matrix `` `[M, N]` ``
    ///
    /// # Performance
    ///
    /// - CSC format is optimal for this operation
    fn dsmm(&self, a: &Tensor<R>, b: &SparseTensor<R>) -> Result<Tensor<R>>;

    // =========================================================================
    // Sparse-Sparse Operations
    // =========================================================================

    /// Sparse matrix addition: C = A + B
    ///
    /// Both matrices must have the same shape.
    fn sparse_add(&self, a: &SparseTensor<R>, b: &SparseTensor<R>) -> Result<SparseTensor<R>>;

    /// Sparse matrix subtraction: C = A - B
    fn sparse_sub(&self, a: &SparseTensor<R>, b: &SparseTensor<R>) -> Result<SparseTensor<R>>;

    /// Sparse matrix-sparse matrix multiplication: C = A * B
    ///
    /// # Returns
    ///
    /// Sparse matrix (result may be denser than inputs)
    fn sparse_matmul(&self, a: &SparseTensor<R>, b: &SparseTensor<R>) -> Result<SparseTensor<R>>;

    /// Element-wise multiplication (Hadamard product): C = A .* B
    ///
    /// Result is sparse (only non-zero where both are non-zero).
    fn sparse_mul(&self, a: &SparseTensor<R>, b: &SparseTensor<R>) -> Result<SparseTensor<R>>;

    // =========================================================================
    // Sparse-Scalar Operations
    // =========================================================================

    /// Multiply sparse matrix by scalar: B = alpha * A
    fn sparse_scale(&self, a: &SparseTensor<R>, scalar: f64) -> Result<SparseTensor<R>>;

    /// Add scalar to non-zero elements: B = A + alpha (on non-zeros only)
    ///
    /// Note: This does NOT add to zero elements (would destroy sparsity).
    fn sparse_add_scalar(&self, a: &SparseTensor<R>, scalar: f64) -> Result<SparseTensor<R>>;

    // =========================================================================
    // Reductions
    // =========================================================================

    /// Sum of all non-zero elements
    fn sparse_sum(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    /// Sum along rows: `` `result[i]` `` = `` `sum(A[i, :])` ``
    ///
    /// # Returns
    ///
    /// Dense vector `` `[M]` ``
    fn sparse_sum_rows(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    /// Sum along columns: `` `result[j]` `` = `` `sum(A[:, j])` ``
    ///
    /// # Returns
    ///
    /// Dense vector `` `[N]` ``
    fn sparse_sum_cols(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    /// Number of non-zeros per row
    ///
    /// # Returns
    ///
    /// Dense vector `` `[M]` `` with counts
    fn sparse_nnz_per_row(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    /// Number of non-zeros per column
    ///
    /// # Returns
    ///
    /// Dense vector `` `[N]` `` with counts
    fn sparse_nnz_per_col(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    // =========================================================================
    // Conversion
    // =========================================================================

    /// Convert sparse tensor to dense
    fn sparse_to_dense(&self, a: &SparseTensor<R>) -> Result<Tensor<R>>;

    /// Convert dense tensor to sparse (COO format)
    ///
    /// # Arguments
    ///
    /// * `a` - Dense 2D tensor
    /// * `threshold` - Values with |value| < threshold become zero
    fn dense_to_sparse(&self, a: &Tensor<R>, threshold: f64) -> Result<SparseTensor<R>>;

    /// Convert dense tensor directly to CSR format
    ///
    /// This is a convenience method that chains `dense_to_sparse()` with
    /// format conversion. All operations stay on device (no GPU→CPU transfers).
    ///
    /// # Arguments
    ///
    /// * `a` - Dense 2D tensor
    /// * `threshold` - Values with |value| < threshold become zero
    ///
    /// # Returns
    ///
    /// CSR sparse matrix data
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # #[cfg(feature = "sparse")]
    /// # {
    /// # use numr::sparse::{SparseOps, SparseStorage};
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// # let dense_matrix = Tensor::try_from_slice(&[1.0, 0.0, 2.0f32], &[1, 3], &device)?;
    /// # let vector = Tensor::try_from_slice(&[1.0, 2.0, 3.0f32], &[3], &device)?;
    /// let csr = client.dense_to_csr(&dense_matrix, 1e-15)?;
    /// let result = client.spmv_csr::<f32>(
    ///     csr.row_ptrs(), csr.col_indices(), csr.values(),
    ///     &vector, SparseStorage::shape(&csr)
    /// )?;
    /// # }
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn dense_to_csr(&self, a: &Tensor<R>, threshold: f64) -> Result<CsrData<R>> {
        let sparse = self.dense_to_sparse(a, threshold)?;
        let csr = sparse.to_csr()?;
        match csr {
            SparseTensor::Csr(data) => Ok(data),
            _ => unreachable!("to_csr() always returns SparseTensor::Csr"),
        }
    }

    /// Convert dense tensor directly to CSC format
    ///
    /// This is a convenience method that chains `dense_to_sparse()` with
    /// format conversion. All operations stay on device (no GPU→CPU transfers).
    ///
    /// # Arguments
    ///
    /// * `a` - Dense 2D tensor
    /// * `threshold` - Values with |value| < threshold become zero
    ///
    /// # Returns
    ///
    /// CSC sparse matrix data
    fn dense_to_csc(&self, a: &Tensor<R>, threshold: f64) -> Result<CscData<R>> {
        let sparse = self.dense_to_sparse(a, threshold)?;
        let csc = sparse.to_csc()?;
        match csc {
            SparseTensor::Csc(data) => Ok(data),
            _ => unreachable!("to_csc() always returns SparseTensor::Csc"),
        }
    }

    // =========================================================================
    // Format Conversions (Low-Level)
    // =========================================================================
    //
    // These methods convert between sparse formats while staying on device.
    // They are implemented by backends to avoid GPU→CPU transfers.

    /// COO → CSR: Sort by row and build row pointers
    ///
    /// # Arguments
    ///
    /// * `row_indices` - COO row indices `` `[nnz]` ``
    /// * `col_indices` - COO column indices `` `[nnz]` ``
    /// * `values` - COO values `` `[nnz]` ``
    /// * `shape` - Matrix shape `` `[nrows, ncols]` ``
    ///
    /// # Returns
    ///
    /// Tuple of `` `(row_ptrs [nrows+1], col_indices [nnz], values [nnz])` ``
    ///
    /// # Performance
    ///
    /// - CPU: Parallel sort + prefix sum
    /// - CUDA: Radix sort + CUB prefix sum
    fn coo_to_csr<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// COO → CSC: Sort by column and build column pointers
    ///
    /// # Arguments
    ///
    /// * `row_indices` - COO row indices `` `[nnz]` ``
    /// * `col_indices` - COO column indices `` `[nnz]` ``
    /// * `values` - COO values `` `[nnz]` ``
    /// * `shape` - Matrix shape `` `[nrows, ncols]` ``
    ///
    /// # Returns
    ///
    /// Tuple of `` `(col_ptrs [ncols+1], row_indices [nnz], values [nnz])` ``
    fn coo_to_csc<T: crate::dtype::Element>(
        &self,
        row_indices: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSR → COO: Expand row pointers to explicit row indices
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers `` `[nrows+1]` ``
    /// * `col_indices` - CSR column indices `` `[nnz]` ``
    /// * `values` - CSR values `` `[nnz]` ``
    /// * `shape` - Matrix shape `[nrows, ncols]`
    ///
    /// # Returns
    ///
    /// Tuple of `` `(row_indices [nnz], col_indices [nnz], values [nnz])` ``
    fn csr_to_coo<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSC → COO: Expand column pointers to explicit column indices
    ///
    /// # Arguments
    ///
    /// * `col_ptrs` - CSC column pointers `` `[ncols+1]` ``
    /// * `row_indices` - CSC row indices `` `[nnz]` ``
    /// * `values` - CSC values `` `[nnz]` ``
    /// * `shape` - Matrix shape `[nrows, ncols]`
    ///
    /// # Returns
    ///
    /// Tuple of `` `(row_indices [nnz], col_indices [nnz], values [nnz])` ``
    fn csc_to_coo<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<R>,
        row_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSR → CSC: Direct transpose without COO intermediary
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers `` `[nrows+1]` ``
    /// * `col_indices` - CSR column indices `` `[nnz]` ``
    /// * `values` - CSR values `` `[nnz]` ``
    /// * `shape` - Matrix shape `[nrows, ncols]`
    ///
    /// # Returns
    ///
    /// Tuple of `` `(col_ptrs [ncols+1], row_indices [nnz], values [nnz])` ``
    ///
    /// # Performance
    ///
    /// Much faster than CSR→COO→CSC as it avoids sorting.
    fn csr_to_csc<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    /// CSC → CSR: Direct transpose without COO intermediary
    ///
    /// # Arguments
    ///
    /// * `col_ptrs` - CSC column pointers `` `[ncols+1]` ``
    /// * `row_indices` - CSC row indices `` `[nnz]` ``
    /// * `values` - CSC values `` `[nnz]` ``
    /// * `shape` - Matrix shape `[nrows, ncols]`
    ///
    /// # Returns
    ///
    /// Tuple of `` `(row_ptrs [nrows+1], col_indices [nnz], values [nnz])` ``
    fn csc_to_csr<T: crate::dtype::Element>(
        &self,
        col_ptrs: &Tensor<R>,
        row_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<(Tensor<R>, Tensor<R>, Tensor<R>)>;

    // =========================================================================
    // Transpose
    // =========================================================================

    /// Transpose sparse matrix
    ///
    /// For CSR, this produces CSC (and vice versa) efficiently.
    fn sparse_transpose(&self, a: &SparseTensor<R>) -> Result<SparseTensor<R>>;

    // =========================================================================
    // Diagonal Extraction
    // =========================================================================

    /// Extract diagonal elements from a CSR matrix as a dense vector.
    ///
    /// Returns a tensor of length `min(nrows, ncols)` containing the diagonal
    /// entries. Missing diagonal entries are returned as zeros.
    ///
    /// Thread-per-row: each thread scans its row for `col_index == row_index`.
    ///
    /// # Arguments
    ///
    /// * `row_ptrs` - CSR row pointers `[nrows + 1]` (dtype I64)
    /// * `col_indices` - CSR column indices `[nnz]` (dtype I64)
    /// * `values` - CSR values `[nnz]` (dtype T)
    /// * `shape` - Matrix shape `[nrows, ncols]`
    fn extract_diagonal_csr<T: crate::dtype::Element>(
        &self,
        row_ptrs: &Tensor<R>,
        col_indices: &Tensor<R>,
        values: &Tensor<R>,
        shape: [usize; 2],
    ) -> Result<Tensor<R>>;

    /// Extract diagonal from a sparse matrix (format-agnostic).
    ///
    /// Converts to CSR if needed, then delegates to `extract_diagonal_csr`.
    fn sparse_extract_diagonal(&self, a: &SparseTensor<R>) -> Result<Tensor<R>> {
        let csr = match a {
            SparseTensor::Csr(data) => data.clone(),
            SparseTensor::Coo(data) => data.to_csr()?,
            SparseTensor::Csc(data) => data.to_csr()?,
        };

        let shape = csr.shape;
        let dtype = csr.values.dtype();

        crate::dispatch_dtype!(dtype, T => {
            self.extract_diagonal_csr::<T>(
                &csr.row_ptrs,
                &csr.col_indices,
                &csr.values,
                shape,
            )
        }, "sparse_extract_diagonal")
    }
}

// ============================================================================
// Sparse Scaling Operations
// ============================================================================

/// Norm type for computing row/column norms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormType {
    /// L1 norm: sum of absolute values
    L1,
    /// L2 norm: square root of sum of squares
    L2,
    /// L-infinity norm: maximum absolute value
    Linf,
}

/// Sparse matrix scaling and equilibration operations
///
/// These utilities improve numerical stability for sparse linear systems,
/// particularly important for ODE/DAE Jacobians which may be badly scaled.
///
/// # Example
///
/// ```
/// # use numr::prelude::*;
/// # #[cfg(feature = "sparse")]
/// # {
/// # use numr::sparse::{CscData, SparseScaling, NormType};
/// # let device = CpuDevice::new();
/// // Create a sample CSC matrix
/// # let col_ptrs = vec![0i64, 1, 2];  // CSC column pointers
/// # let row_indices = vec![0i64, 1];  // CSC row indices
/// # let values = vec![1.0f64, 2.0];
/// # let csc = CscData::<CpuRuntime>::from_slices(&col_ptrs, &row_indices, &values, [2, 2], &device)?;
/// // Equilibrate a badly-scaled matrix
/// let (scaled, row_scales, col_scales) = csc.equilibrate::<f64>()?;
///
/// // Scaling factors can be applied to solve Ax = b
/// // becomes (R A C)(C⁻¹x) = Rb
/// # }
/// # Ok::<(), numr::error::Error>(())
/// ```
pub trait SparseScaling<R: Runtime> {
    /// Compute row-wise norms of a sparse matrix.
    ///
    /// # Arguments
    ///
    /// * `norm` - Type of norm to compute (L1, L2, or Linf)
    ///
    /// # Returns
    ///
    /// Tensor of shape `` `[nrows]` `` containing the norm of each row.
    fn row_norms<T: crate::dtype::Element + Default + Copy>(
        &self,
        norm: NormType,
    ) -> Result<Tensor<R>>;

    /// Compute column-wise norms of a sparse matrix.
    ///
    /// # Arguments
    ///
    /// * `norm` - Type of norm to compute (L1, L2, or Linf)
    ///
    /// # Returns
    ///
    /// Tensor of shape `` `[ncols]` `` containing the norm of each column.
    fn col_norms<T: crate::dtype::Element + Default + Copy>(
        &self,
        norm: NormType,
    ) -> Result<Tensor<R>>;

    /// Scale rows of a sparse matrix by given factors.
    ///
    /// Computes `` `B[i,j] = scales[i] * A[i,j]` `` for all nonzeros.
    ///
    /// # Arguments
    ///
    /// * `scales` - Row scaling factors `` `[nrows]` ``
    ///
    /// # Returns
    ///
    /// New sparse matrix with scaled rows (same sparsity pattern).
    fn scale_rows<T: crate::dtype::Element + Default + Copy + std::ops::Mul<Output = T>>(
        &self,
        scales: &[T],
    ) -> Result<Self>
    where
        Self: Sized;

    /// Scale columns of a sparse matrix by given factors.
    ///
    /// Computes `` `B[i,j] = A[i,j] * scales[j]` `` for all nonzeros.
    ///
    /// # Arguments
    ///
    /// * `scales` - Column scaling factors `` `[ncols]` ``
    ///
    /// # Returns
    ///
    /// New sparse matrix with scaled columns (same sparsity pattern).
    fn scale_cols<T: crate::dtype::Element + Default + Copy + std::ops::Mul<Output = T>>(
        &self,
        scales: &[T],
    ) -> Result<Self>
    where
        Self: Sized;

    /// Equilibrate the matrix for better numerical stability.
    ///
    /// Computes row and column scalings to make all row and column infinity-norms
    /// approximately 1. Uses iterative scaling: row-scale, then column-scale,
    /// then row-scale again (one iteration is usually sufficient).
    ///
    /// # Returns
    ///
    /// Tuple of:
    /// - Scaled matrix (same sparsity pattern)
    /// - Row scales `` `[nrows]` ``
    /// - Column scales `` `[ncols]` ``
    ///
    /// To undo the scaling on the solution:
    /// - If solving Ax = b, the scaled system is (R*A*C) * (C⁻¹*x) = R*b
    /// - After solving, recover x = C * scaled_x
    fn equilibrate<T: crate::dtype::Element + Default + Copy + num_traits::Float>(
        &self,
    ) -> Result<(Self, Vec<T>, Vec<T>)>
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests require a backend implementation
    // These are placeholder tests for the trait definition

    #[test]
    fn test_sparse_ops_trait_exists() {
        // Trait compiles correctly
        fn _accepts_sparse_ops<R: Runtime<DType = DType>, T: SparseOps<R>>(_: &T) {}
    }

    #[test]
    fn test_norm_type() {
        // NormType enum exists and can be used
        let _l1 = NormType::L1;
        let _l2 = NormType::L2;
        let _linf = NormType::Linf;
        assert_eq!(NormType::L1, NormType::L1);
    }
}
