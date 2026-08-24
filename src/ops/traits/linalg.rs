//! Linear algebra operations trait
//!
//! This module defines the trait for linear algebra operations:
//! - Matrix decompositions (QR, SVD, LU, Cholesky)
//! - Solving linear systems (solve, lstsq, pinverse)
//! - Matrix properties (determinant, inverse, trace, rank)
//! - Norms and diagonal operations

use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

/// Linear algebra operations trait
///
/// Provides matrix decompositions, solving linear systems, and
/// computing matrix properties.
///
/// # Implementation Notes
///
/// All linear algebra operations must support floating-point types (F32, F64).
/// Some operations may have limited support on certain backends (e.g., WebGPU with F32 only).
pub trait LinalgOps<R: Runtime> {
    /// Solve linear system Ax = b using LU decomposition
    ///
    /// Computes the solution x to the linear equation Ax = b, where A is a square
    /// coefficient matrix.
    ///
    /// # Algorithm
    ///
    /// Uses LU decomposition with partial pivoting:
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. Solve Ly = Pb (forward substitution)
    /// 3. Solve Ux = y (backward substitution)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix `` `[n, n]` ``
    /// * `b` - Right-hand side vector/matrix `` `[n]` `` or `` `[n, k]` ``
    ///
    /// # Returns
    ///
    /// Solution tensor x `` `[n]` `` or `` `[n, k]` ``
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if dimensions are incompatible or A is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if matrix is singular (not invertible)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Solve 2x + 3y = 5
    /// //       4x + 5y = 11
    /// let a = Tensor::<CpuRuntime>::from_slice(&[2.0, 3.0, 4.0, 5.0], &[2, 2], &device)?;
    /// let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 11.0], &[2], &device)?;
    /// let x = client.solve(&a, &b)?;
    /// // x = [2.0, 1.0]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn solve(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LinalgOps::solve",
        })
    }

    /// Least squares solution: minimize ||Ax - b||²
    ///
    /// Computes the solution x that minimizes the 2-norm of the residual ||Ax - b||².
    /// Uses QR decomposition (Householder reflections) followed by back-substitution.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute QR decomposition: A = QR
    /// 2. Transform: y = Q^T @ b
    /// 3. Solve: R @ x = y (back-substitution)
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Coefficient matrix `` `[m, n]` `` (can be non-square)
    /// * `b` - Right-hand side vector/matrix `` `[m]` `` or `` `[m, k]` ``
    ///
    /// # Returns
    ///
    /// Solution tensor x `` `[n]` `` or `` `[n, k]` `` that minimizes ||Ax - b||²
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if dimensions are incompatible
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // Fit line y = mx + c to overdetermined system
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[3, 2], &device)?;
    /// let b = Tensor::<CpuRuntime>::from_slice(&[2.0, 4.0, 6.0], &[3], &device)?;
    /// let x = client.lstsq(&a, &b)?; // [m, c]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn lstsq(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LinalgOps::lstsq",
        })
    }

    /// Moore-Penrose pseudo-inverse via SVD: A^+ = V @ diag(1/S) @ U^T
    ///
    /// Computes the pseudo-inverse of a matrix using SVD. For a matrix A with
    /// SVD decomposition A = U @ diag(S) @ V^T, the pseudo-inverse is:
    ///
    /// ```text
    /// A^+ = V @ diag(1/S_i where S_i > rcond*max(S), else 0) @ U^T
    /// ```
    ///
    /// # Algorithm
    ///
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. Invert non-zero singular values: `` `S_inv[i]` `` = `` `1/S[i]` `` if `` `S[i]` `` > `` `rcond*max(S)` ``, else 0
    /// 3. Compute: A^+ = V @ diag(S_inv) @ U^T
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix `[m, n]`
    /// * `rcond` - Relative condition number threshold (singular values below rcond*max(S) are treated as zero)
    ///   If None, uses default: max(m,n) * machine_epsilon
    ///
    /// # Returns
    ///
    /// Pseudo-inverse matrix `[n, m]`
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    /// let a_pinv = client.pinverse(&a, None)?; // Shape: [3, 2]
    /// // Verify: a @ a_pinv @ a ≈ a
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn pinverse(&self, a: &Tensor<R>, rcond: Option<f64>) -> Result<Tensor<R>> {
        let _ = (a, rcond);
        Err(Error::NotImplemented {
            feature: "LinalgOps::pinverse",
        })
    }

    /// Matrix norm
    ///
    /// Computes the matrix norm of the input tensor.
    ///
    /// # Supported Norms
    ///
    /// - **Frobenius**: `` `sqrt(sum(A[i,j]²))` `` - Euclidean norm of the matrix
    /// - **Spectral** (2-norm): Maximum singular value (requires SVD)
    /// - **Nuclear** (trace norm): Sum of singular values (requires SVD)
    ///
    /// # Algorithm
    ///
    /// **Frobenius norm:**
    /// ```text
    /// ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2) = sqrt(trace(A^T @ A))
    /// ```
    ///
    /// **Spectral norm:**
    /// ```text
    /// ||A||_2 = max singular value of A
    /// ```
    ///
    /// **Nuclear norm:**
    /// ```text
    /// ||A||_* = sum of singular values of A
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix tensor
    /// * `ord` - Norm order (Frobenius, Spectral, Nuclear)
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the norm value
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # use numr::algorithm::linalg::MatrixNormOrder;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let fro = client.matrix_norm(&a, MatrixNormOrder::Frobenius)?;
    /// let spec = client.matrix_norm(&a, MatrixNormOrder::Spectral)?;
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn matrix_norm(
        &self,
        a: &Tensor<R>,
        ord: crate::algorithm::linalg::MatrixNormOrder,
    ) -> Result<Tensor<R>> {
        let _ = (a, ord);
        Err(Error::NotImplemented {
            feature: "LinalgOps::matrix_norm",
        })
    }

    /// Matrix inverse using LU decomposition
    ///
    /// Computes the multiplicative inverse of a square matrix.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. Solve for A^{-1}: each column j of A^{-1} solves A @ x_j = e_j
    ///    where e_j is the j-th standard basis vector
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Inverse matrix [n, n] such that A @ A^{-1} = I
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    /// - `Internal` if matrix is singular (determinant = 0)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[4.0, 7.0, 2.0, 6.0], &[2, 2], &device)?;
    /// let a_inv = client.inverse(&a)?;
    /// // Verify: a @ a_inv ≈ I
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn inverse(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::inverse",
        })
    }

    /// Matrix determinant using LU decomposition
    ///
    /// Computes the determinant of a square matrix.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. det(A) = (-1)^{number of row swaps} * product(diag(U))
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Scalar tensor containing the determinant
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let det = client.det(&a)?;
    /// // det = 1*4 - 2*3 = -2
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn det(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::det",
        })
    }

    /// Matrix trace: sum of diagonal elements
    ///
    /// Computes the sum of the diagonal elements of a matrix.
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// Scalar tensor containing `` `trace(A)` `` = `` `sum_i A[i,i]` ``
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let tr = client.trace(&a)?;
    /// // tr = 1 + 4 = 5
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn trace(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::trace",
        })
    }

    /// Extract diagonal elements
    ///
    /// Returns the diagonal elements of a 2D matrix as a 1D tensor.
    ///
    /// # Arguments
    ///
    /// * `a` - 2D matrix [m, n]
    ///
    /// # Returns
    ///
    /// 1D tensor [min(m,n)] containing diagonal elements
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 2D
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)?;
    /// let d = client.diag(&a)?;
    /// // d = [1, 5]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn diag(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::diag",
        })
    }

    /// Create diagonal matrix from 1D tensor
    ///
    /// Creates a 2D square matrix with the input elements on the diagonal.
    ///
    /// # Arguments
    ///
    /// * `a` - 1D tensor `` `[n]` ``
    ///
    /// # Returns
    ///
    /// 2D diagonal matrix [n, n]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if input is not 1D
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0], &[3], &device)?;
    /// let d = client.diagflat(&a)?;
    /// // d = [[1, 0, 0],
    /// //      [0, 2, 0],
    /// //      [0, 0, 3]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn diagflat(&self, a: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::diagflat",
        })
    }

    /// Matrix rank via SVD
    ///
    /// Computes the numerical rank of a matrix (number of non-zero singular values).
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute SVD: A = U @ diag(S) @ V^T
    /// 2. Count singular values: rank = #{i : `` `S[i]` `` > tol}
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Input matrix [m, n]
    /// * `tol` - Singular value threshold (values below this are treated as zero)
    ///   If None, uses default: max(m,n) * eps * max(S)
    ///
    /// # Returns
    ///
    /// Scalar integer tensor containing the rank
    ///
    /// # Errors
    ///
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 2.0, 4.0], &[2, 2], &device)?;
    /// let rank = client.matrix_rank(&a, None)?;
    /// // rank = 1 (rank-deficient: rows are linearly dependent)
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn matrix_rank(&self, a: &Tensor<R>, tol: Option<f64>) -> Result<Tensor<R>> {
        let _ = (a, tol);
        Err(Error::NotImplemented {
            feature: "LinalgOps::matrix_rank",
        })
    }

    /// Kronecker product: A ⊗ B
    ///
    /// Computes the Kronecker product (tensor product) of two matrices.
    ///
    /// # Definition
    ///
    /// For matrices A of shape [m, n] and B of shape [p, q], the Kronecker product
    /// A ⊗ B has shape [m*p, n*q] and is defined as:
    ///
    /// ```text
    /// (A ⊗ B)[i*p + k, j*q + l] = A[i, j] * B[k, l]
    /// ```
    ///
    /// Equivalently, it replaces each element a_ij of A with the block a_ij * B.
    ///
    /// # Properties
    ///
    /// - (A ⊗ B) ⊗ C = A ⊗ (B ⊗ C) (associative)
    /// - A ⊗ (B + C) = A ⊗ B + A ⊗ C (distributive)
    /// - (A ⊗ B)^T = A^T ⊗ B^T
    /// - (A ⊗ B)(C ⊗ D) = (AC) ⊗ (BD) (mixed-product property)
    /// - det(A ⊗ B) = det(A)^q * det(B)^m for square matrices
    /// - tr(A ⊗ B) = tr(A) * tr(B)
    ///
    /// # Arguments
    ///
    /// * `a` - First matrix [m, n]
    /// * `b` - Second matrix [p, q]
    ///
    /// # Returns
    ///
    /// Kronecker product [m*p, n*q]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if inputs are not 2D
    /// - `DTypeMismatch` if dtypes don't match
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Use Cases
    ///
    /// - Quantum computing (tensor products of quantum states/operators)
    /// - Control theory (Sylvester/Lyapunov equation solvers)
    /// - Signal processing (2D filtering, separable convolutions)
    /// - Graph theory (graph products)
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let b = Tensor::<CpuRuntime>::from_slice(&[0.0, 5.0, 6.0, 7.0], &[2, 2], &device)?;
    /// let c = client.kron(&a, &b)?;
    /// // c has shape [4, 4]:
    /// // [[0, 5, 0, 10],
    /// //  [6, 7, 12, 14],
    /// //  [0, 15, 0, 20],
    /// //  [18, 21, 24, 28]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn kron(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LinalgOps::kron",
        })
    }

    /// Solve banded linear system using LAPACK-style band storage
    ///
    /// # Arguments
    ///
    /// * `ab` - Band matrix `[kl + ku + 1, n]`
    /// * `b` - Right-hand side `` `[n]` `` or `` `[n, nrhs]` ``
    /// * `kl` - Number of subdiagonals
    /// * `ku` - Number of superdiagonals
    fn solve_banded(
        &self,
        ab: &Tensor<R>,
        b: &Tensor<R>,
        kl: usize,
        ku: usize,
    ) -> Result<Tensor<R>> {
        let _ = (ab, b, kl, ku);
        Err(Error::NotImplemented {
            feature: "LinalgOps::solve_banded",
        })
    }

    /// Khatri-Rao product (column-wise Kronecker product)
    ///
    /// Computes the column-wise Kronecker product of two matrices.
    ///
    /// # Definition
    ///
    /// For matrices A of shape [m, k] and B of shape [n, k] (same number of columns),
    /// the Khatri-Rao product A ⊙ B has shape [m*n, k] where each column is the
    /// Kronecker product of the corresponding columns:
    ///
    /// ```text
    /// (A ⊙ B)[:, j] = A[:, j] ⊗ B[:, j]
    /// ```
    ///
    /// # Properties
    ///
    /// - Used in tensor decompositions (CP/PARAFAC, Tucker)
    /// - (A ⊙ B)^T (A ⊙ B) = (A^T A) * (B^T B) (element-wise product)
    /// - Related to mode-n products in tensor algebra
    ///
    /// # Arguments
    ///
    /// * `a` - First matrix [m, k]
    /// * `b` - Second matrix [n, k]
    ///
    /// # Returns
    ///
    /// Khatri-Rao product [m*n, k]
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if inputs are not 2D or have different number of columns
    /// - `DTypeMismatch` if dtypes don't match
    /// - `UnsupportedDType` if input is not F32 or F64 (WebGPU: F32 only)
    ///
    /// # Use Cases
    ///
    /// - CP/PARAFAC tensor decomposition (ALS updates)
    /// - Tucker decomposition factor updates
    /// - Multi-linear algebra operations
    /// - Compressed sensing
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// // A = [[1, 2], [3, 4]]  (2x2)
    /// // B = [[5, 6], [7, 8]]  (2x2)
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let b = Tensor::<CpuRuntime>::from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2], &device)?;
    /// let c = client.khatri_rao(&a, &b)?;
    /// // c has shape [4, 2]:
    /// // [[5, 12],   // 1*5, 2*6
    /// //  [7, 16],   // 1*7, 2*8
    /// //  [15, 24],  // 3*5, 4*6
    /// //  [21, 32]]  // 3*7, 4*8
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn khatri_rao(&self, a: &Tensor<R>, b: &Tensor<R>) -> Result<Tensor<R>> {
        let _ = (a, b);
        Err(Error::NotImplemented {
            feature: "LinalgOps::khatri_rao",
        })
    }

    /// Upper triangular part of a matrix
    ///
    /// Returns a copy of the matrix with elements below the k-th diagonal zeroed.
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix [m, n]
    /// * `diagonal` - Diagonal offset. 0 = main diagonal, positive = above, negative = below.
    ///
    /// # Returns
    ///
    /// Matrix of same shape and dtype with lower triangle zeroed.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], &device)?;
    /// let u = client.triu(&a, 0)?;
    /// // u = [[1, 2, 3],
    /// //      [0, 5, 6],
    /// //      [0, 0, 9]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn triu(&self, a: &Tensor<R>, diagonal: i64) -> Result<Tensor<R>> {
        let _ = (a, diagonal);
        Err(Error::NotImplemented {
            feature: "LinalgOps::triu",
        })
    }

    /// Lower triangular part of a matrix
    ///
    /// Returns a copy of the matrix with elements above the k-th diagonal zeroed.
    ///
    /// # Arguments
    ///
    /// * `a` - Input 2D matrix [m, n]
    /// * `diagonal` - Diagonal offset. 0 = main diagonal, positive = above, negative = below.
    ///
    /// # Returns
    ///
    /// Matrix of same shape and dtype with upper triangle zeroed.
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3], &device)?;
    /// let l = client.tril(&a, 0)?;
    /// // l = [[1, 0, 0],
    /// //      [4, 5, 0],
    /// //      [7, 8, 9]]
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn tril(&self, a: &Tensor<R>, diagonal: i64) -> Result<Tensor<R>> {
        let _ = (a, diagonal);
        Err(Error::NotImplemented {
            feature: "LinalgOps::tril",
        })
    }

    /// Sign and log-absolute-determinant
    ///
    /// Computes sign(det(A)) and log(|det(A)|) separately for numerical stability.
    /// This avoids overflow/underflow issues with large matrices where det(A) may
    /// be astronomically large or vanishingly small.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// 1. Compute PA = LU (pivoted LU decomposition)
    /// 2. sign = (-1)^num_swaps * product(sign(`` `U[i,i]` ``))
    /// 3. logabsdet = sum(log(|`` `U[i,i]` ``|))
    /// 4. If any `` `U[i,i]` `` == 0: sign = 0, logabsdet = -inf
    /// ```
    ///
    /// # Arguments
    ///
    /// * `a` - Square matrix [n, n]
    ///
    /// # Returns
    ///
    /// `SlogdetResult` with `sign` (scalar) and `logabsdet` (scalar)
    ///
    /// # Errors
    ///
    /// - `ShapeMismatch` if matrix is not square
    /// - `UnsupportedDType` if input is not F32 or F64
    ///
    /// # Example
    ///
    /// ```
    /// # use numr::prelude::*;
    /// # let device = CpuDevice::new();
    /// # let client = CpuRuntime::default_client(&device);
    /// let a = Tensor::<CpuRuntime>::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2], &device)?;
    /// let result = client.slogdet(&a)?;
    /// // sign = -1.0, logabsdet = log(2) ≈ 0.693
    /// # Ok::<(), numr::error::Error>(())
    /// ```
    fn slogdet(&self, a: &Tensor<R>) -> Result<crate::algorithm::linalg::SlogdetResult<R>> {
        let _ = a;
        Err(Error::NotImplemented {
            feature: "LinalgOps::slogdet",
        })
    }
}
