//! Sparse tensor support for numr
//!
//! This module provides sparse tensor formats and operations for efficient
//! storage and computation on tensors with mostly zero elements.
//!
//! # Sparse Formats
//!
//! Three standard sparse formats are supported:
//!
//! - **COO** (Coordinate): Stores (row, col, value) triplets. Best for construction
//!   and format conversion. O(nnz) storage.
//!
//! - **CSR** (Compressed Sparse Row): Row-major compressed format. Best for
//!   row slicing and matrix-vector multiplication (SpMV). O(nnz + nrows) storage.
//!
//! - **CSC** (Compressed Sparse Column): Column-major compressed format. Best for
//!   column slicing and some linear algebra operations. O(nnz + ncols) storage.
//!
//! # Usage
//!
//! ```
//! # use numr::prelude::*;
//! # #[cfg(feature = "sparse")]
//! # {
//! # use numr::sparse::{SparseTensor, SparseFormat};
//! # let device = CpuDevice::new();
//! // Create COO tensor from triplets
//! let rows = vec![0, 1, 2];
//! let cols = vec![1, 0, 2];
//! let values = vec![1.0f32, 2.0, 3.0];
//! let shape = [3, 3];
//!
//! let coo = SparseTensor::<CpuRuntime>::from_coo_slices(
//!     &rows, &cols, &values, shape, &device
//! )?;
//!
//! // Convert to CSR for efficient SpMV
//! let csr = coo.to_csr()?;
//!
//! // Sparse matrix-vector multiplication
//! let x = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device)?;
//! // let sparse_ops = CpuRuntime::default_client(&device);
//! // let y = sparse_ops.spmv(&csr, &x)?;  // y = A * x
//! # }
//! # Ok::<(), numr::error::Error>(())
//! ```
//!
//! # When to Use Sparse
//!
//! Sparse formats are beneficial when:
//! - Sparsity > 90% (less than 10% non-zeros)
//! - Memory is constrained
//! - Operations preserve sparsity (SpMV, element-wise on non-zeros)
//!
//! Dense is usually faster when:
//! - Sparsity < 50%
//! - Operations produce dense results
//! - Using GPU with high memory bandwidth

mod coo;
mod csc;
mod csr;
mod format;
mod ops;
pub mod structured;
mod tensor;

pub use crate::ops::traits::Sparse24Ops;
pub use coo::CooData;
pub use csc::CscData;
pub use csr::CsrData;
pub use format::{SparseFormat, SparseStorage};
pub use ops::{NormType, SparseOps, SparseScaling};
pub use structured::Sparse24Tensor;
pub use tensor::SparseTensor;
