//! Dense × Sparse Matrix Multiplication (DSMM) - CUDA implementation
//!
//! Implements column-parallel DSMM algorithm for CUDA backend.
//!
//! # Algorithm
//!
//! Column-parallel CSC iteration (matches CPU implementation):
//! ```text
//! For each column j in B (0..N):
//!   For each non-zero B[k,j] in column j:
//!     C[:,j] += A[:,k] * B[k,j]
//! ```
//!
//! CUDA parallelization:
//! - One block per column (blockIdx.x = column index)
//! - Threads process rows within each column
//! - Shared memory caching for sparse column data (up to 1024 nnz)
//!
//! This algorithm matches the CPU implementation for backend parity.

use crate::algorithm::sparse::validate_dsmm_shapes;
use crate::error::Result;
use crate::runtime::cuda::kernels::launch_dsmm_csc;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Public function to be called from the combined trait implementation
pub(super) fn column_parallel_dsmm(
    client: &CudaClient,
    dense_a: &Tensor<CudaRuntime>,
    sparse_b_csc: &CscData<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = dense_a.dtype();
    let device = dense_a.device();

    // Validate dimensions
    let ([m, n], k) = validate_dsmm_shapes(dense_a.shape(), sparse_b_csc.shape)?;

    // Ensure A is contiguous for kernel
    let a_contig = dense_a.contiguous()?;

    // Allocate output C [M, N]
    let output = Tensor::<CudaRuntime>::zeros(&[m, n], dtype, device)?;

    // Get raw pointers
    let a_ptr = a_contig.ptr();
    let col_ptrs_ptr = sparse_b_csc.col_ptrs.ptr();
    let row_indices_ptr = sparse_b_csc.row_indices.ptr();
    let values_ptr = sparse_b_csc.values.ptr();
    let output_ptr = output.ptr();

    // Launch CUDA kernel
    unsafe {
        crate::dispatch_dtype!(dtype, T => {
            launch_dsmm_csc::<T>(
                &client.context,
                &client.stream,
                client.device.index,
                a_ptr,
                col_ptrs_ptr,
                row_indices_ptr,
                values_ptr,
                output_ptr,
                m,
                k,
                n,
            )
        }, "dsmm")?;
    }

    Ok(output)
}
