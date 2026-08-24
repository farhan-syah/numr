//! Dense × Sparse Matrix Multiplication (DSMM) - CPU implementation
//!
//! Implements column-parallel DSMM algorithm for CPU backend.
//!
//! # Algorithm
//!
//! Column-parallel CSC iteration:
//! ```text
//! For each column j in B (0..N):
//!   For each non-zero B[k,j] in column j:
//!     C[:,j] += A[:,k] * B[k,j]
//! ```
//!
//! This algorithm matches the CUDA implementation for backend parity.

use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::sparse::CscData;
use crate::tensor::Tensor;

/// Public function to be called from the combined trait implementation
pub(super) fn column_parallel_dsmm(
    _client: &CpuClient,
    dense_a: &Tensor<CpuRuntime>,
    sparse_b_csc: &CscData<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    use crate::algorithm::sparse::validate_dsmm_shapes;

    let [k, n] = sparse_b_csc.shape;
    let dtype = sparse_b_csc.values.dtype();
    let device = dense_a.device();

    // Validate dimensions
    let ([m, _output_n], _k) = validate_dsmm_shapes(dense_a.shape(), sparse_b_csc.shape)?;

    crate::dispatch_dtype!(dtype, T => {
        dsmm_typed::<T>(dense_a, sparse_b_csc, m, k, n, device)
    }, "dsmm")
}

/// Typed DSMM implementation
fn dsmm_typed<T: Element>(
    a: &Tensor<CpuRuntime>,
    csc: &CscData<CpuRuntime>,
    m: usize,
    k: usize,
    n: usize,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<Tensor<CpuRuntime>> {
    let a_data: Vec<T> = a.to_vec();
    let col_ptrs_data: Vec<i64> = csc.col_ptrs.to_vec();
    let row_indices_data: Vec<i64> = csc.row_indices.to_vec();
    let b_values: Vec<T> = csc.values.to_vec();

    // Result C [M, N]
    let mut c_data: Vec<T> = vec![T::zero(); m * n];

    // Column-parallel algorithm (matches CUDA)
    // For each column j in B
    for col in 0..n {
        let start = col_ptrs_data[col] as usize;
        let end = col_ptrs_data[col + 1] as usize;

        // For each non-zero in column j
        for idx in start..end {
            let row_b = row_indices_data[idx] as usize; // row in B
            let b_val = b_values[idx].to_f64();

            // Update column j of C: C[:, j] += A[:, row_b] * b_val
            for row_a in 0..m {
                let a_val = a_data[row_a * k + row_b].to_f64();
                let c_idx = row_a * n + col;
                let current = c_data[c_idx].to_f64();
                c_data[c_idx] = T::from_f64(current + a_val * b_val);
            }
        }
    }

    Tensor::try_from_slice(&c_data, &[m, n], device)
}
