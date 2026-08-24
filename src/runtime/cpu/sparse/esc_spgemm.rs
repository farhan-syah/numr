//! ESC SpGEMM (Exact Symbolic Computation) - CPU implementation
//!
//! Implements the ESC algorithm for Sparse × Sparse matrix multiplication.
//!
//! # Algorithm
//!
//! ## Phase 1: Symbolic (Count NNZ per output row)
//! ```text
//! For each output row i:
//!   Use HashSet to find unique columns in C[i,:]
//!   Store count in row_nnz[i]
//! Build row_ptrs via exclusive scan
//! ```
//!
//! ## Phase 2: Numeric (Compute values with pre-sized hash accumulator)
//! ```text
//! For each output row i:
//!   Create HashMap<col_index, value> with capacity = row_nnz[i]
//!   For each non-zero A[i,k]:
//!     For each non-zero B[k,j]:
//!       Accumulate: map[j] += A[i,k] * B[k,j]
//!   Sort entries by column index
//!   Filter values below zero_tolerance
//!   Write to output arrays
//! ```
//!
//! This algorithm matches the CUDA implementation for backend parity.

use crate::algorithm::sparse::{validate_spgemm_shapes, zero_tolerance};
use crate::dtype::Element;
use crate::error::Result;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::sparse::CsrData;
use crate::tensor::Tensor;
use std::collections::{HashMap, HashSet};

/// Public function to be called from the combined trait implementation
pub(super) fn esc_spgemm_csr(
    _client: &CpuClient,
    a_csr: &CsrData<CpuRuntime>,
    b_csr: &CsrData<CpuRuntime>,
) -> Result<CsrData<CpuRuntime>> {
    // Validate shapes and dtypes
    let ([m, n], _k) = validate_spgemm_shapes(a_csr.shape, b_csr.shape)?;

    let dtype = a_csr.values.dtype();
    let device = a_csr.values.device();

    crate::dispatch_dtype!(dtype, T => {
        esc_spgemm_typed::<T>(a_csr, b_csr, m, n, device)
    }, "esc_spgemm")
}

/// Typed ESC SpGEMM implementation
fn esc_spgemm_typed<T: Element>(
    a_csr: &CsrData<CpuRuntime>,
    b_csr: &CsrData<CpuRuntime>,
    m: usize,
    n: usize,
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
) -> Result<CsrData<CpuRuntime>> {
    let a_row_ptrs: Vec<i64> = a_csr.row_ptrs.to_vec();
    let a_col_indices: Vec<i64> = a_csr.col_indices.to_vec();
    let a_values: Vec<T> = a_csr.values.to_vec();

    let b_row_ptrs: Vec<i64> = b_csr.row_ptrs.to_vec();
    let b_col_indices: Vec<i64> = b_csr.col_indices.to_vec();
    let b_values: Vec<T> = b_csr.values.to_vec();

    // ========================================================================
    // PHASE 1: Symbolic (Count NNZ per output row)
    // ========================================================================
    let mut row_nnz: Vec<i64> = vec![0; m];

    for i in 0..m {
        let a_start = a_row_ptrs[i] as usize;
        let a_end = a_row_ptrs[i + 1] as usize;

        // Use HashSet to track unique column indices in output row i
        let mut col_set: HashSet<usize> = HashSet::new();

        // For each non-zero A[i, k]
        #[allow(clippy::needless_range_loop)]
        for a_idx in a_start..a_end {
            let k = a_col_indices[a_idx] as usize;

            // Scan row k of B and collect column indices
            let b_start = b_row_ptrs[k] as usize;
            let b_end = b_row_ptrs[k + 1] as usize;

            #[allow(clippy::needless_range_loop)]
            for b_idx in b_start..b_end {
                let j = b_col_indices[b_idx] as usize;
                col_set.insert(j);
            }
        }

        row_nnz[i] = col_set.len() as i64;
    }

    // Build row_ptrs via exclusive scan
    let mut c_row_ptrs: Vec<i64> = Vec::with_capacity(m + 1);
    c_row_ptrs.push(0);
    for i in 0..m {
        c_row_ptrs.push(c_row_ptrs[i] + row_nnz[i]);
    }

    // ========================================================================
    // PHASE 2: Numeric (Compute values with pre-sized hash accumulator)
    // ========================================================================
    let mut c_col_indices: Vec<i64> = Vec::new();
    let mut c_values: Vec<T> = Vec::new();

    // Rebuild row_ptrs to account for zero_tolerance filtering
    let mut c_row_ptrs_final: Vec<i64> = Vec::with_capacity(m + 1);
    c_row_ptrs_final.push(0);

    for i in 0..m {
        let a_start = a_row_ptrs[i] as usize;
        let a_end = a_row_ptrs[i + 1] as usize;
        let capacity = row_nnz[i] as usize;

        // Pre-sized hash accumulator (no resizing!)
        let mut row_accum: HashMap<usize, f64> = HashMap::with_capacity(capacity);

        // For each non-zero A[i, k]
        for a_idx in a_start..a_end {
            let k = a_col_indices[a_idx] as usize;
            let a_val = a_values[a_idx].to_f64();

            // Multiply with row k of B
            let b_start = b_row_ptrs[k] as usize;
            let b_end = b_row_ptrs[k + 1] as usize;

            for b_idx in b_start..b_end {
                let j = b_col_indices[b_idx] as usize;
                let b_val = b_values[b_idx].to_f64();

                *row_accum.entry(j).or_insert(0.0) += a_val * b_val;
            }
        }

        // Sort by column index and write to output
        let mut row_entries: Vec<(usize, f64)> = row_accum.into_iter().collect();
        row_entries.sort_by_key(|&(col, _)| col);

        // Filter and write row
        for (col, val) in row_entries {
            if val.abs() > zero_tolerance::<T>() {
                c_col_indices.push(col as i64);
                c_values.push(T::from_f64(val));
            }
        }

        // Update row_ptr
        c_row_ptrs_final.push(c_col_indices.len() as i64);
    }

    let final_nnz = c_col_indices.len();

    // Create result CSR tensors
    let result_row_ptrs = Tensor::from_slice(&c_row_ptrs_final, &[m + 1], device)?;
    let result_col_indices = Tensor::from_slice(&c_col_indices, &[final_nnz], device)?;
    let result_values = Tensor::from_slice(&c_values, &[final_nnz], device)?;

    Ok(CsrData {
        row_ptrs: result_row_ptrs,
        col_indices: result_col_indices,
        values: result_values,
        shape: [m, n],
    })
}
