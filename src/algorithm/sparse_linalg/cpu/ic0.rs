//! CPU implementation of IC(0) factorization
//!
//! Incomplete Cholesky factorization with zero fill-in for symmetric positive definite matrices.

use std::collections::HashMap;

use super::super::traits::validate_square_sparse;
use super::super::types::{IcDecomposition, IcOptions};
use super::validate_cpu_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// IC(0) factorization on CPU: A ≈ L·Lᵀ with zero fill-in
///
/// # Algorithm (row-by-row Cholesky)
///
/// For each row i:
///   For k = 0 to i-1 where `a[i,k]` exists:
///     For j = 0 to k-1 where both `a[i,j]` and `L[k,j]` exist:
///       `a[i,k]` = `a[i,k]` - `a[i,j]` * `L[k,j]`
///     `a[i,k]` = `a[i,k]` / `L[k,k]`
///
///   sum = `a[i,i]`
///   For j = 0 to i-1 where `a[i,j]` exists:
///     sum = sum - `a[i,j]`²
///   `L[i,i]` = sqrt(sum)
///
/// # Arguments
///
/// * `a` - Symmetric positive definite sparse matrix in CSR format
/// * `options` - Factorization options
///
/// # Returns
///
/// IC decomposition with lower triangular factor L
pub fn ic0_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    options: IcOptions,
) -> Result<IcDecomposition<R>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cpu_dtype(dtype)?;

    // Extract CSR components (efficient for CPU)
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Work with f64 for numerical stability
    let values: Vec<f64> = match dtype {
        DType::F32 => a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => a.values().to_vec(),
        _ => return Err(Error::UnsupportedDType { dtype, op: "ic0" }),
    };

    // We only work with lower triangle
    let mut l_values = values.clone();

    // Build a map from (row, col) to value index for fast lookup
    let mut col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j <= i {
                // Only store lower triangle
                col_to_idx[i].insert(j, idx);
            }
        }
    }

    // Row-by-row IC(0)
    for i in 0..n {
        let i_start = row_ptrs[i] as usize;
        let i_end = row_ptrs[i + 1] as usize;

        // Process off-diagonal entries in row i (columns k < i)
        for idx_ik in i_start..i_end {
            let k = col_indices[idx_ik] as usize;
            if k >= i {
                break; // Only lower triangle
            }

            // Get L[k,:] for updates
            let k_start = row_ptrs[k] as usize;
            let k_end = row_ptrs[k + 1] as usize;

            // Compute inner product contribution
            let mut sum = l_values[idx_ik];
            for idx_kj in k_start..k_end {
                let j = col_indices[idx_kj] as usize;
                if j >= k {
                    break; // Only j < k
                }
                // Check if L[i,j] exists
                if let Some(&idx_ij) = col_to_idx[i].get(&j) {
                    sum -= l_values[idx_ij] * l_values[idx_kj];
                }
            }

            // Divide by L[k,k]
            let diag_idx = match col_to_idx[k].get(&k) {
                Some(&idx) => idx,
                None => {
                    return Err(Error::Internal(format!(
                        "Zero diagonal at row {} in IC(0)",
                        k
                    )));
                }
            };

            l_values[idx_ik] = sum / l_values[diag_idx];
        }

        // Compute diagonal L[i,i]
        let diag_idx = match col_to_idx[i].get(&i) {
            Some(&idx) => idx,
            None => {
                return Err(Error::Internal(format!(
                    "Missing diagonal at row {} in IC(0)",
                    i
                )));
            }
        };

        let mut sum = l_values[diag_idx] + options.diagonal_shift;
        for idx_ij in i_start..i_end {
            let j = col_indices[idx_ij] as usize;
            if j >= i {
                break;
            }
            sum -= l_values[idx_ij] * l_values[idx_ij];
        }

        if sum <= 0.0 {
            if options.diagonal_shift > 0.0 {
                sum = options.diagonal_shift;
            } else {
                return Err(Error::Internal(format!(
                    "Matrix not positive definite at row {} (sum = {})",
                    i, sum
                )));
            }
        }

        l_values[diag_idx] = sum.sqrt();
    }

    // Apply drop tolerance if specified
    if options.drop_tolerance > 0.0 {
        for val in &mut l_values {
            if val.abs() < options.drop_tolerance {
                *val = 0.0;
            }
        }
    }

    // Filter to lower triangle only and create output
    let l = extract_lower_triangle::<R>(
        n,
        &row_ptrs,
        &col_indices,
        &l_values,
        dtype,
        a.values().device(),
    )?;

    Ok(IcDecomposition { l })
}

/// Extract lower triangular matrix from full CSR
fn extract_lower_triangle<R: Runtime<DType = DType>>(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    l_values: &[f64],
    dtype: DType,
    device: &R::Device,
) -> Result<CsrData<R>> {
    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_col_indices = Vec::new();
    let mut new_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        let mut count = 0i64;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j > i {
                continue; // Skip upper triangle
            }
            let val = l_values[idx];
            if val.abs() >= 1e-15 {
                new_col_indices.push(j as i64);
                new_values.push(val);
                count += 1;
            }
        }

        new_row_ptrs[i + 1] = new_row_ptrs[i] + count;
    }

    // Create output tensors
    let l_row_ptrs_tensor = Tensor::<R>::from_slice(&new_row_ptrs, &[n + 1], device)?;
    let l_col_indices_tensor =
        Tensor::<R>::from_slice(&new_col_indices, &[new_col_indices.len()], device)?;

    let l_values_tensor = match dtype {
        DType::F32 => {
            let f32_vals: Vec<f32> = new_values.iter().map(|&x| x as f32).collect();
            Tensor::<R>::from_slice(&f32_vals, &[f32_vals.len()], device)?
        }
        DType::F64 => Tensor::<R>::from_slice(&new_values, &[new_values.len()], device)?,
        _ => unreachable!(),
    };

    CsrData::new(
        l_row_ptrs_tensor,
        l_col_indices_tensor,
        l_values_tensor,
        [n, n],
    )
}
