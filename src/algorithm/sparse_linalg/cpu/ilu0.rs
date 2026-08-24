//! CPU implementation of ILU(0) factorization
//!
//! Incomplete LU factorization with zero fill-in using IKJ algorithm variant.

use std::collections::HashMap;

use super::super::traits::validate_square_sparse;
use super::super::types::{IluDecomposition, IluOptions, SymbolicIlu0};
use super::validate_cpu_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(0) factorization on CPU: A ≈ L·U with zero fill-in
///
/// # Algorithm (IKJ variant)
///
/// For each row i:
///   For k = 0 to i-1 where `a[i,k]` exists:
///     `a[i,k]` = `a[i,k]` / `a[k,k]`  (becomes `L[i,k]`)
///     For j = k+1 to n-1 where both `a[i,j]` and `a[k,j]` exist:
///       `a[i,j]` = `a[i,j]` - `a[i,k]` * `a[k,j]`
///
/// # Arguments
///
/// * `a` - Square sparse matrix in CSR format
/// * `options` - Factorization options (drop tolerance, diagonal shift)
///
/// # Returns
///
/// ILU decomposition with L (unit lower triangular) and U (upper triangular)
pub fn ilu0_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    options: IluOptions,
) -> Result<IluDecomposition<R>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cpu_dtype(dtype)?;

    // Extract CSR components (efficient for CPU - just reading RAM)
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
        _ => return Err(Error::UnsupportedDType { dtype, op: "ilu0" }),
    };

    // Create mutable copy for in-place factorization
    let mut lu_values = values;

    // Build a map from (row, col) to value index for fast lookup
    let mut col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            col_to_idx[i].insert(j, idx);
        }
    }

    // IKJ factorization
    for i in 0..n {
        let row_start = row_ptrs[i] as usize;
        let row_end = row_ptrs[i + 1] as usize;

        // Process columns k < i (for L factor)
        for idx_ik in row_start..row_end {
            let k = col_indices[idx_ik] as usize;
            if k >= i {
                break; // Only process lower triangular part for L
            }

            // Get diagonal a[k,k]
            let diag_idx = match col_to_idx[k].get(&k) {
                Some(&idx) => idx,
                None => {
                    return Err(Error::Internal(format!(
                        "Zero diagonal at row {} in ILU(0)",
                        k
                    )));
                }
            };

            let diag_val = lu_values[diag_idx];
            if diag_val.abs() < 1e-15 {
                if options.diagonal_shift > 0.0 {
                    lu_values[diag_idx] = options.diagonal_shift;
                } else {
                    return Err(Error::Internal(format!(
                        "Zero pivot at row {} in ILU(0)",
                        k
                    )));
                }
            }

            // L[i,k] = A[i,k] / A[k,k]
            lu_values[idx_ik] /= lu_values[diag_idx];
            let l_ik = lu_values[idx_ik];

            // Update row i for columns j > k
            let k_start = row_ptrs[k] as usize;
            let k_end = row_ptrs[k + 1] as usize;

            for idx_kj in k_start..k_end {
                let j = col_indices[idx_kj] as usize;
                if j <= k {
                    continue; // Only update upper part
                }

                // Check if a[i,j] exists (zero fill-in constraint)
                if let Some(&idx_ij) = col_to_idx[i].get(&j) {
                    // a[i,j] = a[i,j] - L[i,k] * U[k,j]
                    lu_values[idx_ij] -= l_ik * lu_values[idx_kj];
                }
            }
        }
    }

    // Apply drop tolerance if specified
    if options.drop_tolerance > 0.0 {
        for val in &mut lu_values {
            if val.abs() < options.drop_tolerance {
                *val = 0.0;
            }
        }
    }

    // Split into L and U
    let (l, u) = split_lu::<R>(
        n,
        &row_ptrs,
        &col_indices,
        &lu_values,
        dtype,
        a.values().device(),
        options.drop_tolerance,
    )?;

    Ok(IluDecomposition { l, u })
}

/// Split combined LU values into separate L and U matrices
fn split_lu<R: Runtime<DType = DType>>(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    lu_values: &[f64],
    dtype: DType,
    device: &R::Device,
    drop_tolerance: f64,
) -> Result<(CsrData<R>, CsrData<R>)> {
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut l_values = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();
    let mut u_values = Vec::new();

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        let mut l_count = 0i64;
        let mut u_count = 0i64;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            let val = lu_values[idx];

            if val.abs() < 1e-15 && drop_tolerance > 0.0 {
                continue; // Skip zeros
            }

            if j < i {
                // L: below diagonal
                l_col_indices.push(j as i64);
                l_values.push(val);
                l_count += 1;
            } else {
                // U: diagonal and above
                u_col_indices.push(j as i64);
                u_values.push(val);
                u_count += 1;
            }
        }

        l_row_ptrs[i + 1] = l_row_ptrs[i] + l_count;
        u_row_ptrs[i + 1] = u_row_ptrs[i] + u_count;
    }

    // Create output tensors on the original device
    let l_row_ptrs_tensor = Tensor::<R>::from_slice(&l_row_ptrs, &[n + 1], device)?;
    let l_col_indices_tensor =
        Tensor::<R>::from_slice(&l_col_indices, &[l_col_indices.len()], device)?;
    let u_row_ptrs_tensor = Tensor::<R>::from_slice(&u_row_ptrs, &[n + 1], device)?;
    let u_col_indices_tensor =
        Tensor::<R>::from_slice(&u_col_indices, &[u_col_indices.len()], device)?;

    // Convert back to original dtype
    let (l_values_tensor, u_values_tensor) = match dtype {
        DType::F32 => {
            let l_f32: Vec<f32> = l_values.iter().map(|&x| x as f32).collect();
            let u_f32: Vec<f32> = u_values.iter().map(|&x| x as f32).collect();
            (
                Tensor::<R>::from_slice(&l_f32, &[l_f32.len()], device)?,
                Tensor::<R>::from_slice(&u_f32, &[u_f32.len()], device)?,
            )
        }
        DType::F64 => (
            Tensor::<R>::from_slice(&l_values, &[l_values.len()], device)?,
            Tensor::<R>::from_slice(&u_values, &[u_values.len()], device)?,
        ),
        _ => unreachable!(),
    };

    let l = CsrData::new(
        l_row_ptrs_tensor,
        l_col_indices_tensor,
        l_values_tensor,
        [n, n],
    )?;

    let u = CsrData::new(
        u_row_ptrs_tensor,
        u_col_indices_tensor,
        u_values_tensor,
        [n, n],
    )?;

    Ok((l, u))
}

/// ILU(0) symbolic factorization: precompute update schedule
///
/// Analyzes the sparsity pattern to create an efficient update schedule
/// for numeric factorization. This avoids hash map lookups during the
/// numeric phase.
pub fn ilu0_symbolic_cpu<R: Runtime<DType = DType>>(pattern: &CsrData<R>) -> Result<SymbolicIlu0> {
    let n = validate_square_sparse(pattern.shape)?;

    // Extract CSR structure for CPU-based symbolic analysis
    let row_ptrs: Vec<i64> = pattern.row_ptrs().to_vec();
    let col_indices: Vec<i64> = pattern.col_indices().to_vec();

    // Delegate to shared implementation
    crate::algorithm::sparse_linalg::ilu0_symbolic_impl(n, &row_ptrs, &col_indices)
}

/// ILU(0) numeric factorization using precomputed symbolic data
///
/// Uses the precomputed update schedule to perform factorization efficiently
/// without hash map lookups.
pub fn ilu0_numeric_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    symbolic: &SymbolicIlu0,
    options: IluOptions,
) -> Result<IluDecomposition<R>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cpu_dtype(dtype)?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

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
        _ => return Err(Error::UnsupportedDType { dtype, op: "ilu0" }),
    };

    // Create mutable copy for in-place factorization
    let mut lu_values = values;

    // IKJ factorization using precomputed schedule
    for (i, row_updates) in symbolic.update_schedule.iter().enumerate() {
        for &(k, idx_ik, ref updates) in row_updates {
            // Get diagonal A[k,k]
            let k_start = row_ptrs[k] as usize;
            let k_end = row_ptrs[k + 1] as usize;

            // Find diagonal in row k
            let mut diag_idx = None;
            for idx in k_start..k_end {
                if col_indices[idx] as usize == k {
                    diag_idx = Some(idx);
                    break;
                }
            }

            let diag_idx = match diag_idx {
                Some(idx) => idx,
                None => {
                    return Err(Error::Internal(format!(
                        "Missing diagonal at row {} in ILU(0)",
                        k
                    )));
                }
            };

            let diag_val = lu_values[diag_idx];
            if diag_val.abs() < 1e-15 {
                if options.diagonal_shift > 0.0 {
                    lu_values[diag_idx] = options.diagonal_shift;
                } else {
                    return Err(Error::Internal(format!(
                        "Zero pivot at row {} in ILU(0)",
                        k
                    )));
                }
            }

            // L[i,k] = A[i,k] / A[k,k]
            lu_values[idx_ik] /= lu_values[diag_idx];
            let l_ik = lu_values[idx_ik];

            // Apply precomputed updates
            for &(_j, idx_ij, idx_kj) in updates {
                lu_values[idx_ij] -= l_ik * lu_values[idx_kj];
            }
        }

        // Suppress unused warning for i (used for debugging)
        let _ = i;
    }

    // Apply drop tolerance if specified
    if options.drop_tolerance > 0.0 {
        for val in &mut lu_values {
            if val.abs() < options.drop_tolerance {
                *val = 0.0;
            }
        }
    }

    // Split into L and U
    let (l, u) = split_lu::<R>(
        n,
        &row_ptrs,
        &col_indices,
        &lu_values,
        dtype,
        a.values().device(),
        options.drop_tolerance,
    )?;

    Ok(IluDecomposition { l, u })
}
