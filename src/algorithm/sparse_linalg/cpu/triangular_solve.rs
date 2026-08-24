//! CPU implementation of sparse triangular solve
//!
//! Forward and backward substitution for sparse triangular matrices.

use std::collections::HashMap;

use super::super::traits::validate_triangular_solve_dims;
use super::validate_cpu_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Sparse triangular solve on CPU: L·x = b or U·x = b
///
/// # Algorithm
///
/// Forward substitution (lower triangular):
///   For i = 0 to n-1:
///     `x[i]` = `b[i]` - sum(`L[i,j]` * `x[j]` for j < i)
///     `x[i]` /= `L[i,i]` (unless unit diagonal)
///
/// Backward substitution (upper triangular):
///   For i = n-1 down to 0:
///     `x[i]` = `b[i]` - sum(`U[i,j]` * `x[j]` for j > i)
///     `x[i]` /= `U[i,i]`
///
/// # Arguments
///
/// * `l_or_u` - Sparse triangular matrix in CSR format
/// * `b` - Right-hand side dense vector `[n]` or matrix `[n, k]`
/// * `lower` - True for lower triangular, false for upper triangular
/// * `unit_diagonal` - True if diagonal is implicitly 1 (for unit triangular)
///
/// # Returns
///
/// Solution vector x `[n]` or matrix `[n, k]`
pub fn sparse_solve_triangular_cpu<R: Runtime<DType = DType>>(
    l_or_u: &CsrData<R>,
    b: &Tensor<R>,
    lower: bool,
    unit_diagonal: bool,
) -> Result<Tensor<R>> {
    let (n, nrhs) = validate_triangular_solve_dims(l_or_u.shape, b.shape())?;
    let dtype = l_or_u.values().dtype();
    validate_cpu_dtype(dtype)?;

    // Ensure dtypes match
    if b.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }

    // Extract CSR components (efficient for CPU)
    let row_ptrs: Vec<i64> = l_or_u.row_ptrs().to_vec();
    let col_indices: Vec<i64> = l_or_u.col_indices().to_vec();

    // Work with f64 for numerical stability
    let mat_values: Vec<f64> = match dtype {
        DType::F32 => l_or_u
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => l_or_u.values().to_vec(),
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "sparse_solve_triangular",
            });
        }
    };

    let b_flat: Vec<f64> = match dtype {
        DType::F32 => b.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
        DType::F64 => b.to_vec(),
        _ => unreachable!(),
    };

    // Reshape b if needed
    let b_matrix: Vec<Vec<f64>> = if nrhs == 1 {
        b_flat.into_iter().map(|x| vec![x]).collect()
    } else {
        // b is [n, nrhs]
        (0..n)
            .map(|i| (0..nrhs).map(|j| b_flat[i * nrhs + j]).collect())
            .collect()
    };

    // Build column-to-index map
    let col_to_idx = build_col_index_map(n, &row_ptrs, &col_indices);

    // Solve for each right-hand side
    let x_matrix = if lower {
        solve_lower(
            n,
            nrhs,
            &row_ptrs,
            &col_indices,
            &mat_values,
            &b_matrix,
            &col_to_idx,
            unit_diagonal,
        )?
    } else {
        solve_upper(
            n,
            nrhs,
            &row_ptrs,
            &col_indices,
            &mat_values,
            &b_matrix,
            &col_to_idx,
        )?
    };

    // Flatten result and create output tensor
    let x_flat: Vec<f64> = if nrhs == 1 {
        x_matrix.into_iter().map(|v| v[0]).collect()
    } else {
        x_matrix.into_iter().flatten().collect()
    };

    let device = b.device();
    let shape = if nrhs == 1 { vec![n] } else { vec![n, nrhs] };

    let result = match dtype {
        DType::F32 => {
            let f32_vals: Vec<f32> = x_flat.iter().map(|&x| x as f32).collect();
            Tensor::<R>::from_slice(&f32_vals, &shape, device)?
        }
        DType::F64 => Tensor::<R>::from_slice(&x_flat, &shape, device)?,
        _ => unreachable!(),
    };

    Ok(result)
}

/// Build column-to-index map for fast lookup
fn build_col_index_map(
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
) -> Vec<HashMap<usize, usize>> {
    let mut col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            col_to_idx[i].insert(j, idx);
        }
    }
    col_to_idx
}

/// Forward substitution for lower triangular solve
fn solve_lower(
    n: usize,
    nrhs: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    mat_values: &[f64],
    b_matrix: &[Vec<f64>],
    col_to_idx: &[HashMap<usize, usize>],
    unit_diagonal: bool,
) -> Result<Vec<Vec<f64>>> {
    let mut x_matrix: Vec<Vec<f64>> = vec![vec![0.0; nrhs]; n];

    for i in 0..n {
        for k in 0..nrhs {
            let mut sum = b_matrix[i][k];

            let start = row_ptrs[i] as usize;
            let end = row_ptrs[i + 1] as usize;

            for idx in start..end {
                let j = col_indices[idx] as usize;
                if j < i {
                    sum -= mat_values[idx] * x_matrix[j][k];
                }
            }

            if !unit_diagonal {
                let diag_idx = col_to_idx[i].get(&i);
                match diag_idx {
                    Some(&idx) => {
                        let diag = mat_values[idx];
                        if diag.abs() < 1e-15 {
                            return Err(Error::Internal(format!(
                                "Zero diagonal at row {} in triangular solve",
                                i
                            )));
                        }
                        sum /= diag;
                    }
                    None => {
                        return Err(Error::Internal(format!(
                            "Missing diagonal at row {} in triangular solve",
                            i
                        )));
                    }
                }
            }

            x_matrix[i][k] = sum;
        }
    }

    Ok(x_matrix)
}

/// Backward substitution for upper triangular solve
fn solve_upper(
    n: usize,
    nrhs: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    mat_values: &[f64],
    b_matrix: &[Vec<f64>],
    col_to_idx: &[HashMap<usize, usize>],
) -> Result<Vec<Vec<f64>>> {
    let mut x_matrix: Vec<Vec<f64>> = vec![vec![0.0; nrhs]; n];

    for i in (0..n).rev() {
        for k in 0..nrhs {
            let mut sum = b_matrix[i][k];

            let start = row_ptrs[i] as usize;
            let end = row_ptrs[i + 1] as usize;

            for idx in start..end {
                let j = col_indices[idx] as usize;
                if j > i {
                    sum -= mat_values[idx] * x_matrix[j][k];
                }
            }

            let diag_idx = col_to_idx[i].get(&i);
            match diag_idx {
                Some(&idx) => {
                    let diag = mat_values[idx];
                    if diag.abs() < 1e-15 {
                        return Err(Error::Internal(format!(
                            "Zero diagonal at row {} in triangular solve",
                            i
                        )));
                    }
                    sum /= diag;
                }
                None => {
                    return Err(Error::Internal(format!(
                        "Missing diagonal at row {} in triangular solve",
                        i
                    )));
                }
            }

            x_matrix[i][k] = sum;
        }
    }

    Ok(x_matrix)
}
