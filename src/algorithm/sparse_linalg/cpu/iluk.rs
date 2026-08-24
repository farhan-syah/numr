//! CPU implementation of ILU(k) factorization
//!
//! Incomplete LU factorization with level-k fill-in using level-of-fill algorithm.

use std::collections::HashMap;

use super::super::traits::validate_square_sparse;
use super::super::types::{IluFillLevel, IluMetrics, IlukDecomposition, IlukOptions, IlukSymbolic};
use super::validate_cpu_dtype;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// ILU(k) symbolic factorization on CPU
///
/// Computes the fill pattern for ILU(k) using the level-of-fill algorithm.
///
/// # Algorithm
///
/// The level-of-fill for each position (i,j) is computed as:
/// - `level[i,j]` = 0 if `A[i,j]` ≠ 0
/// - `level[i,j]` = min over all paths i→k→j of: `level[i,k]` + `level[k,j]` + 1
///
/// Positions with `level[i,j]` ≤ k are included in the fill pattern.
pub fn iluk_symbolic_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    level: IluFillLevel,
) -> Result<IlukSymbolic> {
    let n = validate_square_sparse(a.shape)?;

    // Extract CSR structure for CPU-based symbolic analysis
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Delegate to shared implementation
    crate::algorithm::sparse_linalg::iluk_symbolic_impl(n, &row_ptrs, &col_indices, level)
}

/// ILU(k) numeric factorization on CPU using precomputed symbolic data
pub fn iluk_numeric_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    symbolic: &IlukSymbolic,
    opts: &IlukOptions,
) -> Result<IlukDecomposition<R>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_cpu_dtype(dtype)?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Extract CSR components
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();
    let values: Vec<f64> = match dtype {
        DType::F32 => a
            .values()
            .to_vec::<f32>()
            .iter()
            .map(|&x| x as f64)
            .collect(),
        DType::F64 => a.values().to_vec(),
        _ => return Err(Error::UnsupportedDType { dtype, op: "iluk" }),
    };

    // Build col_to_idx for the original matrix
    let mut col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            col_to_idx[i].insert(j, idx);
        }
    }

    // Initialize L and U values arrays based on symbolic pattern
    let nnz_l = symbolic.col_indices_l.len();
    let nnz_u = symbolic.col_indices_u.len();
    let mut l_values = vec![0.0f64; nnz_l];
    let mut u_values = vec![0.0f64; nnz_u];

    // Build index maps for L and U
    let mut l_col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];
    let mut u_col_to_idx: Vec<HashMap<usize, usize>> = vec![HashMap::new(); n];

    for i in 0..n {
        let l_start = symbolic.row_ptrs_l[i] as usize;
        let l_end = symbolic.row_ptrs_l[i + 1] as usize;
        for (offset, idx) in (l_start..l_end).enumerate() {
            let j = symbolic.col_indices_l[idx] as usize;
            l_col_to_idx[i].insert(j, l_start + offset);
        }

        let u_start = symbolic.row_ptrs_u[i] as usize;
        let u_end = symbolic.row_ptrs_u[i + 1] as usize;
        for (offset, idx) in (u_start..u_end).enumerate() {
            let j = symbolic.col_indices_u[idx] as usize;
            u_col_to_idx[i].insert(j, u_start + offset);
        }
    }

    // Copy original values into L and U
    for i in 0..n {
        // L part: columns < i
        if let Some(l_indices) = Some(&l_col_to_idx[i]) {
            for (&col, &l_idx) in l_indices.iter() {
                if let Some(&orig_idx) = col_to_idx[i].get(&col) {
                    l_values[l_idx] = values[orig_idx];
                }
            }
        }

        // U part: columns >= i
        if let Some(u_indices) = Some(&u_col_to_idx[i]) {
            for (&col, &u_idx) in u_indices.iter() {
                if let Some(&orig_idx) = col_to_idx[i].get(&col) {
                    u_values[u_idx] = values[orig_idx];
                }
            }
        }
    }

    // Perform IKJ factorization
    let mut diagonal_shifts_applied = 0;

    for i in 0..n {
        // Get L columns for row i (columns k < i)
        let l_start = symbolic.row_ptrs_l[i] as usize;
        let l_end = symbolic.row_ptrs_l[i + 1] as usize;

        for l_idx in l_start..l_end {
            let k = symbolic.col_indices_l[l_idx] as usize;

            // Get diagonal U[k,k]
            let u_diag_idx = match u_col_to_idx[k].get(&k) {
                Some(&idx) => idx,
                None => {
                    return Err(Error::Internal(format!(
                        "Missing diagonal at row {} in ILU(k)",
                        k
                    )));
                }
            };

            let diag_val = u_values[u_diag_idx];
            if diag_val.abs() < 1e-15 {
                if opts.diagonal_shift > 0.0 {
                    u_values[u_diag_idx] = opts.diagonal_shift;
                    diagonal_shifts_applied += 1;
                } else {
                    return Err(Error::Internal(format!(
                        "Zero pivot at row {} in ILU(k)",
                        k
                    )));
                }
            }

            // L[i,k] = L[i,k] / U[k,k]
            l_values[l_idx] /= u_values[u_diag_idx];
            let l_ik = l_values[l_idx];

            // Update U row i: U[i,j] -= L[i,k] * U[k,j] for j > k
            let u_k_start = symbolic.row_ptrs_u[k] as usize;
            let u_k_end = symbolic.row_ptrs_u[k + 1] as usize;

            for u_k_idx in u_k_start..u_k_end {
                let j = symbolic.col_indices_u[u_k_idx] as usize;
                if j <= k {
                    continue;
                }

                // Check if U[i,j] exists in the fill pattern
                if let Some(&u_ij_idx) = u_col_to_idx[i].get(&j) {
                    u_values[u_ij_idx] -= l_ik * u_values[u_k_idx];
                }
                // Also check L[i,j] for j < i
                if let Some(&l_ij_idx) = l_col_to_idx[i].get(&j).filter(|_| j < i) {
                    l_values[l_ij_idx] -= l_ik * u_values[u_k_idx];
                }
            }
        }
    }

    // Apply drop tolerance if specified
    if opts.drop_tolerance > 0.0 {
        for val in &mut l_values {
            if val.abs() < opts.drop_tolerance {
                *val = 0.0;
            }
        }
        for val in &mut u_values {
            if val.abs() < opts.drop_tolerance {
                *val = 0.0;
            }
        }
    }

    // Create output tensors
    let device = a.values().device();
    let original_nnz = values.len();
    let factored_nnz = nnz_l + nnz_u;

    let l_row_ptrs_tensor = Tensor::<R>::try_from_slice(&symbolic.row_ptrs_l, &[n + 1], device)?;
    let l_col_indices_tensor =
        Tensor::<R>::try_from_slice(&symbolic.col_indices_l, &[nnz_l], device)?;
    let u_row_ptrs_tensor = Tensor::<R>::try_from_slice(&symbolic.row_ptrs_u, &[n + 1], device)?;
    let u_col_indices_tensor =
        Tensor::<R>::try_from_slice(&symbolic.col_indices_u, &[nnz_u], device)?;

    let (l_values_tensor, u_values_tensor) = match dtype {
        DType::F32 => {
            let l_f32: Vec<f32> = l_values.iter().map(|&x| x as f32).collect();
            let u_f32: Vec<f32> = u_values.iter().map(|&x| x as f32).collect();
            (
                Tensor::<R>::try_from_slice(&l_f32, &[nnz_l], device)?,
                Tensor::<R>::try_from_slice(&u_f32, &[nnz_u], device)?,
            )
        }
        DType::F64 => (
            Tensor::<R>::try_from_slice(&l_values, &[nnz_l], device)?,
            Tensor::<R>::try_from_slice(&u_values, &[nnz_u], device)?,
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

    let metrics = IluMetrics {
        original_nnz,
        factored_nnz,
        fill_ratio: factored_nnz as f64 / original_nnz as f64,
        fill_level: symbolic.fill_level,
        diagonal_shifts_applied,
    };

    Ok(IlukDecomposition { l, u, metrics })
}

/// Combined ILU(k) factorization (symbolic + numeric)
pub fn iluk_cpu<R: Runtime<DType = DType>>(
    a: &CsrData<R>,
    opts: IlukOptions,
) -> Result<IlukDecomposition<R>> {
    let symbolic = iluk_symbolic_cpu(a, opts.fill_level)?;
    iluk_numeric_cpu(a, &symbolic, &opts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    fn make_tridiagonal(n: usize, device: &CpuDevice) -> CsrData<CpuRuntime> {
        let mut row_ptrs = vec![0i64; n + 1];
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for i in 0..n {
            if i > 0 {
                col_indices.push((i - 1) as i64);
                values.push(-1.0f64);
            }
            col_indices.push(i as i64);
            values.push(2.0f64);
            if i < n - 1 {
                col_indices.push((i + 1) as i64);
                values.push(-1.0f64);
            }
            row_ptrs[i + 1] = col_indices.len() as i64;
        }

        let row_ptrs_tensor =
            Tensor::<CpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], device);
        let col_indices_tensor =
            Tensor::<CpuRuntime>::from_slice(&col_indices, &[col_indices.len()], device);
        let values_tensor = Tensor::<CpuRuntime>::from_slice(&values, &[values.len()], device);

        CsrData::new(row_ptrs_tensor, col_indices_tensor, values_tensor, [n, n])
            .expect("CSR creation should succeed")
    }

    #[test]
    fn test_iluk_symbolic_level0() {
        let device = CpuDevice::new();
        let a = make_tridiagonal(5, &device);

        let symbolic = iluk_symbolic_cpu::<CpuRuntime>(&a, IluFillLevel::Zero)
            .expect("symbolic should succeed");

        assert_eq!(symbolic.n, 5);
        assert_eq!(symbolic.fill_level, IluFillLevel::Zero);

        // For tridiagonal, ILU(0) has same pattern as original
        // L has 4 subdiagonal entries, U has 5 diagonal + 4 superdiagonal
        assert_eq!(symbolic.col_indices_l.len(), 4);
        assert_eq!(symbolic.col_indices_u.len(), 9);
    }

    #[test]
    fn test_iluk_numeric() {
        let device = CpuDevice::new();
        let a = make_tridiagonal(5, &device);

        let opts = IlukOptions::default();
        let result = iluk_cpu::<CpuRuntime>(&a, opts).expect("iluk should succeed");

        // Check metrics
        assert_eq!(result.metrics.fill_level, IluFillLevel::Zero);
        assert!(result.metrics.fill_ratio >= 1.0);
        assert_eq!(result.metrics.diagonal_shifts_applied, 0);

        // Verify L and U shapes
        assert_eq!(result.l.shape, [5, 5]);
        assert_eq!(result.u.shape, [5, 5]);
    }

    #[test]
    fn test_iluk_level1_more_fill() {
        let device = CpuDevice::new();
        let a = make_tridiagonal(5, &device);

        let symbolic_0 = iluk_symbolic_cpu::<CpuRuntime>(&a, IluFillLevel::Zero)
            .expect("symbolic should succeed");
        let symbolic_1 = iluk_symbolic_cpu::<CpuRuntime>(&a, IluFillLevel::One)
            .expect("symbolic should succeed");

        // ILU(1) should have at least as much fill as ILU(0)
        let nnz_0 = symbolic_0.col_indices_l.len() + symbolic_0.col_indices_u.len();
        let nnz_1 = symbolic_1.col_indices_l.len() + symbolic_1.col_indices_u.len();
        assert!(nnz_1 >= nnz_0);
    }
}
