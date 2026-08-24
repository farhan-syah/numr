//! AMG coarsening and interpolation operators
//!
//! Classical Ruge-Stüben AMG:
//! - Strength-of-connection based on threshold
//! - PMIS (Parallel Modified Independent Set) coarsening
//! - Classical interpolation with truncation

use crate::dtype::DType;
use crate::error::Result;
use crate::runtime::Runtime;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Coarse/fine splitting result
pub struct CfSplitting {
    /// true = coarse point, false = fine point
    pub is_coarse: Vec<bool>,
    /// Coarse-to-fine index mapping
    pub coarse_indices: Vec<usize>,
    /// Number of coarse points
    pub n_coarse: usize,
}

/// Compute strength-of-connection: strong connections where
/// |a_ij| >= theta * max_k(|a_ik|) for k != i
///
/// Returns: for each row, the set of strongly connected column indices
pub fn strength_of_connection(
    row_ptrs: &[i64],
    col_indices: &[i64],
    values: &[f64],
    n: usize,
    theta: f64,
) -> Vec<Vec<usize>> {
    let mut strong = vec![Vec::new(); n];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        // Find max off-diagonal magnitude
        let mut max_off_diag = 0.0_f64;
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j != i {
                max_off_diag = max_off_diag.max(values[idx].abs());
            }
        }

        let threshold = theta * max_off_diag;

        // Collect strong connections
        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j != i && values[idx].abs() >= threshold {
                strong[i].push(j);
            }
        }
    }

    strong
}

/// PMIS coarsening: greedy independent set selection based on
/// connection weights (number of strong connections)
pub fn pmis_coarsening(strong_connections: &[Vec<usize>], n: usize) -> CfSplitting {
    // Weight = number of strong connections (influences)
    let mut weights: Vec<usize> = strong_connections.iter().map(|s| s.len()).collect();
    let mut is_coarse = vec![false; n];
    let mut is_decided = vec![false; n];

    // Greedy: pick highest-weight undecided point as coarse,
    // mark its strong neighbors as fine
    loop {
        // Find undecided point with highest weight
        let mut best_idx = None;
        let mut best_weight = 0;
        for i in 0..n {
            if !is_decided[i] && weights[i] >= best_weight {
                best_weight = weights[i];
                best_idx = Some(i);
            }
        }

        let Some(idx) = best_idx else { break };

        // Mark as coarse
        is_coarse[idx] = true;
        is_decided[idx] = true;

        // Mark strong neighbors as fine
        for &j in &strong_connections[idx] {
            if !is_decided[j] {
                is_decided[j] = true;
                // Reduce weights of neighbors of j
                for &k in &strong_connections[j] {
                    if !is_decided[k] && weights[k] > 0 {
                        weights[k] -= 1;
                    }
                }
            }
        }
    }

    // Build coarse index mapping
    let mut coarse_indices = Vec::new();
    let mut coarse_map = vec![0usize; n];
    for i in 0..n {
        if is_coarse[i] {
            coarse_map[i] = coarse_indices.len();
            coarse_indices.push(i);
        }
    }

    let n_coarse = coarse_indices.len();

    CfSplitting {
        is_coarse,
        coarse_indices,
        n_coarse,
    }
}

/// Build classical interpolation operator P: coarse -> fine
///
/// For coarse points: P[i, coarse_map[i]] = 1
/// For fine points: P[i, j] = -a_ij / a_ii for strongly connected coarse j,
/// normalized to sum to 1
pub fn build_interpolation<R: Runtime<DType = DType>>(
    row_ptrs: &[i64],
    col_indices: &[i64],
    values: &[f64],
    n: usize,
    splitting: &CfSplitting,
    strong_connections: &[Vec<usize>],
    device: &R::Device,
) -> Result<CsrData<R>> {
    let n_coarse = splitting.n_coarse;

    // Build coarse-point index map
    let mut coarse_map = vec![0usize; n];
    for (ci, &fine_idx) in splitting.coarse_indices.iter().enumerate() {
        coarse_map[fine_idx] = ci;
    }

    // Build P in CSR format
    let mut p_row_ptrs = Vec::with_capacity(n + 1);
    let mut p_col_indices = Vec::new();
    let mut p_values = Vec::new();

    p_row_ptrs.push(0i64);

    for i in 0..n {
        if splitting.is_coarse[i] {
            // Injection: P[i, coarse_map[i]] = 1
            p_col_indices.push(coarse_map[i] as i64);
            p_values.push(1.0f64);
        } else {
            // Interpolation from strong coarse neighbors
            let start = row_ptrs[i] as usize;
            let end = row_ptrs[i + 1] as usize;

            // Get diagonal
            let mut diag = 1.0;
            for idx in start..end {
                if col_indices[idx] as usize == i {
                    diag = values[idx];
                    break;
                }
            }

            // Collect strong coarse connections and their weights
            let mut interp_cols = Vec::new();
            let mut interp_vals = Vec::new();
            let mut sum_weights = 0.0;

            for &j in &strong_connections[i] {
                if splitting.is_coarse[j] {
                    // Find a_ij
                    for idx in start..end {
                        if col_indices[idx] as usize == j {
                            let w = -values[idx] / diag;
                            interp_cols.push(coarse_map[j] as i64);
                            interp_vals.push(w);
                            sum_weights += w;
                            break;
                        }
                    }
                }
            }

            // Normalize weights to sum to 1 (if any)
            if sum_weights.abs() > 1e-15 && !interp_vals.is_empty() {
                for v in &mut interp_vals {
                    *v /= sum_weights;
                }
            }

            // If no coarse connections found, use nearest coarse point
            if interp_cols.is_empty() {
                // Fallback: find closest coarse point
                for &j in &strong_connections[i] {
                    if splitting.is_coarse[j] {
                        interp_cols.push(coarse_map[j] as i64);
                        interp_vals.push(1.0);
                        break;
                    }
                }
                // If still empty, just use first coarse point (degenerate case)
                if interp_cols.is_empty() && n_coarse > 0 {
                    interp_cols.push(0);
                    interp_vals.push(1.0);
                }
            }

            for (c, v) in interp_cols.into_iter().zip(interp_vals) {
                p_col_indices.push(c);
                p_values.push(v);
            }
        }

        p_row_ptrs.push(p_col_indices.len() as i64);
    }

    let rp = Tensor::<R>::try_from_slice(&p_row_ptrs, &[p_row_ptrs.len()], device)?;
    let ci = Tensor::<R>::try_from_slice(&p_col_indices, &[p_col_indices.len()], device)?;
    let vv = Tensor::<R>::try_from_slice(&p_values, &[p_values.len()], device)?;

    CsrData::new(rp, ci, vv, [n, n_coarse])
}

/// Build restriction operator R = P^T (transpose of interpolation)
pub fn build_restriction<R: Runtime<DType = DType>>(p: &CsrData<R>) -> Result<CsrData<R>> {
    // P^T: CsrData::transpose() returns CscData, then to_csr() gives CSR of P^T
    let pt = p.transpose().to_csr()?;
    Ok(pt)
}

/// Build Galerkin coarse operator: A_c = R * A * P = P^T * A * P
///
/// This is done via sparse matrix multiplication.
/// For simplicity, we compute it via explicit SpMM on CPU.
pub fn galerkin_coarse_operator<R: Runtime<DType = DType>>(
    row_ptrs: &[i64],
    col_indices: &[i64],
    values: &[f64],
    n_fine: usize,
    p_row_ptrs: &[i64],
    p_col_indices: &[i64],
    p_values: &[f64],
    n_coarse: usize,
    device: &R::Device,
) -> Result<CsrData<R>> {
    // Compute A*P first (n_fine x n_coarse dense, then sparsify)
    // For moderate sizes this is tractable. For very large grids,
    // would need true sparse triple product.

    // Build P as column-major for easy column access
    let mut p_cols: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n_coarse];
    for i in 0..n_fine {
        let start = p_row_ptrs[i] as usize;
        let end = p_row_ptrs[i + 1] as usize;
        for idx in start..end {
            let j = p_col_indices[idx] as usize;
            let v = p_values[idx];
            p_cols[j].push((i, v));
        }
    }

    // Compute R*A*P row by row (R = P^T, so row i of R*A*P = sum over fine points)
    let mut ac_rows: Vec<Vec<(usize, f64)>> = Vec::with_capacity(n_coarse);

    for ci in 0..n_coarse {
        let mut row_map = std::collections::HashMap::new();

        // For each fine point k that interpolates to coarse point ci (column ci of P)
        for &(k, r_val) in &p_cols[ci] {
            // Row k of A
            let a_start = row_ptrs[k] as usize;
            let a_end = row_ptrs[k + 1] as usize;

            for a_idx in a_start..a_end {
                let l = col_indices[a_idx] as usize;
                let a_val = values[a_idx];

                // Multiply by P[l, cj] for each coarse point cj
                let p_start = p_row_ptrs[l] as usize;
                let p_end = p_row_ptrs[l + 1] as usize;

                for p_idx in p_start..p_end {
                    let cj = p_col_indices[p_idx] as usize;
                    let p_val = p_values[p_idx];

                    *row_map.entry(cj).or_insert(0.0) += r_val * a_val * p_val;
                }
            }
        }

        let mut entries: Vec<(usize, f64)> = row_map.into_iter().collect();
        entries.sort_by_key(|&(j, _)| j);
        ac_rows.push(entries);
    }

    // Build CSR from ac_rows
    let mut rp = Vec::with_capacity(n_coarse + 1);
    let mut ci_vec = Vec::new();
    let mut vv = Vec::new();

    rp.push(0i64);
    for row in &ac_rows {
        for &(j, v) in row {
            if v.abs() > 1e-15 {
                ci_vec.push(j as i64);
                vv.push(v);
            }
        }
        rp.push(ci_vec.len() as i64);
    }

    let rp_t = Tensor::<R>::try_from_slice(&rp, &[rp.len()], device)?;
    let ci_t = Tensor::<R>::try_from_slice(&ci_vec, &[ci_vec.len()], device)?;
    let vv_t = Tensor::<R>::try_from_slice(&vv, &[vv.len()], device)?;

    CsrData::new(rp_t, ci_t, vv_t, [n_coarse, n_coarse])
}
