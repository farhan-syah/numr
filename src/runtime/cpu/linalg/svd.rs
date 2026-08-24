//! Singular Value Decomposition (SVD) using One-Sided Jacobi algorithm

use super::super::jacobi::{
    self, JacobiRotation, LinalgElement, apply_rotation_to_columns, argsort_desc,
    compute_gram_elements, identity_matrix, normalize_columns, permute_columns,
};
use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    SvdDecomposition, linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// SVD decomposition using One-Sided Jacobi algorithm
pub fn svd_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<SvdDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => svd_decompose_typed::<f32>(client, &a, m, n),
        DType::F64 => svd_decompose_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    Ok(SvdDecomposition {
        u: linalg_demote(client, result.u, original_dtype)?,
        s: linalg_demote(client, result.s, original_dtype)?,
        vt: linalg_demote(client, result.vt, original_dtype)?,
    })
}

/// SVD decomposition using One-Sided Jacobi algorithm
///
/// Algorithm: One-Sided Jacobi SVD
/// 1. If m < n: Transpose A, compute SVD, swap U↔V^T
/// 2. Initialize: B = A (working copy), V = I_n
/// 3. REPEAT (max 30 sweeps):
///    FOR each pair (p, q) where p < q:
///      - Compute Gram elements: a_pp, a_qq, a_pq = B[:,p]·B[:,q]
///      - If |a_pq| > tol: compute Jacobi rotation (c,s), apply to B and V columns
///      - Check convergence: sqrt(Σ a_pq²) < n * epsilon
/// 4. Extract: S[j] = ||B[:,j]||, U[:,j] = B[:,j]/S[j]
/// 5. Sort S descending, reorder U and V columns accordingly
/// 6. Return U, S, V^T = V.transpose()
fn svd_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<SvdDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // Handle empty matrix
    if m == 0 || n == 0 {
        let u = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[m, k], device)?;
        let s = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[k], device)?;
        let vt = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[k, n], device)?;
        return Ok(SvdDecomposition { u, s, vt });
    }

    // If m < n, transpose and swap U/V at the end
    let transpose = m < n;
    let (work_m, work_n) = if transpose { (n, m) } else { (m, n) };

    // Get input data, transposing if needed
    let mut b: Vec<T> = if transpose {
        // Transpose input: A[i,j] -> A^T[j,i]
        let a_data: Vec<T> = a.to_vec();
        let mut b_transposed = vec![T::zero(); work_m * work_n];
        for i in 0..m {
            for j in 0..n {
                b_transposed[j * work_n + i] = a_data[i * n + j];
            }
        }
        b_transposed
    } else {
        a.to_vec()
    };

    let work_k = work_m.min(work_n);

    // Initialize V as identity [work_n x work_n]
    let mut v: Vec<T> = identity_matrix(work_n);

    // Convergence parameters
    let eps = T::epsilon_val();
    let tol = (work_n as f64) * eps;
    let max_sweeps = 30;

    // One-Sided Jacobi iterations
    for _sweep in 0..max_sweeps {
        let mut off_diag_sum = 0.0f64;

        // Process all column pairs (p, q) where p < q
        for p in 0..work_n {
            for q in (p + 1)..work_n {
                // Compute Gram matrix elements for columns p and q
                let (a_pp, a_qq, a_pq) = compute_gram_elements(&b, work_m, work_n, p, q);

                off_diag_sum += a_pq.to_f64() * a_pq.to_f64();

                // Skip if off-diagonal is essentially zero
                if a_pq.abs_val().to_f64() < tol * (a_pp.to_f64() * a_qq.to_f64()).sqrt() {
                    continue;
                }

                // Compute Jacobi rotation using stable LAPACK formula
                let rot = JacobiRotation::compute(a_pp.to_f64(), a_qq.to_f64(), a_pq.to_f64());

                // Apply rotation to B columns
                apply_rotation_to_columns(&mut b, work_m, work_n, p, q, &rot);

                // Apply rotation to V columns
                apply_rotation_to_columns(&mut v, work_n, work_n, p, q, &rot);
            }
        }

        // Check convergence: sqrt(sum of squared off-diagonals) < tolerance
        if off_diag_sum.sqrt() < tol {
            break;
        }
    }

    // Extract singular values and normalize U columns using shared utility
    // S[j] = ||B[:,j]||, U[:,j] = B[:,j] / S[j]
    let singular_values = normalize_columns(&mut b, work_m, work_n, eps);
    let u_data = b; // b is now U after normalization

    // Sort singular values in descending order and reorder U, V accordingly
    let indices = argsort_desc(&singular_values);

    // Reorder singular values (take first work_k)
    let s_sorted: Vec<T> = jacobi::permute_vector(&singular_values, &indices)
        .into_iter()
        .take(work_k)
        .collect();

    // Reorder U columns (take first work_k)
    let u_sorted = permute_columns(&u_data, work_m, work_n, &indices, work_k);

    // Reorder V columns and transpose to get V^T (take first work_k rows)
    // V^T[i, j] = V[j, perm[i]]
    let mut vt_sorted: Vec<T> = vec![T::zero(); work_k * work_n];
    for (new_idx, &old_idx) in indices.iter().take(work_k).enumerate() {
        for j in 0..work_n {
            vt_sorted[new_idx * work_n + j] = v[j * work_n + old_idx];
        }
    }

    // If we transposed at the beginning, swap U and V^T
    if transpose {
        // Original: A = U @ S @ V^T
        // Transposed: A^T = U' @ S @ V'^T
        // So: A = (U' @ S @ V'^T)^T = V' @ S @ U'^T
        // Therefore: U_final = V', V^T_final = U'^T
        //
        // For original A [m x n] with m < n:
        // - U_final should be [m x k] where k = min(m,n) = m
        // - V^T_final should be [k x n] = [m x n]
        //
        // After A^T SVD:
        // - u_sorted is [work_m x work_k] = [n x m]
        // - vt_sorted is [work_k x work_n] = [m x m]

        // U_final = V' = (vt_sorted)^T, shape [m x k] = [m x m]
        // vt_sorted is [work_k x work_n] = [m x m], so transpose to [m x m]
        let mut u_final: Vec<T> = vec![T::zero(); m * k];
        for i in 0..k {
            for j in 0..m {
                // u_final[j, i] = vt_sorted[i, j]
                u_final[j * k + i] = vt_sorted[i * work_n + j];
            }
        }

        // V^T_final = U'^T, shape [k x n] = [m x n]
        // u_sorted is [work_m x work_k] = [n x m], transpose to [m x n]
        let mut vt_final: Vec<T> = vec![T::zero(); k * n];
        for i in 0..work_m {
            for j in 0..work_k {
                // vt_final[j, i] = u_sorted[i, j]
                vt_final[j * n + i] = u_sorted[i * work_k + j];
            }
        }

        let u_tensor = Tensor::<CpuRuntime>::from_slice(&u_final, &[m, k], device)?;
        let s_tensor = Tensor::<CpuRuntime>::from_slice(&s_sorted, &[k], device)?;
        let vt_tensor = Tensor::<CpuRuntime>::from_slice(&vt_final, &[k, n], device)?;

        Ok(SvdDecomposition {
            u: u_tensor,
            s: s_tensor,
            vt: vt_tensor,
        })
    } else {
        let u_tensor = Tensor::<CpuRuntime>::from_slice(&u_sorted, &[m, k], device)?;
        let s_tensor = Tensor::<CpuRuntime>::from_slice(&s_sorted, &[k], device)?;
        let vt_tensor = Tensor::<CpuRuntime>::from_slice(&vt_sorted, &[k, n], device)?;

        Ok(SvdDecomposition {
            u: u_tensor,
            s: s_tensor,
            vt: vt_tensor,
        })
    }
}
