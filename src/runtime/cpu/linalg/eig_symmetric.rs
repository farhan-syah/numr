//! Eigendecomposition for symmetric matrices using Jacobi algorithm

use super::super::jacobi::{
    self, JacobiRotation, LinalgElement, apply_rotation_to_columns, apply_two_sided_rotation,
    argsort_by_magnitude_desc, identity_matrix, permute_columns,
};
use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    EigenDecomposition, linalg_demote, linalg_promote, validate_linalg_dtype,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Eigendecomposition for symmetric matrices using Jacobi algorithm
pub fn eig_decompose_symmetric_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<EigenDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => eig_decompose_symmetric_typed::<f32>(client, &a, n),
        DType::F64 => eig_decompose_symmetric_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(EigenDecomposition {
        eigenvalues: linalg_demote(client, result.eigenvalues, original_dtype)?,
        eigenvectors: linalg_demote(client, result.eigenvectors, original_dtype)?,
    })
}

/// Eigendecomposition for symmetric matrices using Jacobi algorithm
///
/// Algorithm: Jacobi Eigenvalue Algorithm
/// 1. Initialize: V = I_n (eigenvector matrix starts as identity)
/// 2. REPEAT (max 30 sweeps):
///    FOR each pair (p, q) where p < q:
///      - If |A[p,q]| > tol:
///        a. Compute Jacobi rotation angle θ from A[p,p], A[q,q], A[p,q]
///        b. Apply rotation: A' = J^T @ A @ J (zeros out A[p,q] and A[q,p])
///        c. Update eigenvectors: V = V @ J
///      - Check convergence: max(|A[i,j]| for i≠j) < n * epsilon
/// 3. eigenvalues = diag(A) (diagonal elements after convergence)
/// 4. Sort eigenvalues descending by magnitude, reorder eigenvector columns
fn eig_decompose_symmetric_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<EigenDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle empty matrix
    if n == 0 {
        let eigenvalues = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0], device)?;
        let eigenvectors = Tensor::<CpuRuntime>::from_slice::<T>(&[], &[0, 0], device)?;
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Handle 1x1 matrix
    if n == 1 {
        let a_data: Vec<T> = a.to_vec();
        let eigenvalues = Tensor::<CpuRuntime>::from_slice(&[a_data[0]], &[1], device)?;
        let eigenvectors = Tensor::<CpuRuntime>::from_slice(&[T::one()], &[1, 1], device)?;
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Copy input to working matrix (will be modified in-place)
    // We symmetrize by using only lower triangle: A[i,j] = A[j,i] for i > j
    let a_data: Vec<T> = a.to_vec();
    let mut work: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        for j in 0..=i {
            // Use lower triangular part
            let val = a_data[i * n + j];
            work[i * n + j] = val;
            work[j * n + i] = val;
        }
    }

    // Initialize eigenvector matrix V as identity
    let mut v: Vec<T> = identity_matrix(n);

    // Convergence parameters
    let eps = T::epsilon_val();
    let tol = (n as f64) * eps;
    let max_sweeps = 30;

    // Jacobi iterations
    for _sweep in 0..max_sweeps {
        // Find maximum off-diagonal element
        let mut max_off_diag = 0.0f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = work[i * n + j].abs_val().to_f64();
                if val > max_off_diag {
                    max_off_diag = val;
                }
            }
        }

        // Check convergence
        if max_off_diag < tol {
            break;
        }

        // Process all element pairs (p, q) where p < q
        for p in 0..n {
            for q in (p + 1)..n {
                let a_pq = work[p * n + q];

                // Skip if already essentially zero
                if a_pq.abs_val().to_f64() < tol {
                    continue;
                }

                let a_pp = work[p * n + p];
                let a_qq = work[q * n + q];

                // Compute Jacobi rotation using stable LAPACK formula
                let rot = JacobiRotation::compute(a_pp.to_f64(), a_qq.to_f64(), a_pq.to_f64());

                // Apply two-sided rotation to work matrix: A' = J^T @ A @ J
                apply_two_sided_rotation(&mut work, n, p, q, &rot, a_pp, a_qq, a_pq);

                // Update eigenvector matrix: V = V @ J
                apply_rotation_to_columns(&mut v, n, n, p, q, &rot);
            }
        }
    }

    // Extract eigenvalues (diagonal of converged matrix)
    let mut eigenvalues: Vec<T> = vec![T::zero(); n];
    for i in 0..n {
        eigenvalues[i] = work[i * n + i];
    }

    // Sort eigenvalues by magnitude (descending) and reorder eigenvectors
    let indices = argsort_by_magnitude_desc(&eigenvalues);

    // Reorder eigenvalues and eigenvector columns
    let eigenvalues_sorted = jacobi::permute_vector(&eigenvalues, &indices);
    let v_sorted = permute_columns(&v, n, n, &indices, n);

    let eigenvalues_tensor = Tensor::<CpuRuntime>::from_slice(&eigenvalues_sorted, &[n], device)?;
    let eigenvectors_tensor = Tensor::<CpuRuntime>::from_slice(&v_sorted, &[n, n], device)?;

    Ok(EigenDecomposition {
        eigenvalues: eigenvalues_tensor,
        eigenvectors: eigenvectors_tensor,
    })
}
