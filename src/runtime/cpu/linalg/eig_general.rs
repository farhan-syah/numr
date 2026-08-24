//! General eigendecomposition for non-symmetric matrices

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::schur::schur_decompose_impl;
use crate::algorithm::linalg::{
    GeneralEigenDecomposition, linalg_demote, linalg_promote, validate_linalg_dtype,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::Result;
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// General eigendecomposition for non-symmetric matrices
///
/// Uses Schur decomposition followed by eigenvector extraction via back-substitution.
pub fn eig_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<GeneralEigenDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => eig_decompose_typed::<f32>(client, &a, n),
        DType::F64 => eig_decompose_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(GeneralEigenDecomposition {
        eigenvalues_real: linalg_demote(client, result.eigenvalues_real, original_dtype)?,
        eigenvalues_imag: linalg_demote(client, result.eigenvalues_imag, original_dtype)?,
        eigenvectors_real: linalg_demote(client, result.eigenvectors_real, original_dtype)?,
        eigenvectors_imag: linalg_demote(client, result.eigenvectors_imag, original_dtype)?,
    })
}

fn eig_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<GeneralEigenDecomposition<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0], device)?,
            eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0], device)?,
            eigenvectors_real: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
            eigenvectors_imag: Tensor::<CpuRuntime>::try_from_slice(&[] as &[T], &[0, 0], device)?,
        });
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(&data, &[1], device)?,
            eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&[T::zero()], &[1], device)?,
            eigenvectors_real: Tensor::<CpuRuntime>::try_from_slice(&[T::one()], &[1, 1], device)?,
            eigenvectors_imag: Tensor::<CpuRuntime>::try_from_slice(&[T::zero()], &[1, 1], device)?,
        });
    }

    // Step 1: Compute Schur decomposition A = Z @ T @ Z^T
    let schur = schur_decompose_impl(client, a)?;
    let z_data: Vec<T> = schur.z.to_vec();
    let t_data: Vec<T> = schur.t.to_vec();

    // Step 2: Extract eigenvalues from the quasi-triangular Schur form T
    let mut eigenvalues_real = vec![T::zero(); n];
    let mut eigenvalues_imag = vec![T::zero(); n];

    let mut i = 0;
    while i < n {
        if i == n - 1 {
            // Last diagonal element is a real eigenvalue
            eigenvalues_real[i] = t_data[i * n + i];
            eigenvalues_imag[i] = T::zero();
            i += 1;
        } else {
            // Check if this is a 2x2 block (complex conjugate pair)
            let subdiag = t_data[(i + 1) * n + i].to_f64().abs();
            let eps = T::epsilon_val();
            let diag_scale =
                t_data[i * n + i].to_f64().abs() + t_data[(i + 1) * n + (i + 1)].to_f64().abs();
            let threshold = eps * diag_scale.max(1.0);

            if subdiag > threshold {
                // 2x2 block: extract complex conjugate pair
                // [ a  b ]
                // [ c  d ]  where eigenvalues are (a+d)/2 ± sqrt((a-d)²/4 + bc)
                let a_val = t_data[i * n + i].to_f64();
                let b_val = t_data[i * n + (i + 1)].to_f64();
                let c_val = t_data[(i + 1) * n + i].to_f64();
                let d_val = t_data[(i + 1) * n + (i + 1)].to_f64();

                let trace = a_val + d_val;
                let disc = (a_val - d_val) * (a_val - d_val) / 4.0 + b_val * c_val;

                if disc < 0.0 {
                    // Complex eigenvalues
                    let real_part = trace / 2.0;
                    let imag_part = (-disc).sqrt();
                    eigenvalues_real[i] = T::from_f64(real_part);
                    eigenvalues_imag[i] = T::from_f64(imag_part);
                    eigenvalues_real[i + 1] = T::from_f64(real_part);
                    eigenvalues_imag[i + 1] = T::from_f64(-imag_part);
                } else {
                    // Real eigenvalues (shouldn't happen for properly detected 2x2 block)
                    let sqrt_disc = disc.sqrt();
                    eigenvalues_real[i] = T::from_f64(trace / 2.0 + sqrt_disc);
                    eigenvalues_imag[i] = T::zero();
                    eigenvalues_real[i + 1] = T::from_f64(trace / 2.0 - sqrt_disc);
                    eigenvalues_imag[i + 1] = T::zero();
                }
                i += 2;
            } else {
                // 1x1 block: real eigenvalue
                eigenvalues_real[i] = t_data[i * n + i];
                eigenvalues_imag[i] = T::zero();
                i += 1;
            }
        }
    }

    // Step 3: Compute eigenvectors via back-substitution
    // For each eigenvalue λ, solve (T - λI)y = 0, then V = Z @ Y
    let mut eigenvectors_real = vec![T::zero(); n * n];
    let mut eigenvectors_imag = vec![T::zero(); n * n];

    i = 0;
    while i < n {
        let imag = eigenvalues_imag[i].to_f64();

        if imag.abs() < T::epsilon_val() {
            // Real eigenvalue: solve (T - λI)y = 0 via back-substitution
            let lambda = eigenvalues_real[i].to_f64();
            let y = solve_schur_eigenvector_real::<T>(&t_data, n, i, lambda);

            // Transform back: v = Z @ y
            for row in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += z_data[row * n + k].to_f64() * y[k];
                }
                eigenvectors_real[row * n + i] = T::from_f64(sum);
                eigenvectors_imag[row * n + i] = T::zero();
            }
            i += 1;
        } else {
            // Complex conjugate pair
            let lambda_real = eigenvalues_real[i].to_f64();
            let lambda_imag = eigenvalues_imag[i].to_f64();

            let (y_real, y_imag) =
                solve_schur_eigenvector_complex::<T>(&t_data, n, i, lambda_real, lambda_imag);

            // Transform back: v = Z @ y (for both conjugate eigenvectors)
            for row in 0..n {
                let mut sum_real = 0.0;
                let mut sum_imag = 0.0;
                for k in 0..n {
                    let z_val = z_data[row * n + k].to_f64();
                    sum_real += z_val * y_real[k];
                    sum_imag += z_val * y_imag[k];
                }
                // First eigenvector: v = u + iw
                eigenvectors_real[row * n + i] = T::from_f64(sum_real);
                eigenvectors_imag[row * n + i] = T::from_f64(sum_imag);
                // Second eigenvector: v* = u - iw
                eigenvectors_real[row * n + (i + 1)] = T::from_f64(sum_real);
                eigenvectors_imag[row * n + (i + 1)] = T::from_f64(-sum_imag);
            }
            i += 2;
        }
    }

    Ok(GeneralEigenDecomposition {
        eigenvalues_real: Tensor::<CpuRuntime>::try_from_slice(&eigenvalues_real, &[n], device)?,
        eigenvalues_imag: Tensor::<CpuRuntime>::try_from_slice(&eigenvalues_imag, &[n], device)?,
        eigenvectors_real: Tensor::<CpuRuntime>::try_from_slice(
            &eigenvectors_real,
            &[n, n],
            device,
        )?,
        eigenvectors_imag: Tensor::<CpuRuntime>::try_from_slice(
            &eigenvectors_imag,
            &[n, n],
            device,
        )?,
    })
}

/// Solve (T - λI)y = 0 for a real eigenvalue λ via back-substitution.
/// Returns the eigenvector y in the Schur basis.
fn solve_schur_eigenvector_real<T: Element + LinalgElement>(
    t: &[T],
    n: usize,
    eig_idx: usize,
    lambda: f64,
) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let eps = T::epsilon_val();

    // Start with y[eig_idx] = 1 as the reference
    y[eig_idx] = 1.0;

    // Back-substitute for rows above eig_idx
    // (T - λI)[k, :] · y = 0 for k < eig_idx
    // t[k,k]-λ)y[k] + sum_{j>k}(t[k,j]y[j]) = 0
    for k in (0..eig_idx).rev() {
        let diag = t[k * n + k].to_f64() - lambda;
        let mut rhs = 0.0;
        for j in (k + 1)..n {
            rhs -= t[k * n + j].to_f64() * y[j];
        }

        if diag.abs() > eps {
            y[k] = rhs / diag;
        } else {
            // Near-zero diagonal: handle defective case
            y[k] = 0.0;
        }
    }

    // Normalize the eigenvector
    let mut norm_sq = 0.0;
    for yi in &y {
        norm_sq += yi * yi;
    }
    let norm = norm_sq.sqrt();
    if norm > eps {
        for yi in &mut y {
            *yi /= norm;
        }
    }

    y
}

/// Solve (T - λI)y = 0 for a complex eigenvalue λ = λ_r + i*λ_i.
/// Returns (y_real, y_imag) for the complex eigenvector y = y_real + i*y_imag.
fn solve_schur_eigenvector_complex<T: Element + LinalgElement>(
    t: &[T],
    n: usize,
    eig_idx: usize,
    lambda_real: f64,
    lambda_imag: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut y_real = vec![0.0; n];
    let mut y_imag = vec![0.0; n];
    let eps = T::epsilon_val();

    // For a 2x2 block at (eig_idx, eig_idx), use the structure of the block
    // to find the eigenvector direction
    let a = t[eig_idx * n + eig_idx].to_f64();
    let b = t[eig_idx * n + (eig_idx + 1)].to_f64();

    // Initial guess from the 2x2 block structure
    // (A - λI)v = 0 where λ = a + i*sqrt(-bc) for appropriate sign
    // This gives v = [b, λ - a] or normalized version
    y_real[eig_idx] = b;
    y_imag[eig_idx] = 0.0;
    y_real[eig_idx + 1] = lambda_real - a;
    y_imag[eig_idx + 1] = lambda_imag;

    // Back-substitute for rows above eig_idx
    // Need to solve complex system: (T - λI)y = 0
    for k in (0..eig_idx).rev() {
        let diag_real = t[k * n + k].to_f64() - lambda_real;
        let diag_imag = -lambda_imag;

        let mut rhs_real = 0.0;
        let mut rhs_imag = 0.0;

        for j in (k + 1)..n {
            let t_kj = t[k * n + j].to_f64();
            rhs_real -= t_kj * y_real[j];
            rhs_imag -= t_kj * y_imag[j];
        }

        // Solve: (diag_real + i*diag_imag) * y[k] = rhs_real + i*rhs_imag
        // y[k] = (rhs_real + i*rhs_imag) / (diag_real + i*diag_imag)
        let denom = diag_real * diag_real + diag_imag * diag_imag;
        if denom > eps * eps {
            y_real[k] = (rhs_real * diag_real + rhs_imag * diag_imag) / denom;
            y_imag[k] = (rhs_imag * diag_real - rhs_real * diag_imag) / denom;
        } else {
            y_real[k] = 0.0;
            y_imag[k] = 0.0;
        }
    }

    // Normalize the eigenvector
    let mut norm_sq = 0.0;
    for i in 0..n {
        norm_sq += y_real[i] * y_real[i] + y_imag[i] * y_imag[i];
    }
    let norm = norm_sq.sqrt();
    if norm > eps {
        for i in 0..n {
            y_real[i] /= norm;
            y_imag[i] /= norm;
        }
    }

    (y_real, y_imag)
}
