//! Statistical matrix operations (pseudo-inverse, condition number, covariance, correlation)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::svd::svd_decompose_impl;
use crate::algorithm::linalg::{linalg_demote, linalg_promote, validate_matrix_2d};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Moore-Penrose pseudo-inverse via SVD: A^+ = V @ diag(1/S) @ U^T
pub fn pinverse_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    rcond: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "pinverse",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => pinverse_typed::<f32>(client, &a, m, n, rcond),
        DType::F64 => pinverse_typed::<f64>(client, &a, m, n, rcond),
        _ => unreachable!(),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn pinverse_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    rcond: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        return Tensor::<CpuRuntime>::try_from_slice::<T>(&[], &[n, m], device);
    }

    // Compute SVD: A = U @ S @ V^T
    let svd = svd_decompose_impl(client, a)?;
    let u_data: Vec<T> = svd.u.to_vec();
    let s_data: Vec<T> = svd.s.to_vec();
    let vt_data: Vec<T> = svd.vt.to_vec();

    let k = m.min(n);

    // Compute cutoff threshold
    let eps = T::epsilon_val();
    let max_s = if k > 0 {
        s_data.iter().map(|v| v.to_f64()).fold(0.0f64, f64::max)
    } else {
        1.0
    };
    let cutoff = rcond.unwrap_or_else(|| (m.max(n) as f64) * eps) * max_s;

    // Compute S^{-1} (reciprocal of singular values above cutoff)
    let s_inv: Vec<T> = s_data
        .iter()
        .map(|&s| {
            if s.to_f64() > cutoff {
                T::from_f64(1.0 / s.to_f64())
            } else {
                T::zero()
            }
        })
        .collect();

    // Compute A^+ = V @ diag(S^{-1}) @ U^T
    // V = (V^T)^T, so V[i,j] = vt_data[j * n + i]
    // Result has shape [n, m]

    let mut pinv: Vec<T> = vec![T::zero(); n * m];

    // pinv[i,j] = sum_l V[i,l] * S_inv[l] * U^T[l,j]
    //           = sum_l V[i,l] * S_inv[l] * U[j,l]
    for i in 0..n {
        for j in 0..m {
            let mut sum = T::zero();
            for l in 0..k {
                // V[i,l] = vt_data[l * n + i] (since V = V^T transposed)
                // U[j,l] = u_data[j * k + l]
                let v_il = vt_data[l * n + i];
                let u_jl = u_data[j * k + l];
                sum = sum + v_il * s_inv[l] * u_jl;
            }
            pinv[i * m + j] = sum;
        }
    }

    Tensor::<CpuRuntime>::try_from_slice(&pinv, &[n, m], device)
}

/// Condition number via SVD: cond(A) = σ_max / σ_min
pub fn cond_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "cond",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => cond_typed::<f32>(client, &a, m, n),
        DType::F64 => cond_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn cond_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        // Return infinity for empty matrix (undefined condition number)
        return Tensor::<CpuRuntime>::try_from_slice(&[T::from_f64(f64::INFINITY)], &[], device);
    }

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    if s_data.is_empty() {
        return Tensor::<CpuRuntime>::try_from_slice(&[T::from_f64(f64::INFINITY)], &[], device);
    }

    // Find max and min singular values (s is sorted descending)
    let s_max = s_data[0].to_f64();
    let s_min = s_data[s_data.len() - 1].to_f64();

    // Condition number = s_max / s_min
    // If s_min is essentially zero, return infinity
    let eps = T::epsilon_val();
    let cond = if s_min.abs() < eps {
        T::from_f64(f64::INFINITY)
    } else {
        T::from_f64(s_max / s_min)
    };

    Tensor::<CpuRuntime>::try_from_slice(&[cond], &[], device)
}

/// Covariance matrix
/// cov(X) = (X - mean(X))^T @ (X - mean(X)) / (n_samples - ddof)
pub fn cov_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    ddof: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "cov",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => cov_typed::<f32>(client, &a, n_samples, n_features, ddof),
        DType::F64 => cov_typed::<f64>(client, &a, n_samples, n_features, ddof),
        _ => unreachable!(),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn cov_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_features: usize,
    ddof: Option<usize>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let ddof = ddof.unwrap_or(1); // Default: unbiased (sample) covariance

    // Check we have enough samples
    if n_samples <= ddof {
        return Err(Error::Internal(format!(
            "cov: n_samples ({}) must be greater than ddof ({})",
            n_samples, ddof
        )));
    }

    // Handle edge case: single feature
    if n_features == 0 {
        return Tensor::<CpuRuntime>::try_from_slice::<T>(&[], &[0, 0], device);
    }

    let a_data: Vec<T> = a.to_vec();
    let n_f64 = n_samples as f64;

    // Step 1: Compute mean for each feature (column)
    let mut means: Vec<T> = vec![T::zero(); n_features];
    for j in 0..n_features {
        let mut sum = T::zero();
        for i in 0..n_samples {
            sum = sum + a_data[i * n_features + j];
        }
        means[j] = T::from_f64(sum.to_f64() / n_f64);
    }

    // Step 2: Compute centered data and covariance matrix
    // cov[i,j] = sum_k (X[k,i] - mean[i]) * (X[k,j] - mean[j]) / (n - ddof)
    let divisor = (n_samples - ddof) as f64;
    let mut cov: Vec<T> = vec![T::zero(); n_features * n_features];

    for i in 0..n_features {
        for j in i..n_features {
            // Exploit symmetry: only compute upper triangle
            let mut sum = T::zero();
            for k in 0..n_samples {
                let xi = a_data[k * n_features + i] - means[i];
                let xj = a_data[k * n_features + j] - means[j];
                sum = sum + xi * xj;
            }
            let cov_val = T::from_f64(sum.to_f64() / divisor);
            cov[i * n_features + j] = cov_val;
            cov[j * n_features + i] = cov_val; // Symmetry
        }
    }

    Tensor::<CpuRuntime>::try_from_slice(&cov, &[n_features, n_features], device)
}

/// Correlation coefficient matrix
/// corr[i,j] = cov[i,j] / (std[i] * std[j])
pub fn corrcoef_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    if !a.dtype().is_float() {
        return Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "corrcoef",
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => corrcoef_typed::<f32>(client, &a, n_samples, n_features),
        DType::F64 => corrcoef_typed::<f64>(client, &a, n_samples, n_features),
        _ => unreachable!(),
    }?;
    linalg_demote(client, result, original_dtype)
}

fn corrcoef_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n_samples: usize,
    n_features: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Check we have enough samples (need at least 2 for correlation)
    if n_samples <= 1 {
        return Err(Error::Internal(format!(
            "corrcoef: n_samples ({}) must be greater than 1",
            n_samples
        )));
    }

    // Handle edge case: single feature
    if n_features == 0 {
        return Tensor::<CpuRuntime>::try_from_slice::<T>(&[], &[0, 0], device);
    }

    // Compute covariance matrix (with ddof=1 for sample covariance)
    let cov_tensor = cov_typed::<T>(client, a, n_samples, n_features, Some(1))?;
    let cov_data: Vec<T> = cov_tensor.to_vec();

    // Extract standard deviations (sqrt of diagonal)
    let mut stds: Vec<f64> = vec![0.0; n_features];
    for i in 0..n_features {
        let var = cov_data[i * n_features + i].to_f64();
        stds[i] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }

    // Compute correlation matrix
    let mut corr: Vec<T> = vec![T::zero(); n_features * n_features];

    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                // Diagonal is always 1.0 (unless std is 0)
                corr[i * n_features + j] = if stds[i] > 0.0 { T::one() } else { T::zero() };
            } else {
                // Off-diagonal: cov[i,j] / (std[i] * std[j])
                let std_prod = stds[i] * stds[j];
                let corr_val = if std_prod > 0.0 {
                    T::from_f64(cov_data[i * n_features + j].to_f64() / std_prod)
                } else {
                    T::zero() // Undefined correlation set to 0
                };
                corr[i * n_features + j] = corr_val;
            }
        }
    }

    Tensor::<CpuRuntime>::try_from_slice(&corr, &[n_features, n_features], device)
}
