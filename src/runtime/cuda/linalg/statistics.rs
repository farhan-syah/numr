//! Statistical operations for CUDA (pinverse, cond, cov, corrcoef)
//!
//! Uses linalg_promote/linalg_demote to handle reduced-precision types (F16, BF16, FP8)
//! by promoting to F32 before computation and demoting back afterward.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use crate::algorithm::linalg::helpers::{linalg_demote, linalg_promote};
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, MatmulOps, ReduceOps, UnaryOps};
use crate::runtime::{Allocator, RuntimeClient};
use crate::tensor::Tensor;

/// Moore-Penrose pseudoinverse via SVD
pub fn pinverse_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    rcond: Option<f64>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Promote reduced-precision types to F32
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;

    let (m, n) = validate_matrix_2d(a_promoted.shape())?;
    let dtype = a_promoted.dtype();
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        let out_ptr = client.allocator().allocate(0)?;
        let result =
            unsafe { CudaClient::tensor_from_raw(out_ptr, &[n, m], original_dtype, device) };
        return Ok(result);
    }

    // Compute SVD: A = U @ diag(S) @ V^T
    let svd = client.svd_decompose(&a_promoted)?;

    // Get singular values to determine cutoff
    let k = m.min(n);
    let s_data: Vec<f64> = match dtype {
        DType::F32 => svd
            .s
            .to_vec::<f32>()
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        DType::F64 => svd.s.to_vec::<f64>(),
        _ => unreachable!(), // linalg_promote ensures F32 or F64
    };

    // Determine cutoff threshold
    let max_sv = s_data.iter().cloned().fold(0.0_f64, f64::max);
    let default_rcond = (m.max(n) as f64)
        * match dtype {
            DType::F32 => f32::EPSILON as f64,
            DType::F64 => f64::EPSILON,
            _ => f32::EPSILON as f64,
        };
    let rcond_val = rcond.unwrap_or(default_rcond);
    let cutoff = rcond_val * max_sv;

    // Compute S_inv: invert singular values above cutoff, zero otherwise
    let s_inv_data: Vec<f64> = s_data
        .iter()
        .map(|&s| if s > cutoff { 1.0 / s } else { 0.0 })
        .collect();

    // Create S_inv diagonal matrix [k x k] on GPU
    let s_inv_diag = match dtype {
        DType::F32 => {
            let s_inv_f32: Vec<f32> = s_inv_data.iter().map(|&x| x as f32).collect();
            Tensor::<CudaRuntime>::from_slice(&s_inv_f32, &[k], device)?
        }
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&s_inv_data, &[k], device)?,
        _ => unreachable!(),
    };

    // Create diagonal matrix from vector
    let s_inv_mat = LinearAlgebraAlgorithms::diagflat(client, &s_inv_diag)?;

    // Compute A^+ = V @ S_inv @ U^T
    let v = svd.vt.transpose(0, 1)?;
    let ut = svd.u.transpose(0, 1)?;
    let v_sinv = client.matmul(&v, &s_inv_mat)?;
    let pinv = client.matmul(&v_sinv, &ut)?;

    linalg_demote(client, pinv, original_dtype)
}

/// Condition number via SVD
pub fn cond_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Promote reduced-precision types to F32
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;

    let (m, n) = validate_matrix_2d(a_promoted.shape())?;
    let dtype = a_promoted.dtype();
    let device = client.device();

    // Handle empty matrix
    if m == 0 || n == 0 {
        let inf_val = match dtype {
            DType::F32 => Tensor::<CudaRuntime>::from_slice(&[f32::INFINITY], &[], device)?,
            DType::F64 => Tensor::<CudaRuntime>::from_slice(&[f64::INFINITY], &[], device)?,
            _ => unreachable!(),
        };
        return linalg_demote(client, inf_val, original_dtype);
    }

    // Compute SVD to get singular values
    let svd = client.svd_decompose(&a_promoted)?;

    // Get singular values
    let s_data: Vec<f64> = match dtype {
        DType::F32 => svd
            .s
            .to_vec::<f32>()
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        DType::F64 => svd.s.to_vec::<f64>(),
        _ => unreachable!(),
    };

    // Condition number = max(S) / min(S)
    let max_sv = s_data.iter().cloned().fold(0.0_f64, f64::max);
    let min_sv = s_data.iter().cloned().fold(f64::INFINITY, f64::min);

    let cond_val = if min_sv == 0.0 || !min_sv.is_finite() {
        f64::INFINITY
    } else {
        max_sv / min_sv
    };

    // Create result tensor
    let result = match dtype {
        DType::F32 => Tensor::<CudaRuntime>::from_slice(&[cond_val as f32], &[], device)?,
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&[cond_val], &[], device)?,
        _ => unreachable!(),
    };

    linalg_demote(client, result, original_dtype)
}

/// Covariance matrix
pub fn cov_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    ddof: Option<usize>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Promote reduced-precision types to F32
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;

    let (n_samples, _n_features) = validate_matrix_2d(a_promoted.shape())?;
    let dtype = a_promoted.dtype();
    let device = client.device();
    let ddof_val = ddof.unwrap_or(1);

    // Need at least ddof + 1 samples
    if n_samples <= ddof_val {
        return Err(crate::error::Error::Internal(format!(
            "cov: need at least {} samples for ddof={}, got {}",
            ddof_val + 1,
            ddof_val,
            n_samples
        )));
    }

    // Compute mean along axis 0 (mean of each column/feature)
    let sum = client.sum(&a_promoted, &[0], true)?; // [1, n_features]
    let n_samples_tensor = match dtype {
        DType::F32 => Tensor::<CudaRuntime>::from_slice(&[n_samples as f32], &[], device)?,
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&[n_samples as f64], &[], device)?,
        _ => unreachable!(),
    };
    let mean = client.div(&sum, &n_samples_tensor)?; // [1, n_features]

    // Center the data: X_centered = X - mean (broadcast subtraction)
    let centered = client.sub(&a_promoted, &mean)?; // [n_samples, n_features]

    // Compute covariance: C = X_centered^T @ X_centered / (n - ddof)
    let centered_t = centered.transpose(0, 1)?; // [n_features, n_samples]
    let cov_unnorm = client.matmul(&centered_t, &centered)?; // [n_features, n_features]

    // Normalize by (n - ddof)
    let divisor = (n_samples - ddof_val) as f64;
    let divisor_tensor = match dtype {
        DType::F32 => Tensor::<CudaRuntime>::from_slice(&[divisor as f32], &[], device)?,
        DType::F64 => Tensor::<CudaRuntime>::from_slice(&[divisor], &[], device)?,
        _ => unreachable!(),
    };
    let cov_mat = client.div(&cov_unnorm, &divisor_tensor)?;

    linalg_demote(client, cov_mat, original_dtype)
}

/// Correlation coefficient matrix
pub fn corrcoef_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Promote reduced-precision types to F32
    let (a_promoted, original_dtype) = linalg_promote(client, a)?;

    let (n_samples, n_features) = validate_matrix_2d(a_promoted.shape())?;
    let dtype = a_promoted.dtype();
    let device = client.device();

    // Need at least 2 samples
    if n_samples < 2 {
        return Err(crate::error::Error::Internal(format!(
            "corrcoef: need at least 2 samples, got {}",
            n_samples
        )));
    }

    // Handle edge case: no features
    if n_features == 0 {
        return Tensor::<CudaRuntime>::from_slice::<f32>(&[], &[0, 0], device);
    }

    // Compute covariance matrix (already in promoted dtype)
    let cov_mat = LinearAlgebraAlgorithms::cov(client, &a_promoted, Some(1))?;

    // Extract diagonal (variances) and compute standard deviations
    let variances = LinearAlgebraAlgorithms::diag(client, &cov_mat)?; // [n_features]
    let std_devs = client.sqrt(&variances)?; // [n_features]

    // Pull std_devs to CPU for zero-variance detection
    let std_vec: Vec<f64> = match dtype {
        DType::F32 => std_devs
            .to_vec::<f32>()
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        DType::F64 => std_devs.to_vec::<f64>(),
        _ => unreachable!(),
    };

    // Build correlation matrix with proper zero-variance handling
    let cov_vec: Vec<f64> = match dtype {
        DType::F32 => cov_mat
            .to_vec::<f32>()
            .into_iter()
            .map(|x| x as f64)
            .collect(),
        DType::F64 => cov_mat.to_vec::<f64>(),
        _ => unreachable!(),
    };

    let mut corr_data: Vec<f64> = vec![0.0; n_features * n_features];
    for i in 0..n_features {
        for j in 0..n_features {
            if i == j {
                corr_data[i * n_features + j] = if std_vec[i] > 0.0 { 1.0 } else { 0.0 };
            } else {
                let std_prod = std_vec[i] * std_vec[j];
                corr_data[i * n_features + j] = if std_prod > 0.0 {
                    (cov_vec[i * n_features + j] / std_prod).clamp(-1.0, 1.0)
                } else {
                    0.0
                };
            }
        }
    }

    // Convert back to working dtype
    let result = match dtype {
        DType::F32 => {
            let corr_f32: Vec<f32> = corr_data.iter().map(|&x| x as f32).collect();
            Tensor::<CudaRuntime>::from_slice(&corr_f32, &[n_features, n_features], device)?
        }
        DType::F64 => {
            Tensor::<CudaRuntime>::from_slice(&corr_data, &[n_features, n_features], device)?
        }
        _ => unreachable!(),
    };

    linalg_demote(client, result, original_dtype)
}
