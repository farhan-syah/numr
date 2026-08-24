//! Covariance and correlation coefficient computations.

use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_matrix_2d,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, CompareOps, ConditionalOps, LinalgOps, MatmulOps, ReduceOps, UnaryOps, UtilityOps,
};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Compute covariance matrix.
///
/// Supports F32 and F64 dtypes. All computation runs on GPU.
pub fn cov(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    ddof: Option<usize>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (n_samples, _n_features) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();
    let ddof_val = ddof.unwrap_or(1);

    // Support F32 and F64
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op: "cov" });
    }

    // Need at least ddof + 1 samples
    if n_samples <= ddof_val {
        return Err(Error::Internal(format!(
            "cov: need at least {} samples for ddof={}, got {}",
            ddof_val + 1,
            ddof_val,
            n_samples
        )));
    }

    // Compute mean along axis 0
    let sum = client.sum(a, &[0], true)?;
    let n_samples_tensor = match dtype {
        DType::F32 => Tensor::<WgpuRuntime>::try_from_slice(&[n_samples as f32], &[], device)?,
        DType::F64 => Tensor::<WgpuRuntime>::try_from_slice(&[n_samples as f64], &[], device)?,
        _ => unreachable!(),
    };
    let mean = client.div(&sum, &n_samples_tensor)?;

    // Center the data
    let centered = client.sub(a, &mean)?;

    // Compute covariance: C = X_centered^T @ X_centered / (n - ddof)
    let centered_t = centered.transpose(0, 1)?.contiguous()?;
    let centered_contig = centered.contiguous()?;
    let cov_unnorm = client.matmul(&centered_t, &centered_contig)?;

    // Normalize by (n - ddof)
    let divisor_tensor = match dtype {
        DType::F32 => {
            Tensor::<WgpuRuntime>::try_from_slice(&[(n_samples - ddof_val) as f32], &[], device)?
        }
        DType::F64 => {
            Tensor::<WgpuRuntime>::try_from_slice(&[(n_samples - ddof_val) as f64], &[], device)?
        }
        _ => unreachable!(),
    };
    let cov_mat = client.div(&cov_unnorm, &divisor_tensor)?;

    Ok(cov_mat)
}

/// Compute correlation coefficient matrix.
///
/// Supports F32 and F64 dtypes. All computation runs on GPU (no CPU transfers).
pub fn corrcoef(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (n_samples, n_features) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Support F32 and F64
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "corrcoef",
        });
    }

    // Need at least 2 samples
    if n_samples < 2 {
        return Err(Error::Internal(format!(
            "corrcoef: need at least 2 samples, got {}",
            n_samples
        )));
    }

    // Handle edge case: no features
    if n_features == 0 {
        return match dtype {
            DType::F32 => Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device),
            DType::F64 => Tensor::<WgpuRuntime>::try_from_slice::<f64>(&[], &[0, 0], device),
            _ => unreachable!(),
        };
    }

    // Compute covariance matrix (all on GPU)
    let cov_mat = LinearAlgebraAlgorithms::cov(client, a, Some(1))?;

    // Extract diagonal (variances) and compute standard deviations
    let variances = LinalgOps::diag(client, &cov_mat)?;
    let std_devs = client.sqrt(&variances)?; // [n_features]

    // Create outer product of std_devs: std_outer[i,j] = std[i] * std[j]
    // Use broadcasting: [n,1] * [1,n] -> [n,n]
    let std_col = std_devs.reshape(&[n_features, 1])?; // [n, 1]
    let std_row = std_devs.reshape(&[1, n_features])?; // [1, n]
    let std_outer = client.mul(&std_col, &std_row)?; // [n, n] via broadcasting

    // Compute correlation: corr = cov / std_outer
    // But handle zero-variance (std_outer == 0) case
    let eps = match dtype {
        DType::F32 => Tensor::<WgpuRuntime>::try_from_slice(&[f32::EPSILON], &[], device)?,
        DType::F64 => Tensor::<WgpuRuntime>::try_from_slice(&[f64::EPSILON], &[], device)?,
        _ => unreachable!(),
    };

    // Create mask where std_outer > eps (both stds are positive)
    let mask = client.gt(&std_outer, &eps)?; // bool tensor

    // Compute raw correlation (will have inf/nan where std_outer is 0, but we'll mask those out)
    // Add eps to denominator to avoid division by zero
    let std_outer_safe = client.add(&std_outer, &eps)?;
    let corr_raw = client.div(&cov_mat, &std_outer_safe)?;

    // Clamp to [-1, 1]
    let corr_clamped = client.clamp(&corr_raw, -1.0, 1.0)?;

    // Zero out entries where std_outer <= eps
    let zero = match dtype {
        DType::F32 => Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[], device)?,
        DType::F64 => Tensor::<WgpuRuntime>::try_from_slice(&[0.0f64], &[], device)?,
        _ => unreachable!(),
    };
    let corr_masked = client.where_cond(&mask, &corr_clamped, &zero)?;

    // Set diagonal to 1.0 where variance > 0, else 0.0
    // diag_mask[i,i] = 1 if std_devs[i] > eps else 0
    let std_positive = client.gt(&std_devs, &eps)?; // [n_features] bool

    // Create identity matrix
    let identity = client.eye(n_features, Some(n_features), dtype)?;

    // Create diagonal ones: diag_vals[i,i] = 1.0 if std_devs[i] > 0
    let one = match dtype {
        DType::F32 => Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[], device)?,
        DType::F64 => Tensor::<WgpuRuntime>::try_from_slice(&[1.0f64], &[], device)?,
        _ => unreachable!(),
    };
    let diag_ones = client.where_cond(&std_positive, &one, &zero)?; // [n_features]
    let diag_matrix = LinalgOps::diagflat(client, &diag_ones)?; // [n, n]

    // Replace diagonal of corr_masked with diag_matrix values:
    // corr_final = (corr_masked - corr_masked * I) + diag_matrix
    // This zeros out the diagonal of corr_masked, then adds the proper diagonal
    let corr_diag_zeroed = client.mul(&corr_masked, &identity)?;
    let corr_no_diag = client.sub(&corr_masked, &corr_diag_zeroed)?;
    let corr_final = client.add(&corr_no_diag, &diag_matrix)?;

    Ok(corr_final)
}
