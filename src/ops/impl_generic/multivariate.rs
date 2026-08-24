//! Generic implementations of multivariate random distribution operations.
//!
//! These implementations are shared across all backends to ensure numerical parity.
//! All operations stay entirely on the device - NO GPU-to-CPU transfers of sample data.
//!
//! # Parameter vs Sample Data
//!
//! - **Parameters** (alpha, probs, scale): Small user-provided inputs. May be extracted
//!   to CPU for validation since they're typically tiny (k categories, d dimensions).
//! - **Sample data**: Generated random values. NEVER transferred to CPU.

use crate::algorithm::linalg::LinearAlgebraAlgorithms;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, MatmulOps, RandomOps, ReduceOps, ShapeOps, UnaryOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// DType Support Configuration
// ============================================================================

/// Configuration for dtype support per backend.
#[derive(Debug, Clone, Copy)]
pub struct DTypeSupport {
    /// Whether F64 is supported (false for WebGPU)
    pub f64_supported: bool,
}

impl DTypeSupport {
    /// CPU and CUDA support both F32 and F64
    pub const FULL: Self = Self {
        f64_supported: true,
    };

    /// WebGPU only supports F32
    #[allow(dead_code)]
    pub const F32_ONLY: Self = Self {
        f64_supported: false,
    };
}

// ============================================================================
// Validation Helpers (parameter extraction is OK - these are small user inputs)
// ============================================================================

fn validate_multivariate_normal_inputs<R: Runtime<DType = DType>>(
    mean: &Tensor<R>,
    cov: &Tensor<R>,
    n_samples: usize,
    dtype_support: DTypeSupport,
) -> Result<usize> {
    let dtype = mean.dtype();

    if dtype_support.f64_supported {
        if dtype != DType::F32 && dtype != DType::F64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "multivariate_normal",
            });
        }
    } else if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "multivariate_normal (F32 only on this backend)",
        });
    }

    if cov.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            lhs: dtype,
            rhs: cov.dtype(),
        });
    }

    let mean_shape = mean.shape();
    if mean_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "mean",
            reason: format!("mean must be 1D, got shape {:?}", mean_shape),
        });
    }
    let d = mean_shape[0];

    let cov_shape = cov.shape();
    if cov_shape.len() != 2 || cov_shape[0] != cov_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "cov",
            reason: format!("cov must be a square 2D matrix, got shape {:?}", cov_shape),
        });
    }
    if cov_shape[0] != d {
        return Err(Error::InvalidArgument {
            arg: "cov",
            reason: format!(
                "cov dimension {} must match mean dimension {}",
                cov_shape[0], d
            ),
        });
    }

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: "n_samples must be > 0".to_string(),
        });
    }

    Ok(d)
}

fn validate_wishart_inputs<R: Runtime<DType = DType>>(
    scale: &Tensor<R>,
    df: usize,
    n_samples: usize,
    dtype_support: DTypeSupport,
) -> Result<usize> {
    let dtype = scale.dtype();

    if dtype_support.f64_supported {
        if dtype != DType::F32 && dtype != DType::F64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "wishart",
            });
        }
    } else if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "wishart (F32 only on this backend)",
        });
    }

    let scale_shape = scale.shape();
    if scale_shape.len() != 2 || scale_shape[0] != scale_shape[1] {
        return Err(Error::InvalidArgument {
            arg: "scale",
            reason: format!(
                "scale must be a square 2D matrix, got shape {:?}",
                scale_shape
            ),
        });
    }
    let d = scale_shape[0];

    if df < d {
        return Err(Error::InvalidArgument {
            arg: "df",
            reason: format!(
                "degrees of freedom {} must be >= matrix dimension {}",
                df, d
            ),
        });
    }

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: "n_samples must be > 0".to_string(),
        });
    }

    Ok(d)
}

/// Validate dirichlet inputs. Extracts alpha values (small parameter vector).
fn validate_dirichlet_inputs<R: Runtime<DType = DType>>(
    alpha: &Tensor<R>,
    n_samples: usize,
) -> Result<(usize, Vec<f64>)> {
    let dtype = alpha.dtype();

    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "dirichlet",
        });
    }

    let alpha_shape = alpha.shape();
    if alpha_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "alpha",
            reason: format!("alpha must be 1D, got shape {:?}", alpha_shape),
        });
    }
    let k = alpha_shape[0];

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: "n_samples must be > 0".to_string(),
        });
    }

    // Extract alpha parameters (small user-provided vector, OK to transfer)
    let alpha_data: Vec<f64> = match dtype {
        DType::F32 => alpha.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
        DType::F64 => alpha.to_vec::<f64>(),
        _ => alpha.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
    };

    for (i, &a) in alpha_data.iter().enumerate() {
        if a <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "alpha",
                reason: format!("all alpha values must be > 0, got alpha[{}] = {}", i, a),
            });
        }
    }

    Ok((k, alpha_data))
}

/// Validate multinomial inputs. Extracts probs and computes CDF (small parameter vector).
fn validate_multinomial_inputs<R: Runtime<DType = DType>>(
    probs: &Tensor<R>,
    n_trials: usize,
    n_samples: usize,
) -> Result<usize> {
    let dtype = probs.dtype();

    if !dtype.is_float() {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "multinomial_samples",
        });
    }

    let probs_shape = probs.shape();
    if probs_shape.len() != 1 {
        return Err(Error::InvalidArgument {
            arg: "probs",
            reason: format!("probs must be 1D, got shape {:?}", probs_shape),
        });
    }
    let k = probs_shape[0];

    if n_trials == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_trials",
            reason: "n_trials must be > 0".to_string(),
        });
    }

    if n_samples == 0 {
        return Err(Error::InvalidArgument {
            arg: "n_samples",
            reason: "n_samples must be > 0".to_string(),
        });
    }

    if k == 0 {
        return Err(Error::InvalidArgument {
            arg: "probs",
            reason: "probs must have at least 1 category".to_string(),
        });
    }

    Ok(k)
}

// ============================================================================
// Algorithm Implementations - ALL OPERATIONS STAY ON GPU
// ============================================================================

/// Multivariate normal sampling: X ~ N(μ, Σ)
///
/// Algorithm: Cholesky decomposition + linear transform
/// 1. Σ = L @ L^T (Cholesky)
/// 2. Z ~ N(0, I) with shape (n_samples, d)
/// 3. X = μ + Z @ L^T
///
/// ALL OPERATIONS ON GPU - no data transfers.
pub fn multivariate_normal_impl<R, C>(
    client: &C,
    mean: &Tensor<R>,
    cov: &Tensor<R>,
    n_samples: usize,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + MatmulOps<R> + BinaryOps<R> + RandomOps<R>,
{
    let d = validate_multivariate_normal_inputs(mean, cov, n_samples, dtype_support)?;
    let dtype = mean.dtype();

    if d == 0 {
        return Ok(Tensor::<R>::empty(&[n_samples, 0], dtype, mean.device())?);
    }

    // All GPU operations
    let chol = client.cholesky_decompose(cov)?;
    let l = &chol.l;
    let z = client.randn(&[n_samples, d], dtype)?;
    let l_t = l.transpose(-2, -1)?;
    let zl = client.matmul(&z, &l_t)?;
    let mean_expanded = mean.unsqueeze(0)?;
    client.add(&zl, &mean_expanded)
}

/// Wishart distribution sampling: W ~ W(V, df)
///
/// Algorithm: Bartlett decomposition
/// 1. V = L @ L^T (Cholesky)
/// 2. Construct Bartlett A matrices (lower triangular):
///    - Diagonal: sqrt(χ²(df-i)) for i = 0..d
///    - Lower triangular: N(0,1)
/// 3. W = L @ A @ A^T @ L^T
///
/// ALL OPERATIONS ON GPU using batched tensor operations.
pub fn wishart_impl<R, C>(
    client: &C,
    scale: &Tensor<R>,
    df: usize,
    n_samples: usize,
    dtype_support: DTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R>
        + MatmulOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + RandomOps<R>
        + ShapeOps<R>
        + ReduceOps<R>,
{
    let d = validate_wishart_inputs(scale, df, n_samples, dtype_support)?;
    let dtype = scale.dtype();

    if d == 0 {
        return Ok(Tensor::<R>::empty(
            &[n_samples, 0, 0],
            dtype,
            scale.device(),
        )?);
    }

    // Step 1: Cholesky decomposition of scale matrix (GPU)
    let chol = client.cholesky_decompose(scale)?;
    let l_scale = &chol.l;

    // Step 2: Generate all random samples on GPU
    // Diagonal elements: d tensors of chi-squared with different df
    let mut diag_tensors: Vec<Tensor<R>> = Vec::with_capacity(d);
    for i in 0..d {
        let chi2_df = (df - i) as f64;
        let chi2_samples = client.chi_squared(chi2_df, &[n_samples], dtype)?;
        // Take sqrt for Bartlett decomposition
        let sqrt_chi2 = client.sqrt(&chi2_samples)?;
        diag_tensors.push(sqrt_chi2);
    }

    // Lower triangular elements: one tensor of shape [n_samples, d*(d-1)/2]
    let n_lower = d * (d - 1) / 2;
    let lower_samples = if n_lower > 0 {
        Some(client.randn(&[n_samples, n_lower], dtype)?)
    } else {
        None
    };

    // Step 3: Construct Bartlett A matrices [n_samples, d, d]
    // We build this using tensor operations to stay on GPU
    let a_matrices = construct_bartlett_matrices(
        client,
        &diag_tensors,
        lower_samples.as_ref(),
        n_samples,
        d,
        dtype,
        scale.device(),
    )?;

    // Step 4: Compute W = L @ A @ A^T @ L^T for each sample
    // Expand L_scale to [n_samples, d, d] by broadcasting
    let l_expanded = l_scale.unsqueeze(0)?.broadcast_to(&[n_samples, d, d])?;

    // LA = L @ A (batched matmul)
    let la = client.matmul(&l_expanded, &a_matrices)?;

    // LA_T = transpose of LA
    let la_t = la.transpose(-2, -1)?;

    // W = LA @ LA^T
    client.matmul(&la, &la_t)
}

/// Helper to construct Bartlett A matrices on GPU using tensor operations.
fn construct_bartlett_matrices<R, C>(
    client: &C,
    diag_tensors: &[Tensor<R>],
    lower_samples: Option<&Tensor<R>>,
    n_samples: usize,
    d: usize,
    dtype: DType,
    device: &R::Device,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: BinaryOps<R> + ShapeOps<R>,
{
    // We need to place values at specific positions.
    // Build row by row and stack.

    // Build each row of A and stack them
    let mut rows: Vec<Tensor<R>> = Vec::with_capacity(d);

    let mut lower_idx = 0;
    for i in 0..d {
        // Row i has:
        // - positions 0..i: values from lower_samples
        // - position i: value from diag_tensors[i]
        // - positions i+1..d: zeros

        let mut row_parts: Vec<Tensor<R>> = Vec::with_capacity(d);

        // Lower triangular part (columns 0..i)
        for _j in 0..i {
            if let Some(lower) = lower_samples {
                // Extract column lower_idx from lower_samples
                // lower_samples has shape [n_samples, n_lower]
                // narrow(dim=1, start=lower_idx, length=1) gives [n_samples, 1]
                let col = lower.narrow(1, lower_idx, 1)?;
                row_parts.push(col);
                lower_idx += 1;
            } else {
                // Should not happen if i > 0
                row_parts.push(Tensor::<R>::zeros(&[n_samples, 1], dtype, device)?);
            }
        }

        // Diagonal element (column i)
        let diag_col = diag_tensors[i].unsqueeze(1)?; // [n_samples] -> [n_samples, 1]
        row_parts.push(diag_col);

        // Upper triangular part (columns i+1..d) - zeros
        for _j in (i + 1)..d {
            row_parts.push(Tensor::<R>::zeros(&[n_samples, 1], dtype, device)?);
        }

        // Concatenate row parts along dim 1 to get row shape [n_samples, d]
        let row_refs: Vec<&Tensor<R>> = row_parts.iter().collect();
        let row = client.cat(&row_refs, 1)?;
        rows.push(row);
    }

    // Stack rows along a new dimension to get [n_samples, d, d]
    // First unsqueeze each row to [n_samples, 1, d], then cat along dim 1
    let mut row_expanded: Vec<Tensor<R>> = Vec::with_capacity(d);
    for row in rows {
        row_expanded.push(row.unsqueeze(1)?);
    }
    let row_refs: Vec<&Tensor<R>> = row_expanded.iter().collect();
    client.cat(&row_refs, 1)
}

/// Dirichlet distribution sampling: X ~ Dir(α)
///
/// Algorithm: Gamma sampling + normalization
/// 1. Generate Y_i ~ Gamma(α_i, 1) for each category
/// 2. Normalize: X_i = Y_i / sum(Y)
///
/// ALL OPERATIONS ON GPU - only alpha parameters extracted (small user input).
pub fn dirichlet_impl<R, C>(client: &C, alpha: &Tensor<R>, n_samples: usize) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RandomOps<R> + ReduceOps<R> + BinaryOps<R> + ShapeOps<R>,
{
    let (k, alpha_data) = validate_dirichlet_inputs(alpha, n_samples)?;
    let dtype = alpha.dtype();

    if k == 0 {
        return Ok(Tensor::<R>::empty(&[n_samples, 0], dtype, alpha.device())?);
    }

    // Generate gamma samples for each category ON GPU
    let mut gamma_tensors: Vec<Tensor<R>> = Vec::with_capacity(k);
    for i in 0..k {
        // Each gamma_col has shape [n_samples]
        let gamma_col = client.gamma(alpha_data[i], 1.0, &[n_samples], dtype)?;
        // Unsqueeze to [n_samples, 1] for stacking
        gamma_tensors.push(gamma_col.unsqueeze(1)?);
    }

    // Concatenate along dim 1 to get [n_samples, k] - ALL ON GPU
    let gamma_refs: Vec<&Tensor<R>> = gamma_tensors.iter().collect();
    let gamma_samples = client.cat(&gamma_refs, 1)?;

    // Sum across categories (axis 1) with keepdim=true - ON GPU
    let sum_gamma = client.sum(&gamma_samples, &[1], true)?;

    // Normalize: X = Y / sum(Y) - ON GPU
    client.div(&gamma_samples, &sum_gamma)
}

/// Multinomial distribution sampling: X ~ Multinomial(probs, n_trials)
///
/// Algorithm: CDF-based categorical sampling
/// 1. Compute CDF from probabilities
/// 2. Generate uniform samples
/// 3. Map uniforms to categories via CDF lookup (GPU kernel)
/// 4. Count occurrences per category
///
/// Requires MultinomialSamplingOps trait for the GPU kernel.
pub fn multinomial_samples_impl<R, C>(
    client: &C,
    probs: &Tensor<R>,
    n_trials: usize,
    n_samples: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MultinomialSamplingOps<R>,
{
    let k = validate_multinomial_inputs(probs, n_trials, n_samples)?;
    let dtype = probs.dtype();

    if k == 0 {
        return Ok(Tensor::<R>::empty(&[n_samples, 0], dtype, probs.device())?);
    }

    // Delegate to backend-specific kernel that does:
    // 1. Normalize probs and compute CDF on GPU
    // 2. Generate uniforms on GPU
    // 3. CDF lookup and counting on GPU
    client.multinomial_sample_kernel(probs, n_trials, n_samples)
}

// ============================================================================
// Required Trait for Multinomial Kernel
// ============================================================================

/// Trait for multinomial sampling kernel.
///
/// This requires a GPU kernel because CDF lookup + counting cannot be
/// efficiently expressed with standard tensor operations.
pub trait MultinomialSamplingOps<R: Runtime<DType = DType>> {
    /// Multinomial sampling kernel.
    ///
    /// Given probability vector, generates n_samples where each sample
    /// contains counts of n_trials draws from the categorical distribution.
    ///
    /// Implementation must:
    /// 1. Normalize probs on GPU
    /// 2. Compute CDF on GPU (cumsum)
    /// 3. Generate uniform samples on GPU
    /// 4. Map uniforms to categories and count (GPU kernel)
    fn multinomial_sample_kernel(
        &self,
        probs: &Tensor<R>,
        n_trials: usize,
        n_samples: usize,
    ) -> Result<Tensor<R>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_support_constants() {
        const { assert!(DTypeSupport::FULL.f64_supported) };
        const { assert!(!DTypeSupport::F32_ONLY.f64_supported) };
    }
}
