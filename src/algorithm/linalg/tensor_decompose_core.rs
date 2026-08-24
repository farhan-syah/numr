//! Tensor decomposition algorithms (impl_generic)
//!
//! This module contains the shared algorithms for tensor decompositions.
//! All backends delegate to these functions to ensure numerical parity.
//!
//! # Algorithm Overview
//!
//! - **Unfold/Fold**: Tensor matricization and tensorization
//! - **Mode-n Product**: Tensor × matrix along specific mode
//! - **HOSVD**: Non-iterative Tucker via SVD on each mode
//! - **Tucker (HOOI)**: Iterative refinement of Tucker decomposition
//! - **CP/PARAFAC (ALS)**: Alternating least squares for CP decomposition
//! - **Tensor-Train (TT-SVD)**: Sequential SVD for TT decomposition

use super::LinearAlgebraAlgorithms;
use super::decompositions::{
    CpDecomposition, CpInit, CpOptions, TensorTrainDecomposition, TuckerDecomposition, TuckerInit,
    TuckerOptions,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::traits::RandomOps;
use crate::ops::traits::{BinaryOps, MatmulOps, ReduceOps, UnaryOps};
use crate::runtime::Runtime;
use crate::tensor::Tensor;

// ============================================================================
// DType Support Configuration
// ============================================================================

/// Configuration for dtype support per backend.
#[derive(Debug, Clone, Copy)]
pub struct TensorDecomposeDTypeSupport {
    /// Whether F64 is supported (false for WebGPU)
    pub f64_supported: bool,
}

impl TensorDecomposeDTypeSupport {
    /// CPU and CUDA support both F32 and F64
    pub const FULL: Self = Self {
        f64_supported: true,
    };

    /// WebGPU only supports F32
    pub const F32_ONLY: Self = Self {
        f64_supported: false,
    };
}

// ============================================================================
// Validation Helpers
// ============================================================================

/// Validate tensor has at least 2 dimensions
fn validate_tensor_nd(shape: &[usize]) -> Result<()> {
    if shape.len() < 2 {
        return Err(Error::Internal(format!(
            "Tensor decomposition requires at least 2D tensor, got shape {:?}",
            shape
        )));
    }
    Ok(())
}

/// Validate mode is within tensor dimensions
fn validate_mode(mode: usize, ndim: usize) -> Result<()> {
    if mode >= ndim {
        return Err(Error::Internal(format!(
            "Mode {} is out of bounds for tensor with {} dimensions",
            mode, ndim
        )));
    }
    Ok(())
}

/// Validate ranks array matches tensor dimensions
fn validate_ranks(ranks: &[usize], shape: &[usize]) -> Result<Vec<usize>> {
    if ranks.len() != shape.len() {
        return Err(Error::Internal(format!(
            "Ranks length {} must match tensor dimensions {}",
            ranks.len(),
            shape.len()
        )));
    }

    // Clamp ranks to dimension sizes
    let clamped: Vec<usize> = ranks
        .iter()
        .zip(shape.iter())
        .map(|(&r, &d)| if r == 0 || r > d { d } else { r })
        .collect();

    Ok(clamped)
}

/// Validate dtype for tensor decomposition
fn validate_decompose_dtype(
    dtype: DType,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<()> {
    if dtype_support.f64_supported {
        if dtype != DType::F32 && dtype != DType::F64 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "tensor_decompose",
            });
        }
    } else if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "tensor_decompose (F32 only on this backend)",
        });
    }
    Ok(())
}

// ============================================================================
// Mode-n Unfolding (Matricization)
// ============================================================================

/// Compute the permutation for mode-n unfolding
///
/// For mode n, the permutation is: [n, 0, 1, ..., n-1, n+1, ..., N-1]
fn unfold_permutation(mode: usize, ndim: usize) -> Vec<usize> {
    let mut perm = Vec::with_capacity(ndim);
    perm.push(mode);
    for i in 0..ndim {
        if i != mode {
            perm.push(i);
        }
    }
    perm
}

/// Mode-n unfolding (matricization)
///
/// Unfolds tensor T of shape [I₁, I₂, ..., Iₙ] along mode n into matrix
/// of shape [Iₙ, ∏ⱼ≠ₙ Iⱼ].
pub fn unfold_impl<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    mode: usize,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<Tensor<R>> {
    let shape = tensor.shape();
    let ndim = shape.len();

    validate_tensor_nd(shape)?;
    validate_mode(mode, ndim)?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    // Step 1: Permute to bring mode to front
    let perm = unfold_permutation(mode, ndim);
    let permuted = tensor.permute(&perm)?;

    // Step 2: Reshape to 2D [I_mode, prod(other dims)]
    let mode_size = shape[mode];
    let other_size: usize = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &d)| d)
        .product();

    // Make contiguous before reshape (permute may have changed strides)
    let permuted = permuted.contiguous()?;
    permuted.reshape(&[mode_size, other_size])
}

/// Mode-n folding (tensorization) - inverse of unfolding
///
/// Reconstructs tensor from its mode-n unfolding.
pub fn fold_impl<R: Runtime<DType = DType>>(
    matrix: &Tensor<R>,
    mode: usize,
    shape: &[usize],
) -> Result<Tensor<R>> {
    let mat_shape = matrix.shape();
    if mat_shape.len() != 2 {
        return Err(Error::Internal(format!(
            "fold expects 2D matrix, got shape {:?}",
            mat_shape
        )));
    }

    let ndim = shape.len();
    validate_mode(mode, ndim)?;

    // Verify dimensions match
    let mode_size = shape[mode];
    let other_size: usize = shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != mode)
        .map(|(_, &d)| d)
        .product();

    if mat_shape[0] != mode_size || mat_shape[1] != other_size {
        return Err(Error::Internal(format!(
            "Matrix shape {:?} doesn't match expected [{}, {}] for mode {} unfolding of shape {:?}",
            mat_shape, mode_size, other_size, mode, shape
        )));
    }

    // Step 1: Reshape to permuted tensor shape
    let perm = unfold_permutation(mode, ndim);
    let permuted_shape: Vec<usize> = perm.iter().map(|&i| shape[i]).collect();
    let permuted = matrix.contiguous()?.reshape(&permuted_shape)?;

    // Step 2: Inverse permutation to restore original order
    let mut inv_perm = vec![0; ndim];
    for (i, &p) in perm.iter().enumerate() {
        inv_perm[p] = i;
    }

    // Return contiguous tensor for easy use
    permuted.permute(&inv_perm)?.contiguous()
}

// ============================================================================
// Mode-n Product
// ============================================================================

/// Mode-n product: T ×ₙ M
///
/// Multiplies tensor T by matrix M along mode n.
/// If T has shape [I₁, ..., Iₙ, ..., Iₙ] and M has shape [J, Iₙ],
/// result has shape [I₁, ..., J, ..., Iₙ].
pub fn mode_n_product_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    matrix: &Tensor<R>,
    mode: usize,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MatmulOps<R>,
{
    let tensor_shape = tensor.shape();
    let matrix_shape = matrix.shape();

    validate_tensor_nd(tensor_shape)?;
    validate_mode(mode, tensor_shape.len())?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    if matrix_shape.len() != 2 {
        return Err(Error::Internal(format!(
            "mode_n_product expects 2D matrix, got shape {:?}",
            matrix_shape
        )));
    }

    let j = matrix_shape[0];
    let mode_dim = matrix_shape[1];

    if tensor_shape[mode] != mode_dim {
        return Err(Error::Internal(format!(
            "Matrix columns {} must match tensor mode {} dimension {}",
            mode_dim, mode, tensor_shape[mode]
        )));
    }

    // Mode-n product = fold(M @ unfold(T, n), n, new_shape)
    let unfolded = unfold_impl(tensor, mode, dtype_support)?;

    // Matrix multiplication: M @ T₍ₙ₎
    let product = client.matmul(matrix, &unfolded)?;

    // New shape with mode dimension replaced
    let mut new_shape = tensor_shape.to_vec();
    new_shape[mode] = j;

    fold_impl(&product, mode, &new_shape)
}

// ============================================================================
// HOSVD (Higher-Order SVD)
// ============================================================================

/// HOSVD: Non-iterative Tucker decomposition via truncated SVD
///
/// For each mode, computes SVD of the mode-n unfolding and takes the
/// leading singular vectors as the factor matrix.
pub fn hosvd_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    ranks: &[usize],
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<TuckerDecomposition<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + MatmulOps<R>,
{
    let shape = tensor.shape();
    validate_tensor_nd(shape)?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    let ranks = validate_ranks(ranks, shape)?;
    let ndim = shape.len();

    // Step 1: Compute factor matrices via SVD of each mode-n unfolding
    let mut factors: Vec<Tensor<R>> = Vec::with_capacity(ndim);

    for mode in 0..ndim {
        let unfolded = unfold_impl(tensor, mode, dtype_support)?;
        let svd = client.svd_decompose(&unfolded)?;

        // Take first ranks[mode] columns of U
        let rank = ranks[mode];
        let u_shape = svd.u.shape();

        if rank >= u_shape[1] {
            // Full rank, use U as-is
            factors.push(svd.u);
        } else {
            // Truncate: take first `rank` columns
            let factor = svd.u.narrow(1, 0, rank)?;
            factors.push(factor.contiguous()?);
        }
    }

    // Step 2: Compute core tensor: G = T ×₁ A₁ᵀ ×₂ A₂ᵀ ... ×ₙ Aₙᵀ
    let mut core = tensor.clone();
    for (mode, factor) in factors.iter().enumerate() {
        let factor_t = factor.t()?;
        core = mode_n_product_impl(client, &core, &factor_t, mode, dtype_support)?;
    }

    Ok(TuckerDecomposition { core, factors })
}

// ============================================================================
// Tucker Decomposition (HOOI)
// ============================================================================

/// Compute Frobenius norm of a tensor - returns GPU scalar tensor (no CPU transfer)
fn frobenius_norm_tensor<R, C>(client: &C, tensor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let sq = client.mul(tensor, tensor)?;
    let sum = client.sum(&sq, &[], false)?;
    client.sqrt(&sum)
}

/// Tucker decomposition via Higher-Order Orthogonal Iteration (HOOI)
///
/// Iteratively refines factor matrices to minimize ||T - G ×₁ A₁ ... ×ₙ Aₙ||²
/// Runs for `options.max_iter` iterations.
pub fn tucker_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    ranks: &[usize],
    options: TuckerOptions,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<TuckerDecomposition<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + MatmulOps<R> + ReduceOps<R> + BinaryOps<R> + RandomOps<R>,
{
    let shape = tensor.shape();
    validate_tensor_nd(shape)?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    let ranks = validate_ranks(ranks, shape)?;
    let ndim = shape.len();
    let dtype = tensor.dtype();

    // Initialize factors
    let mut factors = match options.init {
        TuckerInit::Hosvd => {
            let hosvd = hosvd_impl(client, tensor, &ranks, dtype_support)?;
            hosvd.factors
        }
        TuckerInit::Random => {
            // Random orthogonal initialization
            let mut factors = Vec::with_capacity(ndim);
            for mode in 0..ndim {
                let m = shape[mode];
                let r = ranks[mode];
                // Create random matrix and orthogonalize via QR
                let random = client.randn(&[m, r], dtype)?;
                let qr = client.qr_decompose(&random)?;
                factors.push(qr.q);
            }
            factors
        }
    };

    // HOOI iteration
    for _iter in 0..options.max_iter {
        // Update each factor
        for mode in 0..ndim {
            // Compute Y = T ×₁ A₁ᵀ ... (skip mode n) ... ×ₙ Aₙᵀ
            let mut y = tensor.clone();
            for (n, factor) in factors.iter().enumerate() {
                if n != mode {
                    let factor_t = factor.t()?;
                    y = mode_n_product_impl(client, &y, &factor_t, n, dtype_support)?;
                }
            }

            // Unfold Y along mode and take leading singular vectors
            let y_unfolded = unfold_impl(&y, mode, dtype_support)?;
            let svd = client.svd_decompose(&y_unfolded)?;

            // Truncate to rank
            let rank = ranks[mode];
            if rank < svd.u.shape()[1] {
                factors[mode] = svd.u.narrow(1, 0, rank)?.contiguous()?;
            } else {
                factors[mode] = svd.u;
            }
        }
    }

    // Compute final core tensor
    let mut core = tensor.clone();
    for (mode, factor) in factors.iter().enumerate() {
        let factor_t = factor.t()?;
        core = mode_n_product_impl(client, &core, &factor_t, mode, dtype_support)?;
    }

    Ok(TuckerDecomposition { core, factors })
}

// ============================================================================
// CP/PARAFAC Decomposition (ALS)
// ============================================================================

/// Initialize CP factor matrices
fn initialize_cp_factors<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    rank: usize,
    options: &CpOptions,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<Vec<Tensor<R>>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + RandomOps<R>,
{
    let shape = tensor.shape();
    let ndim = shape.len();
    let dtype = tensor.dtype();

    match options.init {
        CpInit::Random => {
            let mut factors = Vec::with_capacity(ndim);
            for &dim in shape {
                let factor = client.randn(&[dim, rank], dtype)?;
                factors.push(factor);
            }
            Ok(factors)
        }
        CpInit::Svd | CpInit::Hosvd => {
            // Use SVD of mode-0 unfolding for first factor
            let mut factors = Vec::with_capacity(ndim);

            let unfolded = unfold_impl(tensor, 0, dtype_support)?;
            let svd = client.svd_decompose(&unfolded)?;

            // First factor from U
            let u_cols = svd.u.shape()[1].min(rank);
            let first_factor = if u_cols >= rank {
                svd.u.narrow(1, 0, rank)?.contiguous()?
            } else {
                // Pad with random if not enough singular vectors
                // For simplicity, just use random for remaining columns
                client.randn(&[shape[0], rank], dtype)?
            };
            factors.push(first_factor);

            // Initialize remaining factors randomly
            for mode in 1..ndim {
                let factor = client.randn(&[shape[mode], rank], dtype)?;
                factors.push(factor);
            }

            Ok(factors)
        }
    }
}

/// Compute Khatri-Rao product of all factors except the given mode
fn compute_khatri_rao_except<R, C>(
    client: &C,
    factors: &[Tensor<R>],
    skip_mode: usize,
) -> Result<Tensor<R>>
where
    R: Runtime,
    C: LinearAlgebraAlgorithms<R>,
{
    let n = factors.len();
    let mut result: Option<Tensor<R>> = None;

    // Compute in reverse order: n-1, n-2, ..., 1, 0 (skipping skip_mode)
    // This matches the convention for CP reconstruction
    for mode in (0..n).rev() {
        if mode == skip_mode {
            continue;
        }
        result = Some(match result {
            None => factors[mode].clone(),
            Some(acc) => client.khatri_rao(&factors[mode], &acc)?,
        });
    }

    result.ok_or_else(|| Error::Internal("No factors to compute Khatri-Rao product".to_string()))
}

/// Compute Hadamard (element-wise) product of Gram matrices for all factors except mode
fn compute_gram_hadamard_except<R, C>(
    client: &C,
    factors: &[Tensor<R>],
    skip_mode: usize,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MatmulOps<R> + BinaryOps<R>,
{
    let n = factors.len();
    let rank = factors[0].shape()[1];
    let dtype = factors[0].dtype();
    let device = factors[0].device();

    // Start with ones matrix
    let mut result = Tensor::<R>::try_ones(&[rank, rank], dtype, device)?;

    for mode in 0..n {
        if mode == skip_mode {
            continue;
        }
        // Gram matrix: Aᵀ @ A
        let gram = client.matmul(&factors[mode].t()?, &factors[mode])?;
        // Hadamard (element-wise) product
        result = client.mul(&result, &gram)?;
    }

    Ok(result)
}

/// Compute column norms of a factor matrix
fn compute_factor_norms<R, C>(client: &C, factor: &Tensor<R>) -> Result<Tensor<R>>
where
    R: Runtime,
    C: ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let sq = client.mul(factor, factor)?;
    let sum = client.sum(&sq, &[0], false)?;
    client.sqrt(&sum)
}

/// Normalize factor matrix columns to unit norm, returning the norms
fn normalize_factor_columns<R, C>(client: &C, factor: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>)>
where
    R: Runtime,
    C: ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let norms = compute_factor_norms(client, factor)?;
    let norms_expanded = norms.unsqueeze(0)?;
    let normalized = client.div(factor, &norms_expanded)?;
    Ok((normalized, norms))
}

/// CP/PARAFAC decomposition via Alternating Least Squares
pub fn cp_decompose_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    rank: usize,
    options: CpOptions,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<CpDecomposition<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R>
        + MatmulOps<R>
        + ReduceOps<R>
        + BinaryOps<R>
        + UnaryOps<R>
        + RandomOps<R>,
{
    let shape = tensor.shape();
    validate_tensor_nd(shape)?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    if rank == 0 {
        return Err(Error::Internal("CP rank must be positive".to_string()));
    }

    let ndim = shape.len();

    // Initialize factor matrices
    let mut factors = initialize_cp_factors(client, tensor, rank, &options, dtype_support)?;

    // ALS iteration
    for _iter in 0..options.max_iter {
        for mode in 0..ndim {
            // Compute Khatri-Rao product of all factors except mode
            let kr_product = compute_khatri_rao_except(client, &factors, mode)?;

            // Compute Hadamard product of (Aⱼᵀ @ Aⱼ) for all j != mode
            let gram_product = compute_gram_hadamard_except(client, &factors, mode)?;

            // Unfold tensor along mode
            let t_unfolded = unfold_impl(tensor, mode, dtype_support)?;

            // Solve: A_mode = T₍mode₎ @ KR @ (Gram)⁻¹
            // Equivalent to solving Gram @ A_modeᵀ = KRᵀ @ T₍mode₎ᵀ for A_mode
            let rhs = client.matmul(&t_unfolded, &kr_product)?;
            // Make tensors contiguous for solve (which requires contiguous data)
            let gram_t = gram_product.t()?.contiguous()?;
            let rhs_t = rhs.t()?.contiguous()?;
            let new_factor = client.solve(&gram_t, &rhs_t)?.t()?;
            factors[mode] = new_factor.contiguous()?;

            // Normalize factor if requested, but NOT the last factor
            // (the last factor accumulates the scale which becomes the weights)
            if options.normalize && mode < ndim - 1 {
                let (normalized, _norms) = normalize_factor_columns(client, &factors[mode])?;
                factors[mode] = normalized;
            }
        }
    }

    // Extract final weights (column norms of last factor which accumulated all scale)
    let (normalized_last, weights) = normalize_factor_columns(client, &factors[ndim - 1])?;
    factors[ndim - 1] = normalized_last;

    Ok(CpDecomposition { factors, weights })
}

// ============================================================================
// Tensor-Train Decomposition (TT-SVD)
// ============================================================================

/// Tensor-Train decomposition via sequential SVD
///
/// - `max_rank`: Maximum TT-rank (0 = no limit)
/// - `tolerance`: Truncation tolerance based on singular values (0.0 = no truncation)
pub fn tensor_train_impl<R, C>(
    client: &C,
    tensor: &Tensor<R>,
    max_rank: usize,
    tolerance: f64,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<TensorTrainDecomposition<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + ReduceOps<R> + BinaryOps<R> + UnaryOps<R>,
{
    let shape = tensor.shape();
    validate_tensor_nd(shape)?;
    validate_decompose_dtype(tensor.dtype(), dtype_support)?;

    let ndim = shape.len();
    let dtype = tensor.dtype();

    // Compute tolerance threshold if needed
    let tol_threshold_sq = if tolerance > 0.0 {
        let tensor_norm_tensor = frobenius_norm_tensor(client, tensor)?;
        let norm_val: f64 = match dtype {
            DType::F32 => tensor_norm_tensor
                .to_vec::<f32>()
                .first()
                .copied()
                .unwrap_or(0.0) as f64,
            DType::F64 => tensor_norm_tensor
                .to_vec::<f64>()
                .first()
                .copied()
                .unwrap_or(0.0),
            _ => tensor_norm_tensor
                .to_vec::<f32>()
                .first()
                .copied()
                .unwrap_or(0.0) as f64,
        };
        let threshold = tolerance * norm_val;
        threshold * threshold
    } else {
        0.0
    };

    let mut cores: Vec<Tensor<R>> = Vec::with_capacity(ndim);
    let mut ranks: Vec<usize> = Vec::with_capacity(ndim - 1);

    // Working tensor, will be reshaped and decomposed sequentially
    let mut work = tensor.contiguous()?;
    let mut left_rank = 1usize;

    for mode in 0..(ndim - 1) {
        let work_shape = work.shape();

        // Reshape to [left_rank * I_mode, remaining_dims]
        let mode_dim = shape[mode];
        let left_size = left_rank * mode_dim;
        let right_size: usize = work_shape.iter().product::<usize>() / left_size;

        let matrix = work.reshape(&[left_size, right_size])?.contiguous()?;

        // SVD
        let svd = client.svd_decompose(&matrix)?;

        // Get number of singular values from shape (no GPU transfer)
        let num_singular_values = svd.s.shape()[0];
        let mut trunc_rank = num_singular_values;

        // Tolerance-based truncation requires reading singular values to CPU
        if tolerance > 0.0 {
            // Extract singular values to CPU for adaptive rank selection
            let s_vec: Vec<f64> = match dtype {
                DType::F32 => svd.s.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
                DType::F64 => svd.s.to_vec::<f64>(),
                _ => svd.s.to_vec::<f32>().iter().map(|&x| x as f64).collect(),
            };

            // Truncate based on singular value threshold (cumulative sum from smallest)
            let mut cumsum_sq = 0.0f64;
            for (i, &s) in s_vec.iter().rev().enumerate() {
                cumsum_sq += s * s;
                if cumsum_sq > tol_threshold_sq {
                    trunc_rank = s_vec.len() - i;
                    break;
                }
            }
        }

        // Apply max_rank constraint
        if max_rank > 0 && trunc_rank > max_rank {
            trunc_rank = max_rank;
        }

        // Ensure at least rank 1
        trunc_rank = trunc_rank.max(1);

        // Extract truncated U and form core
        let u_trunc = if trunc_rank < svd.u.shape()[1] {
            svd.u.narrow(1, 0, trunc_rank)?.contiguous()?
        } else {
            svd.u.clone()
        };

        // Core has shape [left_rank, I_mode, trunc_rank]
        let core = u_trunc.reshape(&[left_rank, mode_dim, trunc_rank])?;
        cores.push(core.contiguous()?);
        ranks.push(trunc_rank);

        // Prepare for next mode: S @ Vᵀ (truncated)
        let s_trunc = if trunc_rank < num_singular_values {
            svd.s.narrow(0, 0, trunc_rank)?
        } else {
            svd.s.clone()
        };
        let vt_trunc = if trunc_rank < svd.vt.shape()[0] {
            svd.vt.narrow(0, 0, trunc_rank)?.contiguous()?
        } else {
            svd.vt.clone()
        };

        // Scale Vᵀ rows by singular values: diag(S) @ Vᵀ
        let s_expanded = s_trunc.unsqueeze(1)?;
        work = client.mul(&s_expanded, &vt_trunc)?;
        left_rank = trunc_rank;
    }

    // Last core: remaining work reshaped to [left_rank, I_N, 1]
    let last_dim = shape[ndim - 1];
    let last_core = work.reshape(&[left_rank, last_dim, 1])?;
    cores.push(last_core.contiguous()?);

    Ok(TensorTrainDecomposition { cores, ranks })
}

// ============================================================================
// Reconstruction Functions
// ============================================================================

/// Reconstruct tensor from Tucker decomposition
pub fn tucker_reconstruct_impl<R, C>(
    client: &C,
    decomp: &TuckerDecomposition<R>,
    dtype_support: TensorDecomposeDTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MatmulOps<R>,
{
    let mut result = decomp.core.clone();

    for (mode, factor) in decomp.factors.iter().enumerate() {
        result = mode_n_product_impl(client, &result, factor, mode, dtype_support)?;
    }

    Ok(result)
}

/// Reconstruct tensor from CP decomposition
pub fn cp_reconstruct_impl<R, C>(
    client: &C,
    decomp: &CpDecomposition<R>,
    shape: &[usize],
    _dtype_support: TensorDecomposeDTypeSupport,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: LinearAlgebraAlgorithms<R> + MatmulOps<R> + BinaryOps<R>,
{
    let ndim = decomp.factors.len();

    if shape.len() != ndim {
        return Err(Error::Internal(format!(
            "Shape length {} must match number of factors {}",
            shape.len(),
            ndim
        )));
    }

    // Reconstruct using mode-0 formula:
    // T₍₀₎ = A₀ @ diag(λ) @ (Aₙ₋₁ ⊙ ... ⊙ A₁)ᵀ

    // Compute Khatri-Rao of all factors except mode 0
    let mut kr: Option<Tensor<R>> = None;
    for mode in (1..ndim).rev() {
        kr = Some(match kr {
            None => decomp.factors[mode].clone(),
            Some(acc) => client.khatri_rao(&decomp.factors[mode], &acc)?,
        });
    }

    let kr = kr.ok_or_else(|| Error::Internal("Need at least 2 modes".to_string()))?;

    // Scale first factor by weights: A₀ @ diag(λ)
    let weights_expanded = decomp.weights.unsqueeze(0)?;
    let a0_scaled = client.mul(&decomp.factors[0], &weights_expanded)?;

    // T₍₀₎ = A₀_scaled @ KRᵀ
    let t_unfolded = client.matmul(&a0_scaled, &kr.t()?)?;

    // Fold back to tensor
    fold_impl(&t_unfolded, 0, shape)
}

/// Reconstruct tensor from Tensor-Train decomposition
pub fn tt_reconstruct_impl<R, C>(
    client: &C,
    decomp: &TensorTrainDecomposition<R>,
) -> Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: MatmulOps<R>,
{
    if decomp.cores.is_empty() {
        return Err(Error::Internal("Empty TT decomposition".to_string()));
    }

    let ndim = decomp.cores.len();

    // Collect physical dimensions
    let shape: Vec<usize> = decomp.cores.iter().map(|c| c.shape()[1]).collect();

    // Contract cores from left to right
    // Start with first core reshaped to [I_1, R_1]
    let first_core = &decomp.cores[0];
    let i1 = first_core.shape()[1];
    let r1 = first_core.shape()[2];
    let mut result = first_core.reshape(&[i1, r1])?.contiguous()?;

    for mode in 1..ndim {
        let core = &decomp.cores[mode];
        let core_shape = core.shape();
        let r_left = core_shape[0];
        let i_mode = core_shape[1];
        let r_right = core_shape[2];

        // result: [I_1 * ... * I_{mode-1}, R_{mode-1}]
        // core: [R_{mode-1}, I_mode, R_mode]

        // Reshape core to [R_{mode-1}, I_mode * R_mode]
        let core_mat = core.reshape(&[r_left, i_mode * r_right])?.contiguous()?;

        // Contract: result @ core_mat -> [prod(I_1..I_{mode-1}), I_mode * R_mode]
        result = client.matmul(&result, &core_mat)?;

        // Reshape to [prod(I_1..I_mode), R_mode]
        let left_prod: usize = shape[..=mode].iter().product();
        result = result.reshape(&[left_prod, r_right])?.contiguous()?;
    }

    // Final result should be [prod(all I), 1], reshape to original shape
    result.reshape(&shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unfold_permutation() {
        assert_eq!(unfold_permutation(0, 3), vec![0, 1, 2]);
        assert_eq!(unfold_permutation(1, 3), vec![1, 0, 2]);
        assert_eq!(unfold_permutation(2, 3), vec![2, 0, 1]);
        assert_eq!(unfold_permutation(1, 4), vec![1, 0, 2, 3]);
    }

    #[test]
    fn test_validate_ranks() {
        let shape = vec![4, 5, 6];

        // Normal case
        let ranks = validate_ranks(&[2, 3, 4], &shape).unwrap();
        assert_eq!(ranks, vec![2, 3, 4]);

        // Clamp to dimension size
        let ranks = validate_ranks(&[10, 10, 10], &shape).unwrap();
        assert_eq!(ranks, vec![4, 5, 6]);

        // Zero means full rank
        let ranks = validate_ranks(&[0, 3, 0], &shape).unwrap();
        assert_eq!(ranks, vec![4, 3, 6]);
    }
}
