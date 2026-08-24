//! Basic matrix operations (inverse, det, trace, diag, rank, norms)

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::decompositions::{lu_decompose_impl, qr_decompose_impl};
use super::solvers::solve_impl;
use super::svd::svd_decompose_impl;
use crate::algorithm::linalg::{
    MatrixNormOrder, linalg_demote, linalg_promote, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix inverse via LU decomposition
pub fn inverse_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => inverse_typed::<f32>(client, &a, n),
        DType::F64 => inverse_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn inverse_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Create identity matrix
    let mut identity: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        identity[i * n + i] = T::one();
    }
    let identity_tensor = Tensor::<CpuRuntime>::from_slice(&identity, &[n, n], device)?;

    // Solve A @ X = I
    solve_impl(client, a, &identity_tensor)
}

/// Determinant via LU decomposition
pub fn det_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => det_typed::<f32>(client, &a, n),
        DType::F64 => det_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn det_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0
    if n == 0 {
        return Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device);
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // det = (-1)^num_swaps * product(U[i,i])
    let mut det = if lu_decomp.num_swaps % 2 == 0 {
        T::one()
    } else {
        T::one().neg_val()
    };

    for i in 0..n {
        det = det * lu_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&[det], &[], device)
}

/// Trace: sum of diagonal elements
pub fn trace_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => trace_typed::<f32>(client, &a, m, n),
        DType::F64 => trace_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn trace_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut trace = T::zero();
    for i in 0..k {
        trace = trace + a_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&[trace], &[], device)
}

/// Extract diagonal
pub fn diag_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => diag_typed::<f32>(client, &a, m, n),
        DType::F64 => diag_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn diag_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    let k = m.min(n);
    let mut diag: Vec<T> = vec![T::zero(); k];
    for i in 0..k {
        diag[i] = a_data[i * n + i];
    }

    Tensor::<CpuRuntime>::from_slice(&diag, &[k], device)
}

/// Create diagonal matrix from 1D tensor
pub fn diagflat_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.ndim() != 1 {
        return Err(Error::Internal(format!(
            "diagflat expects 1D tensor, got {}D",
            a.ndim()
        )));
    }
    let (a, original_dtype) = linalg_promote(client, a)?;

    let result = match a.dtype() {
        DType::F32 => diagflat_typed::<f32>(client, &a),
        DType::F64 => diagflat_typed::<f64>(client, &a),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn diagflat_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let n = a.shape()[0];
    let a_data: Vec<T> = a.to_vec();

    let mut mat: Vec<T> = vec![T::zero(); n * n];
    for i in 0..n {
        mat[i * n + i] = a_data[i];
    }

    Tensor::<CpuRuntime>::from_slice(&mat, &[n, n], device)
}

/// Kronecker product: A ⊗ B
///
/// For A of shape [m_a, n_a] and B of shape [m_b, n_b],
/// produces output of shape [m_a * m_b, n_a * n_b].
///
/// (A ⊗ B)[i*m_b + k, j*n_b + l] = A[i, j] * B[k, l]
pub fn kron_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let result = match a.dtype() {
        DType::F32 => kron_typed::<f32>(client, &a, &b, m_a, n_a, m_b, n_b),
        DType::F64 => kron_typed::<f64>(client, &a, &b, m_a, n_a, m_b, n_b),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn kron_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m_a: usize,
    n_a: usize,
    m_b: usize,
    n_b: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let b_data: Vec<T> = b.to_vec();

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let mut out: Vec<T> = vec![T::zero(); m_out * n_out];

    // Compute Kronecker product
    // out[i_a * m_b + i_b, j_a * n_b + j_b] = a[i_a, j_a] * b[i_b, j_b]
    for i_a in 0..m_a {
        for j_a in 0..n_a {
            let a_val = a_data[i_a * n_a + j_a];
            for i_b in 0..m_b {
                for j_b in 0..n_b {
                    let i_out = i_a * m_b + i_b;
                    let j_out = j_a * n_b + j_b;
                    out[i_out * n_out + j_out] = a_val * b_data[i_b * n_b + j_b];
                }
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&out, &[m_out, n_out], device)
}

/// Khatri-Rao product (column-wise Kronecker): A ⊙ B
///
/// For A of shape [m, k] and B of shape [n, k],
/// produces output of shape [m * n, k].
///
/// (A ⊙ B)[i*n + j, c] = A[i, c] * B[j, c]
pub fn khatri_rao_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (b, _) = linalg_promote(client, b)?;
    let (m, k_a) = validate_matrix_2d(a.shape())?;
    let (n, k_b) = validate_matrix_2d(b.shape())?;

    if k_a != k_b {
        return Err(Error::Internal(format!(
            "khatri_rao: column count mismatch. A has shape [{}, {}], B has shape [{}, {}]. \
             Matrices must have the same number of columns.",
            m, k_a, n, k_b
        )));
    }

    let k = k_a;

    let result = match a.dtype() {
        DType::F32 => khatri_rao_typed::<f32>(client, &a, &b, m, n, k),
        DType::F64 => khatri_rao_typed::<f64>(client, &a, &b, m, n, k),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn khatri_rao_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    k: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let b_data: Vec<T> = b.to_vec();

    let m_out = m * n;
    let mut out: Vec<T> = vec![T::zero(); m_out * k];

    // Compute Khatri-Rao product (column-wise Kronecker)
    // out[i*n + j, c] = a[i, c] * b[j, c]
    for c in 0..k {
        for i in 0..m {
            let a_val = a_data[i * k + c];
            for j in 0..n {
                let out_row = i * n + j;
                out[out_row * k + c] = a_val * b_data[j * k + c];
            }
        }
    }

    Tensor::<CpuRuntime>::from_slice(&out, &[m_out, k], device)
}

/// Upper triangular part of a matrix
///
/// Supports all numeric dtypes (not just F32/F64).
pub fn triu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    use crate::runtime::cpu::helpers::dispatch_dtype;
    dispatch_dtype!(dtype, T => {
        triu_typed::<T>(client, a, m, n, diagonal)
    }, "triu")
}

fn triu_typed<T: Element>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    // Single allocation: clone input, then zero out the lower triangle in-place
    let mut data: Vec<T> = a.to_vec();

    for row in 0..m {
        // Zero columns below the diagonal: col < row + diagonal
        let threshold = (row as i64 + diagonal).max(0) as usize;
        let end = threshold.min(n);
        for col in 0..end {
            data[row * n + col] = T::zero();
        }
    }

    Tensor::<CpuRuntime>::from_slice(&data, &[m, n], device)
}

/// Lower triangular part of a matrix
///
/// Supports all numeric dtypes (not just F32/F64).
pub fn tril_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    use crate::runtime::cpu::helpers::dispatch_dtype;
    dispatch_dtype!(dtype, T => {
        tril_typed::<T>(client, a, m, n, diagonal)
    }, "tril")
}

fn tril_typed<T: Element>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    diagonal: i64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    // Single allocation: clone input, then zero out the upper triangle in-place
    let mut data: Vec<T> = a.to_vec();

    for row in 0..m {
        // Zero columns above the diagonal: col > row + diagonal
        let threshold = (row as i64 + diagonal + 1).max(0) as usize;
        let start = threshold.min(n);
        for col in start..n {
            data[row * n + col] = T::zero();
        }
    }

    Tensor::<CpuRuntime>::from_slice(&data, &[m, n], device)
}

/// Sign and log-absolute-determinant via LU decomposition
pub fn slogdet_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<crate::algorithm::linalg::SlogdetResult<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => slogdet_typed::<f32>(client, &a, n),
        DType::F64 => slogdet_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(crate::algorithm::linalg::SlogdetResult {
        sign: linalg_demote(client, result.sign, original_dtype)?,
        logabsdet: linalg_demote(client, result.logabsdet, original_dtype)?,
    })
}

fn slogdet_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<crate::algorithm::linalg::SlogdetResult<CpuRuntime>> {
    let device = client.device();

    // Handle special case n=0: det of empty matrix is 1 by convention
    if n == 0 {
        return Ok(crate::algorithm::linalg::SlogdetResult {
            sign: Tensor::<CpuRuntime>::from_slice(&[T::one()], &[], device)?,
            logabsdet: Tensor::<CpuRuntime>::from_slice(&[T::zero()], &[], device)?,
        });
    }

    // Compute LU decomposition
    let lu_decomp = lu_decompose_impl(client, a)?;
    let lu_data: Vec<T> = lu_decomp.lu.to_vec();

    // Start with sign from row swaps
    let mut sign_val: f64 = if lu_decomp.num_swaps % 2 == 0 {
        1.0
    } else {
        -1.0
    };

    let mut logabsdet_val: f64 = 0.0;
    let mut is_zero = false;

    for i in 0..n {
        let diag = lu_data[i * n + i].to_f64();
        if diag == 0.0 {
            is_zero = true;
            break;
        }
        if diag < 0.0 {
            sign_val = -sign_val;
        }
        logabsdet_val += diag.abs().ln();
    }

    if is_zero {
        sign_val = 0.0;
        logabsdet_val = f64::NEG_INFINITY;
    }

    Ok(crate::algorithm::linalg::SlogdetResult {
        sign: Tensor::<CpuRuntime>::from_slice(&[T::from_f64(sign_val)], &[], device)?,
        logabsdet: Tensor::<CpuRuntime>::from_slice(&[T::from_f64(logabsdet_val)], &[], device)?,
    })
}

/// Matrix rank via singular value thresholding
/// Uses QR-based approach since SVD is not yet implemented
pub fn matrix_rank_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, _original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    // matrix_rank returns I64 (integer rank) - no demotion needed
    match a.dtype() {
        DType::F32 => matrix_rank_typed::<f32>(client, &a, m, n, tol),
        DType::F64 => matrix_rank_typed::<f64>(client, &a, m, n, tol),
        _ => unreachable!(),
    }
}

fn matrix_rank_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    tol: Option<f64>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Use QR decomposition to estimate rank (R diagonal gives singular value bounds)
    let qr = qr_decompose_impl(client, a, true)?;
    let r_data: Vec<T> = qr.r.to_vec();

    let k = m.min(n);

    // Find max diagonal element of R (upper bound on max singular value)
    let mut max_diag = T::zero();
    for i in 0..k {
        let val = r_data[i * n + i].abs_val();
        if val.to_f64() > max_diag.to_f64() {
            max_diag = val;
        }
    }

    // Compute tolerance
    let eps = if a.dtype() == DType::F32 {
        f32::EPSILON as f64
    } else {
        f64::EPSILON
    };
    let tolerance = tol.unwrap_or_else(|| m.max(n) as f64 * eps * max_diag.to_f64());

    // Count diagonal elements above tolerance
    let mut rank = 0i64;
    for i in 0..k {
        if r_data[i * n + i].abs_val().to_f64() > tolerance {
            rank += 1;
        }
    }

    Tensor::<CpuRuntime>::from_slice(&[rank], &[], device)
}

/// Matrix norm implementation
pub fn matrix_norm_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;

    let result = match ord {
        MatrixNormOrder::Frobenius => match a.dtype() {
            DType::F32 => frobenius_norm_typed::<f32>(client, &a),
            DType::F64 => frobenius_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
        MatrixNormOrder::Spectral => match a.dtype() {
            DType::F32 => spectral_norm_typed::<f32>(client, &a),
            DType::F64 => spectral_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
        MatrixNormOrder::Nuclear => match a.dtype() {
            DType::F32 => nuclear_norm_typed::<f32>(client, &a),
            DType::F64 => nuclear_norm_typed::<f64>(client, &a),
            _ => unreachable!(),
        },
    }?;

    linalg_demote(client, result, original_dtype)
}

/// Frobenius norm: ||A||_F = sqrt(sum_{i,j} |A[i,j]|^2)
fn frobenius_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();

    // Sum of squares of all elements
    let mut sum_sq = T::zero();
    for &val in &a_data {
        sum_sq = sum_sq + val * val;
    }

    let norm = sum_sq.sqrt_val();
    Tensor::<CpuRuntime>::from_slice(&[norm], &[], device)
}

/// Spectral norm: ||A||_2 = max(singular_values(A))
fn spectral_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    // Find maximum singular value
    let mut max_sv = T::zero();
    for &val in &s_data {
        if val.to_f64() > max_sv.to_f64() {
            max_sv = val;
        }
    }

    Tensor::<CpuRuntime>::from_slice(&[max_sv], &[], device)
}

/// Nuclear norm: ||A||_* = sum(singular_values(A))
fn nuclear_norm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Compute SVD to get singular values
    let svd = svd_decompose_impl(client, a)?;
    let s_data: Vec<T> = svd.s.to_vec();

    // Sum of singular values
    let mut sum_sv = T::zero();
    for &val in &s_data {
        sum_sv = sum_sv + val;
    }

    Tensor::<CpuRuntime>::from_slice(&[sum_sv], &[], device)
}
