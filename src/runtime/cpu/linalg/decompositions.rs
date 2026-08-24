//! LU, Cholesky, and QR decomposition implementations

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::{
    CholeskyDecomposition, LuDecomposition, QrDecomposition, linalg_demote, linalg_promote,
    validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// LU decomposition with partial pivoting (Doolittle algorithm)
pub fn lu_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<LuDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => lu_decompose_typed::<f32>(client, &a, m, n),
        DType::F64 => lu_decompose_typed::<f64>(client, &a, m, n),
        _ => unreachable!(),
    }?;

    Ok(LuDecomposition {
        lu: linalg_demote(client, result.lu, original_dtype)?,
        pivots: result.pivots,
        num_swaps: result.num_swaps,
    })
}

fn lu_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
) -> Result<LuDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // Copy input to working buffer (will be modified in-place)
    let lu_data: Vec<T> = a.to_vec();
    let mut lu: Vec<T> = lu_data;
    let mut pivots: Vec<i64> = vec![0; k];
    let mut num_swaps = 0usize;

    // LU decomposition with partial pivoting
    for col in 0..k {
        // Find pivot: max absolute value in column col, rows col..m
        let mut pivot_row = col;
        let mut max_val = lu[col * n + col].abs_val();

        for row in (col + 1)..m {
            let val = lu[row * n + col].abs_val();
            if val > max_val {
                max_val = val;
                pivot_row = row;
            }
        }

        pivots[col] = pivot_row as i64;

        // Swap rows if needed
        if pivot_row != col {
            for j in 0..n {
                lu.swap(col * n + j, pivot_row * n + j);
            }
            num_swaps += 1;
        }

        // Check for zero pivot (singular matrix)
        let pivot = lu[col * n + col];
        if pivot.abs_val().to_f64() < T::epsilon_val() {
            return Err(Error::Internal("Matrix is singular".to_string()));
        }

        // Compute multipliers (L column)
        for row in (col + 1)..m {
            lu[row * n + col] = lu[row * n + col] / pivot;
        }

        // Update trailing submatrix
        for row in (col + 1)..m {
            let multiplier = lu[row * n + col];
            for j in (col + 1)..n {
                let update = multiplier * lu[col * n + j];
                lu[row * n + j] = lu[row * n + j] - update;
            }
        }
    }

    // Create output tensors
    let lu_tensor = Tensor::<CpuRuntime>::try_from_slice(&lu, &[m, n], device)?;
    let pivots_tensor = Tensor::<CpuRuntime>::try_from_slice(&pivots, &[k], device)?;

    Ok(LuDecomposition {
        lu: lu_tensor,
        pivots: pivots_tensor,
        num_swaps,
    })
}

/// Cholesky decomposition (Cholesky-Banachiewicz algorithm)
pub fn cholesky_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
) -> Result<CholeskyDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => cholesky_decompose_typed::<f32>(client, &a, n),
        DType::F64 => cholesky_decompose_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    Ok(CholeskyDecomposition {
        l: linalg_demote(client, result.l, original_dtype)?,
    })
}

fn cholesky_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<CholeskyDecomposition<CpuRuntime>> {
    let device = client.device();
    let a_data: Vec<T> = a.to_vec();
    let mut l: Vec<T> = vec![T::zero(); n * n];

    for i in 0..n {
        // Compute diagonal element
        let mut sum_sq = T::zero();
        for k in 0..i {
            sum_sq = sum_sq + l[i * n + k] * l[i * n + k];
        }

        let diag = a_data[i * n + i] - sum_sq;
        if diag.to_f64() <= 0.0 {
            return Err(Error::Internal(
                "Matrix is not positive definite".to_string(),
            ));
        }
        l[i * n + i] = diag.sqrt_val();

        // Compute off-diagonal elements in column i
        for j in (i + 1)..n {
            let mut sum_prod = T::zero();
            for k in 0..i {
                sum_prod = sum_prod + l[j * n + k] * l[i * n + k];
            }
            l[j * n + i] = (a_data[j * n + i] - sum_prod) / l[i * n + i];
        }
    }

    let l_tensor = Tensor::<CpuRuntime>::try_from_slice(&l, &[n, n], device)?;
    Ok(CholeskyDecomposition { l: l_tensor })
}

/// QR decomposition using Householder reflections
pub fn qr_decompose_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    thin: bool,
) -> Result<QrDecomposition<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let (m, n) = validate_matrix_2d(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => qr_decompose_typed::<f32>(client, &a, m, n, thin),
        DType::F64 => qr_decompose_typed::<f64>(client, &a, m, n, thin),
        _ => unreachable!(),
    }?;

    Ok(QrDecomposition {
        q: linalg_demote(client, result.q, original_dtype)?,
        r: linalg_demote(client, result.r, original_dtype)?,
    })
}

fn qr_decompose_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    m: usize,
    n: usize,
    thin: bool,
) -> Result<QrDecomposition<CpuRuntime>> {
    let device = client.device();
    let k = m.min(n);

    // R starts as copy of A
    let mut r: Vec<T> = a.to_vec();

    // Q starts as identity (m x m or m x k for thin)
    let q_cols = if thin { k } else { m };
    let mut q: Vec<T> = vec![T::zero(); m * q_cols];
    for i in 0..q_cols.min(m) {
        q[i * q_cols + i] = T::one();
    }

    // Householder vectors storage
    for col in 0..k {
        // Extract column vector below diagonal: x = R[col:m, col]
        let x_len = m - col;
        let mut x: Vec<T> = vec![T::zero(); x_len];
        for i in 0..x_len {
            x[i] = r[(col + i) * n + col];
        }

        // Compute norm of x
        let mut norm_sq = T::zero();
        for &val in &x {
            norm_sq = norm_sq + val * val;
        }
        let norm_x = norm_sq.sqrt_val();

        if norm_x.abs_val().to_f64() < T::epsilon_val() {
            continue;
        }

        // Compute Householder vector v
        // alpha = -sign(x[0]) * ||x||
        let alpha = if x[0].to_f64() >= 0.0 {
            norm_x.neg_val()
        } else {
            norm_x
        };

        // v = x; v[0] -= alpha
        let mut v = x.clone();
        v[0] = v[0] - alpha;

        // Normalize v
        let mut v_norm_sq = T::zero();
        for &val in &v {
            v_norm_sq = v_norm_sq + val * val;
        }
        let v_norm = v_norm_sq.sqrt_val();

        if v_norm.abs_val().to_f64() < T::epsilon_val() {
            continue;
        }

        for val in &mut v {
            *val = *val / v_norm;
        }

        // Apply reflection to R: R[col:m, col:n] -= 2 * v @ (v^T @ R[col:m, col:n])
        // First compute w = v^T @ R[col:m, col:n], shape [n - col]
        let mut w: Vec<T> = vec![T::zero(); n - col];
        for j in 0..(n - col) {
            for i in 0..x_len {
                w[j] = w[j] + v[i] * r[(col + i) * n + (col + j)];
            }
        }

        // R[col:m, col:n] -= 2 * v @ w^T
        let two = T::from_f64(2.0);
        for i in 0..x_len {
            for j in 0..(n - col) {
                let update = two * v[i] * w[j];
                r[(col + i) * n + (col + j)] = r[(col + i) * n + (col + j)] - update;
            }
        }

        // Apply reflection to Q: Q[:, col:m] -= 2 * Q[:, col:m] @ v @ v^T
        // Q[:, col:m] = Q[:, col:m] @ (I - 2*v*v^T)
        // For each row of Q, compute the update
        for row in 0..m {
            // Compute dot product of Q[row, col:m] with v
            let mut dot = T::zero();
            for i in 0..x_len {
                if col + i < q_cols {
                    dot = dot + q[row * q_cols + (col + i)] * v[i];
                }
            }

            // Update Q[row, col:m] -= 2 * dot * v
            for i in 0..x_len {
                if col + i < q_cols {
                    let update = two * dot * v[i];
                    q[row * q_cols + (col + i)] = q[row * q_cols + (col + i)] - update;
                }
            }
        }
    }

    // Create output tensors
    let q_tensor = Tensor::<CpuRuntime>::try_from_slice(&q, &[m, q_cols], device)?;

    // R is m x n, but for thin QR we only need k x n
    let r_rows = if thin { k } else { m };
    let mut r_out: Vec<T> = vec![T::zero(); r_rows * n];
    for i in 0..r_rows {
        for j in 0..n {
            r_out[i * n + j] = r[i * n + j];
        }
    }
    let r_tensor = Tensor::<CpuRuntime>::try_from_slice(&r_out, &[r_rows, n], device)?;

    Ok(QrDecomposition {
        q: q_tensor,
        r: r_tensor,
    })
}
