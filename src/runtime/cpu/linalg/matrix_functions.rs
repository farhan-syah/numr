//! Matrix function implementations for CPU (expm, logm, sqrtm, signm, funm)
//!
//! All numerical algorithms are delegated to the shared `matrix_functions_core` module
//! to ensure DRY compliance and backend parity.

use super::super::jacobi::LinalgElement;
use super::super::{CpuClient, CpuRuntime};
use super::schur::schur_decompose_impl;
use crate::algorithm::linalg::{
    linalg_demote, linalg_promote, matrix_functions_core, validate_linalg_dtype,
    validate_square_matrix,
};
use crate::dtype::{DType, Element};
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

// ============================================================================
// Convergence Constants
// ============================================================================

/// Maximum iterations for Denman-Beavers square root iteration
const SQRTM_MAX_ITER: usize = 50;

/// Maximum iterations for Newton iteration in signm
const SIGNM_MAX_ITER: usize = 100;

// ============================================================================
// Matrix Exponential
// ============================================================================

/// Matrix exponential: e^A using Schur-Parlett algorithm
///
/// Algorithm:
/// 1. Compute Schur decomposition: A = Z @ T @ Z^T (on CPU, type-generic)
/// 2. Compute exp(T) using shared f64 algorithm (via type conversion)
/// 3. Reconstruct: exp(A) = Z @ exp(T) @ Z^T
pub fn expm_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => expm_typed::<f32>(client, &a, n),
        DType::F64 => expm_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn expm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let exp_val = data[0].to_f64().exp();
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(exp_val)],
            &[1, 1],
            device,
        )?);
    }

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = schur_decompose_impl(client, a)?;
    let z_data: Vec<T> = schur.z.to_vec();
    let t_data: Vec<T> = schur.t.to_vec();

    // Convert to f64 for shared algorithm
    let t_f64: Vec<f64> = t_data.iter().map(|x| x.to_f64()).collect();
    let z_f64: Vec<f64> = z_data.iter().map(|x| x.to_f64()).collect();

    // Use shared algorithm for exp(T)
    let exp_t_f64 = matrix_functions_core::exp_quasi_triangular_f64(&t_f64, n);

    // Reconstruct: exp(A) = Z @ exp(T) @ Z^T using shared algorithm
    let result_f64 = matrix_functions_core::reconstruct_from_schur_f64(&z_f64, &exp_t_f64, n);

    // Convert back to original type
    let result: Vec<T> = result_f64.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result,
        &[n, n],
        device,
    )?)
}

// ============================================================================
// Matrix Square Root
// ============================================================================

/// Matrix square root using Denman-Beavers iteration
///
/// This is implemented directly on CPU using type-generic code since
/// Denman-Beavers iteration involves matrix inversions that benefit
/// from the CPU's existing infrastructure.
pub fn sqrtm_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => sqrtm_typed::<f32>(client, &a, n),
        DType::F64 => sqrtm_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn sqrtm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let val = data[0].to_f64();
        if val < 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "sqrtm requires matrix with no negative real eigenvalues".to_string(),
            });
        }
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(val.sqrt())],
            &[1, 1],
            device,
        )?);
    }

    // Check for negative real eigenvalues using Schur decomposition
    let schur = schur_decompose_impl(client, a)?;
    let t_data: Vec<T> = schur.t.to_vec();
    let eps = T::epsilon_val();

    let mut i = 0;
    while i < n {
        if i + 1 < n && t_data[(i + 1) * n + i].to_f64().abs() > eps {
            // 2x2 block: complex eigenvalues (safe)
            i += 2;
        } else {
            // 1x1 block: real eigenvalue
            let eigenvalue = t_data[i * n + i].to_f64();
            if eigenvalue < -eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: format!(
                        "sqrtm requires matrix with no negative real eigenvalues, found {}",
                        eigenvalue
                    ),
                });
            }
            i += 1;
        }
    }

    // Denman-Beavers iteration using shared f64 algorithm
    let a_data: Vec<T> = a.to_vec();
    let a_f64: Vec<f64> = a_data.iter().map(|x| x.to_f64()).collect();

    let result_f64 = denman_beavers_iteration(&a_f64, n, eps, SQRTM_MAX_ITER)?;

    // Convert back to original type
    let result: Vec<T> = result_f64.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result,
        &[n, n],
        device,
    )?)
}

/// Denman-Beavers iteration for matrix square root
fn denman_beavers_iteration(a: &[f64], n: usize, eps: f64, max_iter: usize) -> Result<Vec<f64>> {
    let mut y_data = a.to_vec();
    let mut z_data: Vec<f64> = vec![0.0; n * n];
    for i in 0..n {
        z_data[i * n + i] = 1.0;
    }

    let tol = eps * (n as f64);

    for _iter in 0..max_iter {
        // Compute Y_inv and Z_inv using shared algorithm
        let y_inv = matrix_functions_core::invert_matrix_f64(&y_data, n, eps).ok_or_else(|| {
            Error::Internal("sqrtm: matrix inversion failed during iteration".to_string())
        })?;

        let z_inv = matrix_functions_core::invert_matrix_f64(&z_data, n, eps).ok_or_else(|| {
            Error::Internal("sqrtm: matrix inversion failed during iteration".to_string())
        })?;

        // Y_new = (Y + Z_inv) / 2, Z_new = (Z + Y_inv) / 2
        let mut y_new = vec![0.0; n * n];
        let mut z_new = vec![0.0; n * n];

        for i in 0..(n * n) {
            y_new[i] = (y_data[i] + z_inv[i]) / 2.0;
            z_new[i] = (z_data[i] + y_inv[i]) / 2.0;
        }

        // Check convergence
        let mut diff_norm = 0.0;
        let mut y_norm = 0.0;
        for i in 0..(n * n) {
            let diff = (y_new[i] - y_data[i]).abs();
            diff_norm += diff * diff;
            y_norm += y_data[i] * y_data[i];
        }
        diff_norm = diff_norm.sqrt();
        y_norm = y_norm.sqrt().max(1.0);

        y_data = y_new;
        z_data = z_new;

        if diff_norm / y_norm < tol {
            break;
        }
    }

    Ok(y_data)
}

// ============================================================================
// Matrix Logarithm
// ============================================================================

/// Matrix logarithm using inverse scaling and squaring with Schur decomposition
pub fn logm_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => logm_typed::<f32>(client, &a, n),
        DType::F64 => logm_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn logm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let val = data[0].to_f64();
        if val <= 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "logm requires matrix with no non-positive real eigenvalues".to_string(),
            });
        }
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(val.ln())],
            &[1, 1],
            device,
        )?);
    }

    let eps = T::epsilon_val();

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = schur_decompose_impl(client, a)?;
    let z_data: Vec<T> = schur.z.to_vec();
    let t_data: Vec<T> = schur.t.to_vec();

    // Convert to f64
    let t_f64: Vec<f64> = t_data.iter().map(|x| x.to_f64()).collect();
    let z_f64: Vec<f64> = z_data.iter().map(|x| x.to_f64()).collect();

    // Check for non-positive real eigenvalues
    validate_log_eigenvalues(&t_f64, n, eps)?;

    // Use shared algorithm for log(T)
    let log_t_f64 = matrix_functions_core::log_quasi_triangular_f64(&t_f64, n, eps);

    // Reconstruct: log(A) = Z @ log(T) @ Z^T
    let result_f64 = matrix_functions_core::reconstruct_from_schur_f64(&z_f64, &log_t_f64, n);

    // Convert back to original type
    let result: Vec<T> = result_f64.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result,
        &[n, n],
        device,
    )?)
}

/// Validate eigenvalues for matrix logarithm
fn validate_log_eigenvalues(t: &[f64], n: usize, eps: f64) -> Result<()> {
    let mut i = 0;
    while i < n {
        if i + 1 < n && t[(i + 1) * n + i].abs() > eps {
            // 2x2 block: check if on negative real axis
            let a_val = t[i * n + i];
            let b_val = t[i * n + (i + 1)];
            let c_val = t[(i + 1) * n + i];

            if a_val <= eps && b_val * c_val >= -eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: "logm requires matrix with no non-positive real eigenvalues"
                        .to_string(),
                });
            }
            i += 2;
        } else {
            // 1x1 block: real eigenvalue
            let eigenvalue = t[i * n + i];
            if eigenvalue <= eps {
                return Err(Error::InvalidArgument {
                    arg: "a",
                    reason: format!(
                        "logm requires matrix with no non-positive real eigenvalues, found {}",
                        eigenvalue
                    ),
                });
            }
            i += 1;
        }
    }
    Ok(())
}

// ============================================================================
// Matrix Sign Function
// ============================================================================

/// Matrix sign function using Newton iteration
pub fn signm_impl(client: &CpuClient, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (a, original_dtype) = linalg_promote(client, a)?;
    let n = validate_square_matrix(a.shape())?;

    let result = match a.dtype() {
        DType::F32 => signm_typed::<f32>(client, &a, n),
        DType::F64 => signm_typed::<f64>(client, &a, n),
        _ => unreachable!(),
    }?;

    linalg_demote(client, result, original_dtype)
}

fn signm_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let val = data[0].to_f64();
        if val.abs() < f64::EPSILON {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "signm requires matrix with no zero eigenvalues".to_string(),
            });
        }
        let sign_val = if val > 0.0 { 1.0 } else { -1.0 };
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(sign_val)],
            &[1, 1],
            device,
        )?);
    }

    let eps = T::epsilon_val();

    // Convert to f64 for computation
    let a_data: Vec<T> = a.to_vec();
    let x_f64: Vec<f64> = a_data.iter().map(|x| x.to_f64()).collect();

    let result_f64 = newton_sign_iteration(&x_f64, n, eps, SIGNM_MAX_ITER)?;

    // Convert back to original type
    let result: Vec<T> = result_f64.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result,
        &[n, n],
        device,
    )?)
}

/// Newton iteration for matrix sign function
fn newton_sign_iteration(a: &[f64], n: usize, eps: f64, max_iter: usize) -> Result<Vec<f64>> {
    let mut x = a.to_vec();
    let tol = eps * (n as f64).sqrt();

    for _iter in 0..max_iter {
        // Compute X^{-1}
        let x_inv = matrix_functions_core::invert_matrix_f64(&x, n, eps).ok_or_else(|| {
            Error::Internal("signm: matrix became singular during iteration".to_string())
        })?;

        // X_new = (X + X^{-1}) / 2
        let mut x_new = vec![0.0; n * n];
        let mut max_diff: f64 = 0.0;

        for i in 0..(n * n) {
            let new_val = (x[i] + x_inv[i]) / 2.0;
            let diff = (new_val - x[i]).abs();
            max_diff = max_diff.max(diff);
            x_new[i] = new_val;
        }

        x = x_new;

        if max_diff < tol {
            break;
        }
    }

    Ok(x)
}

// ============================================================================
// Fractional Matrix Power
// ============================================================================

/// Fractional matrix power: A^p using log and exp
pub fn fractional_matrix_power_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    p: f64,
) -> Result<Tensor<CpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;

    match a.dtype() {
        DType::F32 => fractional_matrix_power_typed::<f32>(client, a, n, p),
        DType::F64 => fractional_matrix_power_typed::<f64>(client, a, n, p),
        _ => Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "fractional_matrix_power",
        }),
    }
}

fn fractional_matrix_power_typed<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
    p: f64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    // p = 0: Return identity
    if p.abs() < f64::EPSILON {
        let mut identity = vec![T::zero(); n * n];
        for i in 0..n {
            identity[i * n + i] = T::one();
        }
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &identity,
            &[n, n],
            device,
        )?);
    }

    // p = 1: Return A
    if (p - 1.0).abs() < f64::EPSILON {
        return Ok(a.clone());
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let val = data[0].to_f64();
        if val <= 0.0 && p.fract() != 0.0 {
            return Err(Error::InvalidArgument {
                arg: "a",
                reason: "fractional_matrix_power requires positive eigenvalues for non-integer p"
                    .to_string(),
            });
        }
        let result = val.powf(p);
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(result)],
            &[1, 1],
            device,
        )?);
    }

    // p = -1: Return inverse
    if (p + 1.0).abs() < f64::EPSILON {
        let data: Vec<T> = a.to_vec();
        let data_f64: Vec<f64> = data.iter().map(|x| x.to_f64()).collect();
        let inv = matrix_functions_core::invert_matrix_f64(&data_f64, n, f64::EPSILON).ok_or_else(
            || Error::InvalidArgument {
                arg: "a",
                reason: "matrix is singular (fractional_matrix_power p=-1)".to_string(),
            },
        )?;
        let result: Vec<T> = inv.iter().map(|&x| T::from_f64(x)).collect();
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &result,
            &[n, n],
            device,
        )?);
    }

    // p = 0.5: Use sqrtm (more accurate)
    if (p - 0.5).abs() < f64::EPSILON {
        return sqrtm_typed::<T>(client, a, n);
    }

    // Integer powers: use repeated squaring
    if p.fract() == 0.0 && p.abs() < 100.0 {
        return integer_matrix_power::<T>(client, a, n, p as i64);
    }

    // General case: A^p = exp(p * log(A))
    let log_a = logm_typed::<T>(client, a, n)?;
    let log_a_data: Vec<T> = log_a.to_vec();

    // Scale by p
    let p_log_a: Vec<T> = log_a_data
        .iter()
        .map(|x| T::from_f64(x.to_f64() * p))
        .collect();
    let p_log_a_tensor = Tensor::<CpuRuntime>::try_from_slice(&p_log_a, &[n, n], device)?;

    expm_typed::<T>(client, &p_log_a_tensor, n)
}

/// Integer matrix power using repeated squaring
fn integer_matrix_power<T: Element + LinalgElement>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
    p: i64,
) -> Result<Tensor<CpuRuntime>> {
    let device = client.device();

    if p == 0 {
        let mut identity = vec![T::zero(); n * n];
        for i in 0..n {
            identity[i * n + i] = T::one();
        }
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &identity,
            &[n, n],
            device,
        )?);
    }

    let data: Vec<T> = a.to_vec();
    let data_f64: Vec<f64> = data.iter().map(|x| x.to_f64()).collect();

    // Handle negative powers
    let (mut base, mut exp): (Vec<f64>, u64) = if p < 0 {
        let inv = matrix_functions_core::invert_matrix_f64(&data_f64, n, f64::EPSILON).ok_or_else(
            || Error::InvalidArgument {
                arg: "a",
                reason: "matrix is singular (integer_matrix_power)".to_string(),
            },
        )?;
        (inv, (-p) as u64)
    } else {
        (data_f64, p as u64)
    };

    // Identity matrix
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        result[i * n + i] = 1.0;
    }

    // Repeated squaring
    while exp > 0 {
        if exp & 1 == 1 {
            result = matrix_functions_core::matmul_square_f64(&result, &base, n);
        }
        base = matrix_functions_core::matmul_square_f64(&base, &base, n);
        exp >>= 1;
    }

    // Convert back to original type
    let result_t: Vec<T> = result.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result_t,
        &[n, n],
        device,
    )?)
}

// ============================================================================
// General Matrix Function
// ============================================================================

/// General matrix function f(A) using Schur-Parlett algorithm
pub fn funm_impl<F>(client: &CpuClient, a: &Tensor<CpuRuntime>, f: F) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64) -> f64,
{
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;

    match a.dtype() {
        DType::F32 => funm_typed::<f32, F>(client, a, n, f),
        DType::F64 => funm_typed::<f64, F>(client, a, n, f),
        _ => Err(Error::UnsupportedDType {
            dtype: a.dtype(),
            op: "funm",
        }),
    }
}

fn funm_typed<T: Element + LinalgElement, F>(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    n: usize,
    f: F,
) -> Result<Tensor<CpuRuntime>>
where
    F: Fn(f64) -> f64,
{
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[] as &[T],
            &[0, 0],
            device,
        )?);
    }

    if n == 1 {
        let data: Vec<T> = a.to_vec();
        let val = data[0].to_f64();
        let result = f(val);
        if result.is_nan() || result.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }
        return Ok(Tensor::<CpuRuntime>::try_from_slice(
            &[T::from_f64(result)],
            &[1, 1],
            device,
        )?);
    }

    // Compute Schur decomposition: A = Z @ T @ Z^T
    let schur = schur_decompose_impl(client, a)?;
    let z_data: Vec<T> = schur.z.to_vec();
    let t_data: Vec<T> = schur.t.to_vec();

    // Convert to f64
    let t_f64: Vec<f64> = t_data.iter().map(|x| x.to_f64()).collect();
    let z_f64: Vec<f64> = z_data.iter().map(|x| x.to_f64()).collect();

    // Use shared algorithm for f(T)
    let f_t = matrix_functions_core::funm_quasi_triangular_f64(&t_f64, n, &f)?;

    // Reconstruct: f(A) = Z @ f(T) @ Z^T
    let result_f64 = matrix_functions_core::reconstruct_from_schur_f64(&z_f64, &f_t, n);

    // Convert back to original type
    let result: Vec<T> = result_f64.iter().map(|&x| T::from_f64(x)).collect();
    Ok(Tensor::<CpuRuntime>::try_from_slice(
        &result,
        &[n, n],
        device,
    )?)
}
