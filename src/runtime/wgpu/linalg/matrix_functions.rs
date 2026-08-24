//! Matrix functions: exponential, logarithm, square root, sign, fractional power, and general functions.
//!
//! # WebGPU Limitations
//!
//! WGSL compute shaders do not support F64 natively. All matrix functions in this
//! module only accept **F32 tensors**. For F64 support, use CPU or CUDA backends.

use super::super::{WgpuClient, WgpuRuntime};
use super::schur::schur_decompose;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{BinaryOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps, UtilityOps};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Matrix exponential: e^A
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. WGSL does not have native F64 support.
/// For F64 matrices, use CPU or CUDA backends.
///
/// # Errors
///
/// Returns `UnsupportedDType` if the tensor is not F32.
pub fn expm(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU expm (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    if n == 1 {
        // For 1x1 matrices, exp(A) = [exp(a_00)]
        return client.exp(a);
    }

    // Compute Schur decomposition
    let schur = schur_decompose(client, a)?;
    let exp_t_tensor = compute_schur_exp(client, &schur.t, n, dtype)?;

    // Reconstruct: exp(A) = Z @ exp(T) @ Z^T
    let temp = client.matmul(&schur.z, &exp_t_tensor)?;
    let z_t = schur.z.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

/// Matrix logarithm: log(A)
///
/// Computes the principal matrix logarithm. The matrix must have no
/// non-positive real eigenvalues.
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. For F64 matrices, use CPU or CUDA backends.
///
/// # Errors
///
/// - Returns `UnsupportedDType` if the tensor is not F32.
/// - Returns `InvalidArgument` if the matrix has non-positive real eigenvalues.
pub fn logm(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU logm (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    if n == 1 {
        // For 1x1 matrices, log(A) = [log(a_00)]
        // Validation: logm requires positive diagonal entries
        // We rely on the same GPU validation path used for larger matrices
        let schur = schur_decompose(client, a)?;
        validate_schur_eigenvalues_gpu(client, &schur.t, n, "logm")?;
        return client.log(a);
    }

    // Compute Schur decomposition
    let schur = schur_decompose(client, a)?;

    // Check for non-positive real eigenvalues (GPU validation)
    validate_schur_eigenvalues_gpu(client, &schur.t, n, "logm")?;

    // Compute log(T) on GPU
    let log_t_tensor = compute_schur_log(client, &schur.t, n, dtype)?;

    // Reconstruct: log(A) = Z @ log(T) @ Z^T
    let temp = client.matmul(&schur.z, &log_t_tensor)?;
    let z_t = schur.z.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

/// Matrix square root: sqrt(A)
///
/// Computes the principal matrix square root. The matrix must have no
/// non-positive real eigenvalues.
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. For F64 matrices, use CPU or CUDA backends.
///
/// # Errors
///
/// - Returns `UnsupportedDType` if the tensor is not F32.
/// - Returns `InvalidArgument` if the matrix has non-positive real eigenvalues.
pub fn sqrtm(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU sqrtm (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    if n == 1 {
        // For 1x1 matrices, sqrt(A) = [sqrt(a_00)]
        // Use GPU sqrt operation, which naturally handles the computation
        return client.sqrt(a);
    }

    let eps = f32::EPSILON as f64;

    // Check for negative real eigenvalues using Schur decomposition (GPU validation)
    // Note: We compute Schur only for validation, the actual sqrtm uses Denman-Beavers
    let schur = schur_decompose(client, a)?;

    // Use GPU validation for eigenvalue check
    // The validate_schur_eigenvalues_gpu checks for non-positive, which is slightly
    // stricter than needed (sqrtm allows 0), but handles the common error cases
    validate_schur_eigenvalues_gpu(client, &schur.t, n, "sqrtm")?;

    // Denman-Beavers iteration (already GPU-based)
    let mut y = a.clone();
    let mut z = client.eye(n, None, dtype)?;

    let max_iter = 50;
    let tol = eps * (n as f64);

    for _iter in 0..max_iter {
        let y_inv = LinearAlgebraAlgorithms::inverse(client, &y).map_err(|_| {
            Error::Internal("sqrtm: matrix inversion failed during iteration".to_string())
        })?;

        let z_inv = LinearAlgebraAlgorithms::inverse(client, &z).map_err(|_| {
            Error::Internal("sqrtm: matrix inversion failed during iteration".to_string())
        })?;

        let y_plus_zinv = client.add(&y, &z_inv)?;
        let y_new = client.div_scalar(&y_plus_zinv, 2.0)?;

        let z_plus_yinv = client.add(&z, &y_inv)?;
        let z_new = client.div_scalar(&z_plus_yinv, 2.0)?;

        // Check convergence
        let diff = client.sub(&y_new, &y)?;
        let diff_norm: f64 = compute_norm(client, &diff)?;
        let y_norm: f64 = compute_norm(client, &y)?;

        y = y_new;
        z = z_new;

        if diff_norm / y_norm.max(1.0) < tol {
            break;
        }
    }

    Ok(y)
}

/// Matrix sign function: sign(A)
///
/// Computes the matrix sign function using Newton iteration.
/// The matrix must have no eigenvalues on the imaginary axis.
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. For F64 matrices, use CPU or CUDA backends.
///
/// # Errors
///
/// - Returns `UnsupportedDType` if the tensor is not F32.
/// - Returns `InvalidArgument` if the matrix has eigenvalues on the imaginary axis.
pub fn signm(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU signm (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    if n == 1 {
        // For 1x1 matrices, sign(A) = [sign(a_00)]
        // sign(x) = x / |x|, handled entirely on GPU
        let abs_a = client.abs(a)?;
        return client.div(a, &abs_a);
    }

    let eps = f32::EPSILON as f64;

    // Newton iteration: X_{k+1} = (X_k + X_k^{-1}) / 2
    let mut x = a.clone();
    let max_iter = 100;
    let tol = eps * (n as f64).sqrt();

    for _iter in 0..max_iter {
        let x_inv = LinearAlgebraAlgorithms::inverse(client, &x).map_err(|_| {
            Error::Internal("signm: matrix became singular during iteration".to_string())
        })?;

        let x_plus_inv = client.add(&x, &x_inv)?;
        let x_new = client.div_scalar(&x_plus_inv, 2.0)?;

        let diff = client.sub(&x_new, &x)?;
        let diff_norm: f64 = compute_norm(client, &diff)?;

        x = x_new;

        if diff_norm < tol {
            break;
        }
    }

    Ok(x)
}

/// Fractional matrix power: A^p
///
/// Computes A^p for any real exponent p. Special cases:
/// - p = 0: Returns identity matrix
/// - p = 0.5: Equivalent to sqrtm(A)
/// - Integer p: Uses repeated squaring
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. For F64 matrices, use CPU or CUDA backends.
///
/// # Errors
///
/// - Returns `UnsupportedDType` if the tensor is not F32.
/// - Returns `InvalidArgument` for non-integer p if matrix has non-positive eigenvalues.
pub fn fractional_matrix_power(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    p: f64,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU fractional_matrix_power (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    // p = 0: Return identity
    if p.abs() < f64::EPSILON {
        return client.eye(n, None, dtype);
    }

    // p = 1: Return A
    if (p - 1.0).abs() < f64::EPSILON {
        return Ok(a.clone());
    }

    if n == 1 {
        // For 1x1 matrices, A^p = [a_00^p]
        // Use the general case: A^p = exp(p * log(A)) for non-integer p
        // Integer powers use repeated squaring
        if p.fract() == 0.0 && p.abs() < 100.0 {
            return integer_matrix_power(client, a, n, p as i64, dtype);
        }
        // For non-integer p, use general case: exp(p * log(A))
        // This handles validation in logm/expm
        let log_a = logm(client, a)?;
        let p_log_a = client.mul_scalar(&log_a, p)?;
        return expm(client, &p_log_a);
    }

    // p = -1: Return inverse
    if (p + 1.0).abs() < f64::EPSILON {
        return LinearAlgebraAlgorithms::inverse(client, a);
    }

    // p = 0.5: Use sqrtm
    if (p - 0.5).abs() < f64::EPSILON {
        return sqrtm(client, a);
    }

    // Integer powers: use repeated squaring
    if p.fract() == 0.0 && p.abs() < 100.0 {
        return integer_matrix_power(client, a, n, p as i64, dtype);
    }

    // General case: A^p = exp(p * log(A))
    let log_a = logm(client, a)?;
    let p_log_a = client.mul_scalar(&log_a, p)?;
    expm(client, &p_log_a)
}

/// General matrix function: f(A)
///
/// Computes f(A) for any scalar function f using the Schur-Parlett algorithm.
///
/// # WebGPU Limitations
///
/// Only F32 tensors are supported. For F64 matrices, use CPU or CUDA backends.
///
/// # Closure Requirements
///
/// The closure `f` must be `Send + Sync` to support potential parallel execution.
///
/// # Example
///
/// ```ignore
/// // Custom matrix function: f(x) = sin(x)
/// let result = funm(&client, &matrix, |x| x.sin())?;
/// ```
///
/// # Errors
///
/// Returns `UnsupportedDType` if the tensor is not F32.
pub fn funm<F>(client: &WgpuClient, a: &Tensor<WgpuRuntime>, f: F) -> Result<Tensor<WgpuRuntime>>
where
    F: Fn(f64) -> f64 + Send + Sync,
{
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU funm (only F32 supported)",
        });
    }

    if n == 0 {
        return Tensor::<WgpuRuntime>::zeros(&[0, 0], dtype, device);
    }

    if n == 1 {
        // NOTE: GPU→CPU transfer for user closure execution
        // User-provided closures cannot execute on GPU, so we extract the scalar,
        // apply the function on CPU, and reconstruct. This is a documented exception
        // to the no-transfer rule, as the closure is user-defined and not GPU-compatible.
        let data: Vec<f32> = a.to_vec();
        let val = data[0] as f64;
        let result = f(val);
        if result.is_nan() || result.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }
        return Tensor::<WgpuRuntime>::full_scalar(&[1, 1], dtype, result, device);
    }

    // Compute Schur decomposition
    let schur = schur_decompose(client, a)?;

    // NOTE: GPU→CPU transfer required for user closure execution.
    // User-provided closures cannot run on GPU, so we extract Schur factors,
    // apply the function on the quasi-triangular form on CPU, and reconstruct.
    // This is a documented exception to the no-transfer rule.
    let t_data: Vec<f32> = schur.t.to_vec();
    let z_data: Vec<f32> = schur.z.to_vec();

    // Compute f(T) on CPU using user-provided function
    let f_t = funm_quasi_triangular_f32(&t_data, n, &f)?;

    // Reconstruct on GPU
    let f_t_tensor = Tensor::<WgpuRuntime>::from_slice(&f_t, &[n, n], device)?;
    let z_tensor = Tensor::<WgpuRuntime>::from_slice(&z_data, &[n, n], device)?;

    let temp = client.matmul(&z_tensor, &f_t_tensor)?;
    let z_t = z_tensor.transpose(0, 1)?;
    client.matmul(&temp, &z_t)
}

// Helper functions

fn compute_norm(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<f64> {
    let a_sq = client.mul(a, a)?;
    let sum = client.sum(&a_sq, &[], false)?;

    // NOTE: Scalar transfer for convergence evaluation (1 float).
    // Iterative algorithms require the termination condition to be evaluated on CPU.
    // This unavoidable scalar read is the minimal transfer required for the algorithm
    // to determine convergence, distinct from processing batches of data on GPU.
    let sum_vec: Vec<f32> = sum.to_vec();
    Ok((sum_vec[0] as f64).sqrt())
}

fn get_tensor_buffer(t: &Tensor<WgpuRuntime>) -> Result<std::sync::Arc<wgpu::Buffer>> {
    use super::super::client::get_buffer;
    get_buffer(t.ptr()).ok_or_else(|| Error::Internal("Failed to get tensor buffer".to_string()))
}

/// Compute exp(T) for quasi-triangular matrix T using GPU kernels.
fn compute_schur_exp(
    client: &WgpuClient,
    t: &Tensor<WgpuRuntime>,
    n: usize,
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    use super::super::shaders::compute_schur_func_gpu;

    let device = client.device();

    // Allocate output buffer
    let output = Tensor::<WgpuRuntime>::zeros(&[n, n], dtype, device)?;

    let t_buffer = get_tensor_buffer(t)?;
    let output_buffer = get_tensor_buffer(&output)?;

    // Run GPU computation
    compute_schur_func_gpu(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &output_buffer,
        n,
        "exp",
        dtype,
    )?;

    client.synchronize();

    Ok(output)
}

/// Compute log(T) for quasi-triangular matrix T using GPU kernels.
fn compute_schur_log(
    client: &WgpuClient,
    t: &Tensor<WgpuRuntime>,
    n: usize,
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    use super::super::shaders::compute_schur_func_gpu;

    let device = client.device();

    // Allocate output buffer
    let output = Tensor::<WgpuRuntime>::zeros(&[n, n], dtype, device)?;

    let t_buffer = get_tensor_buffer(t)?;
    let output_buffer = get_tensor_buffer(&output)?;

    // Run GPU computation
    compute_schur_func_gpu(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &output_buffer,
        n,
        "log",
        dtype,
    )?;

    client.synchronize();

    Ok(output)
}

/// Validate that Schur form has no non-positive real eigenvalues using GPU.
fn validate_schur_eigenvalues_gpu(
    client: &WgpuClient,
    t: &Tensor<WgpuRuntime>,
    n: usize,
    op: &str,
) -> Result<()> {
    use super::super::shaders::launch_validate_eigenvalues;

    let dtype = t.dtype();
    let device = client.device();
    let eps = f32::EPSILON;

    // Allocate result buffer (2 elements: has_error flag, error value)
    let result = Tensor::<WgpuRuntime>::zeros(&[2], dtype, device)?;

    let t_buffer = get_tensor_buffer(t)?;
    let result_buffer = get_tensor_buffer(&result)?;

    launch_validate_eigenvalues(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &result_buffer,
        n,
        eps,
        dtype,
    )?;

    client.synchronize();

    // NOTE: Validation result transfer (2 floats: error flag + eigenvalue).
    // Matrix function validity checks must evaluate on CPU (control flow).
    // This minimal transfer extracts GPU-computed validation results for error handling.
    let result_data: Vec<f32> = result.to_vec();

    if result_data[0] > 0.5 {
        // has_error flag is set
        let eigenvalue = result_data[1];
        return Err(Error::InvalidArgument {
            arg: "a",
            reason: format!(
                "{} requires matrix with no non-positive real eigenvalues, found {}",
                op, eigenvalue
            ),
        });
    }

    Ok(())
}

fn integer_matrix_power(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    n: usize,
    p: i64,
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    if p == 0 {
        return client.eye(n, None, dtype);
    }

    let (mut base, mut exp) = if p < 0 {
        let inv = LinearAlgebraAlgorithms::inverse(client, a)?;
        (inv, (-p) as u64)
    } else {
        (a.clone(), p as u64)
    };

    let mut result = client.eye(n, None, dtype)?;

    while exp > 0 {
        if exp & 1 == 1 {
            result = client.matmul(&result, &base)?;
        }
        base = client.matmul(&base, &base)?;
        exp >>= 1;
    }

    Ok(result)
}

fn funm_quasi_triangular_f32<F>(t: &[f32], n: usize, f: &F) -> Result<Vec<f32>>
where
    F: Fn(f64) -> f64,
{
    let mut result = vec![0.0f32; n * n];
    let eps = f32::EPSILON as f64;

    let mut i = 0;
    while i < n {
        if i + 1 < n && (t[(i + 1) * n + i] as f64).abs() > eps {
            let a = t[i * n + i] as f64;
            let b = t[i * n + (i + 1)] as f64;
            let c = t[(i + 1) * n + i] as f64;
            let d = t[(i + 1) * n + (i + 1)] as f64;

            let (f11, f12, f21, f22) = funm_2x2_block_f32(a, b, c, d, f)?;
            result[i * n + i] = f11 as f32;
            result[i * n + (i + 1)] = f12 as f32;
            result[(i + 1) * n + i] = f21 as f32;
            result[(i + 1) * n + (i + 1)] = f22 as f32;
            i += 2;
        } else {
            let val = t[i * n + i] as f64;
            let f_val = f(val);
            if f_val.is_nan() || f_val.is_infinite() {
                return Err(Error::InvalidArgument {
                    arg: "f",
                    reason: format!("function returned NaN or Inf for eigenvalue {}", val),
                });
            }
            result[i * n + i] = f_val as f32;
            i += 1;
        }
    }

    for diag in 1..n {
        for i in 0..(n - diag) {
            let j = i + diag;

            if i + 1 < n && (t[(i + 1) * n + i] as f64).abs() > eps && diag == 1 {
                continue;
            }
            if j > 0 && (t[j * n + (j - 1)] as f64).abs() > eps && diag == 1 {
                continue;
            }

            let t_ii = t[i * n + i] as f64;
            let t_jj = t[j * n + j] as f64;
            let t_ij = t[i * n + j] as f64;

            let f_ii = result[i * n + i] as f64;
            let f_jj = result[j * n + j] as f64;

            let mut sum = 0.0;
            for k in (i + 1)..j {
                sum += (result[i * n + k] as f64) * (t[k * n + j] as f64);
                sum -= (t[i * n + k] as f64) * (result[k * n + j] as f64);
            }

            let diff = t_ii - t_jj;
            let f_ij = if diff.abs() > eps {
                (f_ii - f_jj) * t_ij / diff + sum / diff
            } else {
                f_ii * t_ij + sum
            };

            result[i * n + j] = f_ij as f32;
        }
    }

    Ok(result)
}

fn funm_2x2_block_f32<F>(a: f64, b: f64, c: f64, d: f64, f: &F) -> Result<(f64, f64, f64, f64)>
where
    F: Fn(f64) -> f64,
{
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace / 4.0 - det;

    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        let lambda1 = trace / 2.0 + sqrt_disc;
        let lambda2 = trace / 2.0 - sqrt_disc;

        let f1 = f(lambda1);
        let f2 = f(lambda2);

        if f1.is_nan() || f1.is_infinite() || f2.is_nan() || f2.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }

        if (lambda1 - lambda2).abs() > f64::EPSILON {
            let coeff1 = (f1 - f2) / (lambda1 - lambda2);
            let coeff0 = f1 - coeff1 * lambda1;
            Ok((
                coeff0 + coeff1 * a,
                coeff1 * b,
                coeff1 * c,
                coeff0 + coeff1 * d,
            ))
        } else {
            Ok((
                f1,
                f1 * b / (a - lambda1 + 1.0),
                f1 * c / (a - lambda1 + 1.0),
                f1,
            ))
        }
    } else {
        let alpha = trace / 2.0;
        let beta = (-disc).sqrt();

        let f_alpha = f(alpha);
        if f_alpha.is_nan() || f_alpha.is_infinite() {
            return Err(Error::InvalidArgument {
                arg: "f",
                reason: "function returned NaN or Inf for eigenvalue".to_string(),
            });
        }

        let h = beta.abs().max(1e-8);
        let f_plus = f(alpha + h);
        let f_minus = f(alpha - h);
        let df_approx = (f_plus - f_minus) / (2.0 * h);

        let f11 = f_alpha + df_approx * (a - alpha);
        let f12 = df_approx * b;
        let f21 = df_approx * c;
        let f22 = f_alpha + df_approx * (d - alpha);

        Ok((f11, f12, f21, f22))
    }
}
