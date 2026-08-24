//! Matrix operations for CUDA (inverse, det, trace, diag, diagflat, rank, norm)

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use super::svd::svd_decompose_impl;
use crate::algorithm::linalg::{
    LinearAlgebraAlgorithms, MatrixNormOrder, validate_linalg_dtype, validate_matrix_2d,
    validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CompareOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Matrix inverse via LU decomposition
pub fn inverse_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Compute LU decomposition
    let lu_result = client.lu_decompose(a)?;

    // Allocate output and temporary buffers
    let inv_size = n * n * dtype.size_in_bytes();
    let col_size = n * dtype.size_in_bytes();

    let inv_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let identity_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let pb_guard = AllocGuard::new(client.allocator(), col_size)?;
    let y_guard = AllocGuard::new(client.allocator(), col_size)?;
    let x_guard = AllocGuard::new(client.allocator(), col_size)?;
    let e_guard = AllocGuard::new(client.allocator(), col_size)?;

    let inv_ptr = inv_guard.ptr();
    let identity_ptr = identity_guard.ptr();
    let pb_ptr = pb_guard.ptr();
    let y_ptr = y_guard.ptr();
    let x_ptr = x_guard.ptr();
    let e_ptr = e_guard.ptr();

    // Create identity matrix on GPU (no CPU transfer)
    let result = unsafe {
        kernels::launch_create_identity(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            identity_ptr,
            n,
        )
    };
    result?;

    // Solve for each column of the identity matrix
    for col in 0..n {
        // Extract column from identity matrix (GPU-only)
        let result = unsafe {
            kernels::launch_extract_column(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                identity_ptr,
                e_ptr,
                n,
                n,
                col,
            )
        };
        result?;

        // Apply permutation: pb = P @ e
        let result = unsafe {
            kernels::launch_apply_lu_permutation(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                e_ptr,
                pb_ptr,
                lu_result.pivots.ptr(),
                n,
            )
        };
        result?;

        // Forward substitution: Ly = pb (L has unit diagonal)
        let result = unsafe {
            kernels::launch_forward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.ptr(),
                pb_ptr,
                y_ptr,
                n,
                true, // unit diagonal
            )
        };
        result?;

        // Backward substitution: Ux = y
        let result = unsafe {
            kernels::launch_backward_sub(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                lu_result.lu.ptr(),
                y_ptr,
                x_ptr,
                n,
            )
        };
        result?;

        // Scatter x into column of inverse matrix (GPU-only, no CPU transfer)
        let result = unsafe {
            kernels::launch_scatter_column(
                client.context(),
                client.stream(),
                device.index,
                dtype,
                x_ptr,
                inv_ptr,
                n,
                col,
            )
        };
        result?
    }

    client.synchronize();

    let inv = unsafe { CudaClient::tensor_from_raw(inv_guard.release(), &[n, n], dtype, device) };

    Ok(inv)
}

/// Determinant via LU decomposition
pub fn det_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Compute LU decomposition
    let lu_result = client.lu_decompose(a)?;

    // Allocate output
    let det_size = dtype.size_in_bytes();
    let det_guard = AllocGuard::new(client.allocator(), det_size)?;
    let det_ptr = det_guard.ptr();

    // Compute determinant from LU diagonal
    unsafe {
        kernels::launch_det_from_lu(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            lu_result.lu.ptr(),
            det_ptr,
            n,
            lu_result.num_swaps as i32,
        )?;
    }

    client.synchronize();

    let det = unsafe { CudaClient::tensor_from_raw(det_guard.release(), &[], dtype, device) };

    Ok(det)
}

/// Matrix trace (sum of diagonal elements)
pub fn trace_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    // Allocate output (zero-initialized for atomic add)
    let trace_size = dtype.size_in_bytes();
    let trace_guard = AllocGuard::new(client.allocator(), trace_size)?;
    let trace_ptr = trace_guard.ptr();

    let zero_bytes = vec![0u8; trace_size];
    CudaRuntime::copy_to_device(&zero_bytes, trace_ptr, device)?;

    unsafe {
        kernels::launch_trace(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            trace_ptr,
            min_dim,
            n, // stride (number of columns)
        )?;
    }

    client.synchronize();

    let trace = unsafe { CudaClient::tensor_from_raw(trace_guard.release(), &[], dtype, device) };

    Ok(trace)
}

/// Extract diagonal elements
pub fn diag_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    let diag_size = min_dim * dtype.size_in_bytes();
    let diag_guard = AllocGuard::new(client.allocator(), diag_size)?;
    let diag_ptr = diag_guard.ptr();

    unsafe {
        kernels::launch_diag(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            diag_ptr,
            min_dim,
            n,
        )?;
    }

    client.synchronize();

    let diag =
        unsafe { CudaClient::tensor_from_raw(diag_guard.release(), &[min_dim], dtype, device) };

    Ok(diag)
}

/// Create diagonal matrix from vector
pub fn diagflat_impl(client: &CudaClient, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;

    // Input must be 1D
    if a.shape().len() != 1 {
        return Err(Error::Internal(format!(
            "diagflat requires 1D input tensor, got {}D tensor with shape {:?}",
            a.shape().len(),
            a.shape()
        )));
    }

    let n = a.shape()[0];
    let dtype = a.dtype();
    let device = client.device();

    let out_size = n * n * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_diagflat(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            out_ptr,
            n,
        )?;
    }

    client.synchronize();

    let out = unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[n, n], dtype, device) };

    Ok(out)
}

/// Matrix rank via QR decomposition - runs entirely on GPU (zero CPU transfers)
pub fn matrix_rank_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let k = m.min(n);

    // Handle empty matrix
    if k == 0 {
        return Tensor::<CudaRuntime>::from_slice(&[0i64], &[], a.device());
    }

    // Compute tolerance factor (depends only on dimensions, no GPU data needed)
    let base_tol = tol.unwrap_or_else(|| {
        let eps = match dtype {
            DType::F32 => f32::EPSILON as f64,
            DType::F64 => f64::EPSILON,
            _ => f32::EPSILON as f64,
        };
        (m.max(n) as f64) * eps
    });

    // Use QR decomposition to estimate rank
    let qr = client.qr_decompose(a)?;

    // Get diagonal of R
    let r_diag = LinearAlgebraAlgorithms::diag(client, &qr.r)?;

    // Compute abs(r_diag) on GPU
    let abs_diag = client.abs(&r_diag)?;

    // Compute max(abs(r_diag)) on GPU - returns scalar tensor
    let max_val = client.max(&abs_diag, &[], false)?;

    // Compute threshold = base_tol * max on GPU
    let threshold = client.mul_scalar(&max_val, base_tol)?;

    // Compare abs_diag > threshold on GPU (broadcasts threshold)
    // CUDA comparisons return same dtype (0.0/1.0), not Bool
    let above_mask = client.gt(&abs_diag, &threshold)?;

    // Sum the mask directly (values are 0.0 or 1.0)
    let rank_float = client.sum(&above_mask, &[], false)?;

    // Cast to I64 for integer result
    let rank_tensor = client.cast(&rank_float, DType::I64)?;

    Ok(rank_tensor)
}

/// Matrix norm
pub fn matrix_norm_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;

    match ord {
        MatrixNormOrder::Frobenius => {
            // Frobenius norm: ||A||_F = sqrt(sum(A²))
            // Use existing tensor ops to keep data on GPU
            let squared = client.square(a)?;
            let sum_sq = client.sum(&squared, &[], false)?;
            client.sqrt(&sum_sq)
        }
        MatrixNormOrder::Spectral => {
            // Spectral norm: ||A||_2 = max(singular_values(A))
            let svd = svd_decompose_impl(client, a)?;
            client.max(&svd.s, &[], false)
        }
        MatrixNormOrder::Nuclear => {
            // Nuclear norm: ||A||_* = sum(singular_values(A))
            let svd = svd_decompose_impl(client, a)?;
            client.sum(&svd.s, &[], false)
        }
    }
}

/// Kronecker product: A ⊗ B
pub fn kron_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

    let (m_a, n_a) = validate_matrix_2d(a.shape())?;
    let (m_b, n_b) = validate_matrix_2d(b.shape())?;

    let dtype = a.dtype();
    let device = client.device();

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let out_size = m_out * n_out * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_kron(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            b.ptr(),
            out_ptr,
            m_a,
            n_a,
            m_b,
            n_b,
        )?;
    }

    client.synchronize();

    let out =
        unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[m_out, n_out], dtype, device) };

    Ok(out)
}

/// Khatri-Rao product (column-wise Kronecker): A ⊙ B
///
/// For A of shape [m, k] and B of shape [n, k],
/// produces output of shape [m * n, k].
///
/// (A ⊙ B)[i*n + j, c] = A[i, c] * B[j, c]
pub fn khatri_rao_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            lhs: a.dtype(),
            rhs: b.dtype(),
        });
    }

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
    let dtype = a.dtype();
    let device = client.device();

    let m_out = m * n;
    let out_size = m_out * k * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();

    unsafe {
        kernels::launch_khatri_rao(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            a.ptr(),
            b.ptr(),
            out_ptr,
            m,
            n,
            k,
        )?;
    }

    client.synchronize();

    let out =
        unsafe { CudaClient::tensor_from_raw(out_guard.release(), &[m_out, k], dtype, device) };

    Ok(out)
}

/// Upper triangular part of a matrix — delegates to impl_generic
pub fn triu_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    diagonal: i64,
) -> Result<Tensor<CudaRuntime>> {
    crate::ops::impl_generic::triu_impl(client, a, diagonal)
}

/// Lower triangular part of a matrix — delegates to impl_generic
pub fn tril_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    diagonal: i64,
) -> Result<Tensor<CudaRuntime>> {
    crate::ops::impl_generic::tril_impl(client, a, diagonal)
}

/// Sign and log-absolute-determinant — delegates to impl_generic
pub fn slogdet_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<crate::algorithm::linalg::SlogdetResult<CudaRuntime>> {
    crate::ops::impl_generic::slogdet_impl(client, a)
}
