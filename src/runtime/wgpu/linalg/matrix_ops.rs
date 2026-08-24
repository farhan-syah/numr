//! Matrix operations: inverse, determinant, trace, diagonal, rank, and norm.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use crate::algorithm::linalg::{
    MatrixNormOrder, validate_linalg_dtype, validate_matrix_2d, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CompareOps, LinalgOps, ReduceOps, ScalarOps, TypeConversionOps, UnaryOps};
use crate::runtime::{AllocGuard, RuntimeClient};
use crate::tensor::Tensor;

pub fn inverse(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU inverse (only F32 supported)",
        });
    }

    // Compute LU decomposition using decompositions module
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Allocate output and temporary buffers
    let inv_size = n * n * dtype.size_in_bytes();
    let inv_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let inv_ptr = inv_guard.ptr();
    let inv_buffer = get_buffer(inv_ptr)
        .ok_or_else(|| Error::Internal("Failed to get inv buffer".to_string()))?;

    let col_size = n * dtype.size_in_bytes();

    // Create identity matrix on GPU
    let identity_guard = AllocGuard::new(client.allocator(), inv_size)?;
    let identity_ptr = identity_guard.ptr();
    let identity_buffer = get_buffer(identity_ptr)
        .ok_or_else(|| Error::Internal("Failed to get identity buffer".to_string()))?;

    let id_params: [u32; 1] = [n as u32];
    let id_params_buffer = client.create_uniform_buffer("identity_params", 4);
    client.write_buffer(&id_params_buffer, &id_params);

    kernels::launch_create_identity(
        client.pipeline_cache(),
        &client.queue,
        &identity_buffer,
        &id_params_buffer,
        n,
        dtype,
    )?;

    // Get LU and pivots buffers (both already on GPU, no transfers needed)
    let lu_buffer = get_buffer(lu_result.lu.ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;
    let pivots_buffer = get_buffer(lu_result.pivots.ptr())
        .ok_or_else(|| Error::Internal("Failed to get pivots buffer".to_string()))?;

    // Allocate temporary buffers
    let e_guard = AllocGuard::new(client.allocator(), col_size)?;
    let e_ptr = e_guard.ptr();
    let pb_guard = AllocGuard::new(client.allocator(), col_size)?;
    let pb_ptr = pb_guard.ptr();
    let y_guard = AllocGuard::new(client.allocator(), col_size)?;
    let y_ptr = y_guard.ptr();
    let x_guard = AllocGuard::new(client.allocator(), col_size)?;
    let x_ptr = x_guard.ptr();

    let e_buffer =
        get_buffer(e_ptr).ok_or_else(|| Error::Internal("Failed to get e buffer".to_string()))?;
    let pb_buffer =
        get_buffer(pb_ptr).ok_or_else(|| Error::Internal("Failed to get pb buffer".to_string()))?;
    let y_buffer =
        get_buffer(y_ptr).ok_or_else(|| Error::Internal("Failed to get y buffer".to_string()))?;
    let x_buffer =
        get_buffer(x_ptr).ok_or_else(|| Error::Internal("Failed to get x buffer".to_string()))?;

    // Solve for each column of the identity matrix
    for col in 0..n {
        // Extract column from identity matrix
        let extract_params: [u32; 3] = [n as u32, n as u32, col as u32];
        let extract_params_buffer = client.create_uniform_buffer("extract_params", 12);
        client.write_buffer(&extract_params_buffer, &extract_params);

        kernels::launch_extract_column(
            client.pipeline_cache(),
            &client.queue,
            &identity_buffer,
            &e_buffer,
            &extract_params_buffer,
            n,
            dtype,
        )?;

        // Apply permutation: pb = P @ e
        let perm_params: [u32; 1] = [n as u32];
        let perm_params_buffer = client.create_uniform_buffer("perm_params", 4);
        client.write_buffer(&perm_params_buffer, &perm_params);

        kernels::launch_apply_lu_permutation(
            client.pipeline_cache(),
            &client.queue,
            &e_buffer,
            &pb_buffer,
            &pivots_buffer,
            &perm_params_buffer,
            dtype,
        )?;

        // Forward substitution: Ly = pb (L has unit diagonal)
        let forward_params: [u32; 2] = [n as u32, 1];
        let forward_params_buffer = client.create_uniform_buffer("forward_params", 8);
        client.write_buffer(&forward_params_buffer, &forward_params);

        kernels::launch_forward_sub(
            client.pipeline_cache(),
            &client.queue,
            &lu_buffer,
            &pb_buffer,
            &y_buffer,
            &forward_params_buffer,
            dtype,
        )?;

        // Backward substitution: Ux = y
        let backward_params: [u32; 1] = [n as u32];
        let backward_params_buffer = client.create_uniform_buffer("backward_params", 4);
        client.write_buffer(&backward_params_buffer, &backward_params);

        kernels::launch_backward_sub(
            client.pipeline_cache(),
            &client.queue,
            &lu_buffer,
            &y_buffer,
            &x_buffer,
            &backward_params_buffer,
            dtype,
        )?;

        // Scatter x into column of inverse matrix
        let scatter_params: [u32; 2] = [n as u32, col as u32];
        let scatter_params_buffer = client.create_uniform_buffer("scatter_params", 8);
        client.write_buffer(&scatter_params_buffer, &scatter_params);

        kernels::launch_scatter_column(
            client.pipeline_cache(),
            &client.queue,
            &x_buffer,
            &inv_buffer,
            &scatter_params_buffer,
            n,
            dtype,
        )?;
    }

    client.synchronize();

    // Guards will automatically deallocate temporary buffers on drop
    drop(identity_guard);
    drop(e_guard);
    drop(pb_guard);
    drop(y_guard);
    drop(x_guard);

    let inv = unsafe { WgpuClient::tensor_from_raw(inv_guard.release(), &[n, n], dtype, device) };

    Ok(inv)
}

pub fn det(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU det (only F32 supported)",
        });
    }

    // Compute LU decomposition
    use super::decompositions::lu_decompose;
    let lu_result = lu_decompose(client, a)?;

    // Allocate output
    let det_size = dtype.size_in_bytes();
    let det_guard = AllocGuard::new(client.allocator(), det_size)?;
    let det_ptr = det_guard.ptr();
    let det_buffer = get_buffer(det_ptr)
        .ok_or_else(|| Error::Internal("Failed to get det buffer".to_string()))?;

    let lu_buffer = get_buffer(lu_result.lu.ptr())
        .ok_or_else(|| Error::Internal("Failed to get lu buffer".to_string()))?;

    // Create params buffer with n and num_swaps
    let params: [u32; 2] = [n as u32, lu_result.num_swaps as u32];
    let params_buffer = client.create_uniform_buffer("det_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_det_from_lu(
        client.pipeline_cache(),
        &client.queue,
        &lu_buffer,
        &det_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    let det = unsafe { WgpuClient::tensor_from_raw(det_guard.release(), &[], dtype, device) };

    Ok(det)
}

pub fn trace(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU trace (only F32 supported)",
        });
    }

    // Allocate output (zero-initialized for reduction)
    let trace_size = dtype.size_in_bytes();
    let trace_guard = AllocGuard::new(client.allocator(), trace_size)?;
    let trace_ptr = trace_guard.ptr();
    let trace_buffer = get_buffer(trace_ptr)
        .ok_or_else(|| Error::Internal("Failed to get trace buffer".to_string()))?;

    let zero: [f32; 1] = [0.0];
    client.write_buffer(&trace_buffer, &zero);

    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 2] = [min_dim as u32, n as u32];
    let params_buffer = client.create_uniform_buffer("trace_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_trace(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &trace_buffer,
        &params_buffer,
        min_dim,
        dtype,
    )?;

    client.synchronize();

    let trace = unsafe { WgpuClient::tensor_from_raw(trace_guard.release(), &[], dtype, device) };

    Ok(trace)
}

pub fn diag(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let min_dim = m.min(n);
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU diag (only F32 supported)",
        });
    }

    let diag_size = min_dim * dtype.size_in_bytes();
    let diag_guard = AllocGuard::new(client.allocator(), diag_size)?;
    let diag_ptr = diag_guard.ptr();
    let diag_buffer = get_buffer(diag_ptr)
        .ok_or_else(|| Error::Internal("Failed to get diag buffer".to_string()))?;

    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 2] = [min_dim as u32, n as u32];
    let params_buffer = client.create_uniform_buffer("diag_params", 8);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_diag(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &diag_buffer,
        &params_buffer,
        min_dim,
        dtype,
    )?;

    client.synchronize();

    let diag =
        unsafe { WgpuClient::tensor_from_raw(diag_guard.release(), &[min_dim], dtype, device) };

    Ok(diag)
}

pub fn diagflat(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;

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

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU diagflat (only F32 supported)",
        });
    }

    let out_size = n * n * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();
    let out_buffer = get_buffer(out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("diagflat_params", 4);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_diagflat(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &out_buffer,
        &params_buffer,
        n,
        dtype,
    )?;

    client.synchronize();

    let out = unsafe { WgpuClient::tensor_from_raw(out_guard.release(), &[n, n], dtype, device) };

    Ok(out)
}

/// Matrix rank via QR decomposition - runs entirely on GPU (zero CPU transfers)
pub fn matrix_rank(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    tol: Option<f64>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let k = m.min(n);

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU matrix_rank (only F32 supported)",
        });
    }

    // Handle empty matrix
    if k == 0 {
        return Tensor::<WgpuRuntime>::try_from_slice(&[0i64], &[], a.device());
    }

    // Compute tolerance factor (depends only on dimensions, no GPU data needed)
    let base_tol = tol.unwrap_or_else(|| {
        let eps = f32::EPSILON as f64;
        (m.max(n) as f64) * eps
    });

    // Use QR decomposition to estimate rank
    use super::decompositions::qr_decompose_internal;
    let qr = qr_decompose_internal(client, a, false)?;

    // Get diagonal of R
    let r_diag = client.diag(&qr.r)?;

    // Compute abs(r_diag) on GPU
    let abs_diag = client.abs(&r_diag)?;

    // Compute max(abs(r_diag)) on GPU - returns scalar tensor
    let max_val = client.max(&abs_diag, &[], false)?;

    // Compute threshold = base_tol * max on GPU
    let threshold = client.mul_scalar(&max_val, base_tol)?;

    // Compare abs_diag > threshold on GPU (broadcasts threshold)
    let above_mask = client.gt(&abs_diag, &threshold)?;

    // Cast bool mask to F32 for counting (WebGPU sum only supports F32)
    let above_f32 = client.cast(&above_mask, DType::F32)?;

    // Sum to get rank (count of true values)
    let rank_f32 = client.sum(&above_f32, &[], false)?;

    // Cast to I64 for final result
    let rank_tensor = client.cast(&rank_f32, DType::I64)?;

    Ok(rank_tensor)
}

pub fn matrix_norm(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    ord: MatrixNormOrder,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (_m, _n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU matrix_norm (only F32 supported)",
        });
    }

    match ord {
        MatrixNormOrder::Frobenius => {
            // Frobenius norm: ||A||_F = sqrt(sum(A²))
            // Use existing tensor ops to keep data on GPU
            let squared = client.square(a)?;
            let sum_sq = client.sum(&squared, &[], false)?;
            client.sqrt(&sum_sq)
        }
        MatrixNormOrder::Spectral => {
            // Spectral norm is the largest singular value
            use super::svd::svd_decompose;
            let svd = svd_decompose(client, a)?;
            // S is already sorted descending, so max(S) = s[0] = largest singular value
            // Using max keeps computation on GPU (no transfer needed)
            client.max(&svd.s, &[], false)
        }
        MatrixNormOrder::Nuclear => {
            // Nuclear norm is sum of singular values
            use super::svd::svd_decompose;
            let svd = svd_decompose(client, a)?;
            client.sum(&svd.s, &[], false)
        }
    }
}

/// Kronecker product: A ⊗ B
pub fn kron(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
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

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU kron (only F32 supported)",
        });
    }

    let m_out = m_a * m_b;
    let n_out = n_a * n_b;
    let out_size = m_out * n_out * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();
    let out_buffer = get_buffer(out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let b_buffer =
        get_buffer(b.ptr()).ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    let params: [u32; 4] = [m_a as u32, n_a as u32, m_b as u32, n_b as u32];
    let params_buffer = client.create_uniform_buffer("kron_params", 16);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_kron(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &b_buffer,
        &out_buffer,
        &params_buffer,
        m_a * m_b * n_a * n_b,
        dtype,
    )?;

    client.synchronize();

    let out =
        unsafe { WgpuClient::tensor_from_raw(out_guard.release(), &[m_out, n_out], dtype, device) };

    Ok(out)
}

/// Khatri-Rao product (column-wise Kronecker): A ⊙ B
///
/// For A of shape [m, k] and B of shape [n, k],
/// produces output of shape [m * n, k].
///
/// (A ⊙ B)[i*n + j, c] = A[i, c] * B[j, c]
pub fn khatri_rao(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
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

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU khatri_rao (only F32 supported)",
        });
    }

    let m_out = m * n;
    let out_size = m_out * k * dtype.size_in_bytes();
    let out_guard = AllocGuard::new(client.allocator(), out_size)?;
    let out_ptr = out_guard.ptr();
    let out_buffer = get_buffer(out_ptr)
        .ok_or_else(|| Error::Internal("Failed to get out buffer".to_string()))?;

    let a_buffer =
        get_buffer(a.ptr()).ok_or_else(|| Error::Internal("Failed to get a buffer".to_string()))?;

    let b_buffer =
        get_buffer(b.ptr()).ok_or_else(|| Error::Internal("Failed to get b buffer".to_string()))?;

    // params: [m, n, k, _pad]
    let params: [u32; 4] = [m as u32, n as u32, k as u32, 0];
    let params_buffer = client.create_uniform_buffer("khatri_rao_params", 16);
    client.write_buffer(&params_buffer, &params);

    kernels::launch_khatri_rao(
        client.pipeline_cache(),
        &client.queue,
        &a_buffer,
        &b_buffer,
        &out_buffer,
        &params_buffer,
        m_out * k,
        dtype,
    )?;

    client.synchronize();

    let out =
        unsafe { WgpuClient::tensor_from_raw(out_guard.release(), &[m_out, k], dtype, device) };

    Ok(out)
}

/// Upper triangular part of a matrix — delegates to impl_generic
pub fn triu(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<WgpuRuntime>> {
    crate::ops::impl_generic::triu_impl(client, a, diagonal)
}

/// Lower triangular part of a matrix — delegates to impl_generic
pub fn tril(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    diagonal: i64,
) -> Result<Tensor<WgpuRuntime>> {
    crate::ops::impl_generic::tril_impl(client, a, diagonal)
}

/// Sign and log-absolute-determinant — delegates to impl_generic
pub fn slogdet(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<crate::algorithm::linalg::SlogdetResult<WgpuRuntime>> {
    crate::ops::impl_generic::slogdet_impl(client, a)
}
