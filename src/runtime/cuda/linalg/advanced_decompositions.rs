//! Advanced decomposition algorithms for CUDA: rsf2csf, QZ, and polar decomposition
//!
//! All algorithms use native GPU computation - NO CPU fallback.

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    ComplexSchurDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    PolarDecomposition, SchurDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::error::Result;
use crate::ops::{LinalgOps, MatmulOps};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Convert real Schur form to complex Schur form (rsf2csf)
///
/// Processes 2x2 blocks on the diagonal that represent complex conjugate
/// eigenvalue pairs and transforms them to upper triangular form in complex space.
pub fn rsf2csf_impl(
    client: &CudaClient,
    schur: &SchurDecomposition<CudaRuntime>,
) -> Result<ComplexSchurDecomposition<CudaRuntime>> {
    validate_linalg_dtype(schur.t.dtype())?;
    let n = validate_square_matrix(schur.t.shape())?;
    let dtype = schur.t.dtype();
    let device = client.device();

    // Handle trivial case
    if n == 0 {
        return Ok(ComplexSchurDecomposition {
            z_real: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            z_imag: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            t_real: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            t_imag: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
        });
    }

    // Allocate output buffers
    let matrix_size = n * n * dtype.size_in_bytes();
    let z_real_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_real_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?;

    let z_real_ptr = z_real_guard.ptr();
    let z_imag_ptr = z_imag_guard.ptr();
    let t_real_ptr = t_real_guard.ptr();
    let t_imag_ptr = t_imag_guard.ptr();

    // Launch native rsf2csf kernel
    let result = unsafe {
        kernels::launch_rsf2csf(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            schur.z.ptr(),
            schur.t.ptr(),
            z_real_ptr,
            z_imag_ptr,
            t_real_ptr,
            t_imag_ptr,
            n,
        )
    };

    result?;

    client.synchronize();

    let z_real =
        unsafe { CudaClient::tensor_from_raw(z_real_guard.release(), &[n, n], dtype, device) };
    let z_imag =
        unsafe { CudaClient::tensor_from_raw(z_imag_guard.release(), &[n, n], dtype, device) };
    let t_real =
        unsafe { CudaClient::tensor_from_raw(t_real_guard.release(), &[n, n], dtype, device) };
    let t_imag =
        unsafe { CudaClient::tensor_from_raw(t_imag_guard.release(), &[n, n], dtype, device) };

    Ok(ComplexSchurDecomposition {
        z_real,
        z_imag,
        t_real,
        t_imag,
    })
}

/// QZ decomposition for matrix pencil (A, B)
///
/// Computes the generalized Schur decomposition: A = Q @ S @ Z^T, B = Q @ T @ Z^T
/// Uses native GPU kernels for Hessenberg-triangular reduction and QZ iteration.
pub fn qz_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    b: &Tensor<CudaRuntime>,
) -> Result<GeneralizedSchurDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial case
    if n == 0 {
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            z: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            s: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            t: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            eigenvalues_real: Tensor::<CudaRuntime>::zeros(&[0], dtype, device)?,
            eigenvalues_imag: Tensor::<CudaRuntime>::zeros(&[0], dtype, device)?,
        });
    }

    // Allocate output buffers
    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();
    let flag_size = std::mem::size_of::<i32>();

    let q_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let s_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let eig_real_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let eig_imag_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let flag_guard = AllocGuard::new(client.allocator(), flag_size)?;

    let q_ptr = q_guard.ptr();
    let z_ptr = z_guard.ptr();
    let s_ptr = s_guard.ptr();
    let t_ptr = t_guard.ptr();
    let eig_real_ptr = eig_real_guard.ptr();
    let eig_imag_ptr = eig_imag_guard.ptr();
    let flag_ptr = flag_guard.ptr();

    // Copy input matrices to S and T (will be modified in-place)
    CudaRuntime::copy_within_device(a.ptr(), s_ptr, matrix_size, device)?;
    CudaRuntime::copy_within_device(b.ptr(), t_ptr, matrix_size, device)?;

    // Initialize converged flag to 0
    let zero_flag = [0i32];
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&zero_flag), flag_ptr, device)?;

    // Launch native QZ decomposition kernel
    let result = unsafe {
        kernels::launch_qz_decompose(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            s_ptr,
            t_ptr,
            q_ptr,
            z_ptr,
            eig_real_ptr,
            eig_imag_ptr,
            flag_ptr,
            n,
        )
    };

    result?;

    client.synchronize();

    let q = unsafe { CudaClient::tensor_from_raw(q_guard.release(), &[n, n], dtype, device) };
    let z = unsafe { CudaClient::tensor_from_raw(z_guard.release(), &[n, n], dtype, device) };
    let s = unsafe { CudaClient::tensor_from_raw(s_guard.release(), &[n, n], dtype, device) };
    let t = unsafe { CudaClient::tensor_from_raw(t_guard.release(), &[n, n], dtype, device) };
    let eigenvalues_real =
        unsafe { CudaClient::tensor_from_raw(eig_real_guard.release(), &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { CudaClient::tensor_from_raw(eig_imag_guard.release(), &[n], dtype, device) };

    Ok(GeneralizedSchurDecomposition {
        q,
        z,
        s,
        t,
        eigenvalues_real,
        eigenvalues_imag,
    })
}

/// Polar decomposition: A = U @ P
///
/// Uses native SVD decomposition and matrix multiplication.
/// - U is unitary (orthogonal for real matrices)
/// - P is positive semi-definite Hermitian
///
/// Algorithm:
/// 1. Compute SVD: A = U_svd @ S @ V^T
/// 2. U_polar = U_svd @ V
/// 3. P = V @ diag(S) @ V^T
pub fn polar_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<PolarDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial case
    if n == 0 {
        return Ok(PolarDecomposition {
            u: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
            p: Tensor::<CudaRuntime>::zeros(&[0, 0], dtype, device)?,
        });
    }

    // Compute SVD using native GPU kernel: A = U_svd @ S @ V^T
    let svd = client.svd_decompose(a)?;

    // Get V from V^T by transposing
    // svd.vt is k x n, so V = (V^T)^T is n x k
    let v = svd.vt.transpose(0, 1)?.contiguous()?;

    // Compute U_polar = U_svd @ V (both are n x k, result is n x n... wait, need to be careful)
    // For square matrix: U is n x n (full), S is n, V^T is n x n
    // So V is n x n after transpose
    // U_polar = U @ V^T^T = U @ V (n x n matrix)

    // U_polar = matmul(U_svd, V)
    let u_polar = client.matmul(&svd.u, &v)?;

    // Create diagonal matrix from singular values
    // diagflat(S) gives n x n diagonal matrix
    let s_diag = LinalgOps::diagflat(client, &svd.s)?;

    // Compute P = V @ diag(S) @ V^T
    // First: temp = V @ diag(S)
    let temp = client.matmul(&v, &s_diag)?;

    // Then: P = temp @ V^T
    let p = client.matmul(&temp, &svd.vt)?;

    Ok(PolarDecomposition { u: u_polar, p })
}
