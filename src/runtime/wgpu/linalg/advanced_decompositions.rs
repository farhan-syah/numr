//! Advanced decomposition algorithms for WebGPU: rsf2csf, QZ, and polar decomposition
//!
//! All algorithms use native WGSL compute shaders - NO CPU FALLBACK.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{
    ComplexSchurDecomposition, GeneralizedSchurDecomposition, LinearAlgebraAlgorithms,
    PolarDecomposition, SchurDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{LinalgOps, MatmulOps};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Convert real Schur form to complex Schur form (rsf2csf)
///
/// Uses native WGSL shader to process 2x2 blocks representing complex
/// conjugate eigenvalue pairs and converts them to upper triangular form.
pub fn rsf2csf(
    client: &WgpuClient,
    schur: &SchurDecomposition<WgpuRuntime>,
) -> Result<ComplexSchurDecomposition<WgpuRuntime>> {
    let n = validate_square_matrix(schur.t.shape())?;
    let dtype = schur.t.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "rsf2csf (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(ComplexSchurDecomposition {
            z_real: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            z_imag: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            t_real: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            t_imag: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
        });
    }

    if n == 1 {
        // GPU-only: copy scalars on device, no CPU transfer
        let elem = dtype.size_in_bytes();
        let t_real_guard = AllocGuard::new(client.allocator(), elem)?;
        let t_real_ptr = t_real_guard.ptr();
        WgpuRuntime::copy_within_device(schur.t.ptr(), t_real_ptr, elem, device)?;
        let z_real_guard = AllocGuard::new(client.allocator(), elem)?;
        let z_real_ptr = z_real_guard.ptr();
        WgpuRuntime::copy_within_device(schur.z.ptr(), z_real_ptr, elem, device)?;
        return Ok(ComplexSchurDecomposition {
            z_real: unsafe {
                WgpuClient::tensor_from_raw(z_real_guard.release(), &[1, 1], dtype, device)
            },
            z_imag: Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[1, 1], device)?,
            t_real: unsafe {
                WgpuClient::tensor_from_raw(t_real_guard.release(), &[1, 1], dtype, device)
            },
            t_imag: Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[1, 1], device)?,
        });
    }

    let matrix_size = n * n * dtype.size_in_bytes();

    // Allocate buffers for T (real and imag)
    let t_real_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_real_ptr = t_real_guard.ptr();
    let t_real_buffer = get_buffer_or_err!(t_real_ptr, "T_real");

    let t_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_imag_ptr = t_imag_guard.ptr();
    let t_imag_buffer = get_buffer_or_err!(t_imag_ptr, "T_imag");

    // Allocate buffers for Z (real and imag)
    let z_real_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_real_ptr = z_real_guard.ptr();
    let z_real_buffer = get_buffer_or_err!(z_real_ptr, "Z_real");

    let z_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_imag_ptr = z_imag_guard.ptr();
    let z_imag_buffer = get_buffer_or_err!(z_imag_ptr, "Z_imag");

    // Copy input T and Z to real buffers
    WgpuRuntime::copy_within_device(schur.t.ptr(), t_real_ptr, matrix_size, device)?;
    WgpuRuntime::copy_within_device(schur.z.ptr(), z_real_ptr, matrix_size, device)?;

    // Zero-initialize imaginary buffers
    let zeros = vec![0.0f32; n * n];
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&zeros), t_imag_ptr, device)?;
    WgpuRuntime::copy_to_device(bytemuck::cast_slice(&zeros), z_imag_ptr, device)?;

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("rsf2csf_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch rsf2csf kernel
    kernels::launch_rsf2csf(
        client.pipeline_cache(),
        &client.queue,
        &t_real_buffer,
        &t_imag_buffer,
        &z_real_buffer,
        &z_imag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Create output tensors
    let z_real =
        unsafe { WgpuClient::tensor_from_raw(z_real_guard.release(), &[n, n], dtype, device) };
    let z_imag =
        unsafe { WgpuClient::tensor_from_raw(z_imag_guard.release(), &[n, n], dtype, device) };
    let t_real =
        unsafe { WgpuClient::tensor_from_raw(t_real_guard.release(), &[n, n], dtype, device) };
    let t_imag =
        unsafe { WgpuClient::tensor_from_raw(t_imag_guard.release(), &[n, n], dtype, device) };

    Ok(ComplexSchurDecomposition {
        z_real,
        z_imag,
        t_real,
        t_imag,
    })
}

/// QZ decomposition for matrix pencil (A, B)
///
/// Uses native WGSL shader implementing Hessenberg-triangular reduction
/// followed by QZ iteration.
pub fn qz_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    b: &Tensor<WgpuRuntime>,
) -> Result<GeneralizedSchurDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let n_b = validate_square_matrix(b.shape())?;

    if n != n_b {
        return Err(Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "qz_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            z: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            s: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            t: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            eigenvalues_real: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0], device)?,
            eigenvalues_imag: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0], device)?,
        });
    }

    if n == 1 {
        // GPU-only: copy scalars on device, compute eigenvalue with GPU div
        let elem = dtype.size_in_bytes();
        let s_guard = AllocGuard::new(client.allocator(), elem)?;
        let s_ptr = s_guard.ptr();
        WgpuRuntime::copy_within_device(a.ptr(), s_ptr, elem, device)?;
        let t_guard = AllocGuard::new(client.allocator(), elem)?;
        let t_ptr = t_guard.ptr();
        WgpuRuntime::copy_within_device(b.ptr(), t_ptr, elem, device)?;
        let s_tensor =
            unsafe { WgpuClient::tensor_from_raw(s_guard.release(), &[1], dtype, device) };
        let t_tensor =
            unsafe { WgpuClient::tensor_from_raw(t_guard.release(), &[1], dtype, device) };

        // eigenvalue = a / b (GPU division handles inf for b≈0)
        use crate::ops::BinaryOps;
        let eigenvalues_real = client.div(&s_tensor, &t_tensor)?;

        // Reshape s, t back to [1,1] for output
        let s_out = s_tensor.reshape(&[1, 1])?;
        let t_out = t_tensor.reshape(&[1, 1])?;

        return Ok(GeneralizedSchurDecomposition {
            q: Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[1, 1], device)?,
            z: Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[1, 1], device)?,
            s: s_out,
            t: t_out,
            eigenvalues_real,
            eigenvalues_imag: Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[1], device)?,
        });
    }

    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();

    // Allocate buffers
    let s_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let s_ptr = s_guard.ptr();
    let s_buffer = get_buffer_or_err!(s_ptr, "S");

    let t_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_ptr = t_guard.ptr();
    let t_buffer = get_buffer_or_err!(t_ptr, "T");

    let q_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let q_ptr = q_guard.ptr();
    let q_buffer = get_buffer_or_err!(q_ptr, "Q");

    let z_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_ptr = z_guard.ptr();
    let z_buffer = get_buffer_or_err!(z_ptr, "Z");

    let eval_real_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let eval_real_ptr = eval_real_guard.ptr();
    let eval_real_buffer = get_buffer_or_err!(eval_real_ptr, "eigenvalues_real");

    let eval_imag_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let eval_imag_ptr = eval_imag_guard.ptr();
    let eval_imag_buffer = get_buffer_or_err!(eval_imag_ptr, "eigenvalues_imag");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_guard = AllocGuard::new(client.allocator(), converged_flag_size)?;
    let converged_flag_ptr = converged_flag_guard.ptr();
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "QZ convergence flag");

    // Copy input matrices
    WgpuRuntime::copy_within_device(a.ptr(), s_ptr, matrix_size, device)?;
    WgpuRuntime::copy_within_device(b.ptr(), t_ptr, matrix_size, device)?;

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("qz_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch QZ kernel
    kernels::launch_qz_decompose(
        client.pipeline_cache(),
        &client.queue,
        &s_buffer,
        &t_buffer,
        &q_buffer,
        &z_buffer,
        &eval_real_buffer,
        &eval_imag_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Guard will automatically deallocate converged flag buffer on drop
    drop(converged_flag_guard);

    // Create output tensors
    let q = unsafe { WgpuClient::tensor_from_raw(q_guard.release(), &[n, n], dtype, device) };
    let z = unsafe { WgpuClient::tensor_from_raw(z_guard.release(), &[n, n], dtype, device) };
    let s = unsafe { WgpuClient::tensor_from_raw(s_guard.release(), &[n, n], dtype, device) };
    let t = unsafe { WgpuClient::tensor_from_raw(t_guard.release(), &[n, n], dtype, device) };
    let eigenvalues_real =
        unsafe { WgpuClient::tensor_from_raw(eval_real_guard.release(), &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { WgpuClient::tensor_from_raw(eval_imag_guard.release(), &[n], dtype, device) };

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
/// U is unitary/orthogonal, P is symmetric positive semi-definite.
pub fn polar_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<PolarDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "polar_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(PolarDecomposition {
            u: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            p: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
        });
    }

    if n == 1 {
        // GPU-only: u = sign(a), p = abs(a)
        use crate::ops::{BinaryOps, UnaryOps};
        let a_abs = client.abs(a)?;
        // sign = a / abs(a) (gives ±1, or NaN for 0 which is acceptable)
        let sign = client.div(a, &a_abs)?;
        let u = sign.reshape(&[1, 1])?;
        let p = a_abs.reshape(&[1, 1])?;
        return Ok(PolarDecomposition { u, p });
    }

    // Compute SVD: A = U_svd @ S @ V^T
    let svd = client.svd_decompose(a)?;

    // Polar decomposition: A = U_polar @ P
    // where U_polar = U_svd @ V^T (orthogonal/unitary)
    // and P = V @ diag(S) @ V^T (symmetric positive semi-definite)

    // U_polar = U @ V^T
    let u_polar = client.matmul(&svd.u, &svd.vt)?;

    // P = V @ diag(S) @ V^T
    // First, create S as diagonal matrix
    let s_diag = LinalgOps::diagflat(client, &svd.s)?;

    // V = Vt^T
    let v = svd.vt.transpose(0, 1)?.contiguous()?;

    // V @ diag(S)
    let v_s = client.matmul(&v, &s_diag)?;

    // P = (V @ S) @ V^T
    let p = client.matmul(&v_s, &svd.vt)?;

    Ok(PolarDecomposition { u: u_polar, p })
}
