//! General (non-symmetric) eigenvalue decomposition.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{
    GeneralEigenDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn eig_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<GeneralEigenDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "eig_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0], device)?,
            eigenvalues_imag: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0], device)?,
            eigenvectors_real: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            eigenvectors_imag: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
        });
    }

    if n == 1 {
        // GPU-only: copy scalar on device, no CPU transfer
        let elem = dtype.size_in_bytes();
        let eval_guard = AllocGuard::new(client.allocator(), elem)?;
        let eval_ptr = eval_guard.ptr();
        WgpuRuntime::copy_within_device(a.ptr(), eval_ptr, elem, device)?;
        let eigenvalues_real =
            unsafe { WgpuClient::tensor_from_raw(eval_guard.release(), &[1], dtype, device) };
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real,
            eigenvalues_imag: Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[1], device)?,
            eigenvectors_real: Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[1, 1], device)?,
            eigenvectors_imag: Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[1, 1], device)?,
        });
    }

    // Allocate buffers for general eigendecomposition
    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();

    let t_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_ptr = t_guard.ptr();
    let t_buffer = get_buffer_or_err!(t_ptr, "T (working matrix)");

    let z_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_ptr = z_guard.ptr();
    let z_buffer = get_buffer_or_err!(z_ptr, "Z (Schur transformation)");

    let eval_real_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let eval_real_ptr = eval_real_guard.ptr();
    let eval_real_buffer = get_buffer_or_err!(eval_real_ptr, "eigenvalues_real");

    let eval_imag_guard = AllocGuard::new(client.allocator(), vector_size)?;
    let eval_imag_ptr = eval_imag_guard.ptr();
    let eval_imag_buffer = get_buffer_or_err!(eval_imag_ptr, "eigenvalues_imag");

    let evec_real_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let evec_real_ptr = evec_real_guard.ptr();
    let evec_real_buffer = get_buffer_or_err!(evec_real_ptr, "eigenvectors_real");

    let evec_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let evec_imag_ptr = evec_imag_guard.ptr();
    let evec_imag_buffer = get_buffer_or_err!(evec_imag_ptr, "eigenvectors_imag");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_guard = AllocGuard::new(client.allocator(), converged_flag_size)?;
    let converged_flag_ptr = converged_flag_guard.ptr();
    let converged_flag_buffer =
        get_buffer_or_err!(converged_flag_ptr, "eig_general convergence flag");

    // Copy input to T buffer
    WgpuRuntime::copy_within_device(a.ptr(), t_ptr, matrix_size, device)?;

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("eig_general_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch general eigendecomposition kernel
    kernels::launch_eig_general(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &z_buffer,
        &eval_real_buffer,
        &eval_imag_buffer,
        &evec_real_buffer,
        &evec_imag_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Guards will automatically deallocate working buffers on drop
    drop(t_guard);
    drop(z_guard);
    drop(converged_flag_guard);

    // Create output tensors from GPU memory
    let eigenvalues_real =
        unsafe { WgpuClient::tensor_from_raw(eval_real_guard.release(), &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { WgpuClient::tensor_from_raw(eval_imag_guard.release(), &[n], dtype, device) };
    let eigenvectors_real =
        unsafe { WgpuClient::tensor_from_raw(evec_real_guard.release(), &[n, n], dtype, device) };
    let eigenvectors_imag =
        unsafe { WgpuClient::tensor_from_raw(evec_imag_guard.release(), &[n, n], dtype, device) };

    Ok(GeneralEigenDecomposition {
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors_real,
        eigenvectors_imag,
    })
}
