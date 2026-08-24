//! General eigendecomposition for CUDA (non-symmetric matrices)

use super::super::CudaRuntime;
use super::super::client::CudaClient;
use super::super::kernels;
use crate::algorithm::linalg::{
    GeneralEigenDecomposition, validate_linalg_dtype, validate_square_matrix,
};
use crate::error::Result;
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// General eigendecomposition for non-symmetric matrices
pub fn eig_decompose_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
) -> Result<GeneralEigenDecomposition<CudaRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    // Handle trivial cases
    if n == 0 {
        return Ok(GeneralEigenDecomposition {
            eigenvalues_real: Tensor::<CudaRuntime>::try_from_slice(&[] as &[f64], &[0], device)?,
            eigenvalues_imag: Tensor::<CudaRuntime>::try_from_slice(&[] as &[f64], &[0], device)?,
            eigenvectors_real: Tensor::<CudaRuntime>::try_from_slice(
                &[] as &[f64],
                &[0, 0],
                device,
            )?,
            eigenvectors_imag: Tensor::<CudaRuntime>::try_from_slice(
                &[] as &[f64],
                &[0, 0],
                device,
            )?,
        });
    }

    // Allocate GPU buffers
    let matrix_size = n * n * dtype.size_in_bytes();
    let vector_size = n * dtype.size_in_bytes();
    let flag_size = std::mem::size_of::<i32>();

    let t_guard = AllocGuard::new(client.allocator(), matrix_size)?; // Working buffer (Schur form)
    let z_guard = AllocGuard::new(client.allocator(), matrix_size)?; // Schur vectors
    let eval_real_guard = AllocGuard::new(client.allocator(), vector_size)?; // Real part of eigenvalues
    let eval_imag_guard = AllocGuard::new(client.allocator(), vector_size)?; // Imag part of eigenvalues
    let evec_real_guard = AllocGuard::new(client.allocator(), matrix_size)?; // Real part of eigenvectors
    let evec_imag_guard = AllocGuard::new(client.allocator(), matrix_size)?; // Imag part of eigenvectors
    let flag_guard = AllocGuard::new(client.allocator(), flag_size)?;

    let t_ptr = t_guard.ptr();
    let z_ptr = z_guard.ptr();
    let eval_real_ptr = eval_real_guard.ptr();
    let eval_imag_ptr = eval_imag_guard.ptr();
    let evec_real_ptr = evec_real_guard.ptr();
    let evec_imag_ptr = evec_imag_guard.ptr();
    let flag_ptr = flag_guard.ptr();

    // Copy A to T (working buffer)
    CudaRuntime::copy_within_device(a.ptr(), t_ptr, matrix_size, device)?;

    // Initialize converged flag to 0
    let zero_flag = [0i32];
    CudaRuntime::copy_to_device(bytemuck::cast_slice(&zero_flag), flag_ptr, device)?;

    // Launch native CUDA general eigenvalue decomposition kernel
    let result = unsafe {
        kernels::launch_eig_general(
            client.context(),
            client.stream(),
            device.index,
            dtype,
            t_ptr,
            z_ptr,
            eval_real_ptr,
            eval_imag_ptr,
            evec_real_ptr,
            evec_imag_ptr,
            flag_ptr,
            n,
        )
    };

    // Handle kernel error
    result?;

    client.synchronize();

    // Create result tensors from GPU pointers
    let eigenvalues_real =
        unsafe { CudaClient::tensor_from_raw(eval_real_guard.release(), &[n], dtype, device) };
    let eigenvalues_imag =
        unsafe { CudaClient::tensor_from_raw(eval_imag_guard.release(), &[n], dtype, device) };
    let eigenvectors_real =
        unsafe { CudaClient::tensor_from_raw(evec_real_guard.release(), &[n, n], dtype, device) };
    let eigenvectors_imag =
        unsafe { CudaClient::tensor_from_raw(evec_imag_guard.release(), &[n, n], dtype, device) };

    Ok(GeneralEigenDecomposition {
        eigenvalues_real,
        eigenvalues_imag,
        eigenvectors_real,
        eigenvectors_imag,
    })
}
