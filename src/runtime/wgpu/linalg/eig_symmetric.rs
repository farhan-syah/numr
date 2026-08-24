//! Symmetric eigenvalue decomposition using Jacobi method.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{EigenDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn eig_decompose_symmetric(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<EigenDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU eig_decompose_symmetric (only F32 supported)",
        });
    }

    // Edge cases
    if n == 0 {
        let eigenvalues = Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0], device)?;
        let eigenvectors = Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?;
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    if n == 1 {
        // GPU-only: copy scalar on device, no CPU transfer
        let elem = dtype.size_in_bytes();
        let eval_guard = AllocGuard::new(client.allocator(), elem)?;
        let eval_ptr = eval_guard.ptr();
        WgpuRuntime::copy_within_device(a.ptr(), eval_ptr, elem, device)?;
        let eigenvalues =
            unsafe { WgpuClient::tensor_from_raw(eval_guard.release(), &[1], dtype, device) };
        let eigenvectors = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[1, 1], device)?;
        return Ok(EigenDecomposition {
            eigenvalues,
            eigenvectors,
        });
    }

    // Allocate buffers for Jacobi computation
    let work_size = n * n * dtype.size_in_bytes();
    let work_guard = AllocGuard::new(client.allocator(), work_size)?;
    let work_ptr = work_guard.ptr();
    let work_buffer = get_buffer_or_err!(work_ptr, "work (working matrix)");

    let eigenvectors_size = n * n * dtype.size_in_bytes();
    let eigenvectors_guard = AllocGuard::new(client.allocator(), eigenvectors_size)?;
    let eigenvectors_ptr = eigenvectors_guard.ptr();
    let eigenvectors_buffer = get_buffer_or_err!(eigenvectors_ptr, "eigenvectors");

    let eigenvalues_size = n * dtype.size_in_bytes();
    let eigenvalues_guard = AllocGuard::new(client.allocator(), eigenvalues_size)?;
    let eigenvalues_ptr = eigenvalues_guard.ptr();
    let eigenvalues_buffer = get_buffer_or_err!(eigenvalues_ptr, "eigenvalues");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_guard = AllocGuard::new(client.allocator(), converged_flag_size)?;
    let converged_flag_ptr = converged_flag_guard.ptr();
    let converged_flag_buffer =
        get_buffer_or_err!(converged_flag_ptr, "eigendecomposition convergence flag");

    // Copy input to work buffer
    WgpuRuntime::copy_within_device(a.ptr(), work_ptr, work_size, device)?;

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("eig_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch eigendecomposition kernel
    kernels::launch_eig_jacobi_symmetric(
        client.pipeline_cache(),
        &client.queue,
        &work_buffer,
        &eigenvectors_buffer,
        &eigenvalues_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Guards will automatically deallocate work buffer and converged flag on drop
    drop(work_guard);
    drop(converged_flag_guard);

    // Create output tensors from GPU memory
    let eigenvalues =
        unsafe { WgpuClient::tensor_from_raw(eigenvalues_guard.release(), &[n], dtype, device) };
    let eigenvectors = unsafe {
        WgpuClient::tensor_from_raw(eigenvectors_guard.release(), &[n, n], dtype, device)
    };

    Ok(EigenDecomposition {
        eigenvalues,
        eigenvectors,
    })
}
