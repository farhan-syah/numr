//! Schur decomposition using QR algorithm.

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{SchurDecomposition, validate_linalg_dtype, validate_square_matrix};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::{AllocGuard, Runtime, RuntimeClient};
use crate::tensor::Tensor;

pub fn schur_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<SchurDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let n = validate_square_matrix(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "schur_decompose (WebGPU)",
        });
    }

    // Handle trivial cases
    if n == 0 {
        return Ok(SchurDecomposition {
            z: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
            t: Tensor::<WgpuRuntime>::try_from_slice::<f32>(&[], &[0, 0], device)?,
        });
    }

    if n == 1 {
        // GPU-only: copy scalar on device, no CPU transfer
        let elem = dtype.size_in_bytes();
        let t_guard = AllocGuard::new(client.allocator(), elem)?;
        let t_ptr = t_guard.ptr();
        WgpuRuntime::copy_within_device(a.ptr(), t_ptr, elem, device)?;
        let t = unsafe { WgpuClient::tensor_from_raw(t_guard.release(), &[1, 1], dtype, device) };
        let z = Tensor::<WgpuRuntime>::try_from_slice(&[1.0f32], &[1, 1], device)?;
        return Ok(SchurDecomposition { z, t });
    }

    // Allocate buffers for Schur decomposition
    let matrix_size = n * n * dtype.size_in_bytes();

    let t_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let t_ptr = t_guard.ptr();
    let t_buffer = get_buffer_or_err!(t_ptr, "T (Schur form matrix)");

    let z_guard = AllocGuard::new(client.allocator(), matrix_size)?;
    let z_ptr = z_guard.ptr();
    let z_buffer = get_buffer_or_err!(z_ptr, "Z (orthogonal matrix)");

    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_guard = AllocGuard::new(client.allocator(), converged_flag_size)?;
    let converged_flag_ptr = converged_flag_guard.ptr();
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "Schur convergence flag");

    // Copy input to T buffer
    WgpuRuntime::copy_within_device(a.ptr(), t_ptr, matrix_size, device)?;

    // Zero-initialize converged flag
    let zero_i32: [i32; 1] = [0];
    client.write_buffer(&converged_flag_buffer, &zero_i32);

    // Create params buffer
    let params: [u32; 1] = [n as u32];
    let params_buffer = client.create_uniform_buffer("schur_params", 4);
    client.write_buffer(&params_buffer, &params);

    // Launch Schur decomposition kernel
    kernels::launch_schur_decompose(
        client.pipeline_cache(),
        &client.queue,
        &t_buffer,
        &z_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Guard will automatically deallocate converged flag buffer on drop
    drop(converged_flag_guard);

    // Create output tensors from GPU memory
    let z = unsafe { WgpuClient::tensor_from_raw(z_guard.release(), &[n, n], dtype, device) };
    let t = unsafe { WgpuClient::tensor_from_raw(t_guard.release(), &[n, n], dtype, device) };

    Ok(SchurDecomposition { z, t })
}
