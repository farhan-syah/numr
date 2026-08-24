//! SVD and related operations (pinverse, cond).
//!
//! All operations run entirely on GPU with zero CPU transfers.

use std::sync::Arc;

use super::super::client::get_buffer;
use super::super::shaders::linalg as kernels;
use super::super::{WgpuClient, WgpuRuntime};
use super::helpers::get_buffer_or_err;
use crate::algorithm::linalg::{SvdDecomposition, validate_linalg_dtype, validate_matrix_2d};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{
    BinaryOps, CompareOps, ConditionalOps, LinalgOps, MatmulOps, ReduceOps, ScalarOps, UnaryOps,
};
use crate::runtime::{AllocGuard, RuntimeClient};
use crate::tensor::Tensor;

/// Helper to get buffer from tensor, with proper error handling.
fn get_tensor_buffer(tensor: &Tensor<WgpuRuntime>) -> Result<Arc<wgpu::Buffer>> {
    let ptr = tensor.ptr();
    get_buffer(ptr).ok_or_else(|| Error::Internal("Failed to get buffer from tensor".to_string()))
}

/// Compute SVD decomposition: A = U @ diag(S) @ VT
///
/// Uses One-Sided Jacobi algorithm, runs entirely on GPU.
pub fn svd_decompose(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
) -> Result<SvdDecomposition<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "WGPU svd_decompose (only F32 supported)",
        });
    }

    // Handle transpose for m < n case - work with transposed matrix
    let transposed = m < n;
    let (work_m, work_n) = if transposed { (n, m) } else { (m, n) };
    let k = work_m.min(work_n);

    // Get working matrix on GPU (transpose if needed)
    let work_tensor = if transposed {
        // Transpose on GPU: use view + contiguous
        a.transpose(0, 1)?.contiguous()?
    } else {
        // Use input directly if already contiguous, otherwise make contiguous
        a.contiguous()?
    };

    // Allocate buffers for SVD computation
    let b_size = work_m * work_n * dtype.size_in_bytes();
    let b_guard = AllocGuard::new(client.allocator(), b_size)?;
    let b_ptr = b_guard.ptr();
    let b_buffer = get_buffer_or_err!(b_ptr, "B (working matrix)");

    let v_size = work_n * work_n * dtype.size_in_bytes();
    let v_guard = AllocGuard::new(client.allocator(), v_size)?;
    let v_ptr = v_guard.ptr();
    let v_buffer = get_buffer_or_err!(v_ptr, "V (right singular vectors)");

    let s_size = work_n * dtype.size_in_bytes();
    let s_guard = AllocGuard::new(client.allocator(), s_size)?;
    let s_ptr = s_guard.ptr();
    let s_buffer = get_buffer_or_err!(s_ptr, "S (singular values)");

    // Convergence flag buffer (required by kernel)
    let converged_flag_size = std::mem::size_of::<i32>();
    let converged_flag_guard = AllocGuard::new(client.allocator(), converged_flag_size)?;
    let converged_flag_ptr = converged_flag_guard.ptr();
    let converged_flag_buffer = get_buffer_or_err!(converged_flag_ptr, "SVD convergence flag");

    // Copy working matrix to B buffer on GPU (no CPU transfer)
    let work_buffer = get_tensor_buffer(&work_tensor)?;
    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("svd_copy_input"),
        });
    encoder.copy_buffer_to_buffer(&work_buffer, 0, &b_buffer, 0, b_size as u64);
    client.queue.submit(std::iter::once(encoder.finish()));

    // Create params buffer
    let params: [u32; 2] = [work_m as u32, work_n as u32];
    let params_buffer = client.create_uniform_buffer("svd_params", 8);
    client.write_buffer(&params_buffer, &params);

    // Launch SVD kernel (Jacobi algorithm with fixed iterations - no convergence check needed)
    kernels::launch_svd_jacobi(
        client.pipeline_cache(),
        &client.queue,
        &b_buffer,
        &v_buffer,
        &s_buffer,
        &converged_flag_buffer,
        &params_buffer,
        dtype,
    )?;

    client.synchronize();

    // Guard will automatically deallocate convergence flag on drop
    drop(converged_flag_guard);

    // Wrap GPU buffers as tensors (no CPU transfer)
    // B contains U columns in [work_m, work_n] layout
    // V contains V (not transposed) in [work_n, work_n] layout
    // S contains singular values in [work_n] layout
    let b_tensor =
        unsafe { WgpuClient::tensor_from_raw(b_guard.release(), &[work_m, work_n], dtype, device) };
    let v_tensor =
        unsafe { WgpuClient::tensor_from_raw(v_guard.release(), &[work_n, work_n], dtype, device) };
    let s_tensor =
        unsafe { WgpuClient::tensor_from_raw(s_guard.release(), &[work_n], dtype, device) };

    // Extract U and VT from the results (all GPU operations)
    let (u, vt) = if transposed {
        // When transposed: V contains U, B^T contains VT
        // U = V[:m, :k] = V[:, :k] narrowed and transposed to get [m, k]
        // Actually: V is [n, n], we need U [m, k] where m < n
        // V[:m, :k] gives us [m, k] directly
        let u = v_tensor.narrow(0, 0, m)?.narrow(1, 0, k)?.contiguous()?;

        // VT = B[:k, :n] transposed -> B is [n, m], we need [k, n]
        // Take first k rows of B^T, which means first k columns of B
        let vt = b_tensor
            .narrow(1, 0, k)? // First k columns: [n, k]
            .transpose(0, 1)? // Transpose to [k, n]
            .contiguous()?;

        (u, vt)
    } else {
        // Normal case: B contains U columns, V contains V
        // U = B[:m, :k] - first k columns
        let u = b_tensor.narrow(1, 0, k)?.contiguous()?;

        // VT = V^T[:k, :n] - transpose V and take first k rows
        let vt = v_tensor
            .transpose(0, 1)? // Transpose to get V^T
            .narrow(0, 0, k)? // First k rows
            .contiguous()?;

        (u, vt)
    };

    // S only uses first k singular values
    let s = s_tensor.narrow(0, 0, k)?.contiguous()?;

    Ok(SvdDecomposition { u, s, vt })
}

/// Compute Moore-Penrose pseudo-inverse.
///
/// All computation runs on GPU.
pub fn pinverse(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    rcond: Option<f64>,
) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "pinverse",
        });
    }

    // Handle empty matrix
    if m == 0 || n == 0 {
        let out_guard = AllocGuard::new(client.allocator(), 0)?;
        return Ok(unsafe {
            WgpuClient::tensor_from_raw(out_guard.release(), &[n, m], dtype, device)
        });
    }

    // Compute SVD
    let svd = svd_decompose(client, a)?;

    // Compute cutoff threshold entirely on GPU: cutoff = rcond * max(S)
    let max_sv_tensor = client.max(&svd.s, &[0], false)?;
    let default_rcond = (m.max(n) as f64) * (f32::EPSILON as f64);
    let rcond_val = rcond.unwrap_or(default_rcond);
    let cutoff_tensor = client.mul_scalar(&max_sv_tensor, rcond_val)?;

    // Compute S_inv on GPU using conditional operations
    // S_inv[i] = 1/S[i] if S[i] > cutoff else 0
    let mask = client.gt(&svd.s, &cutoff_tensor)?; // Boolean mask
    let s_reciprocal = client.recip(&svd.s)?; // 1/S (may have inf for zeros)
    let zero = Tensor::<WgpuRuntime>::try_from_slice(&[0.0f32], &[], device)?;
    let s_inv = client.where_cond(&mask, &s_reciprocal, &zero)?;

    // Create S_inv diagonal matrix on GPU
    let s_inv_mat = LinalgOps::diagflat(client, &s_inv)?;

    // Compute A^+ = V @ S_inv @ U^T
    let v = svd.vt.transpose(0, 1)?.contiguous()?;
    let ut = svd.u.transpose(0, 1)?.contiguous()?;

    let v_sinv = client.matmul(&v, &s_inv_mat)?;
    let pinv = client.matmul(&v_sinv, &ut)?;

    Ok(pinv)
}

/// Compute condition number of a matrix.
///
/// Runs entirely on GPU - no CPU data transfers.
pub fn cond(client: &WgpuClient, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
    validate_linalg_dtype(a.dtype())?;
    let (m, n) = validate_matrix_2d(a.shape())?;
    let dtype = a.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op: "cond" });
    }

    // Handle empty matrix
    if m == 0 || n == 0 {
        return Tensor::<WgpuRuntime>::try_from_slice(&[f32::INFINITY], &[], device);
    }

    // Compute SVD
    let svd = svd_decompose(client, a)?;

    // Condition number = max(S) / min(S) - computed entirely on GPU
    let max_sv_tensor = client.max(&svd.s, &[0], false)?;
    let min_sv_tensor = client.min(&svd.s, &[0], false)?;

    // Compute ratio on GPU
    let ratio = client.div(&max_sv_tensor, &min_sv_tensor)?;

    // Handle edge case: if min_sv <= 0 or not finite, result should be infinity
    // Use GPU conditional: where(min_sv > eps, ratio, infinity)
    let eps = Tensor::<WgpuRuntime>::try_from_slice(&[f32::EPSILON], &[], device)?;
    let infinity = Tensor::<WgpuRuntime>::try_from_slice(&[f32::INFINITY], &[], device)?;
    let valid_mask = client.gt(&min_sv_tensor, &eps)?;
    let cond_result = client.where_cond(&valid_mask, &ratio, &infinity)?;

    Ok(cond_result)
}
