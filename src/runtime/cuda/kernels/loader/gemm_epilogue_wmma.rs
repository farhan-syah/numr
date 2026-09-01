//! Tensor-core WMMA launchers for the fused GEMM epilogues.
//!
//! `gemm_bias_act` and `gemm_bias_residual` for F16 and BF16, 2-D and batched.
//! The kernels are instantiated in `kernels/matmul_wmma.cu` and share the
//! block tile, thread count, and launch geometry of the plain WMMA GEMM, so
//! the grid comes from [`super::matmul_wmma_tile::wmma_launch_config`], which
//! also picks the block tile. Dispatch is
//! gated by [`super::matmul_wmma::use_wmma`], the same predicate plain matmul
//! and matmul_bias use.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Launch 2-D (non-batched) WMMA GEMM with fused bias and activation for F16
/// or BF16: `C[M,N] = activation(A[M,K] @ B[K,N] + bias[N])`.
///
/// The bias add and the activation both run in F32, inside the epilogue,
/// before the narrowing store. `activation_type` is the code
/// `activation_to_u32` emits (`kernels/gemm_epilogue/launcher.rs`), which the
/// kernel feeds to `apply_activation_f32` — the same function the generic
/// `gemm_bias_act_*` kernels call.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, and that `bias_ptr`
/// addresses N elements of `dtype`.
pub unsafe fn launch_gemm_bias_act_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    activation_type: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("gemm_bias_act_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&activation_type);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA gemm_bias_act kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM with fused bias and activation for F16 or BF16.
///
/// A, B and C each advance one matrix per batch index; the bias is `[N]` and
/// broadcasts across rows and batch slices, matching the generic
/// `gemm_bias_act_batched_*` kernels.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, and that `bias_ptr`
/// addresses N elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_act_wmma_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    activation_type: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("gemm_bias_act_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&activation_type);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched gemm_bias_act kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch 2-D (non-batched) WMMA GEMM with fused bias and residual for F16 or
/// BF16: `C[M,N] = A[M,K] @ B[K,N] + bias[N] + residual[M,N]`.
///
/// The residual is elementwise over the output and is read at the flat output
/// offset, matching the generic `gemm_bias_residual_*` kernels. Both addends
/// join in F32 before the narrowing store.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, that `bias_ptr`
/// addresses N elements of `dtype`, and that `residual_ptr` addresses M * N
/// elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_residual_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    residual_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("gemm_bias_residual_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&residual_ptr);
        builder.arg(&c_ptr);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA gemm_bias_residual kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM with fused bias and residual for F16 or BF16.
///
/// The residual carries one `[M,N]` slice per batch index, like C; the bias is
/// `[N]` and broadcasts. Same operand layout as the generic
/// `gemm_bias_residual_batched_*` kernels.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, that `bias_ptr`
/// addresses N elements of `dtype`, and that `residual_ptr` addresses
/// batch * M * N elements of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_residual_wmma_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    residual_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("gemm_bias_residual_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&residual_ptr);
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched gemm_bias_residual kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
