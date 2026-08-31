//! Compile-time-tiled FP32 GEMM epilogue launchers.
//!
//! Selects the `extern "C"` instantiation matching the tile so NVCC can unroll
//! the micro-kernel and keep accumulators in registers. The generic
//! `gemm_bias_act_f32` / `gemm_bias_residual_f32` kernels stay the fallback for
//! unspecialised tiles: they take the tile dims as runtime arguments, so their
//! accumulator spills to local memory.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    f32_tiled_launch_config, f32_tiled_suffix, get_kernel_function, get_or_load_module,
};
use super::launcher::GEMM_EPILOGUE_MODULE;
use crate::algorithm::TileConfig;
use crate::error::{Error, Result};

/// Specialised kernel name for `base` at `tile_cfg`, `None` when unspecialised.
///
/// `base` is the generic kernel family: `gemm_bias_act`,
/// `gemm_bias_act_batched`, `gemm_bias_residual`, or
/// `gemm_bias_residual_batched`. Must match the extern "C" instantiations in
/// `gemm_epilogue.cu`.
pub(super) fn tiled_f32_kernel_name(base: &str, tile_cfg: &TileConfig) -> Option<String> {
    f32_tiled_suffix(tile_cfg).map(|suffix| format!("{base}_f32_tiled_{suffix}"))
}

/// Launch compile-time-tiled FP32 GEMM + bias + activation:
/// C[M,N] = activation(A[M,K] @ B[K,N] + bias[N]), over `batch` matrices.
///
/// `batch` is 1 for the non-batched kernels. A batched launch advances A, B and
/// C by one full matrix per batch; the bias is shared by every batch.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_bias_act_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_fn_name: &str,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: Option<usize>,
    m: usize,
    n: usize,
    k: usize,
    activation_code: u32,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;
    let cfg = f32_tiled_launch_config(m, n, batch.unwrap_or(1), tile_cfg);

    let batch_u32 = batch.unwrap_or(1) as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&bias_ptr);
        builder.arg(&c_ptr);
        if batch.is_some() {
            builder.arg(&batch_u32);
        }
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&activation_code);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA gemm_bias_act F32 tiled kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }
    Ok(())
}

/// Launch compile-time-tiled FP32 GEMM + bias + residual:
/// C[M,N] = A[M,K] @ B[K,N] + bias[N] + residual[M,N], over `batch` matrices.
///
/// The residual is elementwise over the output. `batch` is 1 for the
/// non-batched kernels; a batched launch advances A, B, C and the residual by
/// one full matrix per batch, and the bias is shared by every batch.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_gemm_bias_residual_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_fn_name: &str,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    residual_ptr: u64,
    c_ptr: u64,
    batch: Option<usize>,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;
    let cfg = f32_tiled_launch_config(m, n, batch.unwrap_or(1), tile_cfg);

    let batch_u32 = batch.unwrap_or(1) as u32;
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
        if batch.is_some() {
            builder.arg(&batch_u32);
        }
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA gemm_bias_residual F32 tiled kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }
    Ok(())
}
