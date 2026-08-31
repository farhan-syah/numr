//! Compile-time-tiled FP32 fused matmul+bias launchers.
//!
//! Selects the `extern "C"` instantiation matching the tile so NVCC can unroll
//! the micro-kernel and keep accumulators in registers. The generic
//! `matmul_bias_f32` kernel stays the fallback for unspecialised tiles: it
//! takes the tile dims as runtime arguments, so its accumulator spills to
//! local memory.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::error::{Error, Result};

use super::matmul_config::{f32_tiled_launch_config, f32_tiled_suffix};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Specialised non-batched kernel for `tile_cfg`, `None` when unspecialised.
///
/// Must match the extern "C" instantiations in `matmul.cu`.
pub(super) fn matmul_bias_f32_tiled_name(tile_cfg: &TileConfig) -> Option<String> {
    f32_tiled_suffix(tile_cfg).map(|suffix| format!("matmul_bias_f32_tiled_{suffix}"))
}

/// Specialised batched kernel for `tile_cfg`, `None` when unspecialised.
pub(super) fn matmul_bias_batched_f32_tiled_name(tile_cfg: &TileConfig) -> Option<String> {
    f32_tiled_suffix(tile_cfg).map(|suffix| format!("matmul_bias_batched_f32_tiled_{suffix}"))
}

/// Launch compile-time-tiled FP32 fused matmul+bias:
/// C[M,N] = A[M,K] @ B[K,N] + bias[N].
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_matmul_bias_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_fn_name: &str,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;
    let cfg = f32_tiled_launch_config(m, n, 1, tile_cfg);

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
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA matmul_bias F32 tiled kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }
    Ok(())
}

/// Launch compile-time-tiled FP32 batched fused matmul+bias:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N].
///
/// `a_batch` and `b_batch` are the operand batch counts; an operand with a
/// count below `batch` repeats modulo its count.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn launch_matmul_bias_batched_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    kernel_fn_name: &str,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;
    let cfg = f32_tiled_launch_config(m, n, batch, tile_cfg);

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

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
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA batched matmul_bias F32 tiled kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }
    Ok(())
}
