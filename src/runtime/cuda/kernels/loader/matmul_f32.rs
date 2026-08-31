//! Compile-time-tiled FP32 GEMM launcher.
//!
//! Selects the `extern "C"` instantiation matching the tile so NVCC can unroll
//! the micro-kernel and keep accumulators in registers, with the generic
//! `matmul_f32` kernel as the fallback for unspecialised tiles.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::error::{Error, Result};

use super::launch_dims::{LaunchConfig, check_shared_mem_fits};
use super::matmul_config::{f32_tiled_launch_config, f32_tiled_suffix, matmul_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Launch compile-time-tiled FP32 GEMM: C[M,N] = A[M,K] @ B[K,N].
///
/// Selects the extern "C" kernel instantiation that matches `tile_cfg` so that
/// NVCC can fully unroll the micro-kernel loops and keep all accumulators in
/// registers (no local-memory spill).
///
/// Supported configs (must match the extern "C" instantiations in matmul.cu):
///   128×128×8  TM=8 TN=8  → kernel `matmul_f32_tiled_128x128x8_8x8`  (256 threads)
///   64×64×32   TM=8 TN=4  → kernel `matmul_f32_tiled_64x64x32_8x4`   (128 threads)
///
/// Any other tile_cfg falls back to the generic `matmul_f32` kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub(super) unsafe fn launch_matmul_f32_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    tile_cfg: &TileConfig,
) -> Result<()> {
    // Map tile config to a specialised extern "C" kernel name.
    let specialized: Option<String> =
        f32_tiled_suffix(tile_cfg).map(|suffix| format!("matmul_f32_tiled_{suffix}"));

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;

    if let Some(kernel_fn_name) = specialized {
        let func = get_kernel_function(&module, &kernel_fn_name)?;

        // Grid: (ceil(N/BN), ceil(M/BM), 1)   Block: (BN/TN, BM/TM, 1)
        // The specialized tiled kernels (matmul_f32_tiled_*) use ONLY static
        // __shared__ arrays (no extern __shared__), so f32_tiled_launch_config
        // sets shared_mem_bytes to 0: dynamic smem would stack on top of the
        // static pool, pushing the per-block total past the 48 KB default
        // hardware limit and causing a silent launch failure on sm_86 (Ampere)
        // for the 64×64×32 config (32 KB static + 32 KB dynamic = 64 KB).
        let cfg = f32_tiled_launch_config(m, n, 1, tile_cfg);

        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;

        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&c_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!(
                    "CUDA matmul F32 tiled kernel '{}' launch failed: {:?}",
                    kernel_fn_name, e
                ))
            })?;
        }
        Ok(())
    } else {
        // Fallback to existing generic kernel for any config we didn't specialise.
        let func = get_kernel_function(&module, "matmul_f32")?;

        let elem_size = 4usize; // f32
        let smem_factor: u32 = 2; // double-buffered
        let base_cfg = matmul_launch_config(m, n, tile_cfg, elem_size);
        let shared_mem_bytes = base_cfg.shared_mem_bytes * smem_factor;
        check_shared_mem_fits(device_index, shared_mem_bytes, "matmul", || {
            format!(
                "{}x{}x{} F32 matmul tile",
                tile_cfg.block_m, tile_cfg.block_n, tile_cfg.block_k
            )
        })?;
        let cfg = LaunchConfig {
            shared_mem_bytes,
            ..base_cfg
        };
        let m_u32 = m as u32;
        let n_u32 = n as u32;
        let k_u32 = k as u32;
        let block_m = tile_cfg.block_m as u32;
        let block_n = tile_cfg.block_n as u32;
        let block_k = tile_cfg.block_k as u32;
        let thread_m = tile_cfg.thread_m as u32;
        let thread_n = tile_cfg.thread_n as u32;

        unsafe {
            let mut builder = stream.launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&c_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&block_m);
            builder.arg(&block_n);
            builder.arg(&block_k);
            builder.arg(&thread_m);
            builder.arg(&thread_n);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!(
                    "CUDA matmul F32 generic fallback kernel launch failed: {:?}",
                    e
                ))
            })?;
        }
        Ok(())
    }
}
