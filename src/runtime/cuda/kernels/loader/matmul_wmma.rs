//! Tensor-core WMMA GEMM launchers for F16 and BF16.
//!
//! `use_wmma` decides when the path is legal; the launchers below cover the
//! 2-D and batched forms. The kernels use only static shared memory, so the
//! dynamic request is always zero.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{dtype_suffix, kernel_names};

//
// Block: WARP_ROWS*WARP_COLS warps × 32 threads = 8 warps × 32 = 256 threads.
//   Warp grid: 4 rows × 2 cols. Each warp: WARP_M=2 × WARP_N=4 frags (32×64).
//   8 warps × 32×64 = 128×128 block tile. ✓
// Grid:  ceil(N/128) × ceil(M/128) [× batch]
// Static shared memory per block (single-buffered, no cp.async):
//   smem_A:   128 × 24 × 2 bytes = 6 144
//   smem_B:    16 × 136 × 2 bytes = 4 352
//   scratch:    8 × 256 × 4 bytes = 8 192
//   Total:   18 688 bytes ≈ 18.25 KB  (well within 48 KB)

/// Returns true when the WMMA path should be taken for this dtype and shape.
///
/// Conditions:
/// - dtype is F16 or BF16
/// - M, N, K are all multiples of 16 (WMMA requirement)
/// - M > 16 (keep existing m<=16 GEMV fast path)
#[inline]
pub(super) fn use_wmma(dtype: DType, m: usize, n: usize, k: usize) -> bool {
    // The WMMA kernel is only correct for 16-aligned M/N/K (its sub-16 fragment
    // boundary handling is buggy). The matmul op (src/ops/cuda/matmul.rs) PADS
    // unaligned F16/BF16 operands up to the next multiple of 16 before dispatch,
    // so by the time we get here the dims are aligned — critical for the varlen
    // embedding path where M = total_tokens is rarely a multiple of 16 (without
    // the pad+WMMA, F16 fell to the ~100x-slower generic kernel: 57 vs 8500
    // GFLOP/s). `m > 16` keeps tiny-M matmuls on the GEMV path.
    matches!(dtype, DType::F16 | DType::BF16)
        && m > 16
        && m.is_multiple_of(16)
        && n.is_multiple_of(16)
        && k.is_multiple_of(16)
}

// WMMA block: 16 warps (4×4 warp grid), each warp = 32 threads → 512 threads.
// Each warp computes WARP_M=2 × WARP_N=2 fragments (32×32 outputs).
// 16 warps × 32×32 = 128×128. ✓
const WMMA_BLOCK_THREADS: u32 = 512;
const WMMA_BLOCK_TILE_M: u32 = 128;
const WMMA_BLOCK_TILE_N: u32 = 128;

/// Shared-memory per WMMA block in bytes.
///
/// Single-buffered A+B staging + per-warp F32 epilogue scratch:
///   smem_A:   128 × 24 × 2 bytes =  6 144 bytes
///   smem_B:    16 × 136 × 2 bytes =  4 352 bytes
///   scratch:   16 warps × 256 × 4 bytes = 16 384 bytes = 16 KB
///   Total:    26 880 bytes ≈ 26.25 KB  (well within 48 KB)
// WMMA kernels use only statically-declared __shared__ arrays; there is no
// extern __shared__ (dynamic) allocation.  Pass 0 so CUDA does not add
// extra dynamic smem on top of the static pool (which would push total over
// the 48 KB default per-block limit on sm_86).
const WMMA_SMEM_BYTES: u32 = 0;

/// Launch 2-D (non-batched) WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16.
pub unsafe fn launch_matmul_wmma_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!("matmul_wmma_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &func_name)?;

    let grid_x = ((n as u32) + WMMA_BLOCK_TILE_N - 1) / WMMA_BLOCK_TILE_N;
    let grid_y = ((m as u32) + WMMA_BLOCK_TILE_M - 1) / WMMA_BLOCK_TILE_M;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    };

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
            Error::Internal(format!("CUDA WMMA matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM for F16 or BF16.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16.
pub unsafe fn launch_matmul_wmma_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!(
        "matmul_wmma_batched_{}",
        crate::runtime::cuda::kernels::loader::dtype_suffix(dtype)
    );
    let func = get_kernel_function(&module, &func_name)?;

    let grid_x = ((n as u32) + WMMA_BLOCK_TILE_N - 1) / WMMA_BLOCK_TILE_N;
    let grid_y = ((m as u32) + WMMA_BLOCK_TILE_M - 1) / WMMA_BLOCK_TILE_M;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    };

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
        builder.arg(&c_ptr);
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA WMMA batched matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
