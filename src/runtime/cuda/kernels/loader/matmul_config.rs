//! Tile selection and grid/block sizing for the register-tiled GEMM kernels.
//!
//! `matmul_launch_config` and its batched form turn a `TileConfig` into a
//! launch shape; `default_tile_config` and `f32_batched_tile_config` pick the
//! tile itself.

use crate::algorithm::TileConfig;
use crate::dtype::DType;

use super::launch_dims::LaunchConfig;

/// Calculate launch configuration for register-tiled matrix multiplication.
///
/// Uses configurable tile sizes - no hardcoded values.
/// Grid: ceil(N/block_n) × ceil(M/block_m)
/// Block: (block_n/thread_n) × (block_m/thread_m) threads
#[inline]
pub fn matmul_launch_config(
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    // Dynamic shared memory: As[block_m][block_k] + Bs[block_k][block_n]
    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Calculate launch configuration for batched register-tiled matrix multiplication.
///
/// Uses 3D grid: (tiles_x, tiles_y, batch)
#[inline]
pub fn matmul_batched_launch_config(
    batch: usize,
    m: usize,
    n: usize,
    cfg: &TileConfig,
    elem_size: usize,
) -> LaunchConfig {
    let grid_x = ((n as u32) + cfg.block_n as u32 - 1) / cfg.block_n as u32;
    let grid_y = ((m as u32) + cfg.block_m as u32 - 1) / cfg.block_m as u32;
    let grid_z = batch as u32;
    let threads_x = cfg.block_n / cfg.thread_n;
    let threads_y = cfg.block_m / cfg.thread_m;

    let shared_mem_bytes = (cfg.block_m * cfg.block_k + cfg.block_k * cfg.block_n) * elem_size;

    LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (threads_x as u32, threads_y as u32, 1),
        shared_mem_bytes: shared_mem_bytes as u32,
    }
}

/// Get default tile configuration for a dtype.
///
/// Fixed defaults for the generic runtime-parameter GEMM kernels (matmul,
/// matmul_bias, gemm_epilogue). Not tuned per shape or architecture; no
/// autotuning mechanism selects between alternatives.
#[inline]
pub fn default_tile_config(dtype: DType) -> TileConfig {
    match dtype {
        // F64 uses smaller tiles due to larger element size
        DType::F64 => TileConfig {
            block_m: 64,
            block_n: 64,
            block_k: 8,
            thread_m: 4,
            thread_n: 4,
        },
        // F32/F16/BF16 use larger tiles
        _ => TileConfig::CUDA,
    }
}

/// Shape-aware tile configuration for F32 batched matmul.
///
/// The default 128×128×8 tile is badly inefficient when N or M is small (e.g.
/// N=64 in the context-attention path): half the columns in every block are
/// wasted, and block_k=8 forces 64+ __syncthreads barriers for K=512.
///
/// Rules (all tiles keep smem ≤ 24KB per buffer, 48KB total for double-buffer):
/// - Small-N (N ≤ 64): use 64×64 block tile, block_k=32, thread_m=8, thread_n=4
///   (64×32 + 32×64 = 4096 floats × 2 buffers = 32KB — fits in 48KB limit)
/// - Small-M (M ≤ 64, N large): mirror the small-N tile transposed
/// - Large square (default): 128×128, block_k=16, thread_m=8, thread_n=8
///   (128×16 + 16×128 = 4096 floats × 2 buffers = 32KB — fits in 48KB limit)
///
/// Note: the matmul_f32 and matmul_batched_f32 kernels are double-buffered, so
/// shared memory is allocated as 2 × (block_m*block_k + block_k*block_n) floats.
#[inline]
pub fn f32_batched_tile_config(m: usize, n: usize, _k: usize) -> TileConfig {
    if n <= 64 || m <= 64 {
        // Attention shapes: Scores Q@Kᵀ (M=512, N=512 but K=64 so inner loop short)
        // and Context attn@V (M=512, N=64, K=512).
        // For N≤64: block_n=64 so no wasted columns; block_k=32 halves sync count.
        // thread_m=8, thread_n=4: 8×(64/4)=128 threads/block (2 warps × 4).
        // Smem per buffer: (64×32 + 32×64) × 4 = 16 384 bytes. Two buffers = 32KB.
        TileConfig {
            block_m: 64,
            block_n: 64,
            block_k: 32,
            thread_m: 8,
            thread_n: 4,
        }
    } else {
        // Large square shapes (e.g. 512×512×512, 1024×1024×1024).
        // block_k=8 matches the compile-time-tiled `matmul_f32_tiled_128x128x8_8x8`
        // kernel (register-blocked, unrolled micro-kernel — ~100x the old runtime-
        // param kernel). Smem per buffer: (128×8 + 8×128)×4 = 8 192 B; ×2 = 16 KB.
        TileConfig {
            block_m: 128,
            block_n: 128,
            block_k: 8,
            thread_m: 8,
            thread_n: 8,
        }
    }
}

/// Grid and block shape for a compile-time-tiled F32 kernel.
///
/// Grid: (ceil(N/block_n), ceil(M/block_m), batch)
/// Block: (block_n/thread_n, block_m/thread_m, 1)
///
/// `shared_mem_bytes` is 0. The specialised kernels declare only static
/// `__shared__` arrays; dynamic shared memory would stack on top of that pool
/// and push the per-block total past the 48 KB default limit on sm_86 for the
/// 64x64x32 tile (32 KB static + 32 KB dynamic).
/// Suffix identifying the specialised compile-time-tiled F32 kernel for `cfg`,
/// `None` when `cfg` has no specialised instantiation.
///
/// Single source of truth for which tiles the `*_f32_tiled_*` kernel family
/// (`matmul.cu`, `gemm_epilogue.cu`) specialises. Every F32 tiled-kernel
/// dispatch site derives its extern "C" name from this suffix instead of
/// re-matching the tile tuple, so a new specialisation only needs adding here.
#[inline]
pub fn f32_tiled_suffix(cfg: &TileConfig) -> Option<&'static str> {
    match (
        cfg.block_m,
        cfg.block_n,
        cfg.block_k,
        cfg.thread_m,
        cfg.thread_n,
    ) {
        (128, 128, 8, 8, 8) => Some("128x128x8_8x8"),
        (64, 64, 32, 8, 4) => Some("64x64x32_8x4"),
        _ => None,
    }
}

#[inline]
pub fn f32_tiled_launch_config(m: usize, n: usize, batch: usize, cfg: &TileConfig) -> LaunchConfig {
    let bm = cfg.block_m as u32;
    let bn = cfg.block_n as u32;
    let tm = cfg.thread_m as u32;
    let tn = cfg.thread_n as u32;

    LaunchConfig {
        grid_dim: (
            ((n as u32) + bn - 1) / bn,
            ((m as u32) + bm - 1) / bm,
            batch as u32,
        ),
        block_dim: (bn / tn, bm / tm, 1),
        shared_mem_bytes: 0,
    }
}
