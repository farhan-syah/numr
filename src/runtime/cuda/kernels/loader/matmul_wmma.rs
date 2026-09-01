//! Tensor-core WMMA GEMM launchers for F16 and BF16.
//!
//! `use_wmma` decides when the path is legal; the launchers below cover the
//! 2-D and batched forms, each with and without a fused bias. The bias +
//! activation and bias + residual forms live in `gemm_epilogue_wmma.rs` and
//! share this module's launch geometry. The kernels use only static shared
//! memory, so the dynamic request is always zero.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::traits::profile::DeviceCaps;

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{dtype_suffix, kernel_names};

//
// Block: WARP_ROWS*WARP_COLS warps × 32 threads = 16 warps × 32 = 512 threads.
//   Warp grid: 4 rows × 4 cols. Each warp: WARP_M=2 × WARP_N=2 frags (32×32).
//   16 warps × 32×32 = 128×128 block tile. ✓
// Grid:  ceil(N/128) × ceil(M/128) [× batch]
// Static shared memory per block (single-buffered, no cp.async):
//   smem_A:   128 × 24 × 2 bytes =  6 144
//   smem_B:    16 × 136 × 2 bytes =  4 352
//   scratch:   16 × 256 × 4 bytes = 16 384
//   Total:    26 880 bytes ≈ 26.25 KB  (well within 48 KB)

/// Returns true when the WMMA path should be taken for this dtype, device,
/// and shape. This is the SINGLE source of truth for the decision — both the
/// launcher dispatch below and the pre-launch padding decision in
/// `src/ops/cuda/matmul.rs` (via [`use_wmma_after_padding`]) derive from it.
///
/// Conditions:
/// - dtype is F16 (needs `caps.f16_mma`, sm_70+) or BF16 (needs `caps.bf16`,
///   sm_80+ — the BF16 WMMA kernels are compiled only from sm_80, see
///   `matmul_wmma.cu`)
/// - M, N, K are all multiples of 16 (WMMA requirement)
/// - M > 16 (keep existing m<=16 GEMV fast path)
#[inline]
pub(crate) fn use_wmma(dtype: DType, caps: DeviceCaps, m: usize, n: usize, k: usize) -> bool {
    // The WMMA kernel is only correct for 16-aligned M/N/K (its sub-16 fragment
    // boundary handling is buggy). The matmul op (src/ops/cuda/matmul.rs) PADS
    // unaligned F16/BF16 operands up to the next multiple of 16 before dispatch,
    // so by the time we get here the dims are aligned — critical for the varlen
    // embedding path where M = total_tokens is rarely a multiple of 16 (without
    // the pad+WMMA, F16 fell to the ~100x-slower generic kernel: 57 vs 8500
    // GFLOP/s). `m > 16` keeps tiny-M matmuls on the GEMV path.
    let dtype_ok = match dtype {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    dtype_ok && m > 16 && m.is_multiple_of(16) && n.is_multiple_of(16) && k.is_multiple_of(16)
}

/// Returns true when padding M/N/K up to 16-multiples would make
/// [`use_wmma`] fire for this dtype and device. Derives from `use_wmma`
/// rather than re-encoding the dtype+caps rule, so the padding decision in
/// `src/ops/cuda/matmul.rs` cannot silently disagree with the dispatch
/// decision here (e.g. padding BF16 operands on a device without
/// `caps.bf16`, then not taking the WMMA path — wasted allocation and copy).
#[inline]
pub(crate) fn use_wmma_after_padding(
    dtype: DType,
    caps: DeviceCaps,
    m: usize,
    n: usize,
    k: usize,
) -> bool {
    let aligned = m.is_multiple_of(16) && n.is_multiple_of(16) && k.is_multiple_of(16);
    !aligned
        && use_wmma(
            dtype,
            caps,
            m.next_multiple_of(16),
            n.next_multiple_of(16),
            k.next_multiple_of(16),
        )
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

/// Grid and block for one WMMA launch: one block per 128x128 output tile, one
/// grid-z slice per batch index (`batch` is 1 for the 2-D forms).
#[inline]
pub(super) fn wmma_launch_config(m: usize, n: usize, batch: usize) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            ((n as u32) + WMMA_BLOCK_TILE_N - 1) / WMMA_BLOCK_TILE_N,
            ((m as u32) + WMMA_BLOCK_TILE_M - 1) / WMMA_BLOCK_TILE_M,
            batch as u32,
        ),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    }
}

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

    let cfg = wmma_launch_config(m, n, 1);

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

    let cfg = wmma_launch_config(m, n, batch);

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

/// Launch 2-D (non-batched) WMMA GEMM with fused bias for F16 or BF16:
/// `C[M,N] = A[M,K] @ B[K,N] + bias[N]`.
///
/// The bias is added in F32, inside the epilogue, before the narrowing store.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, and that `bias_ptr`
/// addresses N elements of `dtype`.
pub unsafe fn launch_matmul_bias_wmma_kernel(
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
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!("matmul_bias_wmma_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1);

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
                "CUDA WMMA matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched WMMA GEMM with fused bias for F16 or BF16. The bias is
/// `[N]` and broadcasts across rows and across batch slices.
///
/// # Safety
///
/// Caller must guarantee M, N, K are multiples of 16, and that `bias_ptr`
/// addresses N elements of `dtype`.
pub unsafe fn launch_matmul_bias_wmma_batched_kernel(
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
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let func_name = format!("matmul_bias_wmma_batched_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch);

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
                "CUDA WMMA batched matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Turing (e.g. T4, RTX 20xx): f16 tensor cores, no native bf16 —
    // BF16 WMMA kernels are not even compiled for this arch (matmul_wmma.cu).
    fn turing_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            f16_mma: true,
            bf16: false,
        }
    }

    // Ampere (e.g. A100, RTX 30xx): both f16 and native bf16 tensor cores.
    fn ampere_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            f16_mma: true,
            bf16: true,
        }
    }

    fn no_caps() -> DeviceCaps {
        DeviceCaps::default()
    }

    #[test]
    fn turing_f16_aligned_uses_wmma() {
        assert!(use_wmma(DType::F16, turing_caps(), 32, 32, 32));
    }

    #[test]
    fn turing_bf16_aligned_does_not_use_wmma() {
        // BF16 WMMA symbols do not exist in the sm_75 cubin: requesting them
        // would be a missing-symbol launch failure, not a slow fallback.
        assert!(!use_wmma(DType::BF16, turing_caps(), 32, 32, 32));
    }

    #[test]
    fn ampere_f16_and_bf16_aligned_use_wmma() {
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 32));
        assert!(use_wmma(DType::BF16, ampere_caps(), 32, 32, 32));
    }

    #[test]
    fn no_caps_never_uses_wmma() {
        assert!(!use_wmma(DType::F16, no_caps(), 32, 32, 32));
        assert!(!use_wmma(DType::BF16, no_caps(), 32, 32, 32));
    }

    #[test]
    fn unaligned_dims_do_not_use_wmma_even_with_caps() {
        assert!(!use_wmma(DType::F16, ampere_caps(), 33, 32, 32));
        assert!(!use_wmma(DType::F16, ampere_caps(), 32, 33, 32));
        assert!(!use_wmma(DType::F16, ampere_caps(), 32, 32, 33));
    }

    #[test]
    fn m_boundary_at_16_stays_on_gemv_path() {
        // m == 16 is exactly aligned but must NOT take WMMA — it stays on the
        // GEMV fast path (see use_wmma's `m > 16` condition).
        assert!(!use_wmma(DType::F16, ampere_caps(), 16, 32, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 32));
    }

    #[test]
    fn non_f16_bf16_dtype_never_uses_wmma() {
        assert!(!use_wmma(DType::F32, ampere_caps(), 32, 32, 32));
    }

    #[test]
    fn padding_turing_f16_unaligned_reaches_wmma() {
        assert!(use_wmma_after_padding(
            DType::F16,
            turing_caps(),
            33,
            40,
            17
        ));
    }

    #[test]
    fn padding_turing_bf16_unaligned_does_not_reach_wmma() {
        // Padding a BF16 operand on Turing must NOT be done: the WMMA path
        // never fires afterward (no caps.bf16), so padding would only cost
        // an allocation and a copy for nothing.
        assert!(!use_wmma_after_padding(
            DType::BF16,
            turing_caps(),
            33,
            40,
            17
        ));
    }

    #[test]
    fn padding_ampere_bf16_unaligned_reaches_wmma() {
        assert!(use_wmma_after_padding(
            DType::BF16,
            ampere_caps(),
            33,
            40,
            17
        ));
    }

    #[test]
    fn padding_already_aligned_dims_is_a_noop() {
        // Already-aligned dims never need padding, regardless of caps —
        // `use_wmma` itself already selects the WMMA path for them.
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            32,
            32,
            32
        ));
    }

    #[test]
    fn padding_below_gemv_boundary_never_reaches_wmma() {
        // m <= 16 stays on the GEMV path regardless of alignment or caps.
        assert!(!use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            16,
            33,
            17
        ));
    }
}
