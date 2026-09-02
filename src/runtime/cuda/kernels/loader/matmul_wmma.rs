//! Tensor-core WMMA GEMM launchers for F16 and BF16.
//!
//! `use_wmma` decides when the path is legal; the launchers below cover the
//! 2-D and batched forms, each with and without a fused bias. The bias +
//! activation and bias + residual forms live in `gemm_epilogue_wmma.rs` and
//! share this module's launch geometry. Every kernel family is instantiated at
//! two block tiles; `matmul_wmma_tile` owns the launch geometry and picks the
//! tile per launch. The kernels use only static shared memory, so the dynamic
//! request is always zero.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::traits::profile::DeviceCaps;

use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

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
    // The matmul op (src/ops/cuda/matmul.rs) PADS unaligned F16/BF16 operands up
    // to the next multiple of 16 before dispatch, so by the time we get here the
    // dims are aligned — critical for the varlen embedding path where
    // M = total_tokens is rarely a multiple of 16 (without the pad+WMMA, F16 fell
    // to a ~100x-slower generic kernel). `m > 16` keeps tiny-M matmuls on the
    // GEMV path.
    //
    // The M test is a POLICY of this path, not a kernel limitation. An earlier
    // version of this comment claimed the kernel's sub-16 fragment handling was
    // buggy; that is not true of the kernel as it stands. `WMMA_KERNEL_BODY`
    // bounds-checks its A-tile staging and its epilogue store per ROW against M,
    // so a ragged M is zero-filled and masked rather than mis-read — which is
    // exactly what the grouped path relies on, where M is a per-group count in
    // device memory that the host cannot pad (see `use_wmma_grouped` in
    // grouped_matmul.rs). Keeping the alignment test here is still right for the
    // dense path: M is a host-known launch argument, padding it is cheap, and an
    // aligned M keeps the 128-bit staging path live.
    let dtype_ok = match dtype {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    dtype_ok && m >= 1 && m.is_multiple_of(16) && n.is_multiple_of(16) && k.is_multiple_of(16)
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
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("matmul_wmma", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, 1, tile);

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
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("matmul_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

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
    let tile = select_wmma_tile(m, n, k, 1, device_index);
    let func_name = wmma_kernel_name("matmul_bias_wmma", dtype, tile);
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
    let tile = select_wmma_tile(m, n, k, batch, device_index);
    let func_name = wmma_kernel_name("matmul_bias_wmma_batched", dtype, tile);
    let func = get_kernel_function(&module, &func_name)?;

    let cfg = wmma_launch_config(m, n, batch, tile);

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
    fn small_aligned_m_takes_wmma() {
        // Small m reaches the tensor-core kernel once it is 16-aligned. It used
        // to be held on the GEMV path, which measured slower at every m here.
        assert!(use_wmma(DType::F16, ampere_caps(), 16, 32, 32));
        assert!(use_wmma(DType::F16, ampere_caps(), 32, 32, 32));
        // Still refused when m is not 16-aligned; the op pads before dispatch.
        assert!(!use_wmma(DType::F16, ampere_caps(), 8, 32, 32));
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
    fn small_m_reaches_wmma_after_padding() {
        // A small m pads up to 16 and then takes the tensor-core kernel.
        assert!(use_wmma_after_padding(
            DType::F16,
            ampere_caps(),
            16,
            33,
            17
        ));
    }
}
