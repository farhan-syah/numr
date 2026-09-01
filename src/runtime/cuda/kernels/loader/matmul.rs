//! Dense matmul dispatch.
//!
//! `launch_matmul_kernel` and its batched form pick between the FP8, GEMV,
//! WMMA, integer, and F32 specialisations before falling back to the generic
//! tiled kernel. The `_with_config` entry points skip that choice and launch
//! the generic kernel at an explicit tile.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

use super::gemv::launch_gemv_kernel;
use super::launch_dims::{LaunchConfig, check_shared_mem_fits};
use super::matmul_config::{
    default_tile_config, f32_batched_tile_config, matmul_batched_launch_config,
    matmul_launch_config,
};
use super::matmul_f32::launch_matmul_f32_tiled;
use super::matmul_fp8::launch_matmul_fp8_tiled;
use super::matmul_int::{int_matmul_has_kernel, launch_matmul_int_tiled};
use super::matmul_wmma::{launch_matmul_wmma_batched_kernel, launch_matmul_wmma_kernel, use_wmma};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};

/// Largest `m` routed to the GEMV kernel instead of a tiled GEMM.
///
/// Dtype-dependent, because what the alternative IS differs by dtype.
///
/// F32 has a tiled kernel that handles small `m` well, and it beats GEMV from
/// about `m == 8` upward — GEMV was measurably worse at 8, 12 and 16.
///
/// F16/BF16 have NO tiled path below this point: `use_wmma` requires
/// `m > 16`, so dropping the gate sends them to the generic runtime-parameter
/// kernel, which is orders of magnitude slower. GEMV is their only sane option
/// here, so the threshold stays. F64 and the integer dtypes likewise have no
/// specialised small-`m` path.
///
/// Raising the F16/BF16 case needs `use_wmma`'s `m > 16` relaxed first, so a
/// padded small `m` can reach the tensor-core kernel. That is worth doing —
/// the WMMA path just above this boundary is faster than GEMV below it — but it
/// is a change to WMMA selection, not to this constant.
fn gemv_m_max(dtype: DType) -> usize {
    match dtype {
        DType::F32 => 4,
        _ => 16,
    }
}

/// Launch native tiled matmul kernel: C[M,N] = A[M,K] @ B[K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - C: M * N elements
pub unsafe fn launch_matmul_kernel(
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
    // FP8 first: `gemv.cu` has no FP8 kernels, so the small-M fast path below
    // would look up a name that does not exist.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // Use GEMV kernel for small M (single-token decode in LLM inference)
    // The tiled GEMM wastes 99%+ compute when M < block_m (typically 128)
    //
    // I8 is excluded: its matmul writes I32, and `gemv_int.cu` has no
    // kernel that widens. CPU excludes I8 from its own GEMV-BT fast path for the
    // same reason, so both backends reach the tiled kernel at every M.
    if m <= gemv_m_max(dtype) && dtype != DType::I8 {
        unsafe {
            return launch_gemv_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // Tensor-core WMMA path: F16/BF16 with 16-aligned dims → up to ~100 TFLOPS on Ampere.
    // CudaDevice::new is a zero-cost index wrapper; profile() serves the
    // per-index cache (queried once, on first use of this device index).
    let caps = CudaDevice::new(device_index).profile().caps;
    if use_wmma(dtype, caps, m, n, k) {
        unsafe {
            return launch_matmul_wmma_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k,
            );
        }
    }
    // Integers: same reason as F32 below. Their accumulator is a 16-byte
    // `Numr128`, so a runtime-sized `reg_c` spills four registers per slot.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                1,
                m,
                n,
                k,
                1,
                1,
            );
        }
    }
    // F32: dispatch to compile-time-tiled kernels so NVCC can unroll micro-kernel
    // loops and keep accumulators in registers (avoids local-memory spill).
    if dtype == DType::F32 {
        let tile_cfg = f32_batched_tile_config(m, n, k);
        unsafe {
            return launch_matmul_f32_tiled(
                context,
                stream,
                device_index,
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                n,
                k,
                &tile_cfg,
            );
        }
    }
    let tile_cfg = default_tile_config(dtype);
    unsafe {
        launch_matmul_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            m,
            n,
            k,
            &tile_cfg,
        )
    }
}

/// Launch native tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_kernel_with_config(
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
    tile_cfg: &TileConfig,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation.
    // For F32, the kernel is double-buffered (2 ping-pong smem slots).
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };
    let smem_factor: u32 = if dtype == DType::F32 { 2 } else { 1 };

    let base_cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    let shared_mem_bytes = base_cfg.shared_mem_bytes * smem_factor;
    check_shared_mem_fits(device_index, shared_mem_bytes, "matmul", || {
        format!(
            "{}x{}x{} {dtype} matmul tile",
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

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA matmul kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch native batched tiled matmul kernel: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - C: batch * M * N elements
pub unsafe fn launch_matmul_batched_kernel(
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
    // FP8 first: `gemv.cu` has no FP8 kernels (see launch_matmul_kernel).
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Use GEMV kernel for small M (batched case). I8 is excluded for the same
    // reason as in `launch_matmul_kernel`: it widens to I32 and `gemv_int.cu`
    // has no widening kernel.
    if m <= gemv_m_max(dtype) && dtype != DType::I8 {
        unsafe {
            return launch_gemv_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Tensor-core WMMA path for F16/BF16 with 16-aligned dims.
    let caps = CudaDevice::new(device_index).profile().caps;
    if use_wmma(dtype, caps, m, n, k) {
        unsafe {
            return launch_matmul_wmma_batched_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // Integers use the compile-time-tiled kernels at every batch size.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                None,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
            );
        }
    }
    // F32 uses shape-aware tiles to avoid wasted columns and reduce sync count.
    let tile_cfg = match dtype {
        DType::F32 => f32_batched_tile_config(m, n, k),
        _ => default_tile_config(dtype),
    };
    unsafe {
        launch_matmul_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            batch,
            m,
            n,
            k,
            &tile_cfg,
            a_batch,
            b_batch,
        )
    }
}

/// Launch native batched tiled matmul kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_batched_kernel_with_config(
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
    tile_cfg: &TileConfig,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation.
    // For F32, the kernel is double-buffered (2 ping-pong smem slots).
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };
    let smem_factor: u32 = if dtype == DType::F32 { 2 } else { 1 };

    let base_cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    let shared_mem_bytes = base_cfg.shared_mem_bytes * smem_factor;
    check_shared_mem_fits(device_index, shared_mem_bytes, "matmul", || {
        format!(
            "{}x{}x{} {dtype} batched matmul tile",
            tile_cfg.block_m, tile_cfg.block_n, tile_cfg.block_k
        )
    })?;
    let cfg = LaunchConfig {
        shared_mem_bytes,
        ..base_cfg
    };
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let block_m = tile_cfg.block_m as u32;
    let block_n = tile_cfg.block_n as u32;
    let block_k = tile_cfg.block_k as u32;
    let thread_m = tile_cfg.thread_m as u32;
    let thread_n = tile_cfg.thread_n as u32;
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
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA batched matmul kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}
