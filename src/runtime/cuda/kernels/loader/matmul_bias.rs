//! Fused matmul+bias launchers.
//!
//! The bias joins the GEMM epilogue instead of a separate pass. For integer
//! and FP8 dtypes the fused form is required, not merely faster: the bias has
//! to enter the wide accumulator before the narrowing store.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

use super::launch_dims::check_shared_mem_fits;
use super::matmul_bias_f32::{
    launch_matmul_bias_batched_f32_tiled, launch_matmul_bias_f32_tiled,
    matmul_bias_batched_f32_tiled_name, matmul_bias_f32_tiled_name,
};
use super::matmul_config::{
    default_tile_config, f32_batched_tile_config, matmul_batched_launch_config,
    matmul_launch_config,
};
use super::matmul_fp8::launch_matmul_fp8_tiled;
use super::matmul_int::{int_matmul_has_kernel, launch_matmul_int_tiled};
use super::matmul_wmma::{
    launch_matmul_bias_wmma_batched_kernel, launch_matmul_bias_wmma_kernel, use_wmma,
};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};

/// Launch native tiled fused matmul+bias kernel: C[M,N] = A[M,K] @ B[K,N] + bias[N]
///
/// Uses the same tiled GEMM algorithm as matmul, but fuses bias addition into the
/// epilogue to avoid an extra memory round-trip.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: M * K elements
/// - B: K * N elements
/// - bias: N elements (1D, broadcast across rows)
/// - C: M * N elements (output)
pub unsafe fn launch_matmul_bias_kernel(
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
    // FP8 has its own fused-bias kernels; `matmul.cu` instantiates none.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
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
    // Integers likewise: `matmul.cu` has no integer kernels at all, and the
    // fused form is required rather than merely faster - the bias has to join
    // the 128-bit accumulator before it saturates, which a separate add cannot
    // do. There is no small-M shortcut here because `gemv_int.cu` takes no bias.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
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
    // Tensor-core WMMA path: F16/BF16 with 16-aligned dims, same predicate as
    // plain matmul (`loader/matmul.rs`). Unaligned operands are padded to
    // 16-multiples — bias included — by `src/ops/cuda/matmul.rs` before they
    // reach here. CudaDevice::new is a zero-cost index wrapper; profile()
    // serves the per-index cache.
    let caps = CudaDevice::new(device_index).profile().caps;
    if use_wmma(dtype, caps, m, n, k) {
        unsafe {
            return launch_matmul_bias_wmma_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                bias_ptr,
                c_ptr,
                m,
                n,
                k,
            );
        }
    }
    // F32 uses the same shape-aware tiles as plain matmul so both reach a
    // compile-time-tiled kernel; every other dtype keeps its fixed default.
    let tile_cfg = match dtype {
        DType::F32 => f32_batched_tile_config(m, n, k),
        _ => default_tile_config(dtype),
    };
    unsafe {
        launch_matmul_bias_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
            c_ptr,
            m,
            n,
            k,
            &tile_cfg,
        )
    }
}

/// Launch native tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_kernel_with_config(
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
    tile_cfg: &TileConfig,
) -> Result<()> {
    // F32 with a specialised tile: compile-time-tiled kernel, unrolled
    // micro-kernel, accumulators in registers. Any other tile or dtype falls
    // through to the generic runtime-parameter kernel below.
    if dtype == DType::F32
        && let Some(kernel_fn_name) = matmul_bias_f32_tiled_name(tile_cfg)
    {
        unsafe {
            return launch_matmul_bias_f32_tiled(
                context,
                stream,
                device_index,
                &kernel_fn_name,
                a_ptr,
                b_ptr,
                bias_ptr,
                c_ptr,
                m,
                n,
                k,
                tile_cfg,
            );
        }
    }

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    // For F16/BF16, shared memory uses F32 for accumulation
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4, // F32 accumulator
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, tile_cfg, shared_elem_size);
    check_shared_mem_fits(device_index, cfg.shared_mem_bytes, "matmul", || {
        format!(
            "{}x{}x{} {dtype} matmul_bias tile",
            tile_cfg.block_m, tile_cfg.block_n, tile_cfg.block_k
        )
    })?;
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
        builder.arg(&bias_ptr);
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
            Error::Internal(format!("CUDA matmul_bias kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch native batched tiled fused matmul+bias kernel:
/// C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] + bias[N]
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes:
/// - A: batch * M * K elements
/// - B: batch * K * N elements
/// - bias: N elements (1D, broadcast across all batches and rows)
/// - C: batch * M * N elements (output)
pub unsafe fn launch_matmul_bias_batched_kernel(
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
    // FP8 has its own fused-bias kernels; `matmul.cu` instantiates none.
    if matches!(dtype, DType::FP8E4M3 | DType::FP8E5M2) {
        unsafe {
            return launch_matmul_fp8_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
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
    // Integers: same reason as the non-batched entry point above.
    if int_matmul_has_kernel(dtype) {
        unsafe {
            return launch_matmul_int_tiled(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                Some(bias_ptr),
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
    // Tensor-core WMMA path for F16/BF16 with 16-aligned dims. The bias is
    // [N] and broadcasts across rows and batch slices, matching the generic
    // `matmul_bias_batched_*` kernels.
    let caps = CudaDevice::new(device_index).profile().caps;
    if use_wmma(dtype, caps, m, n, k) {
        unsafe {
            return launch_matmul_bias_wmma_batched_kernel(
                context,
                stream,
                device_index,
                dtype,
                a_ptr,
                b_ptr,
                bias_ptr,
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
    let tile_cfg = match dtype {
        DType::F32 => f32_batched_tile_config(m, n, k),
        _ => default_tile_config(dtype),
    };
    unsafe {
        launch_matmul_bias_batched_kernel_with_config(
            context,
            stream,
            device_index,
            dtype,
            a_ptr,
            b_ptr,
            bias_ptr,
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

/// Launch native batched tiled fused matmul+bias kernel with custom tile configuration.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_matmul_bias_batched_kernel_with_config(
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
    tile_cfg: &TileConfig,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // F32 with a specialised tile: same compile-time-tiled dispatch as the
    // non-batched entry point above.
    if dtype == DType::F32
        && let Some(kernel_fn_name) = matmul_bias_batched_f32_tiled_name(tile_cfg)
    {
        unsafe {
            return launch_matmul_bias_batched_f32_tiled(
                context,
                stream,
                device_index,
                &kernel_fn_name,
                a_ptr,
                b_ptr,
                bias_ptr,
                c_ptr,
                batch,
                m,
                n,
                k,
                a_batch,
                b_batch,
                tile_cfg,
            );
        }
    }

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_MODULE)?;
    let func_name = kernel_name("matmul_bias_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, tile_cfg, shared_elem_size);
    check_shared_mem_fits(device_index, cfg.shared_mem_bytes, "matmul", || {
        format!(
            "{}x{}x{} {dtype} batched matmul_bias tile",
            tile_cfg.block_m, tile_cfg.block_n, tile_cfg.block_k
        )
    })?;
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
        builder.arg(&bias_ptr);
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
            Error::Internal(format!(
                "CUDA batched matmul_bias kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
