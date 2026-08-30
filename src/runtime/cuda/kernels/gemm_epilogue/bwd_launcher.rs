//! CUDA kernel launchers for GEMM epilogue backward operations.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{get_kernel_function, get_or_load_module, kernel_name, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::GemmActivation;

const GEMM_EPILOGUE_BWD_MODULE: &str = "gemm_epilogue_bwd";
const BLOCK_SIZE: u32 = 256;

fn activation_to_u32(activation: GemmActivation) -> u32 {
    match activation {
        GemmActivation::None => 0,
        GemmActivation::ReLU => 1,
        GemmActivation::GELU => 2,
        GemmActivation::SiLU => 3,
        GemmActivation::Sigmoid => 4,
        GemmActivation::Tanh => 5,
    }
}

fn grid_1d(n: u32) -> (u32, u32, u32) {
    ((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1)
}

fn block_1d() -> (u32, u32, u32) {
    (BLOCK_SIZE, 1, 1)
}

/// Launch a single-batch GEMM backward pass (4 kernel launches).
///
/// # Safety
/// All pointers must be valid device memory with correct sizes.
/// `grad_pre_ptr` must point to a temporary buffer of size `m * n * dtype.size_in_bytes()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_act_bwd_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    grad_pre_ptr: u64,
    d_a_ptr: u64,
    d_b_ptr: u64,
    d_bias_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
) -> Result<()> {
    unsafe {
        launch_gemm_bwd_kernels(
            context,
            stream,
            device_index,
            dtype,
            grad_ptr,
            a_ptr,
            b_ptr,
            bias_ptr,
            grad_pre_ptr,
            d_a_ptr,
            d_b_ptr,
            d_bias_ptr,
            m,
            n,
            k,
            activation,
            false, // don't accumulate d_b/d_bias
        )
    }
}

/// Launch batched GEMM backward pass.
///
/// Batch 0 writes d_b/d_bias, batches 1+ accumulate into d_b/d_bias.
/// d_a is written per-batch at offset.
///
/// # Safety
/// All pointers must be valid device memory with correct sizes.
/// `grad_pre_ptr` must point to a temporary buffer of size `m * n * dtype.size_in_bytes()`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_act_bwd_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    grad_pre_ptr: u64,
    d_a_ptr: u64,
    d_b_ptr: u64,
    d_bias_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
) -> Result<()> {
    let elem_size = dtype.size_in_bytes() as u64;
    let mn_bytes = (m * n) as u64 * elem_size;
    let mk_bytes = (m * k) as u64 * elem_size;
    let kn_bytes = (k * n) as u64 * elem_size;

    for batch_idx in 0..batch {
        let grad_off = grad_ptr + batch_idx as u64 * mn_bytes;
        let a_off = a_ptr + batch_idx as u64 * mk_bytes;
        let b_off = b_ptr + batch_idx as u64 * kn_bytes;
        let d_a_off = d_a_ptr + batch_idx as u64 * mk_bytes;
        let accumulate = batch_idx > 0;

        unsafe {
            launch_gemm_bwd_kernels(
                context,
                stream,
                device_index,
                dtype,
                grad_off,
                a_off,
                b_off,
                bias_ptr,
                grad_pre_ptr,
                d_a_off,
                d_b_ptr,
                d_bias_ptr,
                m,
                n,
                k,
                activation,
                accumulate,
            )?;
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_gemm_bwd_kernels(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    grad_ptr: u64,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    grad_pre_ptr: u64,
    d_a_ptr: u64,
    d_b_ptr: u64,
    d_bias_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
    activation: GemmActivation,
    accumulate: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_BWD_MODULE)?;

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let act_u32 = activation_to_u32(activation);
    let mn = (m * n) as u32;
    let mk = (m * k) as u32;
    let kn = (k * n) as u32;

    // Each of the four launches below takes its grid from a product that can be
    // zero — `m * n`, `m * k`, `k * n`, `n`. A grid extent of 0 is a launch error
    // the driver rejects, so a launch whose whole output is empty is skipped
    // rather than floored: the tensor it would write has no element to write.
    unsafe {
        // Kernel 1: grad_pre = grad * act'(A @ B + bias)
        if mn != 0 {
            let func_name = kernel_name("gemm_bias_act_bwd_grad_pre", dtype);
            let func = get_kernel_function(&module, &func_name)?;
            let cfg = launch_config(grid_1d(mn), block_1d(), 0);
            let mut builder = stream.launch_builder(&func);
            builder.arg(&grad_ptr);
            builder.arg(&a_ptr);
            builder.arg(&b_ptr);
            builder.arg(&bias_ptr);
            builder.arg(&grad_pre_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder.arg(&act_u32);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!("CUDA gemm_bwd_grad_pre launch failed: {:?}", e))
            })?;
        }

        // Kernel 2: d_a = grad_pre @ B^T (always write, not accumulate)
        if mk != 0 {
            let func_name = kernel_name("gemm_bwd_da", dtype);
            let func = get_kernel_function(&module, &func_name)?;
            let cfg = launch_config(grid_1d(mk), block_1d(), 0);
            let mut builder = stream.launch_builder(&func);
            builder.arg(&grad_pre_ptr);
            builder.arg(&b_ptr);
            builder.arg(&d_a_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder
                .launch(cfg)
                .map_err(|e| Error::Internal(format!("CUDA gemm_bwd_da launch failed: {:?}", e)))?;
        }

        // Kernel 3: d_b = A^T @ grad_pre (or d_b += for accumulate)
        if kn != 0 {
            let base = if accumulate {
                "gemm_bwd_db_accum"
            } else {
                "gemm_bwd_db"
            };
            let func_name = kernel_name(base, dtype);
            let func = get_kernel_function(&module, &func_name)?;
            let cfg = launch_config(grid_1d(kn), block_1d(), 0);
            let mut builder = stream.launch_builder(&func);
            builder.arg(&a_ptr);
            builder.arg(&grad_pre_ptr);
            builder.arg(&d_b_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.arg(&k_u32);
            builder
                .launch(cfg)
                .map_err(|e| Error::Internal(format!("CUDA gemm_bwd_db launch failed: {:?}", e)))?;
        }

        // Kernel 4: d_bias = sum(grad_pre, dim=0) (or += for accumulate)
        if n_u32 != 0 {
            let base = if accumulate {
                "gemm_bwd_dbias_accum"
            } else {
                "gemm_bwd_dbias"
            };
            let func_name = kernel_name(base, dtype);
            let func = get_kernel_function(&module, &func_name)?;
            let cfg = launch_config(grid_1d(n_u32), block_1d(), 0);
            let mut builder = stream.launch_builder(&func);
            builder.arg(&grad_pre_ptr);
            builder.arg(&d_bias_ptr);
            builder.arg(&m_u32);
            builder.arg(&n_u32);
            builder.launch(cfg).map_err(|e| {
                Error::Internal(format!("CUDA gemm_bwd_dbias launch failed: {:?}", e))
            })?;
        }
    }

    Ok(())
}
