//! CUDA kernel launchers for GEMM epilogue operations.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    default_tile_config, get_kernel_function, get_or_load_module, kernel_name,
    matmul_batched_launch_config, matmul_launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::GemmActivation;

const GEMM_EPILOGUE_MODULE: &str = "gemm_epilogue";

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

/// Launch fused GEMM + bias + activation kernel.
///
/// # Safety
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_act_kernel(
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
    activation: GemmActivation,
) -> Result<()> {
    let tile_cfg = default_tile_config(dtype);
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func_name = kernel_name("gemm_bias_act", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, &tile_cfg, shared_elem_size);
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let act_u32 = activation_to_u32(activation);
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
        builder.arg(&act_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA gemm_bias_act kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch batched fused GEMM + bias + activation kernel.
///
/// # Safety
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_act_batched_kernel(
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
    activation: GemmActivation,
) -> Result<()> {
    let tile_cfg = default_tile_config(dtype);
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func_name = kernel_name("gemm_bias_act_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, &tile_cfg, shared_elem_size);
    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let act_u32 = activation_to_u32(activation);
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
        builder.arg(&batch_u32);
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&act_u32);
        builder.arg(&block_m);
        builder.arg(&block_n);
        builder.arg(&block_k);
        builder.arg(&thread_m);
        builder.arg(&thread_n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA gemm_bias_act_batched kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch fused GEMM + bias + residual kernel.
///
/// # Safety
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_residual_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    residual_ptr: u64,
    c_ptr: u64,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let tile_cfg = default_tile_config(dtype);
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func_name = kernel_name("gemm_bias_residual", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_launch_config(m, n, &tile_cfg, shared_elem_size);
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
        builder.arg(&residual_ptr);
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
                "CUDA gemm_bias_residual kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched fused GEMM + bias + residual kernel.
///
/// # Safety
/// All pointers must be valid device memory.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_gemm_bias_residual_batched_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: u64,
    residual_ptr: u64,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let tile_cfg = default_tile_config(dtype);
    let module = get_or_load_module(context, device_index, GEMM_EPILOGUE_MODULE)?;
    let func_name = kernel_name("gemm_bias_residual_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let elem_size = dtype.size_in_bytes();
    let shared_elem_size = match dtype {
        DType::F16 | DType::BF16 => 4,
        _ => elem_size,
    };

    let cfg = matmul_batched_launch_config(batch, m, n, &tile_cfg, shared_elem_size);
    let batch_u32 = batch as u32;
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
        builder.arg(&residual_ptr);
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

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA gemm_bias_residual_batched kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
