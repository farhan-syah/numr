//! GEMV launchers for the small-M decode path.
//!
//! One warp (or warp pair) per output column, reducing along K with shuffles.
//! `gemv_module` picks the translation unit, which splits on integer dtypes.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};

/// The PTX module holding this dtype's GEMV kernels.
///
/// Integer GEMV lives in its own translation unit: it accumulates in `Numr128`
/// instead of a float register, and `gemv.cu` is already at its size limit.
/// Every dtype that reaches a GEMV launcher has kernels in one module or the
/// other. The small-M fast paths gate on dtype in exactly one place — I8, which
/// `gemv_int.cu` does not instantiate because its matmul widens to I32.
#[inline]
fn gemv_module(dtype: DType) -> &'static str {
    if dtype.is_int() {
        kernel_names::GEMV_INT_MODULE
    } else {
        kernel_names::GEMV_MODULE
    }
}

/// Launch GEMV kernel: C[batch,M,N] = A[batch,M,K] @ B[batch,K,N] for small M
///
/// B is [K,N] row-major (non-transposed). One thread per output column, iterates K.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_gemv_kernel(
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
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N/256), M, batch), block: (256, 1, 1)
    // One thread per output column, each thread iterates over K.
    let block_size: u32 = 256;
    let grid_x = ((n as u32) + block_size - 1) / block_size;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

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
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA GEMV kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch GEMV kernel with transposed B: C[batch,M,N] = A[batch,M,K] @ B^T
///
/// B is stored [N,K] row-major (transposed weight matrix, common for nn.Linear).
/// Warp-cooperative: each warp reduces one output column along K using shuffle.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
/// `b_ptr` points to the raw [N,K] data (NOT the transposed [K,N] view).
pub unsafe fn launch_gemv_kernel_bt(
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
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv_bt", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N/WARPS_PER_BLOCK), M, batch), block: (256, 1, 1)
    // 8 warps per block, each warp handles one output column.
    let warps_per_block: u32 = 8;
    let grid_x = ((n as u32) + warps_per_block - 1) / warps_per_block;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

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
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA GEMV-BT kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch multi-row GEMV kernel with transposed B: C[batch,M,N] = A[batch,M,K] @ B^T
///
/// Each warp computes 2 output columns, sharing the activation vector load across rows.
/// This halves activation memory bandwidth compared to `launch_gemv_kernel_bt`.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
/// `b_ptr` points to the raw [N,K] data (NOT the transposed [K,N] view).
pub unsafe fn launch_gemv_kernel_bt_mr(
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
    let module = get_or_load_module(context, device_index, gemv_module(dtype))?;
    let func_name = kernel_name("gemv_bt_mr", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // grid: (ceil(N / (WARPS_PER_BLOCK * ROWS_PER_WARP)), M, batch), block: (256, 1, 1)
    // 8 warps per block, each warp handles 2 output columns.
    let warps_per_block: u32 = 8;
    let rows_per_warp: u32 = 2;
    let cols_per_block = warps_per_block * rows_per_warp; // 16
    let grid_x = ((n as u32) + cols_per_block - 1) / cols_per_block;
    let grid_y = m as u32;
    let grid_z = batch as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    };

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
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA GEMV-BT-MR kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}
