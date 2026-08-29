//! Semiring matrix multiplication launchers.
//!
//! One thread per output element, with the semiring op passed as a kernel
//! argument rather than specialised into the kernel name.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{kernel_name, kernel_names};

/// Launch semiring matrix multiplication kernel.
///
/// Uses a simple one-thread-per-element kernel parameterized by semiring op.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_semiring_matmul_kernel(
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
    semiring_op: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SEMIRING_MATMUL_MODULE)?;
    let func_name = kernel_name("semiring_matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    // Simple 16x16 thread blocks, one thread per output element
    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (n as u32 + block_x - 1) / block_x;
    let grid_y = (m as u32 + block_y - 1) / block_y;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, 1),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
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
        builder.arg(&semiring_op);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA semiring matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch batched semiring matrix multiplication kernel.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub unsafe fn launch_semiring_matmul_batched_kernel(
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
    semiring_op: u32,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SEMIRING_MATMUL_MODULE)?;
    let func_name = kernel_name("semiring_matmul_batched", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let block_x = 16u32;
    let block_y = 16u32;
    let grid_x = (n as u32 + block_x - 1) / block_x;
    let grid_y = (m as u32 + block_y - 1) / block_y;
    let grid_z = batch as u32;

    let cfg = LaunchConfig {
        grid_dim: (grid_x, grid_y, grid_z),
        block_dim: (block_x, block_y, 1),
        shared_mem_bytes: 0,
    };

    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let batch_u32 = batch as u32;
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
        builder.arg(&semiring_op);
        builder.arg(&batch_u32);
        builder.arg(&a_batch_u32);
        builder.arg(&b_batch_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA batched semiring matmul kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
