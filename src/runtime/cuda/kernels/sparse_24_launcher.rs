//! CUDA kernel launchers for 2:4 structured sparsity
//!
//! Kernel source: sparse_24.cu

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};

const MODULE_NAME: &str = "sparse_24";

/// Launch prune-to-2:4 kernel.
///
/// # Safety
/// All pointers must be valid device memory of correct size.
pub unsafe fn launch_sparse_24_prune(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    dense_ptr: u64,
    compressed_ptr: u64,
    metadata_ptr: u64,
    m: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, MODULE_NAME)?;
    let func_name = kernel_name("sparse_24_prune", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let total_groups = (m * (k / 4)) as u32;
    let grid = elementwise_launch_config(total_groups as usize)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let m_u32 = m as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&dense_ptr);
        builder.arg(&compressed_ptr);
        builder.arg(&metadata_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA sparse_24_prune launch failed: {e:?}")))?;
    }

    Ok(())
}

/// Launch decompress-from-2:4 kernel.
///
/// # Safety
/// All pointers must be valid device memory of correct size.
pub unsafe fn launch_sparse_24_decompress(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    compressed_ptr: u64,
    metadata_ptr: u64,
    dense_ptr: u64,
    m: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, MODULE_NAME)?;
    let func_name = kernel_name("sparse_24_decompress", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let total_groups = (m * (k / 4)) as u32;
    let grid = elementwise_launch_config(total_groups as usize)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let m_u32 = m as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&compressed_ptr);
        builder.arg(&metadata_ptr);
        builder.arg(&dense_ptr);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sparse_24_decompress launch failed: {e:?}"))
        })?;
    }

    Ok(())
}

/// Launch 2:4 sparse matmul kernel: C = A @ B^T where B is 2:4 compressed.
///
/// # Safety
/// All pointers must be valid device memory of correct size.
pub unsafe fn launch_sparse_24_matmul(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,            // [N, K]
    b_compressed_ptr: u64, // [M, K/2]
    b_metadata_ptr: u64,   // [M, meta_cols]
    c_ptr: u64,            // [N, M]
    n: usize,
    m: usize,
    k: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, MODULE_NAME)?;
    let func_name = kernel_name("sparse_24_matmul", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let tile_size = 16u32;
    let grid_x = (m as u32 + tile_size - 1) / tile_size;
    let grid_y = (n as u32 + tile_size - 1) / tile_size;
    let grid = (grid_x, grid_y, 1);
    let block = (tile_size, tile_size, 1);
    let cfg = launch_config(grid, block, 0);

    let n_u32 = n as u32;
    let m_u32 = m as u32;
    let k_u32 = k as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_compressed_ptr);
        builder.arg(&b_metadata_ptr);
        builder.arg(&c_ptr);
        builder.arg(&n_u32);
        builder.arg(&m_u32);
        builder.arg(&k_u32);
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA sparse_24_matmul launch failed: {e:?}")))?;
    }

    Ok(())
}
