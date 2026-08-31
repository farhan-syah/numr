//! CUDA kernel launchers for quasi-random sequence generation

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_names,
    launch_config,
};
use crate::error::{Error, Result};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

/// Launch Sobol sequence generation kernel (F32).
///
/// `dv_ptr` must be a valid device pointer holding `dimension * 32` `u32` direction
/// vectors that were loaded **before** any CUDA graph capture began. Passing a
/// pointer that was H2D-copied inside a capture region will embed a
/// `cuMemcpyHtoD` memcpy node referencing freed host memory and cause
/// `CUDA_ERROR_ILLEGAL_ADDRESS` on graph replay.
///
/// Use [`CudaClient::warmup_sobol`] to pre-populate the persistent device buffer
/// before entering a capture region.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension`
///   elements of `f32`.
/// - `dv_ptr` must be a valid device pointer with at least `dimension * 32` `u32`
///   elements and must remain valid for the duration of the kernel launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sobol_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dv_ptr: u64,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "sobol_f32";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sobol_f32 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Sobol sequence generation kernel (F64).
///
/// `dv_ptr` must be a valid device pointer holding `dimension * 32` `u32` direction
/// vectors that were loaded **before** any CUDA graph capture began. Passing a
/// pointer that was H2D-copied inside a capture region will embed a
/// `cuMemcpyHtoD` memcpy node referencing freed host memory and cause
/// `CUDA_ERROR_ILLEGAL_ADDRESS` on graph replay.
///
/// Use [`CudaClient::warmup_sobol`] to pre-populate the persistent device buffer
/// before entering a capture region.
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension`
///   elements of `f64`.
/// - `dv_ptr` must be a valid device pointer with at least `dimension * 32` `u32`
///   elements and must remain valid for the duration of the kernel launch.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_sobol_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dv_ptr: u64,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "sobol_f64";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&dv_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA sobol_f64 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Halton sequence generation kernel (F32).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_halton_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "halton_f32";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA halton_f32 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Halton sequence generation kernel (F64).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_points * dimension` elements
pub unsafe fn launch_halton_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_points: usize,
    dimension: usize,
    skip: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "halton_f64";
    let func = get_kernel_function(&module, func_name)?;

    let grid = elementwise_launch_config(n_points)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = n_points as u32;
    let dim = dimension as u32;
    let skip_u32 = skip as u32;
    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&skip_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA halton_f64 kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch Latin Hypercube Sampling kernel (F32).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_samples * dimension` elements
pub unsafe fn launch_latin_hypercube_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_samples: usize,
    dimension: usize,
    seed: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "latin_hypercube_f32";
    let func = get_kernel_function(&module, func_name)?;

    // Each block handles one dimension
    let grid = (dimension as u32, 1, 1);
    let block = (BLOCK_SIZE.min(256), 1, 1); // Use smaller block for shared memory

    // Shared memory for interval shuffling: n_samples * sizeof(u32)
    let shared_mem_bytes = (n_samples * std::mem::size_of::<u32>()) as u32;
    let n = n_samples as u32;
    let dim = dimension as u32;
    let cfg = launch_config(grid, block, shared_mem_bytes);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA latin_hypercube_f32 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch Latin Hypercube Sampling kernel (F64).
///
/// # Safety
/// - `out_ptr` must be a valid device pointer with at least `n_samples * dimension` elements
pub unsafe fn launch_latin_hypercube_f64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    out_ptr: u64,
    n_samples: usize,
    dimension: usize,
    seed: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::QUASIRANDOM_MODULE)?;
    let func_name = "latin_hypercube_f64";
    let func = get_kernel_function(&module, func_name)?;

    // Each block handles one dimension
    let grid = (dimension as u32, 1, 1);
    let block = (BLOCK_SIZE.min(256), 1, 1); // Use smaller block for shared memory

    // Shared memory for interval shuffling: n_samples * sizeof(u32)
    let shared_mem_bytes = (n_samples * std::mem::size_of::<u32>()) as u32;
    let n = n_samples as u32;
    let dim = dimension as u32;
    let cfg = launch_config(grid, block, shared_mem_bytes);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&out_ptr);
        builder.arg(&n);
        builder.arg(&dim);
        builder.arg(&seed);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA latin_hypercube_f64 kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}
