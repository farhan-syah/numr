//! FFT CUDA kernel launchers
//!
//! Provides launchers for FFT operations using the Stockham autosort algorithm.
//! Supports Complex64 (float2) and Complex128 (double2) types.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// FFT module name
pub const FFT_MODULE: &str = "fft";

/// Maximum FFT size that can use shared memory (limited by shared memory size)
/// For Complex64: 1024 * 8 bytes * 2 buffers = 16KB
/// For Complex128: 1024 * 16 bytes * 2 buffers = 32KB
pub const MAX_SHARED_MEM_FFT_SIZE: usize = 1024;

/// Launch batched Stockham FFT for small transforms (N <= 1024)
///
/// Uses shared memory for efficient butterfly operations.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `n` must be a power of 2
/// - Input and output must have at least `batch_size * n` elements
pub unsafe fn launch_stockham_fft_batched(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    batch_size: usize,
    inverse: bool,
    scale: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let log_n = (n as f64).log2() as u32;

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "stockham_fft_batched_c64")?;

            // Shared memory: 2 buffers of n complex64 elements
            let shared_mem = (2 * n * std::mem::size_of::<[f32; 2]>()) as u32;

            // One block per FFT, threads cooperate on the FFT
            let block_size = BLOCK_SIZE.min(n as u32);
            let grid = (batch_size as u32, 1, 1);
            let block = (block_size, 1, 1);

            let cfg = launch_config(grid, block, shared_mem);
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let inverse_i32 = if inverse { 1i32 } else { 0i32 };
            let scale_f32 = scale as f32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&log_n);
            builder.arg(&inverse_i32);
            builder.arg(&scale_f32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA FFT kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "stockham_fft_batched_c128")?;

            // Shared memory: 2 buffers of n complex128 elements
            let shared_mem = (2 * n * std::mem::size_of::<[f64; 2]>()) as u32;

            let block_size = BLOCK_SIZE.min(n as u32);
            let grid = (batch_size as u32, 1, 1);
            let block = (block_size, 1, 1);

            let cfg = launch_config(grid, block, shared_mem);
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let inverse_i32 = if inverse { 1i32 } else { 0i32 };
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&log_n);
            builder.arg(&inverse_i32);
            builder.arg(&scale);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA FFT kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType { dtype, op: "fft" });
        }
    }

    Ok(())
}

/// Launch single stage of Stockham FFT for large transforms (N > 1024)
///
/// For large FFTs, we run multiple stages, each doing one level of the butterfly.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `n` must be a power of 2
pub unsafe fn launch_stockham_fft_stage(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
    stage: usize,
    batch_size: usize,
    inverse: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    // Each thread handles one butterfly pair
    let num_butterflies = n / 2;
    let grid_x = ((num_butterflies as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "stockham_fft_stage_c64")?;

            let grid = (grid_x, batch_size as u32, 1);
            let block = (BLOCK_SIZE, 1, 1);

            let cfg = launch_config(grid, block, 0);
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let stage_u32 = stage as u32;
            let inverse_i32 = if inverse { 1i32 } else { 0i32 };
            let batch_u32 = batch_size as u32;

            builder.arg(&src_ptr);
            builder.arg(&dst_ptr);
            builder.arg(&n_u32);
            builder.arg(&stage_u32);
            builder.arg(&inverse_i32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA FFT stage kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "stockham_fft_stage_c128")?;

            let grid = (grid_x, batch_size as u32, 1);
            let block = (BLOCK_SIZE, 1, 1);

            let cfg = launch_config(grid, block, 0);
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let stage_u32 = stage as u32;
            let inverse_i32 = if inverse { 1i32 } else { 0i32 };
            let batch_u32 = batch_size as u32;

            builder.arg(&src_ptr);
            builder.arg(&dst_ptr);
            builder.arg(&n_u32);
            builder.arg(&stage_u32);
            builder.arg(&inverse_i32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA FFT stage kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType { dtype, op: "fft" });
        }
    }

    Ok(())
}

/// Launch scale kernel for complex data
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_scale_complex(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    data_ptr: u64,
    scale: f64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid = elementwise_launch_config(n)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "scale_complex_c64")?;
            let mut builder = stream.launch_builder(&func);

            let scale_f32 = scale as f32;
            let n_u32 = n as u32;

            builder.arg(&data_ptr);
            builder.arg(&scale_f32);
            builder.arg(&n_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA scale kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "scale_complex_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;

            builder.arg(&data_ptr);
            builder.arg(&scale);
            builder.arg(&n_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA scale kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "scale_complex",
            });
        }
    }

    Ok(())
}

/// Launch rfft pack kernel (real -> complex with zero imaginary)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_rfft_pack(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match input_dtype {
        DType::F32 => {
            let func = get_kernel_function(&module, "rfft_pack_c64")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA rfft_pack kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::F64 => {
            let func = get_kernel_function(&module, "rfft_pack_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA rfft_pack kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "rfft_pack",
            });
        }
    }

    Ok(())
}

/// Launch irfft unpack kernel (complex -> real, extracting real parts)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_irfft_unpack(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    output_dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    scale: f64,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match output_dtype {
        DType::F32 => {
            let func = get_kernel_function(&module, "irfft_unpack_c64")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let scale_f32 = scale as f32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&scale_f32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA irfft_unpack kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::F64 => {
            let func = get_kernel_function(&module, "irfft_unpack_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&scale);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA irfft_unpack kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: output_dtype,
                op: "irfft_unpack",
            });
        }
    }

    Ok(())
}

/// Launch Hermitian extension kernel (N/2+1 complex -> N complex)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_hermitian_extend(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    half_n: usize,
    full_n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((full_n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "hermitian_extend_c64")?;
            let mut builder = stream.launch_builder(&func);

            let half_n_u32 = half_n as u32;
            let full_n_u32 = full_n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&half_n_u32);
            builder.arg(&full_n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA hermitian_extend kernel launch failed: {:?}",
                        e
                    ))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "hermitian_extend_c128")?;
            let mut builder = stream.launch_builder(&func);

            let half_n_u32 = half_n as u32;
            let full_n_u32 = full_n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&half_n_u32);
            builder.arg(&full_n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA hermitian_extend kernel launch failed: {:?}",
                        e
                    ))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "hermitian_extend",
            });
        }
    }

    Ok(())
}

/// Launch rfft truncation kernel (N complex -> N/2+1 complex)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_rfft_truncate(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    full_n: usize,
    half_n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((half_n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "rfft_truncate_c64")?;
            let mut builder = stream.launch_builder(&func);

            let full_n_u32 = full_n as u32;
            let half_n_u32 = half_n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&full_n_u32);
            builder.arg(&half_n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA rfft_truncate kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "rfft_truncate_c128")?;
            let mut builder = stream.launch_builder(&func);

            let full_n_u32 = full_n as u32;
            let half_n_u32 = half_n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&full_n_u32);
            builder.arg(&half_n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA rfft_truncate kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "rfft_truncate",
            });
        }
    }

    Ok(())
}

/// Launch fftshift kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_fftshift(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "fftshift_c64")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA fftshift kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "fftshift_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA fftshift kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "fftshift",
            });
        }
    }

    Ok(())
}

/// Launch ifftshift kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_ifftshift(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid_x = ((n as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, batch_size as u32, 1);
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "ifftshift_c64")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA ifftshift kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "ifftshift_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;
            let batch_u32 = batch_size as u32;

            builder.arg(&input_ptr);
            builder.arg(&output_ptr);
            builder.arg(&n_u32);
            builder.arg(&batch_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA ifftshift kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "ifftshift",
            });
        }
    }

    Ok(())
}

/// Launch copy kernel for complex data
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
#[allow(dead_code)]
pub unsafe fn launch_copy_complex(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    src_ptr: u64,
    dst_ptr: u64,
    n: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_MODULE)?;

    let grid = elementwise_launch_config(n)?;
    let block = (BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "copy_complex_c64")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;

            builder.arg(&src_ptr);
            builder.arg(&dst_ptr);
            builder.arg(&n_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA copy_complex kernel launch failed: {:?}", e))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "copy_complex_c128")?;
            let mut builder = stream.launch_builder(&func);

            let n_u32 = n as u32;

            builder.arg(&src_ptr);
            builder.arg(&dst_ptr);
            builder.arg(&n_u32);

            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!("CUDA copy_complex kernel launch failed: {:?}", e))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "copy_complex",
            });
        }
    }

    Ok(())
}
