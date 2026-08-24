// ESC + Hash Accumulation SpGEMM Kernel Launchers
// Two-phase sparse matrix-matrix multiplication
//
// This implements the SAME algorithm as CPU for backend parity

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use super::loader::{BLOCK_SIZE, get_kernel_function, get_or_load_module, launch_config};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

/// CUDA module name for sparse matrix-matrix multiplication (SpGEMM) kernels.
pub const SPGEMM_MODULE: &str = "spgemm";

/// Phase 1: Symbolic - Count NNZ per output row of C = A * B
///
/// Uses a bitmap approach per thread to count unique column indices produced by each output row.
/// Allocates dynamic shared memory of `block_size * ceil(n / 8)` bytes.
///
/// # Safety
///
/// - All tensor arguments must be valid `CudaRuntime` tensors on the device associated with
///   `context` with consistent CSR structure.
/// - `m` must equal the number of rows in `a`; `n` must equal the number of columns in `b`.
/// - `m * ceil(n / 8)` bytes of shared memory must be available on the device.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn spgemm_symbolic_phase(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    a_row_ptrs: &Tensor<CudaRuntime>,
    a_col_indices: &Tensor<CudaRuntime>,
    b_row_ptrs: &Tensor<CudaRuntime>,
    b_col_indices: &Tensor<CudaRuntime>,
    m: usize,
    n: usize,
) -> Result<Tensor<CudaRuntime>> {
    let module = get_or_load_module(context, device_index, SPGEMM_MODULE)?;
    let func = get_kernel_function(&module, "spgemm_symbolic_phase")?;

    // Output: row_nnz[m]
    let row_nnz = Tensor::<CudaRuntime>::zeros(&[m], DType::I32, device)?;

    // Dynamic shared memory allocation for flexible matrix sizes
    // Allocate for ALL launched threads (block_size), not just working threads
    let block_size = BLOCK_SIZE;
    let grid_size = (m as u32 + block_size - 1) / block_size;
    let m_u32 = m as u32;
    let n_u32 = n as u32;

    // Each thread needs (n + 7) / 8 bytes for bitmap
    // CRITICAL: Allocate for block_size (256) threads to match CUDA launch config
    let bytes_per_thread = (n + 7) / 8;
    let shared_mem_bytes = ((block_size as usize) * bytes_per_thread) as u32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), shared_mem_bytes);

    let a_row_ptrs_ptr = a_row_ptrs.ptr();
    let a_col_indices_ptr = a_col_indices.ptr();
    let b_row_ptrs_ptr = b_row_ptrs.ptr();
    let b_col_indices_ptr = b_col_indices.ptr();
    let row_nnz_ptr = row_nnz.ptr();

    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_row_ptrs_ptr);
    builder.arg(&a_col_indices_ptr);
    builder.arg(&b_row_ptrs_ptr);
    builder.arg(&b_col_indices_ptr);
    builder.arg(&row_nnz_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);

    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA spgemm_symbolic_phase launch failed: {:?}", e))
        })?;
    }

    // CRITICAL: Synchronize to ensure kernel completes before returning
    // Otherwise the next kernel (exclusive_scan) will read uninitialized data
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("CUDA synchronization failed: {:?}", e)))?;

    Ok(row_nnz)
}

/// Phase 2: Numeric - Compute values of C = A * B
///
/// Fills the pre-allocated output CSR arrays (`c_row_ptrs`, `c_col_indices`, `c_values`) with
/// the computed product. Must be called after `spgemm_symbolic_phase` and exclusive scan.
///
/// # Safety
///
/// - All tensor arguments must be valid `CudaRuntime` tensors on the device associated with
///   `context` with consistent CSR structure.
/// - `c_row_ptrs` and `c_col_indices` must be pre-allocated (from the symbolic phase and scan).
/// - `c_values` must be pre-allocated to match the NNZ count from the symbolic phase.
/// - `m` must equal the number of rows in `a`; `n` must equal the number of columns in `b`.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
pub unsafe fn spgemm_numeric_phase<T: CudaTypeName + Copy + cudarc::driver::DeviceRepr>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_row_ptrs: &Tensor<CudaRuntime>,
    a_col_indices: &Tensor<CudaRuntime>,
    a_values: &Tensor<CudaRuntime>,
    b_row_ptrs: &Tensor<CudaRuntime>,
    b_col_indices: &Tensor<CudaRuntime>,
    b_values: &Tensor<CudaRuntime>,
    c_row_ptrs: &Tensor<CudaRuntime>,
    c_col_indices: &Tensor<CudaRuntime>,
    c_values: &Tensor<CudaRuntime>,
    m: usize,
    n: usize,
    threshold: T,
) -> Result<()> {
    let dtype_suffix = match T::NAME {
        "float" => "f32",
        "double" => "f64",
        "__half" => "f16",
        "__nv_bfloat16" => "bf16",
        _ => {
            return Err(Error::Internal(format!(
                "Unsupported dtype for SpGEMM: {}",
                T::NAME
            )));
        }
    };

    let kernel_name = format!("spgemm_numeric_phase_{}", dtype_suffix);
    let module = get_or_load_module(context, device_index, SPGEMM_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    // One block per output row
    let grid_size = m as u32;
    let block_size = 256u32; // Threads per block

    // Estimate shared memory need (conservative)
    // Each row needs space for (col_indices, values)
    // We'll use max row capacity
    let max_row_capacity = 1024; // Conservative estimate
    let shared_mem_bytes =
        max_row_capacity * (std::mem::size_of::<i64>() + std::mem::size_of::<T>());

    let cfg = launch_config(
        (grid_size, 1, 1),
        (block_size, 1, 1),
        shared_mem_bytes as u32,
    );

    let a_row_ptrs_ptr = a_row_ptrs.ptr();
    let a_col_indices_ptr = a_col_indices.ptr();
    let a_values_ptr = a_values.ptr();
    let b_row_ptrs_ptr = b_row_ptrs.ptr();
    let b_col_indices_ptr = b_col_indices.ptr();
    let b_values_ptr = b_values.ptr();
    let c_row_ptrs_ptr = c_row_ptrs.ptr();
    let c_col_indices_ptr = c_col_indices.ptr();
    let c_values_ptr = c_values.ptr();

    let m_u32 = m as u32;
    let n_u32 = n as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&a_row_ptrs_ptr);
    builder.arg(&a_col_indices_ptr);
    builder.arg(&a_values_ptr);
    builder.arg(&b_row_ptrs_ptr);
    builder.arg(&b_col_indices_ptr);
    builder.arg(&b_values_ptr);
    builder.arg(&c_row_ptrs_ptr);
    builder.arg(&c_col_indices_ptr);
    builder.arg(&c_values_ptr);
    builder.arg(&m_u32);
    builder.arg(&n_u32);
    builder.arg(&threshold);

    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA spgemm_numeric_phase launch failed: {:?}", e))
        })?;
    }

    Ok(())
}
