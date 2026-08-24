//! Generic two-pass merge implementations for sparse matrices
//!
//! Zero-duplication generic merge using the strategy pattern.
//! Both CSR and CSC formats are handled here.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use cudarc::types::CudaTypeName;
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::Runtime;
use crate::runtime::cuda::CudaRuntime;
use crate::tensor::Tensor;

use super::super::loader::{
    BLOCK_SIZE, get_kernel_function, get_or_load_module, kernel_names, launch_config,
};
use super::super::sparse_strategy::{MergeStrategy, SparseFormat};
use super::helpers::exclusive_scan_i32;

/// Generic two-pass CSR merge using strategy pattern
///
/// Eliminates code duplication across add/sub/mul/div operations by abstracting
/// the merge semantics through the MergeStrategy trait.
///
/// # Type Parameters
///
/// * `T` - Element type (f32, f64, etc.)
/// * `S` - Merge strategy (AddMerge, SubMerge, MulMerge, DivMerge)
///
/// # Algorithm
///
/// 1. **Count**: Determine output size per row using strategy-specific semantics
/// 2. **Scan**: Compute row_ptrs via exclusive prefix sum
/// 3. **Compute**: Merge values using strategy-specific operation
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSR format. `nrows` must match the sparse matrix dimensions.
/// The CUDA stream and context must be valid and associated with the correct device.
pub unsafe fn generic_csr_merge<T: CudaTypeName, S: MergeStrategy>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    row_ptrs_a: &Tensor<CudaRuntime>,
    col_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    row_ptrs_b: &Tensor<CudaRuntime>,
    col_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    nrows: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    // Pass 1: Count output size per row
    let row_counts = Tensor::<CudaRuntime>::try_zeros(&[nrows], DType::I32, device)?;

    // Launch count kernel (union vs intersection semantics determined by strategy)
    let count_kernel_name = S::count_kernel_name(SparseFormat::Csr);
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let function = get_kernel_function(&module, count_kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (nrows as u32 + block_size - 1) / block_size;
    let nrows_i32 = nrows as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let row_ptrs_a_ptr = row_ptrs_a.ptr();
    let col_indices_a_ptr = col_indices_a.ptr();
    let row_ptrs_b_ptr = row_ptrs_b.ptr();
    let col_indices_b_ptr = col_indices_b.ptr();
    let row_counts_ptr = row_counts.ptr();

    builder.arg(&row_ptrs_a_ptr);
    builder.arg(&col_indices_a_ptr);
    builder.arg(&row_ptrs_b_ptr);
    builder.arg(&col_indices_b_ptr);
    builder.arg(&row_counts_ptr);
    builder.arg(&nrows_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel accesses GPU memory
    // Safety requirements satisfied:
    // - All pointers are valid GPU memory addresses from CudaRuntime tensors
    // - Tensor lifetimes ensure memory is valid during kernel execution
    // - nrows matches the actual tensor dimensions
    // - Stream synchronization ensures no data races
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (nrows={}, strategy={:?}): {:?}",
                count_kernel_name,
                nrows,
                S::OP,
                e
            ))
        })?;
    }

    // Synchronize to ensure counts are ready
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get row_ptrs and total_nnz
    let (out_row_ptrs, total_nnz) = exclusive_scan_i32(context, stream, device_index, &row_counts)?;

    // Pass 2: Allocate output and compute merged result
    let out_col_indices = Tensor::<CudaRuntime>::try_zeros(&[total_nnz], DType::I32, device)?;
    let out_values = Tensor::<CudaRuntime>::try_zeros(&[total_nnz], dtype, device)?;

    // Launch compute kernel (operation-specific)
    let compute_kernel_name = S::compute_kernel_name(SparseFormat::Csr, T::NAME);
    let function = get_kernel_function(&module, &compute_kernel_name)?;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let row_ptrs_a_ptr = row_ptrs_a.ptr();
    let col_indices_a_ptr = col_indices_a.ptr();
    let values_a_ptr = values_a.ptr();
    let row_ptrs_b_ptr = row_ptrs_b.ptr();
    let col_indices_b_ptr = col_indices_b.ptr();
    let values_b_ptr = values_b.ptr();
    let out_row_ptrs_ptr = out_row_ptrs.ptr();
    let out_col_indices_ptr = out_col_indices.ptr();
    let out_values_ptr = out_values.ptr();

    builder.arg(&row_ptrs_a_ptr);
    builder.arg(&col_indices_a_ptr);
    builder.arg(&values_a_ptr);
    builder.arg(&row_ptrs_b_ptr);
    builder.arg(&col_indices_b_ptr);
    builder.arg(&values_b_ptr);
    builder.arg(&out_row_ptrs_ptr);
    builder.arg(&out_col_indices_ptr);
    builder.arg(&out_values_ptr);
    builder.arg(&nrows_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel writes to output tensors
    // Safety requirements satisfied:
    // - All input pointers are valid GPU memory from input tensors
    // - Output tensors allocated with correct size (total_nnz from scan)
    // - Tensor ownership prevents concurrent modification
    // - Stream ordering ensures count kernel completed before compute kernel
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (nrows={}, total_nnz={}, strategy={:?}): {:?}",
                compute_kernel_name,
                nrows,
                total_nnz,
                S::OP,
                e
            ))
        })?;
    }

    Ok((out_row_ptrs, out_col_indices, out_values))
}

/// Generic two-pass CSC merge using strategy pattern
///
/// CSC variant of generic_csr_merge. See generic_csr_merge for details.
///
/// # Safety
///
/// All tensor arguments must contain valid CUDA device pointers with correct sizes
/// for the given sparse CSC format. `ncols` must match the sparse matrix dimensions.
/// The CUDA stream and context must be valid and associated with the correct device.
pub unsafe fn generic_csc_merge<T: CudaTypeName, S: MergeStrategy>(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &<CudaRuntime as Runtime>::Device,
    dtype: DType,
    col_ptrs_a: &Tensor<CudaRuntime>,
    row_indices_a: &Tensor<CudaRuntime>,
    values_a: &Tensor<CudaRuntime>,
    col_ptrs_b: &Tensor<CudaRuntime>,
    row_indices_b: &Tensor<CudaRuntime>,
    values_b: &Tensor<CudaRuntime>,
    ncols: usize,
) -> Result<(
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
    Tensor<CudaRuntime>,
)> {
    // Pass 1: Count output size per column
    let col_counts = Tensor::<CudaRuntime>::try_zeros(&[ncols], DType::I32, device)?;

    // Launch count kernel
    let count_kernel_name = S::count_kernel_name(SparseFormat::Csc);
    let module = get_or_load_module(context, device_index, kernel_names::SPARSE_MERGE_MODULE)?;
    let function = get_kernel_function(&module, count_kernel_name)?;

    let block_size = BLOCK_SIZE;
    let grid_size = (ncols as u32 + block_size - 1) / block_size;
    let ncols_i32 = ncols as i32;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let col_ptrs_a_ptr = col_ptrs_a.ptr();
    let row_indices_a_ptr = row_indices_a.ptr();
    let col_ptrs_b_ptr = col_ptrs_b.ptr();
    let row_indices_b_ptr = row_indices_b.ptr();
    let col_counts_ptr = col_counts.ptr();

    builder.arg(&col_ptrs_a_ptr);
    builder.arg(&row_indices_a_ptr);
    builder.arg(&col_ptrs_b_ptr);
    builder.arg(&row_indices_b_ptr);
    builder.arg(&col_counts_ptr);
    builder.arg(&ncols_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel accesses GPU memory
    // Safety requirements satisfied:
    // - All pointers are valid GPU memory addresses from CudaRuntime tensors
    // - Tensor lifetimes ensure memory is valid during kernel execution
    // - ncols matches the actual tensor dimensions
    // - Stream synchronization ensures no data races
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (ncols={}, strategy={:?}): {:?}",
                count_kernel_name,
                ncols,
                S::OP,
                e
            ))
        })?;
    }

    // Synchronize to ensure counts are ready
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Stream synchronize failed: {:?}", e)))?;

    // Exclusive scan to get col_ptrs and total_nnz
    let (out_col_ptrs, total_nnz) = exclusive_scan_i32(context, stream, device_index, &col_counts)?;

    // Pass 2: Allocate output and compute merged result
    let out_row_indices = Tensor::<CudaRuntime>::try_zeros(&[total_nnz], DType::I32, device)?;
    let out_values = Tensor::<CudaRuntime>::try_zeros(&[total_nnz], dtype, device)?;

    // Launch compute kernel
    let compute_kernel_name = S::compute_kernel_name(SparseFormat::Csc, T::NAME);
    let function = get_kernel_function(&module, &compute_kernel_name)?;

    let cfg = launch_config((grid_size, 1, 1), (block_size, 1, 1), 0);
    let mut builder = stream.launch_builder(&function);

    // Store pointers to avoid temporary value issues
    let col_ptrs_a_ptr = col_ptrs_a.ptr();
    let row_indices_a_ptr = row_indices_a.ptr();
    let values_a_ptr = values_a.ptr();
    let col_ptrs_b_ptr = col_ptrs_b.ptr();
    let row_indices_b_ptr = row_indices_b.ptr();
    let values_b_ptr = values_b.ptr();
    let out_col_ptrs_ptr = out_col_ptrs.ptr();
    let out_row_indices_ptr = out_row_indices.ptr();
    let out_values_ptr = out_values.ptr();

    builder.arg(&col_ptrs_a_ptr);
    builder.arg(&row_indices_a_ptr);
    builder.arg(&values_a_ptr);
    builder.arg(&col_ptrs_b_ptr);
    builder.arg(&row_indices_b_ptr);
    builder.arg(&values_b_ptr);
    builder.arg(&out_col_ptrs_ptr);
    builder.arg(&out_row_indices_ptr);
    builder.arg(&out_values_ptr);
    builder.arg(&ncols_i32);

    // SAFETY: Kernel launch is unsafe because:
    // 1. Raw pointers are passed to CUDA kernel
    // 2. Kernel writes to output tensors
    // Safety requirements satisfied:
    // - All input pointers are valid GPU memory from input tensors
    // - Output tensors allocated with correct size (total_nnz from scan)
    // - Tensor ownership prevents concurrent modification
    // - Stream ordering ensures count kernel completed before compute kernel
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed (ncols={}, total_nnz={}, strategy={:?}): {:?}",
                compute_kernel_name,
                ncols,
                total_nnz,
                S::OP,
                e
            ))
        })?;
    }

    Ok((out_col_ptrs, out_row_indices, out_values))
}
