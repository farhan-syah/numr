//! CUDA kernel launchers for sorting and search operations

use super::loader::{
    BLOCK_SIZE, check_shared_mem_fits, dtype_suffix, elementwise_launch_config,
    get_kernel_function, get_or_load_module, kernel_name, launch_config,
};
use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for sort kernels
pub const SORT_MODULE: &str = "sort";

/// Calculate shared memory size for sort operations.
///
/// Computed in `u64` so a huge `sort_size` cannot overflow before the
/// `u32::MAX` check below — `next_power_of_two` and the final byte total can
/// each exceed 32 bits long before they exceed real device shared memory.
fn sort_shared_mem_size(sort_size: usize, elem_size: usize) -> Result<u32> {
    // Need space for values and indices
    // Pad to next power of 2 for bitonic sort
    let n = (sort_size as u64).next_power_of_two();
    let vals_bytes = n * elem_size as u64;
    // Align to 8 bytes for long long indices (matches kernel alignment logic)
    let aligned_offset = (vals_bytes + 7) & !7;
    let total = aligned_offset + n * 8;
    u32::try_from(total).map_err(|_| Error::BackendLimitation {
        backend: "cuda",
        operation: "sort",
        reason: format!(
            "sort dimension of size {sort_size} needs {total} bytes of shared memory, \
             which overflows the u32 launch config field"
        ),
    })
}

/// Launch sort kernel with indices
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_sort(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("sort", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size)?;
    check_shared_mem_fits(device_index, shared_mem, "sort", || {
        format!("sort dimension of size {sort_size}")
    })?;

    // 2D grid: (outer, inner)
    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA sort kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch sort kernel (values only, no indices)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_sort_values_only(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = format!("sort_values_only_{}", dtype_suffix(dtype));
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size)?;
    check_shared_mem_fits(device_index, shared_mem, "sort", || {
        format!("sort dimension of size {sort_size}")
    })?;

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA sort_values_only kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch argsort kernel (indices only, no values)
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_argsort(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("argsort", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size)?;
    check_shared_mem_fits(device_index, shared_mem, "sort", || {
        format!("sort dimension of size {sort_size}")
    })?;

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let desc_u32 = descending as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&desc_u32);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA argsort kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch topk kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_topk(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    values_ptr: u64,
    indices_ptr: u64,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    k: usize,
    largest: bool,
    sorted: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("topk", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let elem_size = dtype.size_in_bytes();
    let shared_mem = sort_shared_mem_size(sort_size, elem_size)?;
    check_shared_mem_fits(device_index, shared_mem, "sort", || {
        format!("sort dimension of size {sort_size}")
    })?;

    let grid = (outer_size as u32, inner_size as u32, 1);
    let block = (BLOCK_SIZE.min(sort_size as u32).max(1), 1, 1);

    let cfg = launch_config(grid, block, shared_mem);

    let outer_u32 = outer_size as u32;
    let sort_u32 = sort_size as u32;
    let inner_u32 = inner_size as u32;
    let k_u32 = k as u32;
    let largest_u32 = largest as u32;
    let sorted_u32 = sorted as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&values_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&outer_u32);
    builder.arg(&sort_u32);
    builder.arg(&inner_u32);
    builder.arg(&k_u32);
    builder.arg(&largest_u32);
    builder.arg(&sorted_u32);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA topk kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

/// Launch count_nonzero kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_count_nonzero(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    count_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("count_nonzero", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel)?;
    let grid = (grid_size.min(256), 1, 1); // Limit grid size for atomic efficiency
    let block = (BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&count_ptr);
    builder.arg(&n);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA count_nonzero kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch gather_nonzero kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_gather_nonzero(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    indices_ptr: u64,
    counter_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("gather_nonzero", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel)?;
    let grid = (grid_size.min(256), 1, 1);
    let block = (BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&indices_ptr);
    builder.arg(&counter_ptr);
    builder.arg(&n);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA gather_nonzero kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch flat_to_multi_index kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_flat_to_multi_index(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    flat_indices_ptr: u64,
    multi_indices_ptr: u64,
    nnz: usize,
    ndim: usize,
    shape_ptr: u64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let func = get_kernel_function(&module, "flat_to_multi_index")?;

    let (grid_size, _, _) = elementwise_launch_config(nnz)?;
    let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let nnz_u32 = nnz as u32;
    let ndim_u32 = ndim as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&flat_indices_ptr);
    builder.arg(&multi_indices_ptr);
    builder.arg(&nnz_u32);
    builder.arg(&ndim_u32);
    builder.arg(&shape_ptr);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA flat_to_multi_index kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Launch searchsorted kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_searchsorted(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    seq_ptr: u64,
    values_ptr: u64,
    output_ptr: u64,
    seq_len: usize,
    num_values: usize,
    right: bool,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("searchsorted", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(num_values)?;
    let cfg = launch_config((grid_size, 1, 1), (BLOCK_SIZE, 1, 1), 0);

    let seq_len_u32 = seq_len as u32;
    let num_values_u32 = num_values as u32;
    let right_u32 = right as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&seq_ptr);
    builder.arg(&values_ptr);
    builder.arg(&output_ptr);
    builder.arg(&seq_len_u32);
    builder.arg(&num_values_u32);
    builder.arg(&right_u32);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA searchsorted kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch count_unique kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_count_unique(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_input_ptr: u64,
    count_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("count_unique", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel)?;
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&sorted_input_ptr);
    builder.arg(&count_ptr);
    builder.arg(&n);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA count_unique kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch extract_unique kernel
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_extract_unique(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    sorted_input_ptr: u64,
    output_ptr: u64,
    counter_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let kname = kernel_name("extract_unique", dtype);
    let func = get_kernel_function(&module, &kname)?;

    let (grid_size, _, _) = elementwise_launch_config(numel)?;
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);
    let n = numel as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&sorted_input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&counter_ptr);
    builder.arg(&n);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA extract_unique kernel launch failed: {:?}", e))
        })?;
    }

    Ok(())
}

/// Launch bincount kernel - counts occurrences of each index
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments (`*_ptr`) point to valid GPU memory
/// allocated on `device_index` with sufficient size for the operation.
pub unsafe fn launch_bincount(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    indices_ptr: u64,
    counts_ptr: u64,
    numel: usize,
    num_bins: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, SORT_MODULE)?;
    let func = get_kernel_function(&module, "bincount")?;

    let (grid_size, _, _) = elementwise_launch_config(numel)?;
    let grid = (grid_size.min(256), 1, 1);
    let cfg = launch_config(grid, (BLOCK_SIZE, 1, 1), 0);

    let n = numel as u32;
    let bins = num_bins as u32;

    let mut builder = stream.launch_builder(&func);
    builder.arg(&indices_ptr);
    builder.arg(&counts_ptr);
    builder.arg(&n);
    builder.arg(&bins);

    // SAFETY: Kernel arguments match the CUDA kernel signature and pointers are valid
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA bincount kernel launch failed: {:?}", e)))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_a_typical_shared_mem_budget() {
        // 2048 f32 values + indices, well under a typical 48KB per-block budget.
        let bytes = sort_shared_mem_size(2048, 4).unwrap();
        assert!(bytes < 48 * 1024, "expected < 48KiB, got {bytes}");
    }

    #[test]
    fn rejects_when_computation_overflows_u32() {
        // next_power_of_two(2^28) * 8 (f64) already exceeds u32::MAX, and the
        // check must catch it instead of wrapping the final `as u32` cast.
        let result = sort_shared_mem_size(1 << 28, 8);
        assert!(result.is_err());
    }

    #[test]
    fn does_not_overflow_near_u32_max_sort_size() {
        // A sort_size whose next_power_of_two is right at the u32 boundary
        // must be handled in u64 without panicking or wrapping.
        let result = sort_shared_mem_size(u32::MAX as usize, 1);
        assert!(result.is_err(), "this size cannot fit in a u32 byte count");
    }

    #[test]
    fn small_size_computation_is_exact() {
        // n = 8 (already a power of 2), elem_size = 4: vals_bytes = 32,
        // aligned_offset = 32 (already 8-aligned), + n*8 = 64 => 96 total.
        let bytes = sort_shared_mem_size(8, 4).unwrap();
        assert_eq!(bytes, 96);
    }

    #[test]
    fn device_limit_rejects_oversized_request() {
        let result = check_shared_mem_fits(0, u32::MAX, "sort", || {
            format!("sort dimension of size {}", 1 << 20)
        });
        // No CUDA device is guaranteed present in unit tests; either the
        // profile query reports a real limit (rejects) or falls back to
        // `unknown()` with limit 0 (also rejects). Both paths return Err.
        assert!(result.is_err());
    }
}
