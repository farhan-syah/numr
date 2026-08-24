//! GPU-accelerated parallel prefix sum (exclusive scan)
//!
//! Provides efficient exclusive scan operations on CUDA tensors, used primarily
//! for sparse matrix operations that need to compute CSR row_ptrs from per-row
//! counts.
//!
//! # Performance
//!
//! - Small arrays (n ≤ 512): Single-block scan, ~10-100 µs
//! - Large arrays (n > 512): Multi-block scan with recursive block sum scan
//! - **Unlimited size support**: Recursive multi-level scan handles arbitrarily large arrays
//! - Zero CPU-GPU transfers (fully on-device)
//!
//! # Example
//!
//! ```ignore
//! // Input: [3, 1, 4, 1, 5] (per-row non-zero counts)
//! // Output: [0, 3, 4, 8, 9, 14] (CSR row_ptrs)
//! let (row_ptrs, total_nnz) = exclusive_scan_i32_gpu(context, stream, device_index, device, &counts)?;
//! assert_eq!(total_nnz, 14);
//! ```

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use crate::tensor::Tensor;
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

use super::loader::{get_kernel_function, get_or_load_module, kernel_names, launch_config};

/// Scan-specific block size (must match SCAN_BLOCK_SIZE in scan.cu)
const SCAN_BLOCK_SIZE: u32 = 512;

/// Maximum recursion depth for multi-level scan.
/// Each level handles SCAN_BLOCK_SIZE^level elements.
/// Depth 10 supports up to 512^10 ≈ 10^27 elements (far beyond any practical use).
/// This prevents stack overflow from malformed inputs or algorithmic errors.
const MAX_SCAN_RECURSION_DEPTH: usize = 10;

// ============================================================================
// I32 Exclusive Scan
// ============================================================================

/// Perform GPU-accelerated exclusive scan on I32 tensor.
///
/// Returns a tuple of:
/// - Output tensor with size n+1, where output[0] = 0 and output[i] = sum(input[0..i])
/// - Total sum (last element of output)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `device` - CUDA device
/// * `input` - Input tensor of I32 values (size n)
///
/// # Returns
///
/// `(output_tensor, total_sum)` where output has size n+1
///
/// # Safety
///
/// - `input` must be a valid `CudaRuntime` tensor of `DType::I32` on the device associated with
///   `context`. Passing a tensor with a different dtype returns an error.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
/// - A single scalar GPU-to-CPU transfer is performed at the end to read the total sum; this is
///   intentional and documented as acceptable for control-flow purposes.
pub unsafe fn exclusive_scan_i32_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    input: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, usize)> {
    let n = input.numel();

    if input.dtype() != DType::I32 {
        return Err(Error::Internal(format!(
            "exclusive_scan_i32_gpu expects I32 input, got {:?}",
            input.dtype()
        )));
    }

    // Allocate output tensor with size n+1
    let output = Tensor::<CudaRuntime>::try_zeros(&[n + 1], DType::I32, device)?;

    let input_ptr = input.ptr();
    let output_ptr = output.ptr();

    if n <= SCAN_BLOCK_SIZE as usize {
        // Small array: use single-block scan
        unsafe {
            launch_scan_single_block_i32(
                context,
                stream,
                device_index,
                input_ptr,
                output_ptr,
                n as u32,
            )?;
        }
    } else {
        // Large array: use multi-block scan (start at depth 0)
        unsafe {
            launch_scan_multi_block_i32(
                context,
                stream,
                device_index,
                device,
                input_ptr,
                output_ptr,
                n as u32,
                0, // Initial recursion depth
            )?;
        }
    }

    // Synchronize to ensure scan is complete before reading total
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Failed to synchronize after scan: {:?}", e)))?;

    // Read the total sum from output[n] — single scalar GPU→CPU read (acceptable control flow)
    let mut total_i32: i32 = 0;
    let offset_bytes = n * std::mem::size_of::<i32>();
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            &mut total_i32 as *mut i32 as *mut std::ffi::c_void,
            output.ptr() + offset_bytes as u64,
            std::mem::size_of::<i32>(),
        );
    }
    let total = total_i32 as usize;

    Ok((output, total))
}

/// Launch single-block exclusive scan kernel
unsafe fn launch_scan_single_block_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_ptr: u64,
    output_ptr: u64,
    n: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SCAN_MODULE)?;
    let func = get_kernel_function(&module, "exclusive_scan_i32")?;

    // Single block with SCAN_BLOCK_SIZE threads
    let grid = (1, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan single-block kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch multi-block exclusive scan with recursive support for unlimited sizes.
///
/// # Arguments
/// * `depth` - Current recursion depth (0 at entry, incremented on each recursive call)
///
/// # Safety
/// Caller must ensure input_ptr and output_ptr point to valid device memory.
unsafe fn launch_scan_multi_block_i32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    input_ptr: u64,
    output_ptr: u64,
    n: u32,
    depth: usize,
) -> Result<()> {
    // Check recursion depth to prevent stack overflow
    if depth >= MAX_SCAN_RECURSION_DEPTH {
        return Err(Error::Internal(format!(
            "Scan recursion depth {} exceeds maximum {}. \
             This indicates an algorithmic error or impossibly large input.",
            depth, MAX_SCAN_RECURSION_DEPTH
        )));
    }

    let module = get_or_load_module(context, device_index, kernel_names::SCAN_MODULE)?;

    // Calculate number of blocks needed
    let num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

    // Allocate temporary buffer for block sums
    let block_sums = Tensor::<CudaRuntime>::try_zeros(&[num_blocks as usize], DType::I32, device)?;
    let block_sums_ptr = block_sums.ptr();

    // Step 1: Scan each block independently
    let func_step1 = get_kernel_function(&module, "scan_blocks_i32_step1")?;
    let grid = (num_blocks, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let mut builder = stream.launch_builder(&func_step1);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&block_sums_ptr);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA scan step 1 kernel launch failed: {:?}", e)))?;

    // Synchronize after step 1
    stream.synchronize().map_err(|e| {
        Error::Internal(format!("Failed to synchronize after scan step 1: {:?}", e))
    })?;

    // Step 2: Recursively scan the block sums
    // Allocate buffer for scanned block sums (size num_blocks + 1)
    let scanned_block_sums =
        Tensor::<CudaRuntime>::try_zeros(&[num_blocks as usize + 1], DType::I32, device)?;
    let scanned_block_sums_ptr = scanned_block_sums.ptr();

    if num_blocks <= SCAN_BLOCK_SIZE {
        // Block sums fit in single block - use simple scan
        let func_scan = get_kernel_function(&module, "exclusive_scan_i32")?;
        let grid = (1, 1, 1);
        let block = (SCAN_BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let mut builder = stream.launch_builder(&func_scan);
        builder.arg(&block_sums_ptr);
        builder.arg(&scanned_block_sums_ptr);
        builder.arg(&num_blocks);
        unsafe { builder.launch(cfg) }.map_err(|e| {
            Error::Internal(format!(
                "CUDA scan block sums kernel launch failed: {:?}",
                e
            ))
        })?;

        stream.synchronize().map_err(|e| {
            Error::Internal(format!("Failed to synchronize after scan step 2: {:?}", e))
        })?;
    } else {
        // Block sums don't fit in single block - recursively scan them
        // Treat block_sums as input tensor for recursive scan
        unsafe {
            launch_scan_multi_block_i32(
                context,
                stream,
                device_index,
                device,
                block_sums_ptr,
                scanned_block_sums_ptr,
                num_blocks,
                depth + 1, // Increment recursion depth
            )?;
        }
    }

    // Step 3: Add scanned block sums as offsets
    let func_step3 = get_kernel_function(&module, "add_block_offsets_i32_step3")?;
    let grid = (num_blocks, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let mut builder = stream.launch_builder(&func_step3);
    builder.arg(&output_ptr);
    builder.arg(&scanned_block_sums_ptr);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }
        .map_err(|e| Error::Internal(format!("CUDA scan step 3 kernel launch failed: {:?}", e)))?;

    // Synchronize after step 3
    stream.synchronize().map_err(|e| {
        Error::Internal(format!("Failed to synchronize after scan step 3: {:?}", e))
    })?;

    // Step 4: Write total sum to output[n]
    let func_total = get_kernel_function(&module, "write_total_i32")?;
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let mut builder = stream.launch_builder(&func_total);
    builder.arg(&output_ptr);
    builder.arg(&scanned_block_sums_ptr);
    builder.arg(&n);
    builder.arg(&num_blocks);
    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan total write kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// I64 Exclusive Scan
// ============================================================================

/// Perform GPU-accelerated exclusive scan on I64 tensor.
///
/// Returns a tuple of:
/// - Output tensor with size n+1, where output[0] = 0 and output[i] = sum(input[0..i])
/// - Total sum (last element of output)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `device` - CUDA device
/// * `input` - Input tensor of I64 values (size n)
///
/// # Returns
///
/// `(output_tensor, total_sum)` where output has size n+1
///
/// # Safety
///
/// - `input` must be a valid `CudaRuntime` tensor of `DType::I64` on the device associated with
///   `context`. Passing a tensor with a different dtype returns an error.
/// - The stream must be from the same context and must not be destroyed while the kernel runs.
/// - A single scalar GPU-to-CPU transfer is performed at the end to read the total sum; this is
///   intentional and documented as acceptable for control-flow purposes.
pub unsafe fn exclusive_scan_i64_gpu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    input: &Tensor<CudaRuntime>,
) -> Result<(Tensor<CudaRuntime>, usize)> {
    let n = input.numel();

    if input.dtype() != DType::I64 {
        return Err(Error::Internal(format!(
            "exclusive_scan_i64_gpu expects I64 input, got {:?}",
            input.dtype()
        )));
    }

    // Allocate output tensor with size n+1
    let output = Tensor::<CudaRuntime>::try_zeros(&[n + 1], DType::I64, device)?;

    let input_ptr = input.ptr();
    let output_ptr = output.ptr();

    if n <= SCAN_BLOCK_SIZE as usize {
        // Small array: use single-block scan
        unsafe {
            launch_scan_single_block_i64(
                context,
                stream,
                device_index,
                input_ptr,
                output_ptr,
                n as u32,
            )?;
        }
    } else {
        // Large array: use multi-block scan (start at depth 0)
        unsafe {
            launch_scan_multi_block_i64(
                context,
                stream,
                device_index,
                device,
                input_ptr,
                output_ptr,
                n as u32,
                0, // Initial recursion depth
            )?;
        }
    }

    // Synchronize to ensure scan is complete before reading total
    stream
        .synchronize()
        .map_err(|e| Error::Internal(format!("Failed to synchronize after scan: {:?}", e)))?;

    // Read the total sum from output[n] — single scalar GPU→CPU read (acceptable control flow)
    let mut total_i64: i64 = 0;
    let offset_bytes = n * std::mem::size_of::<i64>();
    unsafe {
        cudarc::driver::sys::cuMemcpyDtoH_v2(
            &mut total_i64 as *mut i64 as *mut std::ffi::c_void,
            output.ptr() + offset_bytes as u64,
            std::mem::size_of::<i64>(),
        );
    }
    let total = total_i64 as usize;

    Ok((output, total))
}

/// Launch single-block exclusive scan kernel for i64
unsafe fn launch_scan_single_block_i64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_ptr: u64,
    output_ptr: u64,
    n: u32,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::SCAN_MODULE)?;
    let func = get_kernel_function(&module, "exclusive_scan_i64")?;

    // Single block with SCAN_BLOCK_SIZE threads
    let grid = (1, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&n);

    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan single-block i64 kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

/// Launch multi-block exclusive scan for i64 with recursive support for unlimited sizes.
///
/// # Arguments
/// * `depth` - Current recursion depth (0 at entry, incremented on each recursive call)
///
/// # Safety
/// Caller must ensure input_ptr and output_ptr point to valid device memory.
unsafe fn launch_scan_multi_block_i64(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    input_ptr: u64,
    output_ptr: u64,
    n: u32,
    depth: usize,
) -> Result<()> {
    // Check recursion depth to prevent stack overflow
    if depth >= MAX_SCAN_RECURSION_DEPTH {
        return Err(Error::Internal(format!(
            "Scan recursion depth {} exceeds maximum {}. \
             This indicates an algorithmic error or impossibly large input.",
            depth, MAX_SCAN_RECURSION_DEPTH
        )));
    }

    let module = get_or_load_module(context, device_index, kernel_names::SCAN_MODULE)?;

    // Calculate number of blocks needed
    let num_blocks = (n + SCAN_BLOCK_SIZE - 1) / SCAN_BLOCK_SIZE;

    // Allocate temporary buffer for block sums
    let block_sums = Tensor::<CudaRuntime>::try_zeros(&[num_blocks as usize], DType::I64, device)?;
    let block_sums_ptr = block_sums.ptr();

    // Step 1: Scan each block independently
    let func_step1 = get_kernel_function(&module, "scan_blocks_i64_step1")?;
    let grid = (num_blocks, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let mut builder = stream.launch_builder(&func_step1);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&block_sums_ptr);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan i64 step 1 kernel launch failed: {:?}",
            e
        ))
    })?;

    // Synchronize after step 1
    stream.synchronize().map_err(|e| {
        Error::Internal(format!(
            "Failed to synchronize after scan i64 step 1: {:?}",
            e
        ))
    })?;

    // Step 2: Recursively scan the block sums
    // Allocate buffer for scanned block sums (size num_blocks + 1)
    let scanned_block_sums =
        Tensor::<CudaRuntime>::try_zeros(&[num_blocks as usize + 1], DType::I64, device)?;
    let scanned_block_sums_ptr = scanned_block_sums.ptr();

    if num_blocks <= SCAN_BLOCK_SIZE {
        // Block sums fit in single block - use simple scan
        let func_scan = get_kernel_function(&module, "exclusive_scan_i64")?;
        let grid = (1, 1, 1);
        let block = (SCAN_BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let mut builder = stream.launch_builder(&func_scan);
        builder.arg(&block_sums_ptr);
        builder.arg(&scanned_block_sums_ptr);
        builder.arg(&num_blocks);
        unsafe { builder.launch(cfg) }.map_err(|e| {
            Error::Internal(format!(
                "CUDA scan i64 block sums kernel launch failed: {:?}",
                e
            ))
        })?;

        stream.synchronize().map_err(|e| {
            Error::Internal(format!(
                "Failed to synchronize after scan i64 step 2: {:?}",
                e
            ))
        })?;
    } else {
        // Block sums don't fit in single block - recursively scan them
        unsafe {
            launch_scan_multi_block_i64(
                context,
                stream,
                device_index,
                device,
                block_sums_ptr,
                scanned_block_sums_ptr,
                num_blocks,
                depth + 1, // Increment recursion depth
            )?;
        }
    }

    // Step 3: Add scanned block sums as offsets
    let func_step3 = get_kernel_function(&module, "add_block_offsets_i64_step3")?;
    let grid = (num_blocks, 1, 1);
    let block = (SCAN_BLOCK_SIZE, 1, 1);
    let cfg = launch_config(grid, block, 0);

    let mut builder = stream.launch_builder(&func_step3);
    builder.arg(&output_ptr);
    builder.arg(&scanned_block_sums_ptr);
    builder.arg(&n);
    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan i64 step 3 kernel launch failed: {:?}",
            e
        ))
    })?;

    // Synchronize after step 3
    stream.synchronize().map_err(|e| {
        Error::Internal(format!(
            "Failed to synchronize after scan i64 step 3: {:?}",
            e
        ))
    })?;

    // Step 4: Write total sum to output[n]
    let func_total = get_kernel_function(&module, "write_total_i64")?;
    let cfg = launch_config((1, 1, 1), (1, 1, 1), 0);
    let mut builder = stream.launch_builder(&func_total);
    builder.arg(&output_ptr);
    builder.arg(&scanned_block_sums_ptr);
    builder.arg(&n);
    builder.arg(&num_blocks);
    unsafe { builder.launch(cfg) }.map_err(|e| {
        Error::Internal(format!(
            "CUDA scan i64 total write kernel launch failed: {:?}",
            e
        ))
    })?;

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_small() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: [3, 1, 4, 1, 5]
        let input =
            Tensor::<CudaRuntime>::try_from_slice(&[3i32, 1, 4, 1, 5], &[5], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("scan failed");

        // Expected output: [0, 3, 4, 8, 9, 14]
        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec, vec![0, 3, 4, 8, 9, 14]);
        assert_eq!(total, 14);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_large() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 1024 elements, all ones
        let input_vec = vec![1i32; 1024];
        let input = Tensor::<CudaRuntime>::try_from_slice(&input_vec, &[1024], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("scan failed");

        // Expected: [0, 1, 2, ..., 1023, 1024]
        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec.len(), 1025);
        assert_eq!(output_vec[0], 0);
        assert_eq!(output_vec[1024], 1024);
        assert_eq!(total, 1024);

        // Verify it's actually a scan
        for i in 0..1024 {
            assert_eq!(output_vec[i], i as i32);
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_zeros() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let input = Tensor::<CudaRuntime>::try_from_slice(&[0i32, 0, 0, 0], &[4], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("scan failed");

        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec, vec![0, 0, 0, 0, 0]);
        assert_eq!(total, 0);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_single_element() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let input = Tensor::<CudaRuntime>::try_from_slice(&[42i32], &[1], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("scan failed");

        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec, vec![0, 42]);
        assert_eq!(total, 42);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_very_large() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        // Test with 500,000 elements (requires recursive multi-level scan)
        // This exceeds 262,144 = 512^2 which was the previous limit
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let n = 500_000;
        let input_vec = vec![1i32; n];
        let input = Tensor::<CudaRuntime>::try_from_slice(&input_vec, &[n], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("large scan failed");

        // Expected: [0, 1, 2, ..., n-1, n]
        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec.len(), n + 1);
        assert_eq!(output_vec[0], 0);
        assert_eq!(output_vec[n], n as i32);
        assert_eq!(total, n);

        // Verify a few sample values
        assert_eq!(output_vec[1000], 1000);
        assert_eq!(output_vec[100_000], 100_000);
        assert_eq!(output_vec[250_000], 250_000);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_boundary_size() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        // Test at the boundary of single-level multi-block (512 * 512 = 262,144)
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Just above the boundary to trigger recursive path
        let n = 262_145;
        let input_vec = vec![1i32; n];
        let input = Tensor::<CudaRuntime>::try_from_slice(&input_vec, &[n], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i32_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("boundary scan failed");

        let output_vec: Vec<i32> = output.to_vec();
        assert_eq!(output_vec.len(), n + 1);
        assert_eq!(output_vec[0], 0);
        assert_eq!(output_vec[n], n as i32);
        assert_eq!(total, n);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_exclusive_scan_i64_very_large() {
        if !crate::runtime::cuda::is_cuda_available() {
            return;
        }
        // Test i64 with large values that would overflow i32
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let n = 100_000;
        // Use large values to test i64 arithmetic
        let input_vec = vec![50_000i64; n];
        let input = Tensor::<CudaRuntime>::try_from_slice(&input_vec, &[n], &device).unwrap();

        let (output, total) = unsafe {
            exclusive_scan_i64_gpu(
                &client.context,
                &client.stream,
                device.index,
                &device,
                &input,
            )
        }
        .expect("i64 scan failed");

        // Total should be 50_000 * 100_000 = 5_000_000_000 (exceeds i32 max)
        let expected_total: i64 = 50_000 * 100_000;
        assert_eq!(total, expected_total as usize);

        let output_vec: Vec<i64> = output.to_vec();
        assert_eq!(output_vec[0], 0);
        assert_eq!(output_vec[n], expected_total);
    }
}
