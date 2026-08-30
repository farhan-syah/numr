//! Binary operation CUDA kernel launchers
//!
//! Provides launchers for element-wise binary operations (add, sub, mul, div, etc.)
//! on two tensors of the same shape.
//!
//! Also supports broadcasting operations using strided access patterns.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_binary_kernel, launch_config,
};
use super::broadcast_strides::{
    MAX_BROADCAST_DIMS, compute_broadcast_strides, compute_magic_divisor,
    detect_fast_trailing_broadcast,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::CudaDevice;

/// Launch a binary operation kernel.
///
/// Performs element-wise operation: `output[i] = op(a[i], b[i])`
///
/// # Supported Operations
///
/// - `add`: Element-wise addition
/// - `sub`: Element-wise subtraction
/// - `mul`: Element-wise multiplication
/// - `div`: Element-wise division
/// - `pow`: Element-wise power
/// - `max`: Element-wise maximum
/// - `min`: Element-wise minimum
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
/// - `a` and `b` must have the same dtype
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "add", "mul")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_binary_kernel(
            context,
            stream,
            device_index,
            kernel_names::BINARY_MODULE,
            op,
            dtype,
            a_ptr,
            b_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch a logical_and kernel.
///
/// Performs element-wise logical AND: `output[i] = a[i] && b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_and_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_and_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_and kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_or kernel.
///
/// Performs element-wise logical OR: `output[i] = a[i] || b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_or_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_or_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_or kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a logical_xor kernel.
///
/// Performs element-wise logical XOR: `output[i] = a[i] ^ b[i]`
/// All tensors are U8 (boolean: 0 = false, non-zero = true).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` U8 elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `a_ptr` - Device pointer to first input tensor (U8)
/// * `b_ptr` - Device pointer to second input tensor (U8)
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `numel` - Number of elements
pub unsafe fn launch_logical_xor_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
        let func_name = "logical_xor_u8";
        let func = get_kernel_function(&module, func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA logical_xor kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch a broadcast binary operation kernel.
///
/// Performs element-wise operation with broadcasting:
/// `output[i] = op(a[broadcast_idx], b[broadcast_idx])`
///
/// # CUDA Graph Compatibility
///
/// This function uses the `*_broadcast_*_inline` kernel variants that accept
/// strides and shape as individual scalar u32 arguments baked into the
/// kernel-parameter block.  Unlike the pointer-based variants, the inline
/// kernels do NOT trigger H2D memcpy nodes during CUDA graph capture, so the
/// graph's kernel nodes never contain stale host-side pointers.
///
/// # Supported Operations
///
/// - `add`: Element-wise addition
/// - `sub`: Element-wise subtraction
/// - `mul`: Element-wise multiplication
/// - `div`: Element-wise division
/// - `pow`: Element-wise power
/// - `max`: Element-wise maximum
/// - `min`: Element-wise minimum
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - `out_shape.len()` must be ≤ `MAX_BROADCAST_DIMS` (= 8)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `op` - Operation name (e.g., "add", "mul")
/// * `dtype` - Data type of the tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `a_shape` - Shape of tensor a
/// * `b_shape` - Shape of tensor b
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_broadcast_binary_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    _device: &CudaDevice,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(());
    }

    let ndim = out_shape.len();
    if ndim > MAX_BROADCAST_DIMS {
        return Err(Error::Internal(format!(
            "launch_broadcast_binary_op: ndim={ndim} exceeds MAX_BROADCAST_DIMS={MAX_BROADCAST_DIMS}"
        )));
    }

    let module = get_or_load_module(context, device_index, kernel_names::BINARY_MODULE)?;
    let dtype_str = kernel_name("", dtype).trim_start_matches('_').to_owned();
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    // ----------------------------------------------------------------
    // FAST PATH: contiguous trailing-broadcast
    //
    // When a is contiguous and has the same shape as out, and b is a
    // contiguous tensor that just repeats along the leading dimensions
    // (b_index = idx % b_numel), we dispatch a specialized 3-arg kernel
    // that avoids multi-dim coordinate decomposition entirely.
    // ----------------------------------------------------------------
    if let Some(b_numel) = detect_fast_trailing_broadcast(a_shape, b_shape, out_shape) {
        let func_name = format!("{}_broadcast_fast_trailing_{}", op, dtype_str);
        if let Ok(func) = get_kernel_function(&module, &func_name) {
            let (b_magic, b_shift) = compute_magic_divisor(b_numel as u32);
            let b_numel_u32 = b_numel as u32;
            unsafe {
                let mut builder = stream.launch_builder(&func);
                builder.arg(&a_ptr);
                builder.arg(&b_ptr);
                builder.arg(&out_ptr);
                builder.arg(&b_magic);
                builder.arg(&b_shift);
                builder.arg(&b_numel_u32);
                builder.arg(&n);
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA broadcast fast-trailing kernel '{}' launch failed: {:?}",
                        func_name, e
                    ))
                })?;
            }
            return Ok(());
        }
        // If the fast-trailing kernel is missing for some reason, fall through to general path.
    }

    // ----------------------------------------------------------------
    // GENERAL PATH: magic-number inline broadcast
    //
    // Compute broadcast strides and magic-divisor constants for each
    // output dimension. Pass all 40 scalar args inline (CUDA-graph safe).
    // ----------------------------------------------------------------

    // Compute broadcast strides.
    let a_strides_vec = compute_broadcast_strides(a_shape, out_shape)?;
    let b_strides_vec = compute_broadcast_strides(b_shape, out_shape)?;
    let shape_vec: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Pack into fixed-size arrays (zero-padded to MAX_BROADCAST_DIMS).
    let mut a_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut b_strides = [0u32; MAX_BROADCAST_DIMS];
    let mut shape = [0u32; MAX_BROADCAST_DIMS];
    let mut magic = [0u32; MAX_BROADCAST_DIMS];
    let mut pshift = [0u32; MAX_BROADCAST_DIMS];
    for i in 0..ndim {
        a_strides[i] = a_strides_vec[i];
        b_strides[i] = b_strides_vec[i];
        shape[i] = shape_vec[i];
        let (m, s) = compute_magic_divisor(shape_vec[i]);
        magic[i] = m;
        pshift[i] = s;
    }
    // Zero-padded dims: shape=0 means magic=0, shift=0. The kernel skips them via ndim.

    let func_name = format!("{}_broadcast_{}_inline", op, dtype_str);
    let func = get_kernel_function(&module, &func_name)?;
    let ndim_u32 = ndim as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        // a_strides[0..7]
        builder.arg(&a_strides[0]);
        builder.arg(&a_strides[1]);
        builder.arg(&a_strides[2]);
        builder.arg(&a_strides[3]);
        builder.arg(&a_strides[4]);
        builder.arg(&a_strides[5]);
        builder.arg(&a_strides[6]);
        builder.arg(&a_strides[7]);
        // b_strides[0..7]
        builder.arg(&b_strides[0]);
        builder.arg(&b_strides[1]);
        builder.arg(&b_strides[2]);
        builder.arg(&b_strides[3]);
        builder.arg(&b_strides[4]);
        builder.arg(&b_strides[5]);
        builder.arg(&b_strides[6]);
        builder.arg(&b_strides[7]);
        // shape[0..7]
        builder.arg(&shape[0]);
        builder.arg(&shape[1]);
        builder.arg(&shape[2]);
        builder.arg(&shape[3]);
        builder.arg(&shape[4]);
        builder.arg(&shape[5]);
        builder.arg(&shape[6]);
        builder.arg(&shape[7]);
        // magic[0..7]
        builder.arg(&magic[0]);
        builder.arg(&magic[1]);
        builder.arg(&magic[2]);
        builder.arg(&magic[3]);
        builder.arg(&magic[4]);
        builder.arg(&magic[5]);
        builder.arg(&magic[6]);
        builder.arg(&magic[7]);
        // pshift[0..7]
        builder.arg(&pshift[0]);
        builder.arg(&pshift[1]);
        builder.arg(&pshift[2]);
        builder.arg(&pshift[3]);
        builder.arg(&pshift[4]);
        builder.arg(&pshift[5]);
        builder.arg(&pshift[6]);
        builder.arg(&pshift[7]);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA broadcast binary kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    Ok(())
}
