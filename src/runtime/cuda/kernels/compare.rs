//! Comparison CUDA kernel launchers
//!
//! Provides launchers for element-wise comparison operations (eq, ne, lt, le, gt, ge)
//! on two tensors of the same shape.
//!
//! Also supports broadcasting operations using strided access patterns.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::binary::compute_broadcast_strides;
use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_binary_kernel, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use crate::tensor::Tensor;

/// Launch a comparison operation kernel.
///
/// Performs element-wise comparison: `output[i] = a[i] op b[i]`
///
/// Output is the same dtype as input (1.0/0.0 for floats, 1/0 for ints).
/// This is intentional - it allows using comparison results directly
/// in arithmetic operations (e.g., `mask * tensor`) without dtype conversion.
///
/// # Supported Operations
///
/// - `eq`: Equal (==)
/// - `ne`: Not equal (!=)
/// - `lt`: Less than (<)
/// - `le`: Less than or equal (<=)
/// - `gt`: Greater than (>)
/// - `ge`: Greater than or equal (>=)
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
/// * `op` - Comparison operation name (e.g., "eq", "lt")
/// * `dtype` - Data type of the input tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_compare_op(
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
            kernel_names::COMPARE_MODULE,
            op,
            dtype,
            a_ptr,
            b_ptr,
            out_ptr,
            numel,
        )
    }
}

/// Launch a broadcast comparison operation kernel.
///
/// Performs element-wise comparison with broadcasting: `output[i] = a[broadcast_idx] op b[broadcast_idx]`
/// Output is U8 (boolean).
///
/// # Supported Operations
///
/// - `eq`: Equal (==)
/// - `ne`: Not equal (!=)
/// - `lt`: Less than (<)
/// - `le`: Less than or equal (<=)
/// - `gt`: Greater than (>)
/// - `ge`: Greater than or equal (>=)
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Shape arrays must be valid
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `device` - CUDA device for tensor allocation
/// * `op` - Comparison operation name (e.g., "eq", "lt")
/// * `dtype` - Data type of the input tensors
/// * `a_ptr` - Device pointer to first input tensor
/// * `b_ptr` - Device pointer to second input tensor
/// * `out_ptr` - Device pointer to output tensor (U8)
/// * `a_shape` - Shape of tensor a
/// * `b_shape` - Shape of tensor b
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_broadcast_compare_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
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

    // Compute broadcast strides
    let a_strides = compute_broadcast_strides(a_shape, out_shape);
    let b_strides = compute_broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Allocate device memory for strides and shape using Tensor
    let a_strides_tensor = Tensor::<CudaRuntime>::try_from_slice(&a_strides, &[ndim], device)?;
    let b_strides_tensor = Tensor::<CudaRuntime>::try_from_slice(&b_strides, &[ndim], device)?;
    let shape_tensor = Tensor::<CudaRuntime>::try_from_slice(&shape_u32, &[ndim], device)?;

    // Get device pointers
    let a_strides_ptr = a_strides_tensor.ptr();
    let b_strides_ptr = b_strides_tensor.ptr();
    let shape_ptr = shape_tensor.ptr();

    // Get kernel function
    let module = get_or_load_module(context, device_index, kernel_names::COMPARE_MODULE)?;
    let func_name = format!(
        "{}_broadcast_{}",
        op,
        kernel_name("", dtype).trim_start_matches('_')
    );
    let func = get_kernel_function(&module, &func_name)?;

    // Launch kernel
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let ndim_u32 = ndim as u32;

    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&a_strides_ptr);
        builder.arg(&b_strides_ptr);
        builder.arg(&shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA broadcast compare kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    // No sync needed: temporary GPU allocations freed via cuMemFreeAsync (stream-ordered).

    Ok(())
}
