//! Ternary operation CUDA kernel launchers
//!
//! Provides launchers for ternary operations like where (conditional select).
//! where(cond, x, y) = cond ? x : y

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::binary::compute_broadcast_strides;
use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::{CudaDevice, CudaRuntime};
use crate::tensor::Tensor;

/// Launch a where (conditional select) kernel.
///
/// Performs element-wise conditional selection: `output[i] = cond[i] ? x[i] : y[i]`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
/// - Condition tensor must be U8 (boolean: 0 = false, non-zero = true)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor (U8)
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_where_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
        let func_name = kernel_name("where", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a broadcast where (conditional select) kernel.
///
/// Performs element-wise conditional selection with broadcasting support:
/// `output[i] = cond[cond_offset] ? x[x_offset] : y[y_offset]`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Shape arrays must be valid
/// - Condition tensor must be U8 (boolean: 0 = false, non-zero = true)
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `device` - CUDA device for tensor allocation
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor (U8)
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `cond_shape` - Shape of condition tensor
/// * `x_shape` - Shape of x tensor
/// * `y_shape` - Shape of y tensor
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_where_broadcast_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    cond_shape: &[usize],
    x_shape: &[usize],
    y_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(());
    }

    let ndim = out_shape.len();

    // Compute broadcast strides
    let cond_strides = compute_broadcast_strides(cond_shape, out_shape);
    let x_strides = compute_broadcast_strides(x_shape, out_shape);
    let y_strides = compute_broadcast_strides(y_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Allocate device memory for strides and shape using Tensor
    let cond_strides_tensor = Tensor::<CudaRuntime>::from_slice(&cond_strides, &[ndim], device)?;
    let x_strides_tensor = Tensor::<CudaRuntime>::from_slice(&x_strides, &[ndim], device)?;
    let y_strides_tensor = Tensor::<CudaRuntime>::from_slice(&y_strides, &[ndim], device)?;
    let shape_tensor = Tensor::<CudaRuntime>::from_slice(&shape_u32, &[ndim], device)?;

    // Get device pointers
    let cond_strides_ptr = cond_strides_tensor.ptr();
    let x_strides_ptr = x_strides_tensor.ptr();
    let y_strides_ptr = y_strides_tensor.ptr();
    let shape_ptr = shape_tensor.ptr();

    // Get kernel function
    let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
    let func_name = format!("where_broadcast_{}", super::loader::dtype_suffix(dtype));
    let func = get_kernel_function(&module, &func_name)?;

    // Launch kernel
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let ndim_u32 = ndim as u32;

    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&cond_strides_ptr);
        builder.arg(&x_strides_ptr);
        builder.arg(&y_strides_ptr);
        builder.arg(&shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where_broadcast kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    // No sync needed: temporary GPU allocations freed via cuMemFreeAsync (stream-ordered).

    Ok(())
}

/// Launch a where (conditional select) kernel with generic condition type.
///
/// Performs element-wise conditional selection: `output[i] = cond[i] != 0 ? x[i] : y[i]`
/// Supports non-U8 condition types (F32, F64, I32, I64, U32).
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - All tensors must have at least `numel` elements
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `cond_dtype` - Data type of condition tensor
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_where_generic_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    cond_dtype: DType,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    // Build kernel name: where_cond_{cond_dtype}_{out_dtype}
    let cond_suffix = super::loader::dtype_suffix(cond_dtype);
    let out_suffix = super::loader::dtype_suffix(dtype);
    let func_name = format!("where_cond_{}_{}", cond_suffix, out_suffix);

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
        let func =
            get_kernel_function(&module, &func_name).map_err(|_| Error::UnsupportedDType {
                dtype: cond_dtype,
                op: "where_cond (condition dtype)",
            })?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where_cond kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch a broadcast where kernel with generic condition type.
///
/// Performs conditional selection with broadcasting support for non-U8 conditions.
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
/// * `cond_dtype` - Data type of condition tensor
/// * `dtype` - Data type of x, y, and output tensors
/// * `cond_ptr` - Device pointer to condition tensor
/// * `x_ptr` - Device pointer to "true" values tensor
/// * `y_ptr` - Device pointer to "false" values tensor
/// * `out_ptr` - Device pointer to output tensor
/// * `cond_shape` - Shape of condition tensor
/// * `x_shape` - Shape of x tensor
/// * `y_shape` - Shape of y tensor
/// * `out_shape` - Shape of output tensor (broadcast result)
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_where_broadcast_generic_op(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    device: &CudaDevice,
    cond_dtype: DType,
    dtype: DType,
    cond_ptr: u64,
    x_ptr: u64,
    y_ptr: u64,
    out_ptr: u64,
    cond_shape: &[usize],
    x_shape: &[usize],
    y_shape: &[usize],
    out_shape: &[usize],
) -> Result<()> {
    let numel: usize = out_shape.iter().product();
    if numel == 0 {
        return Ok(());
    }

    let ndim = out_shape.len();

    // Compute broadcast strides
    let cond_strides = compute_broadcast_strides(cond_shape, out_shape);
    let x_strides = compute_broadcast_strides(x_shape, out_shape);
    let y_strides = compute_broadcast_strides(y_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&x| x as u32).collect();

    // Allocate device memory for strides and shape using Tensor
    let cond_strides_tensor = Tensor::<CudaRuntime>::from_slice(&cond_strides, &[ndim], device)?;
    let x_strides_tensor = Tensor::<CudaRuntime>::from_slice(&x_strides, &[ndim], device)?;
    let y_strides_tensor = Tensor::<CudaRuntime>::from_slice(&y_strides, &[ndim], device)?;
    let shape_tensor = Tensor::<CudaRuntime>::from_slice(&shape_u32, &[ndim], device)?;

    // Get device pointers
    let cond_strides_ptr = cond_strides_tensor.ptr();
    let x_strides_ptr = x_strides_tensor.ptr();
    let y_strides_ptr = y_strides_tensor.ptr();
    let shape_ptr = shape_tensor.ptr();

    // Build kernel name: where_broadcast_cond_{cond_dtype}_{out_dtype}
    let cond_suffix = super::loader::dtype_suffix(cond_dtype);
    let out_suffix = super::loader::dtype_suffix(dtype);
    let func_name = format!("where_broadcast_cond_{}_{}", cond_suffix, out_suffix);

    // Get kernel function
    let module = get_or_load_module(context, device_index, kernel_names::TERNARY_MODULE)?;
    let func = get_kernel_function(&module, &func_name).map_err(|_| Error::UnsupportedDType {
        dtype: cond_dtype,
        op: "where_cond broadcast (condition dtype)",
    })?;

    // Launch kernel
    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let ndim_u32 = ndim as u32;

    let cfg = launch_config(grid, block, 0);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&cond_ptr);
        builder.arg(&x_ptr);
        builder.arg(&y_ptr);
        builder.arg(&out_ptr);
        builder.arg(&cond_strides_ptr);
        builder.arg(&x_strides_ptr);
        builder.arg(&y_strides_ptr);
        builder.arg(&shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA where_broadcast_cond kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;
    }

    // No sync needed: temporary GPU allocations freed via cuMemFreeAsync (stream-ordered).

    Ok(())
}
