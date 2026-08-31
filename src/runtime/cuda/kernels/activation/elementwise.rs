//! Element-wise activation CUDA kernel launchers
//!
//! Kernel source: activation.cu

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    kernel_names, launch_config, launch_unary_kernel,
};

/// Launch a ReLU (Rectified Linear Unit) kernel.
///
/// Computes: `output[i] = max(0, input[i])`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_relu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::ACTIVATION_MODULE,
            "relu",
            dtype,
            input_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch a SiLU (Swish) kernel.
///
/// Computes: `output[i] = input[i] / (1 + exp(-input[i]))`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_silu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::ACTIVATION_MODULE,
            "silu",
            dtype,
            input_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch a GELU (Gaussian Error Linear Unit) kernel.
///
/// Computes: `output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_gelu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::ACTIVATION_MODULE,
            "gelu",
            dtype,
            input_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch a sigmoid kernel.
///
/// Computes: `output[i] = 1 / (1 + exp(-input[i]))`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_sigmoid(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_unary_kernel(
            context,
            stream,
            device_index,
            kernel_names::ACTIVATION_MODULE,
            "sigmoid",
            dtype,
            input_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch a Leaky ReLU kernel.
///
/// Computes: `output[i] = max(negative_slope * input[i], input[i])`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_leaky_relu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
    negative_slope: f32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::ACTIVATION_MODULE)?;
        let func_name = kernel_name("leaky_relu", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);
        builder.arg(&negative_slope);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA leaky_relu kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch an ELU (Exponential Linear Unit) kernel.
///
/// Computes: `output[i] = input[i] if input[i] > 0, else alpha * (exp(input[i]) - 1)`
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Tensors must have at least `numel` elements
pub unsafe fn launch_elu(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
    alpha: f32,
) -> Result<()> {
    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::ACTIVATION_MODULE)?;
        let func_name = kernel_name("elu", dtype);
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);
        builder.arg(&alpha);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA elu kernel launch failed: {:?}", e)))?;

        Ok(())
    }
}
