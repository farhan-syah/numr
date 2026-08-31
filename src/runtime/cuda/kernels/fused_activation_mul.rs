//! Fused activation-mul CUDA kernel launchers
//!
//! Forward: output = activation(a) * b
//! Backward: d_a = grad * b * activation'(a), d_b = grad * activation(a)

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

const FUSED_ACTIVATION_MUL_MODULE: &str = "fused_activation_mul";
const FUSED_ACTIVATION_MUL_BWD_MODULE: &str = "fused_activation_mul_bwd";

/// Launch a fused activation-mul forward kernel.
///
/// Computes: `output[i] = activation(a[i]) * b[i]`
///
/// # Safety
///
/// All pointers must be valid device memory with at least `numel` elements.
unsafe fn launch_fused_activation_mul_fwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FUSED_ACTIVATION_MUL_MODULE)?;
    let func_name = kernel_name(op, dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA {} kernel launch failed: {:?}", op, e)))?;
    }

    Ok(())
}

/// Launch a fused activation-mul backward kernel.
///
/// Computes: `d_b[i] = grad[i] * activation(a[i])`, `d_a[i] = grad[i] * b[i] * activation'(a[i])`
///
/// # Safety
///
/// All pointers must be valid device memory with at least `numel` elements.
unsafe fn launch_fused_activation_mul_bwd(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    grad_ptr: u64,
    a_ptr: u64,
    b_ptr: u64,
    d_a_ptr: u64,
    d_b_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FUSED_ACTIVATION_MUL_BWD_MODULE)?;
    let func_name = kernel_name(op, dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel)?;
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;

    let cfg = launch_config(grid, block, 0);
    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&grad_ptr);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&d_a_ptr);
        builder.arg(&d_b_ptr);
        builder.arg(&n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA {} kernel launch failed: {:?}", op, e)))?;
    }

    Ok(())
}

// ============================================================================
// Public forward launchers
// ============================================================================

macro_rules! fused_activation_mul_fwd {
    ($($(#[doc = $doc:expr])* $name:ident => $op:expr),+ $(,)?) => {
        $(
            $(#[doc = $doc])*
            ///
            /// # Safety
            ///
            /// All pointers must be valid device memory with at least `numel` elements.
            pub unsafe fn $name(
                context: &Arc<CudaContext>,
                stream: &CudaStream,
                device_index: usize,
                dtype: DType,
                a_ptr: u64,
                b_ptr: u64,
                output_ptr: u64,
                numel: usize,
            ) -> Result<()> {
                unsafe {
                    launch_fused_activation_mul_fwd(
                        context, stream, device_index, $op, dtype, a_ptr, b_ptr, output_ptr, numel,
                    )
                }
            }
        )+
    };
}

fused_activation_mul_fwd! {
    /// Launch fused silu_mul: output = silu(a) * b
    launch_silu_mul => "silu_mul",
    /// Launch fused gelu_mul: output = gelu(a) * b
    launch_gelu_mul => "gelu_mul",
    /// Launch fused relu_mul: output = relu(a) * b
    launch_relu_mul => "relu_mul",
    /// Launch fused sigmoid_mul: output = sigmoid(a) * b
    launch_sigmoid_mul => "sigmoid_mul",
}

// ============================================================================
// Public backward launchers
// ============================================================================

macro_rules! fused_activation_mul_bwd {
    ($($(#[doc = $doc:expr])* $name:ident => $op:expr),+ $(,)?) => {
        $(
            $(#[doc = $doc])*
            ///
            /// # Safety
            ///
            /// All pointers must be valid device memory with at least `numel` elements.
            pub unsafe fn $name(
                context: &Arc<CudaContext>,
                stream: &CudaStream,
                device_index: usize,
                dtype: DType,
                grad_ptr: u64,
                a_ptr: u64,
                b_ptr: u64,
                d_a_ptr: u64,
                d_b_ptr: u64,
                numel: usize,
            ) -> Result<()> {
                unsafe {
                    launch_fused_activation_mul_bwd(
                        context, stream, device_index, $op, dtype, grad_ptr, a_ptr, b_ptr,
                        d_a_ptr, d_b_ptr, numel,
                    )
                }
            }
        )+
    };
}

fused_activation_mul_bwd! {
    /// Launch fused silu_mul backward
    launch_silu_mul_bwd => "silu_mul_bwd",
    /// Launch fused gelu_mul backward
    launch_gelu_mul_bwd => "gelu_mul_bwd",
    /// Launch fused relu_mul backward
    launch_relu_mul_bwd => "relu_mul_bwd",
    /// Launch fused sigmoid_mul backward
    launch_sigmoid_mul_bwd => "sigmoid_mul_bwd",
}
