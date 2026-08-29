//! Fused elementwise CUDA kernel launchers
//!
//! - fused_mul_add: out = a * b + c
//! - fused_add_mul: out = (a + b) * c
//! - fused_mul_add_scalar: out = a * scale + bias

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, kernel_name,
    launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

const MODULE: &str = "fused_elementwise";

/// Launch fused_mul_add: out = a * b + c
///
/// # Safety
/// All pointers must be valid device memory with at least `numel` elements.
pub unsafe fn launch_fused_mul_add(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_ternary_kernel(
            context,
            stream,
            device_index,
            "fused_mul_add",
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch fused_add_mul: out = (a + b) * c
///
/// # Safety
/// All pointers must be valid device memory with at least `numel` elements.
pub unsafe fn launch_fused_add_mul(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    unsafe {
        launch_ternary_kernel(
            context,
            stream,
            device_index,
            "fused_add_mul",
            dtype,
            a_ptr,
            b_ptr,
            c_ptr,
            output_ptr,
            numel,
        )
    }
}

/// Launch fused_mul_add_scalar: out = a * scale + bias
///
/// # Safety
/// All pointers must be valid device memory with at least `numel` elements.
pub unsafe fn launch_fused_mul_add_scalar(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    output_ptr: u64,
    numel: usize,
    scale: f64,
    bias: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, MODULE)?;
    let func_name = kernel_name("fused_mul_add_scalar", dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    let scale_f32 = scale as f32;
    let bias_f32 = bias as f32;

    // Each integer row's kernel takes scale and bias in its own element type.
    // The `as` cast saturates, matching `Element::from_f64` on CPU, so the
    // fused op and `add_scalar(mul_scalar(..))` see the same operands.
    let scale_i64 = scale as i64;
    let bias_i64 = bias as i64;
    let scale_i32 = scale as i32;
    let bias_i32 = bias as i32;
    let scale_i16 = scale as i16;
    let bias_i16 = bias as i16;
    let scale_i8 = scale as i8;
    let bias_i8 = bias as i8;
    let scale_u64 = scale as u64;
    let bias_u64 = bias as u64;
    let scale_u32 = scale as u32;
    let bias_u32 = bias as u32;
    let scale_u16 = scale as u16;
    let bias_u16 = bias as u16;
    let scale_u8 = scale as u8;
    let bias_u8 = bias as u8;

    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&a_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        match dtype {
            DType::F64 => {
                builder.arg(&scale);
                builder.arg(&bias);
            }
            DType::I64 => {
                builder.arg(&scale_i64);
                builder.arg(&bias_i64);
            }
            DType::I32 => {
                builder.arg(&scale_i32);
                builder.arg(&bias_i32);
            }
            DType::I16 => {
                builder.arg(&scale_i16);
                builder.arg(&bias_i16);
            }
            DType::I8 => {
                builder.arg(&scale_i8);
                builder.arg(&bias_i8);
            }
            DType::U64 => {
                builder.arg(&scale_u64);
                builder.arg(&bias_u64);
            }
            DType::U32 => {
                builder.arg(&scale_u32);
                builder.arg(&bias_u32);
            }
            DType::U16 => {
                builder.arg(&scale_u16);
                builder.arg(&bias_u16);
            }
            DType::U8 => {
                builder.arg(&scale_u8);
                builder.arg(&bias_u8);
            }
            _ => {
                builder.arg(&scale_f32);
                builder.arg(&bias_f32);
            }
        }

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA fused_mul_add_scalar kernel launch failed: {:?}",
                e
            ))
        })?;
    }

    Ok(())
}

/// Internal helper for ternary kernels (a, b, c -> out)
unsafe fn launch_ternary_kernel(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    op: &str,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, MODULE)?;
    let func_name = kernel_name(op, dtype);
    let func = get_kernel_function(&module, &func_name)?;

    let grid = elementwise_launch_config(numel);
    let block = (BLOCK_SIZE, 1, 1);
    let n = numel as u32;
    let cfg = launch_config(grid, block, 0);

    let mut builder = stream.launch_builder(&func);
    unsafe {
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&c_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder
            .launch(cfg)
            .map_err(|e| Error::Internal(format!("CUDA {} kernel launch failed: {:?}", op, e)))?;
    }

    Ok(())
}
