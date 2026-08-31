//! Generic CUDA launcher helpers for special mathematical functions
//!
//! Provides reusable launcher infrastructure for unary, binary, and ternary
//! special function kernels.

use super::super::loader::{
    elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};
use cudarc::driver::{CudaContext, CudaStream, PushKernelArg};
use std::sync::Arc;

pub(crate) const SPECIAL_MODULE: &str = "special";

/// Get kernel name with dtype suffix for special functions
pub(crate) fn special_kernel_name(
    base: &str,
    dtype: DType,
    op_name: &'static str,
) -> Result<String> {
    let suffix = match dtype {
        DType::F32 => "f32",
        DType::F64 => "f64",
        DType::F16 => "f16",
        DType::BF16 => "bf16",
        DType::FP8E4M3 => "fp8_e4m3",
        DType::FP8E5M2 => "fp8_e5m2",
        _ => {
            return Err(Error::UnsupportedDType { dtype, op: op_name });
        }
    };
    Ok(format!("{}_{}", base, suffix))
}

/// Generic launcher for unary special functions (1 input -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for binary special functions (2 inputs -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_binary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for ternary special functions (3 inputs -> 1 output)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_ternary_special(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a_ptr: u64,
    b_ptr: u64,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

// ============================================================================
// Extended Launchers with Parameters
// ============================================================================

/// Generic launcher for unary special functions with one i32 parameter
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special_with_int(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    n_param: i32,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&n_param);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for unary special functions with two i32 parameters
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special_with_two_ints(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    n_param: i32,
    m_param: i32,
    x_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&n_param);
        builder.arg(&m_param);
        builder.arg(&x_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for binary special functions with two i32 parameters (sph_harm)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_binary_special_with_two_ints(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    n_param: i32,
    m_param: i32,
    a_ptr: u64,
    b_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&n_param);
        builder.arg(&m_param);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for unary special functions with two f64 parameters (hyp1f1)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special_with_2f64(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a: f64,
    b: f64,
    z_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    // Convert params to appropriate type based on dtype
    let (a_f32, b_f32) = (a as f32, b as f32);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        if dtype == DType::F64 {
            builder.arg(&a);
            builder.arg(&b);
        } else {
            builder.arg(&a_f32);
            builder.arg(&b_f32);
        }
        builder.arg(&z_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}

/// Generic launcher for unary special functions with three f64 parameters (hyp2f1)
///
/// # Safety
/// Pointers must be valid GPU memory of correct size.
pub(crate) unsafe fn launch_unary_special_with_3f64(
    ctx: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kernel_base: &str,
    op_name: &'static str,
    a: f64,
    b: f64,
    c: f64,
    z_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = special_kernel_name(kernel_base, dtype, op_name)?;
    let module = get_or_load_module(ctx, device_index, SPECIAL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_name)?;

    let grid = elementwise_launch_config(numel)?;
    let cfg = launch_config(grid, (256, 1, 1), 0);
    let n = numel as u32;

    // Convert params to appropriate type based on dtype
    let (a_f32, b_f32, c_f32) = (a as f32, b as f32, c as f32);

    unsafe {
        let mut builder = stream.launch_builder(&func);
        if dtype == DType::F64 {
            builder.arg(&a);
            builder.arg(&b);
            builder.arg(&c);
        } else {
            builder.arg(&a_f32);
            builder.arg(&b_f32);
            builder.arg(&c_f32);
        }
        builder.arg(&z_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;
    }

    Ok(())
}
