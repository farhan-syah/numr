//! Complex number operation CUDA kernel launchers
//!
//! Provides launchers for complex number operations:
//! - conj: Complex conjugate
//! - real: Extract real part
//! - imag: Extract imaginary part
//! - angle: Compute phase angle

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Module name for complex operations
const COMPLEX_MODULE: &str = "complex";

/// Launch complex conjugate kernel.
///
/// Computes: output[i] = conj(input[i])
/// For complex numbers a + bi, returns a - bi
///
/// # Supported Dtypes
/// - Complex64: float2 → float2
/// - Complex128: double2 → double2
///
/// # Safety
/// - All pointers must be valid device memory
/// - Input and output tensors must have at least `numel` elements of appropriate dtype
pub unsafe fn launch_conj(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match dtype {
        DType::Complex64 => "conj_complex64",
        DType::Complex128 => "conj_complex128",
        _ => return Err(Error::UnsupportedDType { dtype, op: "conj" }),
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch real part extraction kernel.
///
/// Extracts real component from complex numbers.
///
/// # Output Dtypes
/// - Complex64 input → F32 output
/// - Complex128 input → F64 output
///
/// # Safety
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` complex elements
/// - Output tensor must have at least `numel` float elements
pub unsafe fn launch_real(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match input_dtype {
        DType::Complex64 => "real_complex64",
        DType::Complex128 => "real_complex128",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "real",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch imaginary part extraction kernel.
///
/// Extracts imaginary component from complex numbers.
///
/// # Output Dtypes
/// - Complex64 input → F32 output
/// - Complex128 input → F64 output
///
/// # Safety
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` complex elements
/// - Output tensor must have at least `numel` float elements
pub unsafe fn launch_imag(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match input_dtype {
        DType::Complex64 => "imag_complex64",
        DType::Complex128 => "imag_complex128",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "imag",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch phase angle computation kernel.
///
/// Computes phase angle of complex numbers using atan2(imag, real).
/// Returns angles in radians, range [-π, π].
///
/// # Output Dtypes
/// - Complex64 input → F32 output (angles in radians)
/// - Complex128 input → F64 output (angles in radians)
///
/// # Safety
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` complex elements
/// - Output tensor must have at least `numel` float elements
pub unsafe fn launch_angle(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match input_dtype {
        DType::Complex64 => "angle_complex64",
        DType::Complex128 => "angle_complex128",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "angle",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch angle kernel for real types.
///
/// Computes phase angle for real numbers: angle(x) = 0 if x >= 0, π if x < 0
///
/// # Supported Dtypes
/// - F32 → F32 output
/// - F64 → F64 output
///
/// # Safety
/// - All pointers must be valid device memory
/// - Input and output tensors must have at least `numel` elements
pub unsafe fn launch_angle_real(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match dtype {
        DType::F32 => "angle_real_f32",
        DType::F64 => "angle_real_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "angle_real",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch from_real_imag kernel.
///
/// Constructs complex tensor from separate real and imaginary arrays.
///
/// # Output Dtypes
/// - F32 real, F32 imag → Complex64 output
/// - F64 real, F64 imag → Complex128 output
///
/// # Safety
/// - All pointers must be valid device memory
/// - Real and imag tensors must have at least `numel` elements
/// - Output tensor must have at least `numel` complex elements
pub unsafe fn launch_from_real_imag(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    input_dtype: DType,
    real_ptr: u64,
    imag_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match input_dtype {
        DType::F32 => "from_real_imag_f32",
        DType::F64 => "from_real_imag_f64",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: input_dtype,
                op: "from_real_imag",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&real_ptr);
        builder.arg(&imag_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch complex × real multiplication kernel.
///
/// Computes (a + bi) * r = ar + br*i element-wise.
///
/// # Supported Dtypes
/// - Complex64 × F32 → Complex64
/// - Complex128 × F64 → Complex128
///
/// # Safety
/// - All pointers must be valid device memory
/// - Complex tensor must have at least `numel` complex elements
/// - Real tensor must have at least `numel` float elements
/// - Output tensor must have at least `numel` complex elements
pub unsafe fn launch_complex_mul_real(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    complex_dtype: DType,
    complex_ptr: u64,
    real_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match complex_dtype {
        DType::Complex64 => "complex64_mul_real",
        DType::Complex128 => "complex128_mul_real",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: complex_dtype,
                op: "complex_mul_real",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&complex_ptr);
        builder.arg(&real_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}

/// Launch complex / real division kernel.
///
/// Computes (a + bi) / r = (a/r) + (b/r)*i element-wise.
///
/// # Supported Dtypes
/// - Complex64 / F32 → Complex64
/// - Complex128 / F64 → Complex128
///
/// # Safety
/// - All pointers must be valid device memory
/// - Complex tensor must have at least `numel` complex elements
/// - Real tensor must have at least `numel` float elements
/// - Output tensor must have at least `numel` complex elements
pub unsafe fn launch_complex_div_real(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    complex_dtype: DType,
    complex_ptr: u64,
    real_ptr: u64,
    out_ptr: u64,
    numel: usize,
) -> Result<()> {
    let kernel_name = match complex_dtype {
        DType::Complex64 => "complex64_div_real",
        DType::Complex128 => "complex128_div_real",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: complex_dtype,
                op: "complex_div_real",
            });
        }
    };

    unsafe {
        let module = get_or_load_module(context, device_index, COMPLEX_MODULE)?;
        let func = get_kernel_function(&module, kernel_name)?;

        let grid = elementwise_launch_config(numel)?;
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&complex_ptr);
        builder.arg(&real_ptr);
        builder.arg(&out_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA {} kernel launch failed: {:?}",
                kernel_name, e
            ))
        })?;

        Ok(())
    }
}
