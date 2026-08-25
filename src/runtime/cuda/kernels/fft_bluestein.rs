//! Launchers for the Bluestein (chirp-z) FFT stages.
//!
//! These wrap `fft_bluestein.cu`, which supplies the pre/post work around the
//! power-of-two convolution that the Stockham kernels in [`super::fft`] already
//! provide. See that `.cu` file for the algorithm.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// Bluestein kernel module name.
pub const FFT_BLUESTEIN_MODULE: &str = "fft_bluestein";

/// Whether the premultiply reads real samples or complex ones.
///
/// `rfft` feeds real input; widening it to complex in a separate pass would
/// cost an extra full-size read and write, so the kernel does both at once.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BluesteinInput {
    /// Input buffer holds `Complex64`/`Complex128` elements.
    Complex,
    /// Input buffer holds `f32`/`f64` elements; imaginary part is taken as zero.
    Real,
}

/// Chirp-premultiply `input` into the zero-padded length-`m` convolution buffer.
///
/// Writes all `batch_size * m` elements of `out`, zeroing the tail past `n`, so
/// the caller does not need to pre-clear it.
///
/// # Safety
///
/// Caller must ensure all raw pointer arguments point to valid GPU memory on
/// `device_index`: `input` holds `batch_size * n` elements of the matching kind,
/// `chirp` holds `n` complex elements, and `out` holds `batch_size * m`.
pub unsafe fn launch_bluestein_premultiply(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    kind: BluesteinInput,
    input_ptr: u64,
    chirp_ptr: u64,
    out_ptr: u64,
    n: usize,
    m: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_BLUESTEIN_MODULE)?;
    let cfg = launch_config(
        elementwise_launch_config(batch_size * m),
        (BLOCK_SIZE, 1, 1),
        0,
    );

    let name = match (dtype, kind) {
        (DType::Complex64, BluesteinInput::Complex) => "bluestein_premultiply_c64",
        (DType::Complex128, BluesteinInput::Complex) => "bluestein_premultiply_c128",
        (DType::Complex64, BluesteinInput::Real) => "bluestein_premultiply_real_c64",
        (DType::Complex128, BluesteinInput::Real) => "bluestein_premultiply_real_c128",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bluestein_premultiply",
            });
        }
    };

    let func = get_kernel_function(&module, name)?;
    let mut builder = stream.launch_builder(&func);
    let (n_i32, m_i32, batch_i32) = (n as i32, m as i32, batch_size as i32);
    builder.arg(&input_ptr);
    builder.arg(&chirp_ptr);
    builder.arg(&out_ptr);
    builder.arg(&n_i32);
    builder.arg(&m_i32);
    builder.arg(&batch_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA bluestein premultiply launch failed: {:?}", e))
        })?;
    }
    Ok(())
}

/// Multiply `spectrum` in place by the length-`m` kernel spectrum.
///
/// The kernel spectrum depends only on `(n, direction)`, so one table serves the
/// whole batch.
///
/// # Safety
///
/// Caller must ensure `spectrum` holds `batch_size * m` complex elements and
/// `kernel_spectrum` holds `m`, both on `device_index`.
pub unsafe fn launch_bluestein_pointwise_mul(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    spectrum_ptr: u64,
    kernel_spectrum_ptr: u64,
    m: usize,
    batch_size: usize,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_BLUESTEIN_MODULE)?;
    let cfg = launch_config(
        elementwise_launch_config(batch_size * m),
        (BLOCK_SIZE, 1, 1),
        0,
    );

    let name = match dtype {
        DType::Complex64 => "bluestein_pointwise_mul_c64",
        DType::Complex128 => "bluestein_pointwise_mul_c128",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bluestein_pointwise_mul",
            });
        }
    };

    let func = get_kernel_function(&module, name)?;
    let mut builder = stream.launch_builder(&func);
    let (m_i32, batch_i32) = (m as i32, batch_size as i32);
    builder.arg(&spectrum_ptr);
    builder.arg(&kernel_spectrum_ptr);
    builder.arg(&m_i32);
    builder.arg(&batch_i32);
    unsafe {
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA bluestein pointwise-mul launch failed: {:?}",
                e
            ))
        })?;
    }
    Ok(())
}

/// Chirp-postmultiply the convolution result, cropping each batch row from `m`
/// back to `out_n` and applying `scale`.
///
/// `out_n` is `n` for a full transform and `n / 2 + 1` for `rfft`, which is what
/// lets the Hermitian half be taken without a second pass.
///
/// # Safety
///
/// Caller must ensure `conv` holds `batch_size * m` complex elements, `chirp`
/// holds at least `out_n`, and `out` holds `batch_size * out_n`, all on
/// `device_index`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_bluestein_postmultiply(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    conv_ptr: u64,
    chirp_ptr: u64,
    out_ptr: u64,
    m: usize,
    out_n: usize,
    batch_size: usize,
    scale: f64,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, FFT_BLUESTEIN_MODULE)?;
    let cfg = launch_config(
        elementwise_launch_config(batch_size * out_n),
        (BLOCK_SIZE, 1, 1),
        0,
    );

    let (m_i32, out_n_i32, batch_i32) = (m as i32, out_n as i32, batch_size as i32);

    match dtype {
        DType::Complex64 => {
            let func = get_kernel_function(&module, "bluestein_postmultiply_c64")?;
            let mut builder = stream.launch_builder(&func);
            let scale_f32 = scale as f32;
            builder.arg(&conv_ptr);
            builder.arg(&chirp_ptr);
            builder.arg(&out_ptr);
            builder.arg(&m_i32);
            builder.arg(&out_n_i32);
            builder.arg(&batch_i32);
            builder.arg(&scale_f32);
            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA bluestein postmultiply launch failed: {:?}",
                        e
                    ))
                })?;
            }
        }
        DType::Complex128 => {
            let func = get_kernel_function(&module, "bluestein_postmultiply_c128")?;
            let mut builder = stream.launch_builder(&func);
            builder.arg(&conv_ptr);
            builder.arg(&chirp_ptr);
            builder.arg(&out_ptr);
            builder.arg(&m_i32);
            builder.arg(&out_n_i32);
            builder.arg(&batch_i32);
            builder.arg(&scale);
            unsafe {
                builder.launch(cfg).map_err(|e| {
                    Error::Internal(format!(
                        "CUDA bluestein postmultiply launch failed: {:?}",
                        e
                    ))
                })?;
            }
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bluestein_postmultiply",
            });
        }
    }

    Ok(())
}
