//! Masked fill kernel launchers, in both the same-shape and broadcast-mask
//! forms.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::super::loader::{
    BLOCK_SIZE, elementwise_launch_config, get_kernel_function, get_or_load_module, launch_config,
};
use super::dtype_gate::index_dtype_suffix;
use super::gather::INDEX_MODULE;
use crate::dtype::DType;
use crate::error::{Error, Result};

/// A fill value as the raw bit pattern of its element type, widened to the
/// unsigned integer of the same width.
///
/// A kernel parameter is laid out by size and alignment alone, so passing an
/// `i16`'s bits as a `u16` reaches `masked_fill_i16` byte-for-byte identical to
/// passing the `i16`. Fourteen element types collapse to four widths, which is
/// what keeps this launcher one screen long instead of fourteen match arms
/// repeated per entry point.
#[derive(Clone, Copy)]
enum FillBits {
    B8(u8),
    B16(u16),
    B32(u32),
    B64(u64),
}

/// Convert a fill value to the bit pattern of `dtype`.
///
/// Matches the CPU reference `masked_fill_kernel` in
/// src/runtime/cpu/kernels/index.rs, which fills with `T::from_f64(value)`:
/// Rust's float-to-integer `as` saturates and sends NaN to 0, and the float
/// widths round once from f64.
///
/// # Errors
///
/// Returns [`Error::UnsupportedDType`] for a dtype with no fill kernel, and for
/// F16/BF16/FP8 when their feature is off — those conversions live in the
/// optional dependency.
fn fill_bits(dtype: DType, value: f64, op: &'static str) -> Result<FillBits> {
    match dtype {
        DType::F32 => Ok(FillBits::B32((value as f32).to_bits())),
        DType::F64 => Ok(FillBits::B64(value.to_bits())),
        #[cfg(feature = "f16")]
        DType::F16 => Ok(FillBits::B16(half::f16::from_f64(value).to_bits())),
        #[cfg(feature = "f16")]
        DType::BF16 => Ok(FillBits::B16(half::bf16::from_f64(value).to_bits())),
        #[cfg(feature = "fp8")]
        DType::FP8E4M3 => Ok(FillBits::B8(
            crate::dtype::fp8::FP8E4M3::from_f64(value).to_bits(),
        )),
        #[cfg(feature = "fp8")]
        DType::FP8E5M2 => Ok(FillBits::B8(
            crate::dtype::fp8::FP8E5M2::from_f64(value).to_bits(),
        )),
        DType::I64 => Ok(FillBits::B64(value as i64 as u64)),
        DType::I32 => Ok(FillBits::B32(value as i32 as u32)),
        DType::I16 => Ok(FillBits::B16(value as i16 as u16)),
        DType::I8 => Ok(FillBits::B8(value as i8 as u8)),
        DType::U64 => Ok(FillBits::B64(value as u64)),
        DType::U32 => Ok(FillBits::B32(value as u32)),
        DType::U16 => Ok(FillBits::B16(value as u16)),
        DType::U8 => Ok(FillBits::B8(value as u8)),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

/// Launch masked_fill kernel.
///
/// Fills elements where mask is true with a scalar value.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - input and output must have n elements
pub unsafe fn launch_masked_fill(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let func_name = format!("masked_fill_{}", index_dtype_suffix(dtype, "masked_fill")?);
    let bits = fill_bits(dtype, fill_value, "masked_fill")?;

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        match &bits {
            FillBits::B8(v) => builder.arg(v),
            FillBits::B16(v) => builder.arg(v),
            FillBits::B32(v) => builder.arg(v),
            FillBits::B64(v) => builder.arg(v),
        };
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!("CUDA masked_fill kernel launch failed: {:?}", e))
        })?;

        Ok(())
    }
}

/// Launch broadcast masked_fill kernel.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - mask_strides_ptr, out_shape_ptr must be valid device memory with ndim u32 elements
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_masked_fill_broadcast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    input_ptr: u64,
    mask_ptr: u64,
    output_ptr: u64,
    fill_value: f64,
    mask_strides_ptr: u64,
    out_shape_ptr: u64,
    ndim: usize,
    n: usize,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }

    let func_name = format!(
        "masked_fill_broadcast_{}",
        index_dtype_suffix(dtype, "masked_fill_broadcast")?
    );
    let bits = fill_bits(dtype, fill_value, "masked_fill_broadcast")?;

    unsafe {
        let module = get_or_load_module(context, device_index, INDEX_MODULE)?;
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(n);
        let block = (BLOCK_SIZE, 1, 1);
        let cfg = launch_config(grid, block, 0);

        let ndim_u32 = ndim as u32;
        let n_u32 = n as u32;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&mask_ptr);
        builder.arg(&output_ptr);
        match &bits {
            FillBits::B8(v) => builder.arg(v),
            FillBits::B16(v) => builder.arg(v),
            FillBits::B32(v) => builder.arg(v),
            FillBits::B64(v) => builder.arg(v),
        };
        builder.arg(&mask_strides_ptr);
        builder.arg(&out_shape_ptr);
        builder.arg(&ndim_u32);
        builder.arg(&n_u32);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA masked_fill_broadcast kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_fill_saturates_like_the_cpu_reference() {
        // `Element::from_f64` saturates because Rust's float-to-int `as` does.
        match fill_bits(DType::I32, 4_000_000_000.0, "test").unwrap() {
            FillBits::B32(bits) => assert_eq!(bits as i32, i32::MAX),
            _ => panic!("I32 must produce a 32-bit fill value"),
        }
        match fill_bits(DType::U8, -1.0, "test").unwrap() {
            FillBits::B8(bits) => assert_eq!(bits, 0),
            _ => panic!("U8 must produce an 8-bit fill value"),
        }
    }

    #[test]
    fn negative_signed_fill_keeps_its_bit_pattern() {
        match fill_bits(DType::I16, -3.0, "test").unwrap() {
            FillBits::B16(bits) => assert_eq!(bits as i16, -3),
            _ => panic!("I16 must produce a 16-bit fill value"),
        }
    }
}
