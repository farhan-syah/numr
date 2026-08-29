//! Type casting CUDA kernel launchers
//!
//! Provides launchers for casting tensors between different dtypes.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{
    BLOCK_SIZE, dtype_suffix, elementwise_launch_config, get_kernel_function, get_or_load_module,
    kernel_names, launch_config,
};
use crate::dtype::DType;
use crate::error::{Error, Result};

/// DTypes the `cast.cu` instantiation matrix covers.
///
/// Every ordered pair of these dtypes has a `cast_{src}_{dst}` kernel, so this
/// is the single source of truth for what `launch_cast` accepts. Keep it in
/// step with the row list at the bottom of `cast.cu`.
const CAST_DTYPES: &[DType] = &[
    DType::F32,
    DType::F64,
    DType::F16,
    DType::BF16,
    DType::FP8E4M3,
    DType::FP8E5M2,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
];

/// Launch a cast operation kernel.
///
/// Converts tensor elements from `src_dtype` to `dst_dtype`.
///
/// # Supported Conversions
///
/// All ordered pairs of [`CAST_DTYPES`]. Complex dtypes are not supported.
///
/// # Numerical Semantics
///
/// Conversions go through f64 and saturate on the way to an integer, matching
/// the CPU reference in `src/runtime/cpu/kernels/memory.rs`: NaN becomes 0 and
/// out-of-range values clamp to the nearest bound. Integer to integer saturates
/// too, because the CPU path also routes it through f64.
///
/// # Safety
///
/// - All pointers must be valid device memory
/// - Input tensor must have at least `numel` elements of `src_dtype`
/// - Output tensor must have at least `numel` elements of `dst_dtype`
///
/// # Arguments
///
/// * `context` - CUDA context
/// * `stream` - CUDA stream for async execution
/// * `device_index` - Device index for module caching
/// * `src_dtype` - Source data type
/// * `dst_dtype` - Destination data type
/// * `input_ptr` - Device pointer to input tensor
/// * `output_ptr` - Device pointer to output tensor
/// * `numel` - Number of elements
pub unsafe fn launch_cast(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_dtype: DType,
    dst_dtype: DType,
    input_ptr: u64,
    output_ptr: u64,
    numel: usize,
) -> Result<()> {
    // Same dtype is a no-op (should be handled by caller)
    if src_dtype == dst_dtype {
        return Ok(());
    }

    // Validate supported types
    let is_supported = |d: DType| CAST_DTYPES.contains(&d);

    if !is_supported(src_dtype) {
        return Err(Error::UnsupportedDType {
            dtype: src_dtype,
            op: "cast",
        });
    }
    if !is_supported(dst_dtype) {
        return Err(Error::UnsupportedDType {
            dtype: dst_dtype,
            op: "cast",
        });
    }

    unsafe {
        let module = get_or_load_module(context, device_index, kernel_names::CAST_MODULE)?;
        let func_name = format!(
            "cast_{}_{}",
            dtype_suffix(src_dtype),
            dtype_suffix(dst_dtype)
        );
        let func = get_kernel_function(&module, &func_name)?;

        let grid = elementwise_launch_config(numel);
        let block = (BLOCK_SIZE, 1, 1);
        let n = numel as u32;

        let cfg = launch_config(grid, block, 0);
        let mut builder = stream.launch_builder(&func);
        builder.arg(&input_ptr);
        builder.arg(&output_ptr);
        builder.arg(&n);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA cast kernel '{}' launch failed: {:?}",
                func_name, e
            ))
        })?;

        Ok(())
    }
}
