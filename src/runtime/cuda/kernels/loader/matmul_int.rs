//! Compile-time-tiled integer GEMM launcher, and the dtype gates around it.
//!
//! Integer kernels accumulate in a 128-bit `Numr128`, so `matmul_int.cu`
//! instantiates one fixed tile shape and encodes it in the kernel names.
//! `int_matmul_has_kernel` and `int_matmul_output_dtype` are the dtype rules
//! callers gate on before reaching the launcher.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::{dtype_suffix, kernel_names};

/// Whether `matmul_int.cu` instantiates tiled GEMM kernels for this dtype.
///
/// Every integer dtype has them, and `ops/cuda/matmul.rs` gates the public entry
/// point on this. Bool is the only integer-adjacent dtype left out, and
/// `DType::is_int` already excludes it.
///
/// `gemv_int.cu` instantiates the same list except I8, so the small-M GEMV fast
/// paths carry their own I8 guard — see [`int_matmul_output_dtype`] for why an
/// I8 GEMV kernel would be the wrong shape to write.
#[inline]
pub fn int_matmul_has_kernel(dtype: DType) -> bool {
    dtype.is_int()
}

/// The dtype an integer `matmul` writes for this element type.
///
/// Both forms widen at I8 and neither widens anywhere else, so this is the
/// backend-agnostic rule in [`crate::ops::matmul_output_dtype`] under the name
/// the CUDA launchers use. It is re-exported rather than restated so CUDA and
/// CPU cannot disagree about which widths widen.
#[inline]
pub fn int_matmul_output_dtype(dtype: DType) -> DType {
    crate::ops::matmul_output_dtype(dtype)
}

/// Block and thread tile the compile-time-tiled integer kernels are built at.
///
/// `matmul_int.cu` instantiates exactly this shape and encodes it in the kernel
/// names, so the two must stay in step. A `Numr128` accumulator is 16 bytes, so
/// every `reg_c` slot costs four registers: a 4x4 thread tile spends 64, while
/// an 8x8 tile would need 256 and blow past the 255-register per-thread limit.
const INT_TILE: TileConfig = TileConfig {
    block_m: 64,
    block_n: 64,
    block_k: 8,
    thread_m: 4,
    thread_n: 4,
};

/// Launch the compile-time-tiled integer GEMM, batched or not, with or without
/// a fused bias.
///
/// `bias_ptr` selects the fused entry point, which seeds the 128-bit
/// accumulator with the bias instead of adding it after the narrow-back store.
///
/// The tile dimensions are template parameters in `matmul_int.cu` rather than
/// kernel arguments, so `reg_c` is sized exactly and stays in registers. That
/// fixes the shape of the launch here: four kernel names per dtype, and a grid
/// derived from `INT_TILE`.
///
/// Shared memory in those kernels is static, so the dynamic shared-memory
/// request must be zero - adding the tile formula on top would push the block
/// past the 48 KB limit and fail the launch silently.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub(super) unsafe fn launch_matmul_int_tiled(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    dtype: DType,
    a_ptr: u64,
    b_ptr: u64,
    bias_ptr: Option<u64>,
    c_ptr: u64,
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    a_batch: usize,
    b_batch: usize,
) -> Result<()> {
    // A single batch with no broadcasting takes the 2-D entry point, which skips
    // the per-block batch offset arithmetic.
    let plain = batch == 1 && a_batch == 1 && b_batch == 1;
    if !int_matmul_has_kernel(dtype) {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "matmul_int_tiled",
        });
    }
    // `matmul_int.cu` builds all four names from the same dtype suffix and tile
    // shape, so they are composed here rather than listed per dtype.
    let base = match (bias_ptr.is_some(), plain) {
        (false, true) => "matmul",
        (false, false) => "matmul_batched",
        (true, true) => "matmul_bias",
        (true, false) => "matmul_bias_batched",
    };
    // The I8 kernels write I32 in both forms, so they carry an `i8_i32` suffix
    // instead of the bare element suffix.
    let suffix = if dtype == DType::I8 {
        "i8_i32"
    } else {
        dtype_suffix(dtype)
    };
    let kernel_fn_name = format!("{}_{}_tiled_64x64x8_4x4", base, suffix);

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_INT_MODULE)?;
    let func = get_kernel_function(&module, &kernel_fn_name)?;

    let bm = INT_TILE.block_m as u32;
    let bn = INT_TILE.block_n as u32;
    let tm = INT_TILE.thread_m as u32;
    let tn = INT_TILE.thread_n as u32;
    let cfg = LaunchConfig {
        grid_dim: (
            ((n as u32) + bn - 1) / bn,
            ((m as u32) + bm - 1) / bm,
            batch as u32,
        ),
        block_dim: (bn / tn, bm / tm, 1),
        shared_mem_bytes: 0,
    };

    let batch_u32 = batch as u32;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let a_batch_u32 = a_batch as u32;
    let b_batch_u32 = b_batch as u32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        // The bias parameter sits between B and C in the fused kernels, so it is
        // bound before C rather than appended.
        if let Some(bias) = bias_ptr.as_ref() {
            builder.arg(bias);
        }
        builder.arg(&c_ptr);
        if !plain {
            builder.arg(&batch_u32);
        }
        builder.arg(&m_u32);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        if !plain {
            builder.arg(&a_batch_u32);
            builder.arg(&b_batch_u32);
        }
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA integer matmul kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }

    Ok(())
}
