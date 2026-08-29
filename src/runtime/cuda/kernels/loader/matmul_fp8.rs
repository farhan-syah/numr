//! Compile-time-tiled FP8 GEMM launcher.
//!
//! Same dtype in and out, accumulating in F32 to match CPU. The
//! mixed-precision kernels behind `Fp8MatmulOps` are a different operation and
//! are not reached from here.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::algorithm::TileConfig;
use crate::dtype::DType;
use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Block and thread tile the compile-time-tiled FP8 kernels are built at.
///
/// `matmul_fp8.cu` instantiates exactly this shape and encodes it in the kernel
/// names, so the two must stay in step.
const FP8_TILE: TileConfig = TileConfig {
    block_m: 64,
    block_n: 64,
    block_k: 8,
    thread_m: 4,
    thread_n: 4,
};

/// Launch the compile-time-tiled FP8 GEMM, batched or not, with or without bias.
///
/// Same dtype in and out, accumulating in F32 to match CPU. The mixed-precision
/// kernels behind `Fp8MatmulOps` are a different operation and are not used here.
///
/// `bias_ptr` selects the fused-bias entry point, which seeds the accumulator
/// with the bias instead of adding it to a narrowed result.
///
/// Shared memory in these kernels is static, so the dynamic shared-memory
/// request must be zero.
///
/// # Safety
///
/// All pointers must be valid device memory with correct sizes.
pub(super) unsafe fn launch_matmul_fp8_tiled(
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
    let kernel_fn_name = match (dtype, plain, bias_ptr.is_some()) {
        (DType::FP8E4M3, true, false) => "matmul_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, true, false) => "matmul_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, false, false) => "matmul_batched_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, false, false) => "matmul_batched_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, true, true) => "matmul_bias_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, true, true) => "matmul_bias_fp8_e5m2_tiled_64x64x8_4x4",
        (DType::FP8E4M3, false, true) => "matmul_bias_batched_fp8_e4m3_tiled_64x64x8_4x4",
        (DType::FP8E5M2, false, true) => "matmul_bias_batched_fp8_e5m2_tiled_64x64x8_4x4",
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "matmul_fp8_tiled",
            });
        }
    };

    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_FP8_MODULE)?;
    let func = get_kernel_function(&module, kernel_fn_name)?;

    let bm = FP8_TILE.block_m as u32;
    let bn = FP8_TILE.block_n as u32;
    let tm = FP8_TILE.thread_m as u32;
    let tn = FP8_TILE.thread_n as u32;
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
        // The bias entry points take it as the third argument, before C.
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
                "CUDA FP8 matmul kernel '{}' launch failed: {:?}",
                kernel_fn_name, e
            ))
        })?;
    }

    Ok(())
}
