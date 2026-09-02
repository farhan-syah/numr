//! Grouped FP32 GEMM launcher.
//!
//! Wraps the entry points in `kernels/grouped_matmul.cu`, which reuse the same
//! compile-time-tiled core as the dense F32 path. Tile choice mirrors the dense
//! picker so the two stay in step.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::error::{Error, Result};

use super::launch_dims::LaunchConfig;
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Tile the grouped kernel is instantiated for, as `(BM, BN, suffix)`.
///
/// The 128×128 tile needs a wide output to be worth its shared memory; below
/// that the 64×64 instantiation wastes far less on the ragged edge. Both are
/// the tiles the dense F32 path already specialises.
fn grouped_tile(n: usize) -> (usize, usize, &'static str) {
    if n >= 128 {
        (128, 128, "128x128x8_8x8")
    } else {
        (64, 64, "64x64x32_8x4")
    }
}

/// Threads per block for a tile, matching the `extern "C"` instantiations:
/// `(BN / TN, BM / TM, 1)`.
fn grouped_block_dim(suffix: &str) -> (u32, u32, u32) {
    match suffix {
        "128x128x8_8x8" => (16, 16, 1),
        _ => (16, 8, 1),
    }
}

/// Launch a grouped FP32 GEMM.
///
/// `activation` selects the fused-epilogue kernel; `None` takes the plain one,
/// which has no per-element switch.
///
/// # Safety
///
/// All pointers must be valid device memory with the sizes the shapes imply.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_grouped_matmul_f32(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    a_ptr: u64,
    b_ptr: u64,
    offsets_ptr: u64,
    c_ptr: u64,
    total_rows: usize,
    n: usize,
    k: usize,
    num_groups: usize,
    activation: Option<u32>,
) -> Result<()> {
    let (bm, bn, suffix) = grouped_tile(n);
    let stem = if activation.is_some() {
        "grouped_matmul_act_f32"
    } else {
        "grouped_matmul_f32"
    };
    let kernel_fn_name = format!("{stem}_{suffix}");

    let module = get_or_load_module(context, device_index, kernel_names::GROUPED_MATMUL_MODULE)?;
    let func = get_kernel_function(&module, &kernel_fn_name)?;

    // grid.y covers the TOTAL row count because the per-group counts are on the
    // device; blocks past their own group's count return immediately.
    let cfg = LaunchConfig {
        grid_dim: (
            n.div_ceil(bn) as u32,
            total_rows.div_ceil(bm) as u32,
            num_groups as u32,
        ),
        block_dim: grouped_block_dim(suffix),
        // The tiled core uses only static __shared__ arrays; asking for dynamic
        // shared memory on top would push past the per-block limit.
        shared_mem_bytes: 0,
    };

    let n_u32 = n as u32;
    let k_u32 = k as u32;
    let groups_i32 = num_groups as i32;

    unsafe {
        let mut builder = stream.launch_builder(&func);
        builder.arg(&a_ptr);
        builder.arg(&b_ptr);
        builder.arg(&offsets_ptr);
        builder.arg(&c_ptr);
        builder.arg(&n_u32);
        builder.arg(&k_u32);
        builder.arg(&groups_i32);
        let act_u32 = activation.unwrap_or(0);
        if activation.is_some() {
            builder.arg(&act_u32);
        }
        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA grouped matmul kernel '{kernel_fn_name}' launch failed: {e:?}"
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wide_output_takes_the_large_tile() {
        assert_eq!(grouped_tile(4096).2, "128x128x8_8x8");
    }

    #[test]
    fn narrow_output_takes_the_small_tile() {
        assert_eq!(grouped_tile(48).2, "64x64x32_8x4");
    }

    #[test]
    fn block_dims_match_the_instantiated_tiles() {
        // (BN / TN, BM / TM): 128/8 = 16 both ways, and 64/4 = 16, 64/8 = 8.
        assert_eq!(grouped_block_dim("128x128x8_8x8"), (16, 16, 1));
        assert_eq!(grouped_block_dim("64x64x32_8x4"), (16, 8, 1));
    }
}
