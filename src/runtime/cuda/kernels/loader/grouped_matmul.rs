//! Grouped GEMM launcher: tiled core for F32, tensor-core WMMA for
//! 16-aligned F16/BF16.
//!
//! The tiled path wraps `kernels/grouped_matmul.cu`, which reuses the same
//! compile-time-tiled core as the dense F32 path; tile choice mirrors the
//! dense picker so the two stay in step. The WMMA path wraps the
//! `grouped_matmul_wmma`/`grouped_matmul_act_wmma` families in
//! `kernels/matmul_wmma.cu` and reuses the dense WMMA launcher's tile
//! selection and launch-geometry helpers rather than re-deriving them.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::traits::profile::DeviceCaps;

use super::launch_dims::LaunchConfig;
use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

/// Returns true when the WMMA path should be taken for a grouped GEMM of this
/// dtype, device, and `(N, K)` shape.
///
/// Conditions:
/// - dtype is F16 (needs `caps.f16_mma`) or BF16 (needs `caps.bf16` — the
///   BF16 WMMA symbols are compiled only from sm_80, see `matmul_wmma.cu`,
///   and `caps.bf16` already gates on that)
/// - N and K are both multiples of 16 (WMMA fragment requirement)
///
/// Deliberately does NOT test M alignment, unlike the dense [`use_wmma`]
/// (`matmul_wmma.rs`). There, M is a launch argument the host already knows
/// and can pad. Here M is a PER-GROUP row count read from `offsets` in
/// device memory — the host sees only `total_rows`, the sum across groups,
/// and cannot see or pad any individual group's count. The grouped WMMA
/// kernel is written for this: its A-tile staging and its epilogue store are
/// both bounds-checked per row against the group's `count` (`matmul_wmma.cu`,
/// `DEFINE_WMMA_GROUPED`), so a ragged M is masked off rather than mis-read
/// or mis-written — no host-side alignment check is needed or possible.
///
/// [`use_wmma`]: super::matmul_wmma::use_wmma
#[inline]
pub(crate) fn use_wmma_grouped(dtype: DType, caps: DeviceCaps, n: usize, k: usize) -> bool {
    let dtype_ok = match dtype {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    dtype_ok && n.is_multiple_of(16) && k.is_multiple_of(16)
}

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

/// Kernel-name dtype suffix. The core accumulates in F32 for all of these.
fn grouped_dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => Err(Error::Internal(format!(
            "grouped matmul supports F32/F16/BF16, got {other:?}"
        ))),
    }
}

/// Launch a grouped GEMM.
///
/// `activation` selects the fused-epilogue kernel; `None` takes the plain one,
/// which has no per-element switch. `caps` is this device's capability
/// snapshot; [`use_wmma_grouped`] is the single place that decides between
/// the tensor-core and tiled paths, so callers just forward it rather than
/// pre-deciding.
///
/// # Safety
///
/// All pointers must be valid device memory with the sizes the shapes imply.
#[allow(clippy::too_many_arguments)]
pub unsafe fn launch_grouped_matmul(
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
    dtype: DType,
    caps: DeviceCaps,
    activation: Option<u32>,
) -> Result<()> {
    if use_wmma_grouped(dtype, caps, n, k) {
        return unsafe {
            launch_grouped_matmul_wmma(
                context,
                stream,
                device_index,
                a_ptr,
                b_ptr,
                offsets_ptr,
                c_ptr,
                total_rows,
                n,
                k,
                num_groups,
                dtype,
                activation,
            )
        };
    }

    let (bm, bn, suffix) = grouped_tile(n);
    let dt = grouped_dtype_suffix(dtype)?;
    let stem = if activation.is_some() {
        "grouped_matmul_act"
    } else {
        "grouped_matmul"
    };
    let kernel_fn_name = format!("{stem}_{dt}_{suffix}");

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

/// Launch a grouped GEMM on the tensor-core WMMA path. Only reached once
/// [`use_wmma_grouped`] has already accepted the dtype, caps, and N/K shape.
///
/// Tile choice and launch geometry are the dense WMMA launcher's own helpers
/// (`matmul_wmma_tile`), called with `total_rows` standing in for M — the
/// tile only needs to know the row extent the grid must cover, which is
/// `total_rows` here since per-group counts are on the device. `batch` is
/// passed as the group count so grid.z carries one slice per group, exactly
/// as the dense batched WMMA launcher uses `batch` for its own batch
/// dimension.
///
/// # Safety
///
/// All pointers must be valid device memory with the sizes the shapes imply.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_grouped_matmul_wmma(
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
    dtype: DType,
    activation: Option<u32>,
) -> Result<()> {
    let module = get_or_load_module(context, device_index, kernel_names::MATMUL_WMMA_MODULE)?;
    let tile = select_wmma_tile(total_rows, n, k, 1, device_index);
    let base = if activation.is_some() {
        "grouped_matmul_act_wmma"
    } else {
        "grouped_matmul_wmma"
    };
    let kernel_fn_name = wmma_kernel_name(base, dtype, tile);
    let func = get_kernel_function(&module, &kernel_fn_name)?;

    // grid.z is the group count and grid.y covers total_rows, the same
    // "cover the total, let ragged blocks return early" convention the tiled
    // path above uses — the per-group row count lives in `offsets`, not on
    // the host.
    let cfg = wmma_launch_config(total_rows, n, num_groups, tile);

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
                "CUDA grouped WMMA matmul kernel '{kernel_fn_name}' launch failed: {e:?}"
            ))
        })?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_suffixes_cover_the_instantiated_kernels() {
        assert_eq!(grouped_dtype_suffix(DType::F32).unwrap(), "f32");
        assert_eq!(grouped_dtype_suffix(DType::F16).unwrap(), "f16");
        assert_eq!(grouped_dtype_suffix(DType::BF16).unwrap(), "bf16");
        assert!(grouped_dtype_suffix(DType::I32).is_err());
    }

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

    // ---- use_wmma_grouped ----

    fn turing_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            f16_mma: true,
            bf16: false,
        }
    }

    fn ampere_caps() -> DeviceCaps {
        DeviceCaps {
            dp4a: true,
            int8_mma: true,
            f16_mma: true,
            bf16: true,
        }
    }

    fn no_caps() -> DeviceCaps {
        DeviceCaps::default()
    }

    #[test]
    fn f16_takes_wmma_only_with_f16_mma() {
        assert!(use_wmma_grouped(DType::F16, turing_caps(), 32, 32));
        assert!(!use_wmma_grouped(DType::F16, no_caps(), 32, 32));
    }

    #[test]
    fn bf16_takes_wmma_only_with_bf16_cap() {
        assert!(use_wmma_grouped(DType::BF16, ampere_caps(), 32, 32));
        // Turing has f16_mma but not native bf16 — the BF16 WMMA symbols
        // are not even compiled for sm_75, so this must stay off the WMMA
        // path regardless of N/K alignment.
        assert!(!use_wmma_grouped(DType::BF16, turing_caps(), 32, 32));
    }

    #[test]
    fn f32_never_takes_wmma() {
        assert!(!use_wmma_grouped(DType::F32, ampere_caps(), 32, 32));
    }

    #[test]
    fn unaligned_n_or_k_rejected() {
        assert!(!use_wmma_grouped(DType::F16, ampere_caps(), 33, 32));
        assert!(!use_wmma_grouped(DType::F16, ampere_caps(), 32, 33));
    }
}
