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

use super::grouped_matmul_tile::{
    grouped_block_dim, grouped_dtype_suffix, grouped_row_hint, grouped_tile, use_wmma_grouped,
};
use super::launch_dims::LaunchConfig;
use super::matmul_wmma_tile::{select_wmma_tile, wmma_kernel_name, wmma_launch_config};
use super::module_cache::{get_kernel_function, get_or_load_module};
use super::names::kernel_names;

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

    let row_hint = grouped_row_hint(total_rows, num_groups);
    let (bm, bn, suffix) = grouped_tile(row_hint, n);
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
/// (`matmul_wmma_tile`). `batch` is passed as the group count so grid.z
/// carries one slice per group, exactly as the dense batched WMMA launcher
/// uses `batch` for its own batch dimension.
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
    // Deliberately NOT total_rows: tile selection must see one group's row
    // count (see `grouped_row_hint`), while `wmma_launch_config` below keeps
    // consuming `total_rows` unchanged — grid.y and each kernel's per-group
    // guard are what keep this correctness-safe regardless of tile size.
    let row_hint = grouped_row_hint(total_rows, num_groups);
    let tile = select_wmma_tile(row_hint, n, k, 1, device_index);
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
