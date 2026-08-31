//! Grid and block sizing for CUDA kernel launches.
//!
//! Each helper returns the grid/block shape for one launch pattern:
//! element-wise, global reduction, dimension-wise reduction, or softmax.

pub use cudarc::driver::safe::LaunchConfig;

use crate::error::{Error, Result};
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

/// Block size for element-wise operations (256 threads is optimal for most GPUs)
pub const BLOCK_SIZE: u32 = 256;

/// Calculate optimal grid dimensions for element-wise operations.
///
/// Uses a 1D grid with blocks of `BLOCK_SIZE` threads each.
///
/// # Errors
///
/// Returns [`Error::InvalidArgument`] when `numel` needs more than
/// `MAX_GRID_DIM_X` blocks. The kernels this config drives decode a flat
/// `idx < n` index with no `y`/`z` component, so spreading the excess into
/// another grid axis would compute the wrong elements silently — rejecting
/// the launch is the only correct outcome. Computing the grid size in `u64`
/// keeps this check exact instead of wrapping through a truncating `as u32`
/// cast for `numel > u32::MAX`.
#[inline]
pub fn elementwise_launch_config(numel: usize) -> Result<(u32, u32, u32)> {
    // Floored at 1: a grid extent of 0 is itself a launch error, even though
    // `numel == 0` needs no work. Callers that reach this helper with an
    // empty tensor rely on the kernel's own bounds guard (`idx < n`) to no-op.
    let grid_size = ((numel as u64) + BLOCK_SIZE as u64 - 1) / BLOCK_SIZE as u64;
    let grid_size = grid_size.max(1);
    if grid_size > MAX_GRID_DIM_X as u64 {
        return Err(Error::InvalidArgument {
            arg: "numel",
            reason: format!(
                "{numel} elements need a 1-D grid of {grid_size} blocks, exceeding the \
                 CUDA max grid extent of {MAX_GRID_DIM_X}"
            ),
        });
    }
    Ok((grid_size as u32, 1, 1))
}

/// Calculate launch configuration for global reduction kernels.
///
/// Limits grid size to prevent excessive block overhead for small inputs.
#[inline]
#[allow(dead_code)] // Kept for potential future optimization of global reductions
pub fn reduce_launch_config(numel: usize) -> (u32, u32) {
    let block_size = BLOCK_SIZE;
    let grid_size = ((numel as u32) + block_size - 1) / block_size;
    // Limit grid size to ensure we don't launch too many blocks
    let grid_size = grid_size.min(1024);
    (grid_size, block_size)
}

/// Maximum CUDA grid extent along `x`, for every compute capability numr targets.
const MAX_GRID_DIM_X: u32 = 2_147_483_647;

/// Maximum CUDA grid extent along `y` and `z`, for every compute capability numr targets.
const MAX_GRID_DIM_YZ: u32 = 65_535;

/// Calculate launch configuration for dimension-wise reduction.
///
/// Uses a 2D grid over the `[outer, inner]` output plane: `outer` on `x`,
/// `inner` on `y`. Each axis is clamped to its architectural maximum, and the
/// dim-reduction kernels grid-stride over both axes to cover the remainder — so
/// an `inner` past 65535 (e.g. summing `[1, 32, 4096, 64]` over dim 1, where
/// `inner` is 262144) launches and computes every output instead of being
/// rejected with `CUDA_ERROR_INVALID_VALUE`.
///
/// Each axis is also floored at 1: a grid extent of 0 is itself a launch error.
/// The floor stays here rather than at the callers, which pass `outer`/`inner`
/// unclamped and guard on a zero-element output before launching at all; the
/// dim-reduction kernels bound both loops by `outer_size`/`inner_size`, so a
/// floored axis over a zero extent does no work.
#[inline]
pub fn reduce_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), u32) {
    let grid_x = (outer.min(MAX_GRID_DIM_X as usize) as u32).max(1);
    let grid_y = (inner.min(MAX_GRID_DIM_YZ as usize) as u32).max(1);
    let block = BLOCK_SIZE;
    ((grid_x, grid_y, 1), block)
}

/// Calculate launch configuration for softmax over the last dimension.
///
/// One block per row, with threads cooperating to compute the softmax.
/// Returns (grid_size, block_size, shared_memory_bytes).
#[inline]
pub fn softmax_launch_config(outer: usize, dim_size: usize) -> (u32, u32, u32) {
    // One block per row, threads handle the dimension
    // Block size must be a power of 2 for the shared-memory tree reduction to work correctly
    let block_size = BLOCK_SIZE.min(dim_size as u32).next_power_of_two();
    let block_size = block_size.min(BLOCK_SIZE);
    let grid_size = outer as u32;
    // Shared memory: 2 arrays of block_size floats (for max and sum reduction)
    let shared_mem = 2 * block_size * 4; // f32
    (grid_size, block_size, shared_mem)
}

/// Calculate launch configuration for softmax over a non-last dimension.
///
/// Uses a 2D grid to process all (outer, inner) pairs in parallel.
/// Each thread processes one element position across the reduction dimension.
#[inline]
#[allow(dead_code)] // Available for future optimized softmax_dim kernel
pub fn softmax_dim_launch_config(outer: usize, inner: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    // Use 2D grid: one thread per (outer, inner) pair
    // Each thread sequentially processes the dim_size elements
    let total_elements = (outer * inner) as u32;
    let grid_x = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid = (grid_x, 1, 1);
    let block = (BLOCK_SIZE, 1, 1);
    (grid, block)
}

/// Check a computed dynamic shared-memory request against the device's actual
/// per-block budget before launching. `CudaDevice::new` is a free index
/// wrapper and `profile()` is served from a per-index cache, so this is an
/// atomic load, not a driver query.
///
/// `operation` names the failing op for [`Error::BackendLimitation`];
/// `context` describes what was being sized (e.g. `"sort dimension of size
/// 4096"`, `"128x128 matmul tile"`) so the message is actionable without this
/// helper knowing about sorts or tiles. `context` is a closure rather than a
/// `&str` so the (often `format!`-built) description is only allocated on the
/// error path, not on every launch.
pub fn check_shared_mem_fits(
    device_index: usize,
    shared_mem: u32,
    operation: &'static str,
    context: impl FnOnce() -> String,
) -> Result<()> {
    let limit = CudaDevice::new(device_index).profile().shared_mem_per_block;
    if shared_mem > limit {
        let context = context();
        return Err(Error::BackendLimitation {
            backend: "cuda",
            operation,
            reason: format!(
                "{context} needs {shared_mem} bytes of shared memory, exceeding this \
                 device's {limit}-byte per-block limit"
            ),
        });
    }
    Ok(())
}

/// Create a launch configuration from grid, block, and shared memory sizes.
#[inline]
pub fn launch_config(
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    shared_mem: u32,
) -> LaunchConfig {
    LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: shared_mem,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_numel_floors_to_one_block() {
        assert_eq!(elementwise_launch_config(0).unwrap(), (1, 1, 1));
    }

    #[test]
    fn one_numel_needs_one_block() {
        assert_eq!(elementwise_launch_config(1).unwrap(), (1, 1, 1));
    }

    #[test]
    fn numel_at_block_size_minus_one_needs_one_block() {
        assert_eq!(
            elementwise_launch_config(BLOCK_SIZE as usize - 1).unwrap(),
            (1, 1, 1)
        );
    }

    #[test]
    fn numel_at_block_size_needs_one_block() {
        assert_eq!(
            elementwise_launch_config(BLOCK_SIZE as usize).unwrap(),
            (1, 1, 1)
        );
    }

    #[test]
    fn numel_at_block_size_plus_one_needs_two_blocks() {
        assert_eq!(
            elementwise_launch_config(BLOCK_SIZE as usize + 1).unwrap(),
            (2, 1, 1)
        );
    }

    #[test]
    fn numel_past_1d_grid_capacity_is_rejected_not_truncated() {
        // One past what a 1-D grid of MAX_GRID_DIM_X blocks can cover.
        let numel = (MAX_GRID_DIM_X as u64) * (BLOCK_SIZE as u64) + 1;
        let result = elementwise_launch_config(numel as usize);
        assert!(
            result.is_err(),
            "numel beyond 1-D grid capacity must error, not silently truncate the grid"
        );
    }

    #[test]
    fn numel_at_1d_grid_capacity_still_fits() {
        let numel = (MAX_GRID_DIM_X as u64) * (BLOCK_SIZE as u64);
        let (grid_x, grid_y, grid_z) = elementwise_launch_config(numel as usize).unwrap();
        assert_eq!(grid_x, MAX_GRID_DIM_X);
        assert_eq!((grid_y, grid_z), (1, 1));
    }

    #[test]
    fn shared_mem_check_accepts_a_typical_request() {
        // No CUDA device is guaranteed present in unit tests; `profile()` then
        // falls back to `unknown()`, whose `shared_mem_per_block` is 0, so a
        // strictly-positive request cannot be asserted to fit here. Zero bytes
        // always fits regardless of the device's reported budget.
        let result = check_shared_mem_fits(0, 0, "matmul", || "128x128 matmul tile".to_string());
        assert!(result.is_ok());
    }

    #[test]
    fn shared_mem_check_rejects_an_oversized_request() {
        let result =
            check_shared_mem_fits(0, u32::MAX, "matmul", || "128x128 matmul tile".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn shared_mem_check_error_names_request_limit_and_context() {
        let err =
            check_shared_mem_fits(0, u32::MAX, "matmul", || "128x128 matmul tile".to_string())
                .unwrap_err()
                .to_string();
        assert!(
            err.contains("128x128 matmul tile"),
            "missing context: {err}"
        );
        assert!(
            err.contains(&u32::MAX.to_string()),
            "missing requested bytes: {err}"
        );
        assert!(err.contains("byte per-block limit"), "missing limit: {err}");
    }
}
