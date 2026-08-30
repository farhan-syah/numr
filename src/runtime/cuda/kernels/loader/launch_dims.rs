//! Grid and block sizing for CUDA kernel launches.
//!
//! Each helper returns the grid/block shape for one launch pattern:
//! element-wise, global reduction, dimension-wise reduction, or softmax.

pub use cudarc::driver::safe::LaunchConfig;

/// Block size for element-wise operations (256 threads is optimal for most GPUs)
pub const BLOCK_SIZE: u32 = 256;

/// Calculate optimal grid dimensions for element-wise operations.
///
/// Uses a 1D grid with blocks of `BLOCK_SIZE` threads each.
#[inline]
pub fn elementwise_launch_config(numel: usize) -> (u32, u32, u32) {
    // Floored at 1: a grid extent of 0 is itself a launch error, even though
    // `numel == 0` needs no work. Callers that reach this helper with an
    // empty tensor rely on the kernel's own bounds guard (`idx < n`) to no-op.
    let grid_size = (((numel as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
    (grid_size, 1, 1)
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
