//! Block-tile choice for the tensor-core WMMA GEMM kernels.
//!
//! `kernels/matmul_wmma.cu` instantiates every WMMA kernel family at three
//! block tiles and puts the tile in the symbol name. This module owns the
//! launch geometry and the rule that picks one tile per launch; the launchers
//! in `matmul_wmma.rs` and `gemm_epilogue_wmma.rs` both go through it.

use crate::dtype::DType;
use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

use super::launch_dims::LaunchConfig;
use super::names::dtype_suffix;

// WMMA block: 16 warps (4×4 warp grid), each warp = 32 threads → 512 threads.
// The thread count is identical for every block tile: the tile varies through
// the per-warp fragment count, not through the warp grid (matmul_wmma.cuh).
const WMMA_BLOCK_THREADS: u32 = 512;

/// Resident blocks per SM the WMMA kernels are compiled for. This is the
/// `minBlocksPerMultiprocessor` argument of `WMMA_LAUNCH_BOUNDS`
/// (`matmul_wmma.cuh`): ptxas caps registers so at least this many blocks stay
/// resident, and the shared-memory footprint of every tile is small enough
/// not to lower it.
const WMMA_BLOCKS_PER_UNIT: u32 = 2;

/// Device waves the larger tile's grid must reach when the tile divides the
/// output exactly. One wave is `compute_units * WMMA_BLOCKS_PER_UNIT` blocks.
/// No block computes a partial tile, so every block does full-value work and
/// the tile pays off as soon as it can fill the device once.
const WMMA_MIN_WAVES_EXACT: u32 = 1;

/// Device waves the larger tile's grid must reach when it leaves partial edge
/// tiles. Edge blocks compute and discard the overhanging fragments, so that
/// waste has to be amortized over more waves before the tile pays off.
const WMMA_MIN_WAVES_PARTIAL: u32 = 2;

/// Device waves 128x64 must reach when it divides the output exactly.
///
/// Measured, and it does NOT inherit [`WMMA_MIN_WAVES_EXACT`]. This tile
/// covers half the area of 128x128, so a block reuses each staged B tile over
/// half as many rows. One wave is not enough to pay for that: an exactly
/// dividing shape at just over one wave was faster on 64x64. Two waves keeps
/// those shapes on 64x64 while still taking 128x64 where it wins.
const WMMA_MIN_WAVES_EXACT_128X64: u32 = 2;

/// Block tiles considered before the smallest tile, ordered
/// largest-coverage-first (`tile_m * tile_n` descending). `select_wmma_tile`
/// walks this list and takes the first candidate that clears the coverage and
/// wave gates; the smallest tile is the unconditional fallback and is not in
/// this list.
const WMMA_LARGE_CANDIDATES: &[WmmaTile] = &[WmmaTile::Tile128, WmmaTile::Tile128x64];

/// Block tile of a WMMA kernel instantiation.
///
/// `matmul_wmma.cu` instantiates every kernel family at all three tiles and
/// puts the tile in the symbol name, so a launch selects a tile by picking a
/// symbol. Every tile uses the same block size and launch geometry.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(super) enum WmmaTile {
    /// 128x128 outputs per block, 2x2 fragments per warp.
    Tile128,
    /// 128x64 outputs per block, 2x1 fragments per warp.
    Tile128x64,
    /// 64x64 outputs per block, one fragment per warp.
    Tile64,
}

impl WmmaTile {
    /// Rows of C one block computes.
    #[inline]
    const fn tile_m(self) -> u32 {
        match self {
            WmmaTile::Tile128 => 128,
            WmmaTile::Tile128x64 => 128,
            WmmaTile::Tile64 => 64,
        }
    }

    /// Columns of C one block computes.
    #[inline]
    const fn tile_n(self) -> u32 {
        match self {
            WmmaTile::Tile128 => 128,
            WmmaTile::Tile128x64 => 64,
            WmmaTile::Tile64 => 64,
        }
    }

    /// Symbol-name suffix of the kernels built at this tile.
    #[inline]
    const fn suffix(self) -> &'static str {
        match self {
            WmmaTile::Tile128 => "128x128",
            WmmaTile::Tile128x64 => "128x64",
            WmmaTile::Tile64 => "64x64",
        }
    }
}

/// Symbol name of one WMMA kernel instantiation: `{base}_{dtype}_{tile}`, e.g.
/// `matmul_wmma_f16_128x128`. The single place that builds this name, so the
/// launchers in `matmul_wmma.rs` and `gemm_epilogue_wmma.rs` cannot drift
/// apart on the naming convention `matmul_wmma.cu` instantiates against.
#[inline]
pub(super) fn wmma_kernel_name(base: &str, dtype: DType, tile: WmmaTile) -> String {
    format!("{base}_{}_{}", dtype_suffix(dtype), tile.suffix())
}

/// Blocks `tile` launches for this output shape.
#[inline]
fn wmma_grid_blocks(m: usize, n: usize, batch: usize, tile: WmmaTile) -> u64 {
    let tiles_m = (m as u64).div_ceil(u64::from(tile.tile_m()));
    let tiles_n = (n as u64).div_ceil(u64::from(tile.tile_n()));
    tiles_m * tiles_n * (batch as u64)
}

/// Pick the block tile for one WMMA launch.
///
/// Every tile runs the same kernel body and trades parallelism against reuse.
/// A larger tile computes more output per block and reads each operand
/// element into shared memory fewer times, so it wins once the device is
/// already full of blocks. A smaller tile emits more blocks for the same
/// output, so it wins when the larger tile's grid is too short to keep every
/// SM busy for more than about one wave: the launch is then bound by how few
/// blocks are in flight, and the extra reuse buys nothing.
///
/// [`WMMA_LARGE_CANDIDATES`] lists every tile above the smallest,
/// largest-coverage-first. The walk takes the first candidate that clears
/// both gates and falls back to the smallest tile if none do:
/// - coverage: the output covers the tile in both dimensions. A tile wider or
///   taller than the problem still stages, multiplies and accumulates the
///   overhanging fragments, then discards their stores — pure waste.
/// - waves: the candidate's grid reaches the wave count required for how it
///   divides the output, a wave being `compute_units * WMMA_BLOCKS_PER_UNIT`
///   blocks.
///
/// How many waves are required depends on divisibility, not on the grid alone.
/// A tile that divides both output dimensions exactly wastes nothing on partial
/// edges, so every block does full-value work and the tile is worth taking as
/// soon as it fills the device once (`WMMA_MIN_WAVES_EXACT`). A tile that
/// leaves partial edges pays for the discarded fragments of every edge block,
/// and that cost is only amortized over a longer grid
/// (`WMMA_MIN_WAVES_PARTIAL`). Two shapes can produce the same grid and still
/// want opposite tiles for this reason, so the wave count alone cannot decide.
///
/// Wave thresholds are empirical; `benches/matmul.rs` forces each tile and is
/// how they are re-derived when the kernels change. 128x64 carries its own
/// exact-case threshold ([`WMMA_MIN_WAVES_EXACT_128X64`]) rather than
/// inheriting the shared one.
///
/// K does not enter the rule: every tile iterates the whole K extent, so K
/// scales the per-block work of all of them by the same factor.
///
/// `compute_units` comes from the cached device profile — an atomic load, not
/// a driver query per launch. An unknown profile reports zero compute units,
/// which makes the wave tests trivially true and leaves the first candidate
/// (the largest tile) as the default.
///
/// This runs only after [`super::matmul_wmma::use_wmma`] has already chosen
/// the WMMA path; it never changes whether that path is taken.
#[inline]
pub(super) fn select_wmma_tile(
    m: usize,
    n: usize,
    _k: usize,
    batch: usize,
    device_index: usize,
) -> WmmaTile {
    // CudaDevice::new is a zero-cost index wrapper; profile() reads the cached
    // profile, so this is an atomic load rather than a driver query.
    let compute_units = CudaDevice::new(device_index).profile().compute_units;
    wmma_tile_for_units(m, n, batch, compute_units)
}

/// The tile rule itself, separated from the device query so it is testable
/// without a device.
#[inline]
fn wmma_tile_for_units(m: usize, n: usize, batch: usize, compute_units: u32) -> WmmaTile {
    for &tile in WMMA_LARGE_CANDIDATES {
        if wmma_tile_clears_gates(m, n, batch, compute_units, tile) {
            return tile;
        }
    }
    WmmaTile::Tile64
}

/// The coverage gate and the wave gate for one candidate tile, the two
/// primitives every candidate in the walk is judged by.
#[inline]
fn wmma_tile_clears_gates(
    m: usize,
    n: usize,
    batch: usize,
    compute_units: u32,
    tile: WmmaTile,
) -> bool {
    if m < tile.tile_m() as usize || n < tile.tile_n() as usize {
        return false;
    }

    let wave_blocks = u64::from(compute_units) * u64::from(WMMA_BLOCKS_PER_UNIT);
    let divides_exactly =
        m.is_multiple_of(tile.tile_m() as usize) && n.is_multiple_of(tile.tile_n() as usize);
    let min_waves = match (divides_exactly, tile) {
        (true, WmmaTile::Tile128x64) => WMMA_MIN_WAVES_EXACT_128X64,
        (true, _) => WMMA_MIN_WAVES_EXACT,
        (false, _) => WMMA_MIN_WAVES_PARTIAL,
    };
    let needed = wave_blocks * u64::from(min_waves);

    wmma_grid_blocks(m, n, batch, tile) >= needed
}

// WMMA kernels use only statically-declared __shared__ arrays; there is no
// extern __shared__ (dynamic) allocation.  Pass 0 so CUDA does not add
// extra dynamic smem on top of the static pool (which would push total over
// the 48 KB default per-block limit on sm_86).
const WMMA_SMEM_BYTES: u32 = 0;

/// Grid and block for one WMMA launch: one block per `tile` output tile, one
/// grid-z slice per batch index (`batch` is 1 for the 2-D forms).
#[inline]
pub(super) fn wmma_launch_config(m: usize, n: usize, batch: usize, tile: WmmaTile) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            (n as u32).div_ceil(tile.tile_n()),
            (m as u32).div_ceil(tile.tile_m()),
            batch as u32,
        ),
        block_dim: (WMMA_BLOCK_THREADS, 1, 1),
        shared_mem_bytes: WMMA_SMEM_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- tile selection ----

    // A mid-size device; the rule scales with this number, it is not tuned to
    // any particular value.
    const UNITS: u32 = 28;

    #[test]
    fn small_grid_takes_the_smaller_tile() {
        // 4x8 = 32 blocks at the large tile, under one wave on this device:
        // too few blocks in flight for the extra reuse to buy anything, even
        // though the tile divides the output exactly.
        assert_eq!(wmma_tile_for_units(512, 1024, 1, UNITS), WmmaTile::Tile64);
    }

    #[test]
    fn large_grid_takes_the_larger_tile() {
        // Batching multiplies the grid past the wave threshold.
        assert_eq!(wmma_tile_for_units(512, 1024, 4, UNITS), WmmaTile::Tile128);
        assert_eq!(wmma_tile_for_units(512, 512, 64, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn exact_divide_takes_the_larger_tile_at_one_wave() {
        // 8x8 = 64 blocks, just over one wave. The tile divides 1024 exactly,
        // so no block computes a partial edge and the tile is worth taking.
        assert_eq!(wmma_tile_for_units(1024, 1024, 1, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn partial_edge_at_the_same_grid_drops_below_128x128() {
        // Same 8x8 = 64 blocks as the exact case above, but 1000 leaves a
        // partial edge tile in both dimensions; one wave is too short to
        // amortize the discarded fragments. 128x64 halves the discarded
        // columns and clears two waves here, and measured faster than both
        // 128x128 and the 64x64 fallback on this shape.
        assert_eq!(
            wmma_tile_for_units(1000, 1000, 1, UNITS),
            WmmaTile::Tile128x64
        );
    }

    #[test]
    fn partial_edge_takes_the_larger_tile_once_the_grid_is_long() {
        // The same partial-edge shape at twice the batch reaches the longer
        // grid the edge waste needs.
        assert_eq!(wmma_tile_for_units(1000, 1000, 2, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn overhanging_output_drops_below_128x128() {
        // N below the large tile width: half of every block's columns would be
        // computed and then discarded, however long the grid is. 128x64 covers
        // this N exactly, and measured faster than the 64x64 fallback.
        assert_eq!(
            wmma_tile_for_units(512, 64, 64, UNITS),
            WmmaTile::Tile128x64
        );
        // Same in the M direction, where no 64-wide-M tile exists to catch it.
        assert_eq!(wmma_tile_for_units(64, 512, 64, UNITS), WmmaTile::Tile64);
    }

    #[test]
    fn exact_threshold_is_exactly_the_wave_count() {
        let needed = (UNITS * WMMA_BLOCKS_PER_UNIT * WMMA_MIN_WAVES_EXACT) as usize;
        // 128x128 divides the large tile exactly and gives one block per batch
        // slice, so batch == grid blocks.
        assert_eq!(
            wmma_tile_for_units(128, 128, needed, UNITS),
            WmmaTile::Tile128
        );
        assert_eq!(
            wmma_tile_for_units(128, 128, needed - 1, UNITS),
            WmmaTile::Tile64
        );
    }

    #[test]
    fn partial_threshold_is_exactly_the_wave_count() {
        let wave = (UNITS * WMMA_BLOCKS_PER_UNIT * WMMA_MIN_WAVES_PARTIAL) as usize;
        // 129x129 leaves a partial edge in both dimensions and gives four
        // blocks per batch slice.
        let batch = wave / 4;
        assert_eq!(
            wmma_tile_for_units(129, 129, batch, UNITS),
            WmmaTile::Tile128
        );
        // One block short, so 128x128 is rejected. It falls to 128x64: same
        // class as the measured 1000x1000 case, partial edges clearing two
        // waves. Pinned exactly so a later change cannot silently drop it to
        // the 64x64 fallback.
        assert_eq!(
            wmma_tile_for_units(129, 129, batch - 1, UNITS),
            WmmaTile::Tile128x64
        );
    }

    #[test]
    fn unknown_profile_keeps_the_larger_tile() {
        // DeviceProfile::unknown reports zero compute units; both wave tests
        // are then trivially satisfied and the default tile stands.
        assert_eq!(wmma_tile_for_units(1024, 1024, 1, 0), WmmaTile::Tile128);
        // Including for a shape the large tile does not divide.
        assert_eq!(wmma_tile_for_units(1000, 1000, 1, 0), WmmaTile::Tile128);
    }

    #[test]
    fn grid_blocks_counts_partial_tiles() {
        assert_eq!(wmma_grid_blocks(129, 129, 1, WmmaTile::Tile128), 4);
        assert_eq!(wmma_grid_blocks(129, 129, 1, WmmaTile::Tile64), 9);
        assert_eq!(wmma_grid_blocks(256, 256, 3, WmmaTile::Tile128), 12);
    }

    // ---- generalized candidate walk ----
    //
    // Shapes routed back through wmma_tile_for_units to pin the walk's
    // ordering: the largest covering candidate wins first, and the fallback
    // still applies when no candidate clears its wave gate.

    #[test]
    fn tile_128x64_takes_narrow_n_but_not_thin_exact_grids() {
        // N below 128 cannot use the 128-wide tile, so 128x64 is the only
        // candidate above the fallback. Measured faster than 64x64 here.
        assert_eq!(
            wmma_tile_for_units(512, 64, 64, UNITS),
            WmmaTile::Tile128x64,
            "narrow-N shape with a long grid should take 128x64"
        );

        // Partial edges in both dimensions: 128x128 cannot clear two waves,
        // 128x64 can, and measured faster than the 64x64 fallback.
        assert_eq!(
            wmma_tile_for_units(1000, 1000, 1, UNITS),
            WmmaTile::Tile128x64,
            "unaligned square should take 128x64 over the fallback"
        );

        // Divides 128x64 exactly but reaches only just over one wave. This is
        // the case WMMA_MIN_WAVES_EXACT_128X64 exists to reject: measured
        // faster on 64x64.
        assert_eq!(
            wmma_tile_for_units(512, 1024, 1, UNITS),
            WmmaTile::Tile64,
            "exact but thin grid should fall back rather than take 128x64"
        );

        // Both dimensions clear 128, so the largest tile still wins first.
        assert_eq!(
            wmma_tile_for_units(1024, 1024, 1, UNITS),
            WmmaTile::Tile128,
            "128x128 must still win where it already did"
        );
    }

    // ---- shapes 128x64 must not steal ----
    //
    // Every shape here decided the same way before 128x64 existed. Each one
    // either lets 128x128 win outright or evaluates 128x64 and rejects it, so
    // adding a candidate moved nothing.

    #[test]
    fn walk_reproduces_small_grid_takes_the_smaller_tile() {
        assert_eq!(wmma_tile_for_units(512, 1024, 1, UNITS), WmmaTile::Tile64);
    }

    #[test]
    fn walk_reproduces_large_grid_takes_the_larger_tile() {
        assert_eq!(wmma_tile_for_units(512, 1024, 4, UNITS), WmmaTile::Tile128);
        assert_eq!(wmma_tile_for_units(512, 512, 64, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn walk_reproduces_exact_divide_threshold() {
        assert_eq!(wmma_tile_for_units(1024, 1024, 1, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn walk_still_promotes_a_longer_partial_grid() {
        assert_eq!(wmma_tile_for_units(1000, 1000, 2, UNITS), WmmaTile::Tile128);
    }

    #[test]
    fn walk_reproduces_unknown_profile_default() {
        assert_eq!(wmma_tile_for_units(1024, 1024, 1, 0), WmmaTile::Tile128);
        assert_eq!(wmma_tile_for_units(1000, 1000, 1, 0), WmmaTile::Tile128);
    }
}
