//! Split count for two-stage dimension-wise reductions.
//!
//! The dim-reduce grid from `reduce_dim_launch_config` is one block per output
//! element (`outer * inner`) and does not depend on
//! `reduce_size`, so a long axis collapsing to few outputs runs on a handful of
//! blocks however much data it reads. Cutting the reduced axis into `splits`
//! equal chunks widens that grid by the same factor: stage 1 reduces each chunk
//! and stage 2 reduces the per-chunk results. Both stages are the existing
//! kernel, so this module owns only the choice of `splits`.

use crate::runtime::Device;
use crate::runtime::cuda::CudaDevice;

use super::launch_dims::BLOCK_SIZE;

/// Device waves the single-stage grid must already cover before the launch is
/// left alone. Same fill target `conv1d_ox` and `softmax_dim` use.
const REDUCE_SPLIT_MIN_WAVES: usize = 2;

/// Candidates tried below the target split count before the split is
/// abandoned.
///
/// `splits` must divide `reduce_size` exactly, and a prime or awkward
/// `reduce_size` has no such divisor near the target. Walking all the way down
/// to 2 to discover that would cost more than the launch the split is trying to
/// improve, so the scan is bounded and an unsplittable extent keeps the
/// single-stage path.
const REDUCE_SPLIT_DIVISOR_SCAN: usize = 64;

/// Number of equal chunks to cut the reduced axis into, or `None` to keep the
/// single-stage launch.
///
/// A returned `splits` always divides `reduce` exactly and leaves at least
/// [`BLOCK_SIZE`] elements per chunk. Equal chunks are what let the existing
/// kernel serve both stages unchanged and what makes a chunk-wise mean of the
/// chunk results equal the whole-axis mean.
#[inline]
pub(crate) fn reduce_split_count(
    device_index: usize,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Option<usize> {
    // CudaDevice::new is a zero-cost index wrapper; profile() reads the cached
    // profile, so this is an atomic load rather than a driver query.
    let compute_units = CudaDevice::new(device_index).profile().compute_units as usize;
    reduce_split_for_units(compute_units, outer, reduce, inner)
}

/// The split rule itself, separated from the device query so it is testable
/// without a device.
#[inline]
fn reduce_split_for_units(
    compute_units: usize,
    outer: usize,
    reduce: usize,
    inner: usize,
) -> Option<usize> {
    let outputs = outer.saturating_mul(inner);
    // An unknown profile reports zero compute units. The target is then zero,
    // no grid underfills it, and every shape keeps the single-stage launch.
    let target_blocks = compute_units.saturating_mul(REDUCE_SPLIT_MIN_WAVES);
    if outputs == 0 || outputs >= target_blocks {
        return None;
    }

    // Every chunk must still hold one element per thread of a full block. Below
    // that the threads of stage 1 idle and the second launch costs more than the
    // widened grid returns.
    let max_splits = reduce / BLOCK_SIZE as usize;
    if max_splits < 2 {
        return None;
    }
    let target_splits = target_blocks.div_ceil(outputs).min(max_splits);
    if target_splits < 2 {
        return None;
    }

    // Largest exact divisor at or below the target.
    let floor = target_splits
        .saturating_sub(REDUCE_SPLIT_DIVISOR_SCAN)
        .max(2);
    (floor..=target_splits)
        .rev()
        .find(|&splits| reduce.is_multiple_of(splits))
}

#[cfg(test)]
mod tests {
    use super::*;

    const BLOCK: usize = BLOCK_SIZE as usize;

    #[test]
    fn unknown_profile_never_splits() {
        assert_eq!(reduce_split_for_units(0, 1, 1_000_000, 1), None);
    }

    #[test]
    fn grid_that_already_fills_the_device_is_left_alone() {
        // 1024 outputs against a 2-wave target of 200 blocks.
        assert_eq!(reduce_split_for_units(100, 1024, 1024, 1), None);
    }

    #[test]
    fn whole_tensor_reduction_splits_into_an_exact_divisor() {
        let splits = reduce_split_for_units(100, 1, 1_000_000, 1).expect("1-D sum must split");
        assert!(1_000_000usize.is_multiple_of(splits), "splits={splits}");
        assert!(splits <= 200, "splits={splits} exceeds the 2-wave target");
        assert!(
            1_000_000 / splits >= BLOCK,
            "chunk shorter than a block: splits={splits}"
        );
    }

    #[test]
    fn short_reduce_axis_is_not_worth_splitting() {
        // Two full blocks is the smallest axis that can be cut in two.
        assert_eq!(reduce_split_for_units(100, 1, 2 * BLOCK - 1, 1), None);
        assert!(reduce_split_for_units(100, 1, 2 * BLOCK, 1).is_some());
    }

    #[test]
    fn prime_reduce_axis_falls_back_to_single_stage() {
        // 1000003 is prime, so no divisor exists at any target.
        assert_eq!(reduce_split_for_units(100, 1, 1_000_003, 1), None);
    }

    #[test]
    fn chosen_split_never_starves_a_chunk() {
        for reduce in [BLOCK * 2, BLOCK * 7, 4096, 65_536, 1_048_576] {
            for outputs in [1usize, 3, 17, 64] {
                if let Some(splits) = reduce_split_for_units(100, outputs, reduce, 1) {
                    assert!(reduce.is_multiple_of(splits), "{reduce} % {splits} != 0");
                    assert!(
                        reduce / splits >= BLOCK,
                        "reduce={reduce} splits={splits} leaves a short chunk"
                    );
                    assert!(splits >= 2);
                }
            }
        }
    }

    #[test]
    fn zero_sized_output_is_left_alone() {
        assert_eq!(reduce_split_for_units(100, 0, 1_000_000, 4), None);
        assert_eq!(reduce_split_for_units(100, 4, 1_000_000, 0), None);
    }
}
