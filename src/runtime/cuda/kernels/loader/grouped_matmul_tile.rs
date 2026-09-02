//! Tile selection and WMMA eligibility for the grouped GEMM launchers.
//!
//! `grouped_matmul.rs` owns the launchers; this module owns the decisions
//! that pick what they launch — the tiled-core tile/suffix/block-dim choice
//! and the `use_wmma_grouped` gate that routes between the tiled and WMMA
//! paths. Split out to mirror `matmul_wmma.rs` / `matmul_wmma_tile.rs`, where
//! the dense path already separates launchers from selection.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::traits::profile::DeviceCaps;

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
pub(super) fn use_wmma_grouped(dtype: DType, caps: DeviceCaps, n: usize, k: usize) -> bool {
    let dtype_ok = match dtype {
        DType::F16 => caps.f16_mma,
        DType::BF16 => caps.bf16,
        _ => false,
    };
    dtype_ok && n.is_multiple_of(16) && k.is_multiple_of(16)
}

/// Tile-selection row hint: the ceiling average rows per group.
///
/// What a grouped tile's row dimension must cover is ONE group's row count,
/// not the sum across groups — a grouped GEMM launches one GEMM per group,
/// and a block only ever computes rows for the group it was assigned. The
/// host cannot see individual group counts: they live in `offsets` in device
/// memory, and reading them back would force a stream sync just to pick a
/// tile. The average is the best cheap signal available without one.
/// Rounds up rather than down — a floor average can under-round and nudge
/// selection toward a smaller tile than the groups actually warrant, while a
/// ceiling average only ever biases toward the larger tile.
///
/// This is a PERFORMANCE hint only. Correctness comes from `grid.y` being
/// sized off `total_rows` (not this hint) in both launch-geometry helpers,
/// and from each kernel's per-group bounds check against its own
/// device-resident `count` — neither of which this value feeds into.
#[inline]
pub(super) fn grouped_row_hint(total_rows: usize, num_groups: usize) -> usize {
    if num_groups == 0 {
        return total_rows;
    }
    total_rows.div_ceil(num_groups)
}

/// Tile the grouped kernel is instantiated for, as `(BM, BN, suffix)`.
///
/// Two independent questions, one tile: N picks between the two shared-memory
/// footprints the same way the dense F32 path does — 128×128 needs a wide
/// output to be worth it, 64×64 wastes far less on a narrow one. `row_hint`
/// (the average rows per group, see [`grouped_row_hint`]) picks the same way
/// on M — a wide-N group that only has 64 rows to give still gets the 64-row
/// tile, since the 128-row tile would compute rows it never stores.
pub(super) fn grouped_tile(row_hint: usize, n: usize) -> (usize, usize, &'static str) {
    if n >= 128 && row_hint >= 128 {
        (128, 128, "128x128x8_8x8")
    } else {
        (64, 64, "64x64x32_8x4")
    }
}

/// Threads per block for a tile, matching the `extern "C"` instantiations:
/// `(BN / TN, BM / TM, 1)`.
pub(super) fn grouped_block_dim(suffix: &str) -> (u32, u32, u32) {
    match suffix {
        "128x128x8_8x8" => (16, 16, 1),
        _ => (16, 8, 1),
    }
}

/// Kernel-name dtype suffix. The core accumulates in F32 for all of these.
pub(super) fn grouped_dtype_suffix(dtype: DType) -> Result<&'static str> {
    match dtype {
        DType::F32 => Ok("f32"),
        DType::F16 => Ok("f16"),
        DType::BF16 => Ok("bf16"),
        other => Err(Error::Internal(format!(
            "grouped matmul supports F32/F16/BF16, got {other:?}"
        ))),
    }
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
        assert_eq!(grouped_tile(128, 4096).2, "128x128x8_8x8");
    }

    #[test]
    fn narrow_output_takes_the_small_tile() {
        assert_eq!(grouped_tile(128, 48).2, "64x64x32_8x4");
    }

    #[test]
    fn many_small_groups_with_wide_n_take_the_small_tile() {
        // 32 groups of 64 rows each: N is wide enough for the large tile but
        // each group only has 64 rows to give it, so the small tile wins.
        let hint = grouped_row_hint(32 * 64, 32);
        assert_eq!(grouped_tile(hint, 4096).2, "64x64x32_8x4");
    }

    #[test]
    fn few_large_groups_with_wide_n_take_the_large_tile() {
        // 2 groups of 256 rows each: both N and the per-group row count
        // clear the large-tile threshold.
        let hint = grouped_row_hint(2 * 256, 2);
        assert_eq!(grouped_tile(hint, 4096).2, "128x128x8_8x8");
    }

    // ---- grouped_row_hint ----

    #[test]
    fn hint_exact_division() {
        assert_eq!(grouped_row_hint(256, 4), 64);
    }

    #[test]
    fn hint_ragged_division_rounds_up() {
        // 100 rows over 3 groups: floor would give 33, which under-covers a
        // 34-row group; ceiling gives 34.
        assert_eq!(grouped_row_hint(100, 3), 34);
    }

    #[test]
    fn hint_single_group_is_total_rows() {
        assert_eq!(grouped_row_hint(777, 1), 777);
    }

    #[test]
    fn hint_zero_groups_falls_back_to_total_rows() {
        assert_eq!(grouped_row_hint(512, 0), 512);
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
