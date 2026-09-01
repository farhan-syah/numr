//! Tiled-transpose fast path for strided-to-contiguous copies.
//!
//! The general `strided_copy` kernel assigns one destination element per thread.
//! That is fine when the view's innermost runs stay contiguous in the source
//! (a `narrow`, say), but on a permuted view every warp read scatters across as
//! many sectors as it has lanes.
//!
//! Any permutation that materializes to contiguous memory is, at the
//! memory-access level, a 2-D transpose of two axes once the rest collapse into
//! a batch dimension: the axis whose *source* stride is 1, and the axis whose
//! *destination* stride is 1. [`TransposePlan::detect`] recognizes that
//! structure and [`launch_strided_transpose`] runs it through a shared-memory
//! tile, so both the reads and the writes are coalesced.
//!
//! Detection is deliberately conservative. Every shape it declines falls back to
//! the general kernel, which is always correct.

use cudarc::driver::PushKernelArg;
use cudarc::driver::safe::{CudaContext, CudaStream};
use std::sync::Arc;

use super::loader::{MAX_GRID_DIM_YZ, get_kernel_function, get_or_load_module, launch_config};
use super::strided_copy::MAX_DIMS;
use crate::error::{Error, Result};

/// Module name for the tiled transpose kernels
pub const STRIDED_TRANSPOSE_MODULE: &str = "strided_transpose";

/// Tile edge, in elements. Must match `STRIDED_TRANSPOSE_TILE_DIM` in
/// `strided_transpose.cu`: the grid is sized from it here while the kernel
/// bounds its shared tile by it, so a mismatch leaves output uncomputed instead
/// of failing to build.
const TILE_DIM: usize = 32;

/// Thread rows per block. Must match `STRIDED_TRANSPOSE_BLOCK_ROWS` in
/// `strided_transpose.cu`: the block is sized from it here while the kernel
/// strides its tile loop by it. `32 x 8` threads, each thread covering
/// `TILE_DIM / BLOCK_ROWS` tile rows, is the conventional shape for this kernel.
const BLOCK_ROWS: u32 = 8;

/// A view whose materialization is a batched 2-D transpose.
///
/// The destination is contiguous `[batch, rows, cols]` and the source element
/// for `(b, r, c)` sits at `b * batch_stride + r + c * col_stride` elements past
/// `src_byte_offset`. `rows` is the extent of the axis with source stride 1;
/// `cols` is the extent of the axis with destination stride 1.
pub struct TransposePlan {
    batch: u32,
    rows: u32,
    cols: u32,
    batch_stride: i64,
    col_stride: i64,
    elem_size: usize,
}

/// Merge adjacent axes whose source strides already stand in the destination's
/// row-major relation, iterating to a fixed point because one merge can expose
/// another. Returns `None` if a merged extent or stride overflows.
///
/// The merge is offset-preserving: if `strides[d] == strides[d + 1] * shape[d + 1]`
/// then for every `(i, j)` in the pair's index space
/// `i * strides[d] + j * strides[d + 1] == (i * shape[d + 1] + j) * strides[d + 1]`,
/// and `i * shape[d + 1] + j` is exactly the destination's row-major index over
/// the pair. The merged axis therefore generates the same source offsets in the
/// same destination order as the pair it replaces.
fn collapse_axes(
    ext: &[usize],
    strd: &[i64],
    n: usize,
) -> Option<([usize; MAX_DIMS], [i64; MAX_DIMS], usize)> {
    let mut e = [0usize; MAX_DIMS];
    let mut s = [0i64; MAX_DIMS];
    e[..n].copy_from_slice(&ext[..n]);
    s[..n].copy_from_slice(&strd[..n]);
    let mut len = n;

    loop {
        let mut merged = false;
        let mut d = 0;
        while d + 1 < len {
            let span = s[d + 1].checked_mul(i64::try_from(e[d + 1]).ok()?)?;
            if s[d] == span {
                e[d] = e[d].checked_mul(e[d + 1])?;
                s[d] = s[d + 1];
                for i in d + 1..len - 1 {
                    e[i] = e[i + 1];
                    s[i] = s[i + 1];
                }
                len -= 1;
                merged = true;
                // Retry at `d`: the widened axis may now merge with its new neighbour.
            } else {
                d += 1;
            }
        }
        if !merged {
            break;
        }
    }

    Some((e, s, len))
}

impl TransposePlan {
    /// Recognize a transposing view, or return `None` to use the general kernel.
    ///
    /// Axes of extent 1 are dropped, adjacent axes that are already one run are
    /// merged, and the result is matched against [`TransposePlan::from_axes`].
    ///
    /// Returns `None` for: element sizes the kernel is not instantiated for,
    /// more than [`MAX_DIMS`] dimensions, fewer strides than axes, fewer than
    /// two axes of extent above 1, any negative stride, and anything
    /// [`TransposePlan::from_axes`] rejects on both the merged and the unmerged
    /// axis list.
    pub fn detect(shape: &[usize], strides: &[isize], elem_size: usize) -> Option<Self> {
        if !matches!(elem_size, 1 | 2 | 4 | 8)
            || shape.len() > MAX_DIMS
            || strides.len() < shape.len()
        {
            return None;
        }

        // Axes of extent 1 add no offset on either side, so they take no part in
        // the layout and are dropped before any of the reasoning below.
        let mut ext = [0usize; MAX_DIMS];
        let mut strd = [0i64; MAX_DIMS];
        let mut n = 0;
        for d in 0..shape.len() {
            if shape[d] > 1 {
                if strides[d] < 0 {
                    return None;
                }
                ext[n] = shape[d];
                strd[n] = i64::try_from(strides[d]).ok()?;
                n += 1;
            }
        }
        if n < 2 {
            return None;
        }

        // Merging only widens tiles, so it goes first. The unmerged list is
        // retried because a merge that pushes an extent past a grid limit must
        // not cost a plan the unmerged list would still have produced.
        collapse_axes(&ext, &strd, n)
            .and_then(|(e, s, len)| Self::from_axes(&e[..len], &s[..len], elem_size))
            .or_else(|| Self::from_axes(&ext[..n], &strd[..n], elem_size))
    }

    /// Match one axis list - extents and source strides in destination order,
    /// all of extent above 1 - against the batched 2-D transpose shape.
    ///
    /// Returns `None` for: fewer than two axes, no axis or more than one axis
    /// with source stride 1, a source-contiguous axis that is already the
    /// destination-contiguous one (reads are coalesced already), an axis sitting
    /// between the two transposed axes, leading axes whose strides do not
    /// collapse to one batch stride, and extents that overflow the kernel's
    /// index types or the grid's `y`/`z` limits.
    fn from_axes(ext: &[usize], strd: &[i64], elem_size: usize) -> Option<Self> {
        let n = ext.len();
        if n < 2 {
            return None;
        }

        // The destination is contiguous row-major, so its stride-1 axis is last.
        let dst_inner = n - 1;

        // Exactly one source-contiguous axis. Two would mean overlapping runs.
        let mut src_inner = None;
        for (d, &stride) in strd.iter().enumerate() {
            if stride == 1 {
                if src_inner.is_some() {
                    return None;
                }
                src_inner = Some(d);
            }
        }
        let src_inner = src_inner?;

        // Same axis: the source runs are already contiguous in the destination
        // order, so the general kernel's reads are coalesced.
        if src_inner == dst_inner {
            return None;
        }

        // `dst_inner` is the last axis, so `src_inner` precedes it. For the
        // destination to be `[batch, rows, cols]` the two must be adjacent; an
        // axis between them would sit between them in the destination layout too.
        if src_inner != n - 2 {
            return None;
        }

        let rows = ext[n - 2];
        let cols = ext[n - 1];
        let col_stride = strd[n - 1];

        // The leading axes collapse into one batch dimension only if their
        // source strides follow the destination's row-major progression scaled
        // by a single stride. Anything else is not a plain batched transpose.
        let mut batch = 1usize;
        let mut batch_stride = 0i64;
        if n > 2 {
            batch_stride = strd[n - 3];
            let mut expected = batch_stride;
            for d in (0..n - 2).rev() {
                if strd[d] != expected {
                    return None;
                }
                expected = expected.checked_mul(i64::try_from(ext[d]).ok()?)?;
                batch = batch.checked_mul(ext[d])?;
            }
        }

        // The kernel indexes extents as `unsigned int`.
        if rows > u32::MAX as usize || cols > u32::MAX as usize {
            return None;
        }
        // Row tiles ride the grid's `y` axis and the batch rides `z`, both
        // capped at MAX_GRID_DIM_YZ. `cols` is capped by the `u32` bound above,
        // which keeps its tile count well inside the `x` limit.
        if rows.div_ceil(TILE_DIM) > MAX_GRID_DIM_YZ as usize || batch > MAX_GRID_DIM_YZ as usize {
            return None;
        }

        Some(Self {
            batch: batch as u32,
            rows: rows as u32,
            cols: cols as u32,
            batch_stride,
            col_stride,
            elem_size,
        })
    }

    /// Kernel name for this plan's element width.
    fn kernel_name(&self) -> Result<&'static str> {
        match self.elem_size {
            1 => Ok("strided_transpose_b1"),
            2 => Ok("strided_transpose_b2"),
            4 => Ok("strided_transpose_b4"),
            8 => Ok("strided_transpose_b8"),
            other => Err(Error::Internal(format!(
                "strided_transpose has no kernel for a {} byte element",
                other
            ))),
        }
    }
}

/// Launch the tiled transpose kernel for a detected [`TransposePlan`].
///
/// # Safety
///
/// Same requirements as `launch_strided_copy`: both pointers must be valid
/// device memory on the stream's device, and the destination must have room for
/// `batch * rows * cols * elem_size` bytes.
pub unsafe fn launch_strided_transpose(
    context: &Arc<CudaContext>,
    stream: &CudaStream,
    device_index: usize,
    src_ptr: u64,
    dst_ptr: u64,
    plan: &TransposePlan,
    src_byte_offset: usize,
) -> Result<()> {
    let name = plan.kernel_name()?;

    unsafe {
        let module = get_or_load_module(context, device_index, STRIDED_TRANSPOSE_MODULE)?;
        let func = get_kernel_function(&module, name)?;

        // One block per tile of the `[rows, cols]` plane, one grid `z` slice per
        // batch element. `detect` has already bounded every extent.
        let grid = (
            (plan.cols as usize).div_ceil(TILE_DIM) as u32,
            (plan.rows as usize).div_ceil(TILE_DIM) as u32,
            plan.batch,
        );
        let block = (TILE_DIM as u32, BLOCK_ROWS, 1);
        // The tile is statically sized in the kernel, so no dynamic shared memory.
        let cfg = launch_config(grid, block, 0);

        let batch = plan.batch;
        let rows = plan.rows;
        let cols = plan.cols;
        let batch_stride = plan.batch_stride;
        let col_stride = plan.col_stride;
        let src_offset_u64 = src_byte_offset as u64;

        let mut builder = stream.launch_builder(&func);
        builder.arg(&src_ptr);
        builder.arg(&dst_ptr);
        builder.arg(&batch);
        builder.arg(&rows);
        builder.arg(&cols);
        builder.arg(&batch_stride);
        builder.arg(&col_stride);
        builder.arg(&src_offset_u64);

        builder.launch(cfg).map_err(|e| {
            Error::Internal(format!(
                "CUDA strided_transpose kernel launch failed: {:?}",
                e
            ))
        })?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{BLOCK_ROWS, TILE_DIM, TransposePlan, collapse_axes};

    /// Tile constants that appear in BOTH the kernel source and this launcher.
    /// The launcher sizes the grid and block from them and the kernel bounds its
    /// tile loops by them, so a mismatch does not fail to build - it silently
    /// leaves outputs uncomputed. Parse the kernel and check.
    fn kernel_define(source: &str, name: &str) -> usize {
        let needle = format!("#define {name} ");
        let line = source
            .lines()
            .find(|l| l.starts_with(&needle))
            .unwrap_or_else(|| panic!("{name} is not defined in the kernel source"));
        line[needle.len()..]
            .trim()
            .trim_end_matches('u')
            .parse()
            .unwrap_or_else(|e| panic!("{name} is not a plain integer literal: {e}"))
    }

    #[test]
    fn tile_dim_matches_the_kernel() {
        let source = include_str!("strided_transpose.cu");
        assert_eq!(
            kernel_define(source, "STRIDED_TRANSPOSE_TILE_DIM"),
            TILE_DIM,
            "strided_transpose.cu and strided_transpose.rs disagree on the tile edge"
        );
    }

    #[test]
    fn block_rows_matches_the_kernel() {
        let source = include_str!("strided_transpose.cu");
        assert_eq!(
            kernel_define(source, "STRIDED_TRANSPOSE_BLOCK_ROWS"),
            BLOCK_ROWS as usize,
            "strided_transpose.cu and strided_transpose.rs disagree on the block height"
        );
    }

    #[test]
    fn tile_dim_is_a_multiple_of_block_rows() {
        // Each thread walks TILE_DIM in BLOCK_ROWS steps; a remainder would skip
        // the tail rows of every tile.
        assert_eq!(TILE_DIM % BLOCK_ROWS as usize, 0);
    }

    /// Strides of a contiguous row-major tensor of `shape`.
    fn contiguous_strides(shape: &[usize]) -> Vec<isize> {
        let mut strides = vec![1isize; shape.len()];
        for d in (0..shape.len().saturating_sub(1)).rev() {
            strides[d] = strides[d + 1] * shape[d + 1] as isize;
        }
        strides
    }

    /// Permuted view of a contiguous `base` shape: `perm[i]` is the base axis
    /// that becomes view axis `i`.
    fn permuted(base: &[usize], perm: &[usize]) -> (Vec<usize>, Vec<isize>) {
        let base_strides = contiguous_strides(base);
        (
            perm.iter().map(|&p| base[p]).collect(),
            perm.iter().map(|&p| base_strides[p]).collect(),
        )
    }

    #[test]
    fn contiguous_view_falls_back() {
        let shape = [4usize, 8, 16];
        let strides = contiguous_strides(&shape);
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn narrow_view_with_contiguous_runs_falls_back() {
        // Rows of 16 taken from a 64-wide source: reads are already coalesced.
        let shape = [4usize, 16];
        let strides = [64isize, 1];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn two_dim_transpose_is_detected() {
        let (shape, strides) = permuted(&[64, 32], &[1, 0]);
        let plan = TransposePlan::detect(&shape, &strides, 4).expect("2-D transpose");
        assert_eq!((plan.batch, plan.rows, plan.cols), (1, 32, 64));
        assert_eq!(plan.col_stride, 32);
        assert_eq!(plan.batch_stride, 0);
    }

    #[test]
    fn batched_transpose_collapses_leading_axes() {
        // [b0, b1, h, w] -> swap the last two axes; b0/b1 collapse into a batch.
        let (shape, strides) = permuted(&[2, 3, 16, 8], &[0, 1, 3, 2]);
        let plan = TransposePlan::detect(&shape, &strides, 4).expect("batched transpose");
        assert_eq!((plan.batch, plan.rows, plan.cols), (6, 8, 16));
        assert_eq!(plan.col_stride, 8);
        assert_eq!(plan.batch_stride, 128);
    }

    #[test]
    fn batch_axis_between_the_transposed_axes_falls_back() {
        // The source-contiguous axis is not adjacent to the destination's, so
        // the destination is not [batch, rows, cols].
        let (shape, strides) = permuted(&[8, 4, 16], &[2, 1, 0]);
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn non_uniform_batch_strides_fall_back() {
        // Batch strides that do not follow the row-major progression.
        let shape = [2usize, 3, 4, 5];
        let strides = [1000isize, 7, 1, 40];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn missing_source_contiguous_axis_falls_back() {
        // Every stride above 1: a strided slice, not a transpose.
        let shape = [8usize, 8];
        let strides = [16isize, 2];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn unsupported_element_size_falls_back() {
        let (shape, strides) = permuted(&[64, 32], &[1, 0]);
        assert!(TransposePlan::detect(&shape, &strides, 3).is_none());
        assert!(TransposePlan::detect(&shape, &strides, 16).is_none());
    }

    #[test]
    fn negative_stride_falls_back() {
        let shape = [64usize, 32];
        let strides = [1isize, -64];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn unit_extent_axes_are_ignored() {
        // [1, 32, 1, 64] transposed on its two real axes.
        let shape = [1usize, 32, 1, 64];
        let strides = [0isize, 1, 0, 32];
        let plan = TransposePlan::detect(&shape, &strides, 2).expect("unit axes are inert");
        assert_eq!((plan.batch, plan.rows, plan.cols), (1, 32, 64));
        assert_eq!(plan.col_stride, 32);
    }

    #[test]
    fn broadcast_batch_axis_is_detected_with_a_zero_stride() {
        // A stride-0 batch axis reads the same plane repeatedly, which the
        // kernel's address math handles exactly as the general kernel does.
        let shape = [4usize, 16, 8];
        let strides = [0isize, 1, 16];
        let plan = TransposePlan::detect(&shape, &strides, 4).expect("broadcast batch");
        assert_eq!((plan.batch, plan.rows, plan.cols), (4, 16, 8));
        assert_eq!(plan.batch_stride, 0);
    }

    #[test]
    fn mixed_zero_and_real_batch_strides_fall_back() {
        let shape = [2usize, 4, 16, 8];
        let strides = [0isize, 128, 1, 16];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn rows_past_the_grid_y_limit_fall_back() {
        // More row tiles than the grid's y axis can hold.
        let rows = (super::MAX_GRID_DIM_YZ as usize + 1) * TILE_DIM;
        let shape = [rows, 4];
        let strides = [1isize, rows as isize];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn batch_past_the_grid_z_limit_falls_back() {
        let batch = super::MAX_GRID_DIM_YZ as usize + 1;
        let shape = [batch, 8, 4];
        let strides = [32isize, 1, 8];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn collapse_axes_merges_an_adjacent_run() {
        // strides[1] == strides[2] * shape[2] (65536 == 256 * 256), so axes 1
        // and 2 are one run of 65536 elements with stride 256.
        let (ext, strd, len) =
            collapse_axes(&[256, 256, 256], &[1, 65536, 256], 3).expect("no overflow");
        assert_eq!(len, 2);
        assert_eq!(&ext[..len], &[256, 65536]);
        assert_eq!(&strd[..len], &[1, 256]);
    }

    #[test]
    fn collapse_axes_rejects_a_stride_overflow() {
        assert!(collapse_axes(&[4, 4], &[i64::MAX, i64::MAX], 2).is_none());
    }

    #[test]
    fn collapse_axes_rejects_an_extent_overflow() {
        // The pair merges (2 == 1 * 2) but the merged extent does not fit.
        assert!(collapse_axes(&[usize::MAX, 2], &[2, 1], 2).is_none());
    }

    #[test]
    fn merged_axes_enable_a_leading_axis_permute() {
        // [256, 256, 256] permuted (2, 0, 1): the trailing two axes are one run,
        // leaving a plain 2-D transpose.
        let (shape, strides) = permuted(&[256, 256, 256], &[2, 0, 1]);
        assert_eq!(strides, vec![1, 65536, 256]);
        let plan = TransposePlan::detect(&shape, &strides, 4).expect("merged transpose");
        assert_eq!((plan.batch, plan.rows, plan.cols), (1, 256, 65536));
        assert_eq!(plan.col_stride, 256);
        assert_eq!(plan.batch_stride, 0);
    }

    #[test]
    fn unmergeable_strides_between_the_axes_fall_back() {
        // Same axis layout as the merged case, but strides[1] != strides[2] *
        // shape[2], so the pair is not one run and nothing collapses.
        let shape = [8usize, 8, 8];
        let strides = [1isize, 100, 8];
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn full_axis_reversal_falls_back() {
        // A 4-D reversal is a 4-D transpose, not a 2-D one: no adjacent pair
        // satisfies the merge relation, so no batch collapse exists.
        let (shape, strides) = permuted(&[64, 64, 64, 64], &[3, 2, 1, 0]);
        assert_eq!(strides, vec![1, 64, 4096, 262144]);
        assert!(TransposePlan::detect(&shape, &strides, 4).is_none());
    }

    #[test]
    fn merging_rewrites_an_already_tileable_plan_offset_identically() {
        // Batch stride 4 equals the row extent, so the batch and row axes are
        // one run: the same copy, expressed as a single wider transpose.
        let shape = [2usize, 4, 8];
        let strides = [4isize, 1, 8];
        let plan = TransposePlan::detect(&shape, &strides, 4).expect("merged batch");
        assert_eq!((plan.batch, plan.rows, plan.cols), (1, 8, 8));
        assert_eq!(plan.col_stride, 8);
    }
}
