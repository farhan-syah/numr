// Backend parity for CUDA's tiled `strided_transpose` kernel
// (`src/runtime/cuda/kernels/strided_transpose.rs`), the fast path
// `Tensor::contiguous()` takes on a permuted/transposed view when
// `TransposePlan::detect` recognizes it. Everything `detect` declines falls
// back to the general `strided_copy` kernel.
//
// This is pure data movement — no arithmetic — so CPU and CUDA must match
// EXACTLY, not within a tolerance. `assert_tensor_allclose_tol` with
// `rtol = atol = 0.0` gives bit-exact float comparison and, for integer
// dtypes, exact comparison regardless of the tolerance passed in (see its
// `compare_int_exact` arm in `tests/common/mod.rs`).
//
// Dtypes run under `DTypeDomain::AllNumeric` rather than `FloatsOnly`: the
// tiled kernel is instantiated per element WIDTH (1/2/4/8 bytes), not per
// dtype, so covering only floats would leave the 1-byte and 2-byte
// instantiations (I8/U8, I16/U16) completely untested. `AllNumeric` covers
// all four widths on stock features: I8/U8 (1), I16/U16 (2), F32/I32/U32 (4),
// F64/I64/U64 (8).

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, is_dtype_supported, parity_dtypes,
};

/// Values that vary with the flat index, not a constant or a short repeating
/// run: `i * 37 + 5` scrambles the index before reducing it mod a prime (97)
/// that shares no factor with any tile/shape constant used below (32, 64,
/// 96, ...), so a swapped stride does not accidentally land on an equal
/// value. Range `0..97` fits every dtype under test, including I8 (`0..127`).
fn transpose_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 37 + 5) % 97) as f64).collect()
}

/// Builds `base_shape` contiguous, permutes by `perm`, materializes with
/// `.contiguous()`, and asserts CPU and CUDA agree EXACTLY.
fn assert_contiguous_permute_parity(label: &str, base_shape: &[usize], perm: &[usize]) {
    let numel: usize = base_shape.iter().product();
    let data = transpose_input(numel);

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_base = tensor_from_f64(&data, base_shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor failed for {label} [{dtype:?}]: {e}"));
        let cpu_view = cpu_base
            .permute(perm)
            .unwrap_or_else(|e| panic!("CPU permute failed for {label} [{dtype:?}]: {e}"));
        let cpu_result = cpu_view
            .contiguous()
            .unwrap_or_else(|e| panic!("CPU contiguous failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let base = tensor_from_f64(&data, base_shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor failed for {label} [{dtype:?}]: {e}"));
                let view = base
                    .permute(perm)
                    .unwrap_or_else(|e| panic!("CUDA permute failed for {label} [{dtype:?}]: {e}"));
                let result = view.contiguous().unwrap_or_else(|e| {
                    panic!("CUDA contiguous failed for {label} [{dtype:?}]: {e}")
                });
                assert_tensor_allclose_tol(
                    &result,
                    &cpu_result,
                    0.0,
                    0.0,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Both transposed extents (37, 45) are non-multiples of the 32-wide tile,
/// so every tile in the plane has a partial edge on both axes. This is the
/// partial-tile edge guard — the single most valuable case in this file.
///
/// Base `[3, 45, 37]` permuted `[0, 2, 1]` gives view shape `[3, 37, 45]`
/// with axis 1 (extent 37) source-contiguous and axis 2 (extent 45)
/// destination-contiguous: `TransposePlan::detect` reads this as
/// `batch=3, rows=37, cols=45` and takes the TILED path.
#[test]
fn strided_transpose_ragged_extents_parity() {
    assert_contiguous_permute_parity("strided_transpose_ragged_extents", &[3, 45, 37], &[0, 2, 1]);
}

/// `rows = 64`, `cols = 96` are both exact multiples of the 32-wide tile, so
/// every tile is a full tile and the edge-masking branch never fires.
///
/// Base `[2, 96, 64]` permuted `[0, 2, 1]` gives view shape `[2, 64, 96]`:
/// `detect` reads `batch=2, rows=64, cols=96` and takes the TILED path.
#[test]
fn strided_transpose_exact_tile_multiple_parity() {
    assert_contiguous_permute_parity(
        "strided_transpose_exact_tile_multiple",
        &[2, 96, 64],
        &[0, 2, 1],
    );
}

/// `rows = 31` (one under the tile edge) and `cols = 33` (one over it), so an
/// off-by-one in the row guard and an off-by-one in the column guard cannot
/// hide behind each other — each is wrong on a different one of these two
/// extents.
///
/// Base `[33, 31]` permuted `[1, 0]` gives view shape `[31, 33]`: `detect`
/// reads `batch=1, rows=31, cols=33` and takes the TILED path.
#[test]
fn strided_transpose_one_short_of_tile_parity() {
    assert_contiguous_permute_parity("strided_transpose_one_short_of_tile", &[33, 31], &[1, 0]);
}

/// Four live axes with two leading axes that collapse into one `batch > 1`,
/// so the kernel's `blockIdx.z` batch-stride indexing is exercised, not just
/// its degenerate `batch == 1` case.
///
/// Base `[2, 3, 40, 50]` permuted `[0, 1, 3, 2]` gives view shape
/// `[2, 3, 50, 40]`: axes 0 and 1 collapse into `batch=6` (their strides
/// follow the row-major progression), axis 2 (extent 50) is
/// source-contiguous, axis 3 (extent 40) is destination-contiguous. `detect`
/// reads `batch=6, rows=50, cols=40` and takes the TILED path.
#[test]
fn strided_transpose_batched_parity() {
    assert_contiguous_permute_parity("strided_transpose_batched", &[2, 3, 40, 50], &[0, 1, 3, 2]);
}

/// A plain 2-D transpose: no leading axes at all, so `TransposePlan`'s batch
/// collapse loop runs zero iterations and `batch` stays at its default 1.
///
/// Base `[80, 64]` permuted `[1, 0]` gives view shape `[64, 80]`: `detect`
/// reads `batch=1, rows=64, cols=80` and takes the TILED path.
#[test]
fn strided_transpose_2d_no_batch_parity() {
    assert_contiguous_permute_parity("strided_transpose_2d_no_batch", &[80, 64], &[1, 0]);
}

/// `[b, h, s, d] -> [b, s, h, d]`: axis `d` is innermost (stride 1) on BOTH
/// the source and the destination side, so `detect`'s `src_inner ==
/// dst_inner` check rejects this as already-coalesced and the view falls
/// back to the general `strided_copy` kernel. Correctness still matters on
/// that fallback path, so this guards it rather than the tiled kernel.
///
/// Base `[2, 3, 10, 8]` permuted `[0, 2, 1, 3]` gives view shape
/// `[2, 10, 3, 8]`, with axis 3 (extent 8) source-contiguous AND
/// destination-contiguous: `detect` returns `None` and the FALLBACK kernel
/// runs.
#[test]
fn strided_transpose_falls_back_coalesced_parity() {
    assert_contiguous_permute_parity(
        "strided_transpose_falls_back_coalesced",
        &[2, 3, 10, 8],
        &[0, 2, 1, 3],
    );
}

/// A leading-axis rotation, which reaches the tiled path only via the axis
/// MERGE in `collapse_axes`: for a view of shape `[C, A, B]` with strides
/// `[1, B*C, C]`, the trailing pair is one contiguous run and collapses to a
/// single axis, leaving an ordinary 2-D transpose. Without merging this falls
/// back, because an axis sits between the two transposed ones.
#[test]
fn strided_transpose_merged_rotation_parity() {
    assert_contiguous_permute_parity(
        "strided_transpose_merged_rotation",
        &[40, 3, 50],
        &[2, 0, 1],
    );
}

/// The merged rotation again, with both post-merge extents non-multiples of 32
/// (rows 37, cols 3*53) so the partial-tile guards run at an edge produced by
/// the merge rather than by an original axis extent.
#[test]
fn strided_transpose_merged_rotation_ragged_parity() {
    assert_contiguous_permute_parity(
        "strided_transpose_merged_rotation_ragged",
        &[53, 3, 37],
        &[2, 0, 1],
    );
}

/// A full axis reversal, which `collapse_axes` cannot merge — the strides
/// ascend with axis index, so no adjacent pair forms one run. It must fall back
/// to the general kernel and still be correct. This pins the documented limit
/// of the tiled path so a later change cannot quietly claim to tile it.
#[test]
fn strided_transpose_full_reversal_falls_back_parity() {
    assert_contiguous_permute_parity(
        "strided_transpose_full_reversal",
        &[8, 6, 5, 7],
        &[3, 2, 1, 0],
    );
}
