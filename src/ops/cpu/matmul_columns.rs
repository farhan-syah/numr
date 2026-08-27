//! Splitting a CPU matmul over the N dimension (output columns).
//!
//! # Why this exists
//!
//! With the tiered `copy_strided` and the transposed-B packing in place, a
//! profiled VoxCPM2 decode spends 96.5% of its instructions inside the tiled
//! f32 matmul — 45.0% `microkernel_loop_f32::<16>`, 25.8% `tiled_loop_f32::<16>`,
//! 25.7% `microkernel_6x16_f32` — and no measurable time copying. The process
//! still held 124% CPU on a 24-thread machine, because the only parallel axis
//! in [`super::matmul`] is the batch one and the dominant shapes there are
//! single-batch (`[22, 1024] × [1024, 4096]` and friends). One batch, one
//! thread. Splitting the columns is what fills the machine on those shapes.
//!
//! The split lives at the ops layer rather than in the kernel because it needs
//! the client: the pool from `install_parallelism`. Kernels are free functions
//! with no client handle.
//!
//! # The split is environment-free
//!
//! A column split moves the N-block boundaries the tiled kernel sees, so an
//! output element near a boundary can be produced by a different microkernel
//! variant (full 2×NR block, single block, scalar edge) than the unsplit call
//! would use, and its float accumulation lands in a different order. Results
//! move in the last ulp.
//!
//! That is why **nothing about the machine may reach these boundaries**. The
//! first version of this module sized the chunks as `n / thread_count`, and the
//! same VoxCPM2 sentence then decoded to different audio on different pool
//! sizes: at `RAYON_NUM_THREADS=1` the speech said "dah cuba", at 24 it said
//! "datuh bul" — same model, same seed, same input. Chunk boundaries are now a
//! pure function of the problem shape `(batch_size, m, n, k)`, so a 4-core
//! laptop and a 24-core workstation run the identical arithmetic and only the
//! scheduling differs.
//!
//! The purity has to cover *whether* to split, not just how wide the chunks
//! are: a run that takes the unsplit path is a run with different boundaries.
//! So [`column_chunk_count`] consults neither `rayon::current_num_threads()`
//! nor the client's pool size, and a one-thread pool still walks the same chunk
//! list — rayon simply runs it in sequence. Do not reintroduce a thread count
//! here, in either role.
//!
//! Two further properties are preserved rather than traded:
//!
//! - Accumulation order over `k` is untouched. The `k` blocking in
//!   `tiled_loop_f32` does not depend on `n`, so every chunk sums a given dot
//!   product in exactly the order the unsplit call does.
//! - Every chunk stays on the tiled path. `min_tiled_columns` is the chunk
//!   floor, so a chunk never falls back to the small-matrix kernel — which is
//!   what keeps a transposed B and a materialized B agreeing bit for bit, the
//!   property `matmul_bt_matches_contiguous` gates on.

use crate::dtype::Element;
use crate::ops::Kernel;
use crate::runtime::cpu::kernels::matmul_bt_kernel;
use crate::runtime::cpu::kernels::simd::matmul::min_tiled_columns;
use crate::runtime::cpu::{CpuClient, CpuRuntime};

/// Output columns per chunk — fixed, never derived from the thread count.
///
/// 128 columns:
///
/// - Clear the `min_tiled_columns` floor on the shapes that matter. The hot
///   decode shape `[22, 1024] × [1024, 4096]` needs 94, so 128 keeps every
///   chunk on the tiled kernel without the floor having to raise it.
/// - Fill a many-core machine at the hot shape: `4096 / 128` is 32 chunks, more
///   units than a 24-thread pool has workers, so no worker idles waiting for a
///   long final chunk.
/// - Divide `NC` (512), the tiled loop's own L3 column block, and are a
///   multiple of both `NR` widths (8 for AVX2, 16 for AVX-512). Where the width
///   divides `n`, every chunk boundary then lands on a block boundary the
///   unsplit call already had.
///
/// Narrower would re-pack A more often for no extra occupancy; wider would
/// leave a 24-thread pool short of units at `n = 4096`.
const COLUMN_CHUNK_WIDTH: usize = 128;

/// How many column chunks to split this matmul into, or `None` to leave the
/// batch axis in charge.
///
/// The answer depends on `(batch_size, m, n, k)` and nothing else — see the
/// module docs for why the machine must not enter this decision.
///
/// # Choosing the axis
///
/// The two axes are never nested — one rayon level only. The batch axis offers
/// exactly `batch_size` independent units, the column axis `n / width`. Take
/// whichever offers more, which is the columns on every single-batch decode
/// shape. When columns win and `batch_size > 1`, batches run in sequence and
/// each one gets the whole pool for its columns.
///
/// Comparing the two unit counts is what replaces the old
/// `batch_size >= threads` test: it picks the wider axis just as well, and it
/// does so without letting the pool size decide which arithmetic runs.
///
/// The chunk width is `max(128, min_tiled_columns(m, k))` — wide enough to pay
/// for a thread, and wide enough that every chunk stays on the tiled kernel.
/// `min_tiled_columns` returns `usize::MAX` for a degenerate `m * k`, which
/// divides down to zero chunks and leaves the split off.
pub(crate) fn column_chunk_count(batch_size: usize, m: usize, n: usize, k: usize) -> Option<usize> {
    let width = COLUMN_CHUNK_WIDTH.max(min_tiled_columns(m, k));
    let chunks = n / width;
    (chunks >= 2 && chunks > batch_size).then_some(chunks)
}

/// Run `f(col_start, col_count)` over `chunks` disjoint column ranges of `[0, n)`.
///
/// Ranges are balanced to within one column, so every range is at least
/// `n / chunks` wide — that is what lets [`column_chunk_count`] guarantee a
/// floor, which uniform striding cannot do (a stride leaves a remainder chunk
/// of arbitrary width). Each range writes `out + col_start` with the full
/// output row stride, so destinations are disjoint and nothing synchronises.
fn for_each_column_chunk<F>(client: &CpuClient, n: usize, chunks: usize, f: F)
where
    F: Fn(usize, usize) + Send + Sync,
{
    use rayon::prelude::*;

    let base = n / chunks;
    let remainder = n % chunks;

    client.install_parallelism(|| {
        (0..chunks).into_par_iter().for_each(|chunk| {
            let start = chunk * base + chunk.min(remainder);
            let width = base + usize::from(chunk < remainder);
            f(start, width);
        });
    });
}

/// Column-parallel `C = A @ B` for a contiguous row-major B.
///
/// Chunk `[j0, j1)` reads `b + j0` with the unchanged leading dimension `ldb`
/// and writes `out + j0` with the unchanged `ldc`.
///
/// # Safety
/// Same as [`Kernel::matmul`]: pointers valid for the given dimensions and
/// leading dimensions, `out` not aliasing `a` or `b`.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_columns<T: Element>(
    client: &CpuClient,
    a: *const T,
    b: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    chunks: usize,
) {
    // Pointers cross the thread boundary as addresses: raw pointers are not
    // Send, and the ranges they reach are disjoint by construction.
    let elem = std::mem::size_of::<T>();
    let (a_addr, b_addr, out_addr) = (a as usize, b as usize, out as usize);

    for_each_column_chunk(client, n, chunks, |col_start, cols| unsafe {
        <CpuClient as Kernel<CpuRuntime>>::matmul::<T>(
            client,
            a_addr as *const T,
            (b_addr + col_start * elem) as *const T,
            (out_addr + col_start * elem) as *mut T,
            m,
            cols,
            k,
            lda,
            ldb,
            ldc,
        );
    });
}

/// Column-parallel `C = A @ B` for a transposed B held as a contiguous
/// `[N, K]` buffer.
///
/// Columns of the logical `[K, N]` operand are rows of that buffer, so chunk
/// `[j0, j1)` reads `b_nk + j0 * k` — still contiguous, still row stride `k`.
///
/// # Safety
/// Same as [`matmul_bt_kernel`]: `a` valid for `m * k` elements, `b_nk` for
/// `n * k`, `out` for `m * ldc` writes, and `out` aliasing neither input.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn matmul_bt_columns<T: Element>(
    client: &CpuClient,
    a: *const T,
    b_nk: *const T,
    out: *mut T,
    m: usize,
    n: usize,
    k: usize,
    ldc: usize,
    chunks: usize,
) {
    let elem = std::mem::size_of::<T>();
    let (a_addr, b_addr, out_addr) = (a as usize, b_nk as usize, out as usize);

    for_each_column_chunk(client, n, chunks, |col_start, cols| unsafe {
        matmul_bt_kernel::<T>(
            a_addr as *const T,
            (b_addr + col_start * k * elem) as *const T,
            (out_addr + col_start * elem) as *mut T,
            m,
            cols,
            k,
            ldc,
        );
    });
}

#[cfg(test)]
mod tests {
    use crate::ops::MatmulOps;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use crate::tensor::Tensor;

    /// The shape that motivated the split: the local DiT decode step.
    const DIT_M: usize = 22;
    const DIT_K: usize = 1024;

    fn values(len: usize, seed: usize) -> Vec<f32> {
        (0..len)
            .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.004 - 0.5)
            .collect()
    }

    /// A client pinned to one thread walks the same chunk list as a full pool
    /// — rayon just runs it in sequence — so this is the single-threaded
    /// execution of this module, not a different code path.
    fn serial_client(device: &CpuDevice) -> CpuClient {
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None))
    }

    fn assert_close_rel(got: &[f32], want: &[f32], relative: f32, label: &str) {
        assert_eq!(got.len(), want.len(), "{label}: length");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let tolerance = relative * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tolerance,
                "{label}: element {i} differs ({g} vs {w})"
            );
        }
    }

    /// One thread and many threads run identical arithmetic, so this is a loose
    /// gate on a property `tests/matmul_column_determinism.rs` pins exactly.
    fn assert_close(got: &[f32], want: &[f32], label: &str) {
        assert_close_rel(got, want, 1e-5, label);
    }

    /// Naive `C = A @ B` with an f64 accumulator — independent of every kernel.
    fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a[i * k + p]) * f64::from(b[p * n + j]);
                }
                out[i * n + j] = acc as f32;
            }
        }
        out
    }

    /// Same product, with B held transposed as a contiguous `[N, K]` buffer.
    fn reference_matmul_bt(a: &[f32], b_nk: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += f64::from(a[i * k + p]) * f64::from(b_nk[j * k + p]);
                }
                out[i * n + j] = acc as f32;
            }
        }
        out
    }

    /// `A[batch.., m, k] @ B[k, n]` with a contiguous B, run in parallel and
    /// serially.
    fn contiguous_both_ways(a_shape: &[usize], b_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let device = CpuDevice::new();
        let parallel = CpuClient::new(device.clone());
        let serial = serial_client(&device);

        let a_len: usize = a_shape.iter().product();
        let b_len: usize = b_shape.iter().product();
        let a = Tensor::<CpuRuntime>::from_slice(&values(a_len, 1), a_shape, &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&values(b_len, 2), b_shape, &device).unwrap();

        (
            parallel.matmul(&a, &b).unwrap().to_vec::<f32>(),
            serial.matmul(&a, &b).unwrap().to_vec::<f32>(),
        )
    }

    /// `A[batch.., m, k] @ W[n, k]^T`, the transposed-B path, run in parallel
    /// and serially.
    fn transposed_both_ways(a_shape: &[usize], w_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
        let device = CpuDevice::new();
        let parallel = CpuClient::new(device.clone());
        let serial = serial_client(&device);

        let a_len: usize = a_shape.iter().product();
        let w_len: usize = w_shape.iter().product();
        let a = Tensor::<CpuRuntime>::from_slice(&values(a_len, 1), a_shape, &device).unwrap();
        let w = Tensor::<CpuRuntime>::from_slice(&values(w_len, 2), w_shape, &device).unwrap();

        let last = w_shape.len() - 1;
        let b_view = w.transpose((last - 1) as isize, last as isize).unwrap();

        (
            parallel.matmul(&a, &b_view).unwrap().to_vec::<f32>(),
            serial.matmul(&a, &b_view).unwrap().to_vec::<f32>(),
        )
    }

    #[test]
    fn contiguous_matches_serial_at_dit_shape() {
        let (par, ser) = contiguous_both_ways(&[DIT_M, DIT_K], &[DIT_K, 4096]);
        assert_close(&par, &ser, "contiguous 22x1024x4096");
    }

    #[test]
    fn transposed_matches_serial_at_dit_shape() {
        let (par, ser) = transposed_both_ways(&[DIT_M, DIT_K], &[4096, DIT_K]);
        assert_close(&par, &ser, "transposed 22x1024x4096");
    }

    /// A prime `n` is a multiple of neither the chunk width nor the 16-wide
    /// register block, so both remainder paths run.
    #[test]
    fn prime_n_matches_serial() {
        let (par, ser) = contiguous_both_ways(&[DIT_M, DIT_K], &[DIT_K, 1021]);
        assert_close(&par, &ser, "contiguous n=1021");

        let (par, ser) = transposed_both_ways(&[DIT_M, DIT_K], &[1021, DIT_K]);
        assert_close(&par, &ser, "transposed n=1021");
    }

    /// `n` just over two chunk widths splits into the fewest chunks the split
    /// allows, with an uneven remainder. Checked against the reference too, so
    /// a split that actually happens is pinned to the right answer and not only
    /// to the serial path.
    #[test]
    fn n_just_above_floor_matches_serial() {
        let n = 269;
        let a = values(DIT_M * DIT_K, 1);
        let b = values(DIT_K * n, 2);

        let (par, ser) = contiguous_both_ways(&[DIT_M, DIT_K], &[DIT_K, n]);
        assert_close(&par, &ser, "contiguous n=269 serial");
        assert_close_rel(
            &par,
            &reference_matmul(&a, &b, DIT_M, n, DIT_K),
            1e-3,
            "contiguous n=269 reference",
        );

        let (par, ser) = transposed_both_ways(&[DIT_M, DIT_K], &[n, DIT_K]);
        assert_close(&par, &ser, "transposed n=269 serial");
        assert_close_rel(
            &par,
            &reference_matmul_bt(&a, &b, DIT_M, n, DIT_K),
            1e-3,
            "transposed n=269 reference",
        );
    }

    /// `n` under one chunk width takes the unsplit call, so the answer is
    /// checked against an independent reference rather than against the same
    /// path.
    #[test]
    fn n_below_floor_takes_serial_path() {
        let n = 93;
        let a = values(DIT_M * DIT_K, 1);
        let b = values(DIT_K * n, 2);

        let (par, ser) = contiguous_both_ways(&[DIT_M, DIT_K], &[DIT_K, n]);
        assert_close(&par, &ser, "contiguous n=93 serial");
        assert_close_rel(
            &par,
            &reference_matmul(&a, &b, DIT_M, n, DIT_K),
            1e-3,
            "contiguous n=93 reference",
        );

        let (par, ser) = transposed_both_ways(&[DIT_M, DIT_K], &[n, DIT_K]);
        assert_close(&par, &ser, "transposed n=93 serial");
        assert_close_rel(
            &par,
            &reference_matmul_bt(&a, &b, DIT_M, n, DIT_K),
            1e-3,
            "transposed n=93 reference",
        );
    }

    /// `n` off the register-block width, with `m` large enough that the floor
    /// is the flat `COLUMN_CHUNK_WIDTH` (128) rather than the tiled minimum.
    #[test]
    fn n_off_register_block_matches_serial() {
        let (par, ser) = contiguous_both_ways(&[130, 512], &[512, 519]);
        assert_close(&par, &ser, "contiguous 130x512x519");

        let (par, ser) = transposed_both_ways(&[130, 512], &[519, 512]);
        assert_close(&par, &ser, "transposed 130x512x519");
    }

    /// The chunk count is a pure function of the shape, and every chunk it
    /// asks for clears the tiled-path floor.
    #[test]
    fn chunk_count_depends_only_on_shape() {
        use super::{COLUMN_CHUNK_WIDTH, column_chunk_count};
        use crate::runtime::cpu::kernels::simd::matmul::min_tiled_columns;

        // The hot decode shape: 4096 columns over 128-wide chunks.
        assert_eq!(column_chunk_count(1, DIT_M, 4096, DIT_K), Some(32));
        // A width the chunk size does not divide keeps the floor-division count.
        assert_eq!(column_chunk_count(1, DIT_M, 1021, DIT_K), Some(7));
        // Under two chunks there is nothing to split.
        assert_eq!(column_chunk_count(1, DIT_M, 200, DIT_K), None);
        // The batch axis is wider here, so it stays in charge.
        assert_eq!(column_chunk_count(8, DIT_M, 1021, DIT_K), None);
        // A degenerate shape has no tiled chunk width at all.
        assert_eq!(column_chunk_count(1, 0, 4096, DIT_K), None);

        // Balanced chunks are never narrower than the floor.
        let floor = COLUMN_CHUNK_WIDTH.max(min_tiled_columns(DIT_M, DIT_K));
        for n in [269usize, 1021, 4096] {
            let chunks = column_chunk_count(1, DIT_M, n, DIT_K).unwrap();
            assert!(n / chunks >= floor, "n={n}: narrowest chunk under floor");
        }
    }

    /// Batched: more column chunks than batches, so columns win and batches run
    /// in sequence.
    #[test]
    fn batched_matches_serial() {
        let (par, ser) = contiguous_both_ways(&[2, DIT_M, DIT_K], &[DIT_K, 1021]);
        assert_close(&par, &ser, "contiguous batched n=1021");

        let (par, ser) = transposed_both_ways(&[2, DIT_M, DIT_K], &[1021, DIT_K]);
        assert_close(&par, &ser, "transposed batched n=1021");
    }
}
