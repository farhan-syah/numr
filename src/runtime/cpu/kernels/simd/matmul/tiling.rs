//! Cache-aware tiled matmul algorithm
//!
//! Implements BLIS-style 3-level blocking with:
//! - Thread-local packing buffers (no allocation on hot path)
//! - Beta=0/1 microkernel (no separate zero pass over output)
//! - Optimized pack_b with bulk copies for full NR blocks

use super::packing::{pack_a_f32, pack_a_f64, pack_b_f32, pack_b_f64, pack_b_t_f32, pack_b_t_f64};
use super::scalar::{microkernel_edge_f32, microkernel_edge_f64};
use super::{KC, MC, MR, NC};
use super::{
    call_microkernel_2x_f32, call_microkernel_2x_f64, call_microkernel_f32, call_microkernel_f64,
};
use crate::runtime::cpu::kernels::simd::SimdLevel;
use std::cell::RefCell;

// ---------------------------------------------------------------------------
// Thread-local packing buffers (avoids heap allocation on every matmul call)
// ---------------------------------------------------------------------------

thread_local! {
    static PACK_F32: RefCell<(Vec<f32>, Vec<f32>)> = const { RefCell::new((Vec::new(), Vec::new())) };
    static PACK_F64: RefCell<(Vec<f64>, Vec<f64>)> = const { RefCell::new((Vec::new(), Vec::new())) };
}

/// Ensure packing buffers have sufficient capacity, then call `f` with them.
fn with_pack_f32<R>(f: impl FnOnce(&mut [f32], &mut [f32]) -> R) -> R {
    PACK_F32.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let a_need = MC * KC;
        let b_need = KC * NC;
        if bufs.0.len() < a_need {
            bufs.0.resize(a_need, 0.0);
        }
        if bufs.1.len() < b_need {
            bufs.1.resize(b_need, 0.0);
        }
        let (ref mut pack_a, ref mut pack_b) = *bufs;
        f(&mut pack_a[..a_need], &mut pack_b[..b_need])
    })
}

fn with_pack_f64<R>(f: impl FnOnce(&mut [f64], &mut [f64]) -> R) -> R {
    PACK_F64.with(|cell| {
        let mut bufs = cell.borrow_mut();
        let a_need = MC * KC;
        let b_need = KC * NC;
        if bufs.0.len() < a_need {
            bufs.0.resize(a_need, 0.0);
        }
        if bufs.1.len() < b_need {
            bufs.1.resize(b_need, 0.0);
        }
        let (ref mut pack_a, ref mut pack_b) = *bufs;
        f(&mut pack_a[..a_need], &mut pack_b[..b_need])
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Tiled matmul: C = A @ B (f32)
///
/// No separate zero pass - microkernels use beta=0 on first K-block.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_tiled_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    with_pack_f32(|packed_a, packed_b| {
        tiled_loop_f32::<NR>(
            a, b, c, m, n, k, lda, ldb, ldc, level, packed_a, packed_b, false,
        );
    });
}

/// Tiled matmul with a transposed B operand: C = A @ B (f32)
///
/// `b` points at a contiguous `[N, K]` buffer holding the transpose of the
/// logical `[K, N]` operand, with row stride `ldb_t` (the contraction extent).
/// Nothing is materialized: only the B packing step changes its index
/// arithmetic, so the packed panels, and therefore the accumulation order and
/// the result, are identical to [`matmul_tiled_f32`] on a materialized B.
///
/// # Safety
/// Same as [`matmul_tiled_f32`], with `b` valid for `n * ldb_t` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bt_tiled_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb_t: usize,
    ldc: usize,
    level: SimdLevel,
) {
    with_pack_f32(|packed_a, packed_b| {
        tiled_loop_f32::<NR>(
            a, b, c, m, n, k, lda, ldb_t, ldc, level, packed_a, packed_b, true,
        );
    });
}

/// Tiled matmul with bias: C = A @ B + bias (f32)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_tiled_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    bias: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    // Bias needs C pre-initialized before accumulation
    let bias_slice = std::slice::from_raw_parts(bias, n);
    for i in 0..m {
        let c_row = std::slice::from_raw_parts_mut(c.add(i * ldc), n);
        c_row.copy_from_slice(bias_slice);
    }

    with_pack_f32(|packed_a, packed_b| {
        // All K-blocks use beta=1 since C has bias values
        tiled_loop_f32_beta1::<NR>(a, b, c, m, n, k, lda, ldb, ldc, level, packed_a, packed_b);
    });
}

/// Tiled matmul: C = A @ B (f64)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_tiled_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    with_pack_f64(|packed_a, packed_b| {
        tiled_loop_f64::<NR>(
            a, b, c, m, n, k, lda, ldb, ldc, level, packed_a, packed_b, false,
        );
    });
}

/// Tiled matmul with a transposed B operand: C = A @ B (f64)
///
/// See [`matmul_bt_tiled_f32`]; `b` is a contiguous `[N, K]` buffer with row
/// stride `ldb_t`.
///
/// # Safety
/// Same as [`matmul_tiled_f64`], with `b` valid for `n * ldb_t` elements.
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bt_tiled_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb_t: usize,
    ldc: usize,
    level: SimdLevel,
) {
    with_pack_f64(|packed_a, packed_b| {
        tiled_loop_f64::<NR>(
            a, b, c, m, n, k, lda, ldb_t, ldc, level, packed_a, packed_b, true,
        );
    });
}

/// Tiled matmul with bias: C = A @ B + bias (f64)
#[allow(clippy::too_many_arguments)]
pub unsafe fn matmul_bias_tiled_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    bias: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
) {
    let bias_slice = std::slice::from_raw_parts(bias, n);
    for i in 0..m {
        let c_row = std::slice::from_raw_parts_mut(c.add(i * ldc), n);
        c_row.copy_from_slice(bias_slice);
    }

    with_pack_f64(|packed_a, packed_b| {
        tiled_loop_f64_beta1::<NR>(a, b, c, m, n, k, lda, ldb, ldc, level, packed_a, packed_b);
    });
}

// ---------------------------------------------------------------------------
// Core tiled loops
// ---------------------------------------------------------------------------

/// Core tiled loop for f32 with beta=0 on first K-block
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f32<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f32],
    packed_b: &mut [f32],
    b_transposed: bool,
) {
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);
            let first_k = pc == 0;

            // Tested once per packed panel, never inside the microkernel loops.
            if b_transposed {
                pack_b_t_f32::<NR>(b.add(jc * ldb + pc), packed_b.as_mut_ptr(), nc, kc, ldb);
            } else {
                pack_b_f32::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);
            }

            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f32(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                microkernel_loop_f32::<NR>(
                    packed_a, packed_b, c, ic, jc, mc, nc, kc, ldc, level, first_k,
                );
            }
        }
    }
}

/// Core tiled loop for f32 always using beta=1 (for bias variant)
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f32_beta1<const NR: usize>(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f32],
    packed_b: &mut [f32],
) {
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            pack_b_f32::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);

            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f32(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                microkernel_loop_f32::<NR>(
                    packed_a, packed_b, c, ic, jc, mc, nc, kc, ldc, level, false,
                );
            }
        }
    }
}

/// Inner microkernel dispatch loop for f32
///
/// NR is the double-width (e.g. 32 for AVX-512). Uses the 2x microkernel for
/// full blocks and falls back to single-width or edge for remainders.
///
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn microkernel_loop_f32<const NR: usize>(
    packed_a: &[f32],
    packed_b: &[f32],
    c: *mut f32,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    let nr_half = NR / 2;

    for jr in (0..nc).step_by(NR) {
        let nr_actual = (nc - jr).min(NR);

        for ir in (0..mc).step_by(MR) {
            let mr_actual = (mc - ir).min(MR);

            if mr_actual == MR && nr_actual == NR {
                call_microkernel_2x_f32(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    kc,
                    ldc,
                    level,
                    first_k,
                );
            } else if mr_actual == MR && nr_actual == nr_half {
                // Half block
                call_microkernel_f32(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    kc,
                    ldc,
                    level,
                    first_k,
                );
            } else {
                microkernel_edge_f32(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    mr_actual,
                    nr_actual,
                    kc,
                    ldc,
                    first_k,
                );
            }
        }
    }
}

/// Core tiled loop for f64 with beta=0 on first K-block
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f64<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f64],
    packed_b: &mut [f64],
    b_transposed: bool,
) {
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);
            let first_k = pc == 0;

            // Tested once per packed panel, never inside the microkernel loops.
            if b_transposed {
                pack_b_t_f64::<NR>(b.add(jc * ldb + pc), packed_b.as_mut_ptr(), nc, kc, ldb);
            } else {
                pack_b_f64::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);
            }

            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f64(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                microkernel_loop_f64::<NR>(
                    packed_a, packed_b, c, ic, jc, mc, nc, kc, ldc, level, first_k,
                );
            }
        }
    }
}

/// Core tiled loop for f64 always using beta=1 (for bias variant)
#[allow(clippy::too_many_arguments)]
unsafe fn tiled_loop_f64_beta1<const NR: usize>(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
    level: SimdLevel,
    packed_a: &mut [f64],
    packed_b: &mut [f64],
) {
    for jc in (0..n).step_by(NC) {
        let nc = (n - jc).min(NC);

        for pc in (0..k).step_by(KC) {
            let kc = (k - pc).min(KC);

            pack_b_f64::<NR>(b.add(pc * ldb + jc), packed_b.as_mut_ptr(), nc, kc, ldb);

            for ic in (0..m).step_by(MC) {
                let mc = (m - ic).min(MC);

                pack_a_f64(a.add(ic * lda + pc), packed_a.as_mut_ptr(), mc, kc, lda);

                microkernel_loop_f64::<NR>(
                    packed_a, packed_b, c, ic, jc, mc, nc, kc, ldc, level, false,
                );
            }
        }
    }
}

/// Inner microkernel dispatch loop for f64
#[allow(clippy::too_many_arguments)]
#[inline]
unsafe fn microkernel_loop_f64<const NR: usize>(
    packed_a: &[f64],
    packed_b: &[f64],
    c: *mut f64,
    ic: usize,
    jc: usize,
    mc: usize,
    nc: usize,
    kc: usize,
    ldc: usize,
    level: SimdLevel,
    first_k: bool,
) {
    let nr_half = NR / 2;

    for jr in (0..nc).step_by(NR) {
        let nr_actual = (nc - jr).min(NR);

        for ir in (0..mc).step_by(MR) {
            let mr_actual = (mc - ir).min(MR);

            if mr_actual == MR && nr_actual == NR {
                call_microkernel_2x_f64(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    kc,
                    ldc,
                    level,
                    first_k,
                );
            } else if mr_actual == MR && nr_actual == nr_half {
                call_microkernel_f64(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    kc,
                    ldc,
                    level,
                    first_k,
                );
            } else {
                microkernel_edge_f64(
                    packed_a.as_ptr().add(ir * kc),
                    packed_b.as_ptr().add(jr * kc),
                    c.add((ic + ir) * ldc + jc + jr),
                    mr_actual,
                    nr_actual,
                    kc,
                    ldc,
                    first_k,
                );
            }
        }
    }
}
