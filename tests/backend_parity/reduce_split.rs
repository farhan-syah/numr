// Backend parity for the CUDA two-stage dim-reduction split path.
//
// `src/runtime/cuda/kernels/loader/reduce_split.rs` (`reduce_split_for_units`)
// widens the launch grid when a long reduce axis collapses to few outputs:
// stage 1 reduces `splits` equal chunks in parallel, stage 2 merges the
// `splits` partials. `src/runtime/cuda/kernels/reduce.rs`
// (`reduce_dim_split_count`) additionally restricts which ops may take this
// path to those that merge exactly: `max`/`min`/`any`/`all` for every dtype,
// and `sum`/`prod` only where the accumulator is the element type itself
// (F32 native, F64). Before this file, only two `mean` tests incidentally
// exercised stage 1 (`mean` never takes the split path itself) — no test
// covered the ops that actually split.
//
// Every shape below is derived by hand against the gate in `reduce_split.rs`:
// `outer * inner` must sit below `compute_units * 2`, `reduce / 256` must be
// `>= 2`, and the scan for a `splits` that divides `reduce` exactly must land
// on something `>= 2`. `outer=1` and a power-of-two `reduce` make the divisor
// scan trivial; the `[2, 8192]` case keeps `outer > 1` so the stage-1
// `outer * splits` remapping is exercised, not just the `outer == 1`
// degenerate case.

use numr::dtype::DType;
use numr::ops::ReduceOps;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose_tol, create_cpu_client, gemm_long_k_tolerance,
    is_dtype_supported, parity_dtypes,
};

/// Values with a UNIQUE maximum and a UNIQUE minimum, each parked at a specific
/// index late in the reduced axis.
///
/// `varying_nonneg` repeats every 13 elements, so its extremum occurs hundreds
/// of times and a wrong stage-1 index still lands on an equal value — an
/// injected chunk bug went undetected by max/min while every other op caught
/// it. A single occurrence of each extremum removes that slack. Magnitudes stay
/// inside I8 range so the signed-integer dtypes cover this too.
fn unique_extrema(n: usize) -> Vec<f64> {
    let mut values: Vec<f64> = (0..n).map(|i| (i % 13) as f64 + 5.0).collect();
    if n > 4090 {
        values[4090] = 100.0;
    }
    if n > 3901 {
        values[3901] = 1.0;
    }
    values
}

/// Values centered on zero (small dyadic range), for dtypes that must accept
/// negatives (signed integers, floats) and where exact bit-for-bit
/// reproduction across the split matters (`max`/`min`).
fn varying_signed(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i % 13) as f64 - 6.0) * 0.25).collect()
}

/// Factors clustered tightly around 1.0 so a 4096-long product neither
/// overflows nor flushes to zero, while still varying with the flat index so
/// a dropped chunk changes the product.
fn varying_near_one(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i % 9) as f64 - 4.0) * 0.0003)
        .collect()
}

/// Run `op` (`sum`, `max`, `min`, `prod`, `any`, `all`) on CPU and CUDA for
/// one dtype and assert they match within `(rtol, atol)`.
#[allow(clippy::too_many_arguments)]
fn assert_reduce_dim_parity(
    label: &str,
    op: &'static str,
    input: &[f64],
    input_shape: &[usize],
    dim: usize,
    dtype: DType,
    rtol: f64,
    atol: f64,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_t = tensor_from_f64(input, input_shape, dtype, &cpu_device, &cpu_client)
        .unwrap_or_else(|e| panic!("CPU tensor failed for {label} [{dtype:?}]: {e}"));
    let cpu_result = match op {
        "sum" => cpu_client.sum(&cpu_t, &[dim], false),
        "prod" => cpu_client.prod(&cpu_t, &[dim], false),
        "max" => cpu_client.max(&cpu_t, &[dim], false),
        "min" => cpu_client.min(&cpu_t, &[dim], false),
        "any" => cpu_client.any(&cpu_t, &[dim], false),
        "all" => cpu_client.all(&cpu_t, &[dim], false),
        other => panic!("unknown op {other}"),
    }
    .unwrap_or_else(|e| panic!("CPU {op} failed for {label} [{dtype:?}]: {e}"));

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let t = tensor_from_f64(input, input_shape, dtype, &cuda_device, &cuda_client)
                .unwrap_or_else(|e| panic!("CUDA tensor failed for {label} [{dtype:?}]: {e}"));
            let result = match op {
                "sum" => cuda_client.sum(&t, &[dim], false),
                "prod" => cuda_client.prod(&t, &[dim], false),
                "max" => cuda_client.max(&t, &[dim], false),
                "min" => cuda_client.min(&t, &[dim], false),
                "any" => cuda_client.any(&t, &[dim], false),
                "all" => cuda_client.all(&t, &[dim], false),
                other => panic!("unknown op {other}"),
            }
            .unwrap_or_else(|e| panic!("CUDA {op} failed for {label} [{dtype:?}]: {e}"));
            assert_tensor_allclose_tol(
                &result,
                &cpu_result,
                rtol,
                atol,
                &format!("{label} CUDA vs CPU [{dtype:?}]"),
            );
        });
    }
}

/// Catches: `sum`'s two-stage split (`reduce_dim_split_count` picks a split
/// only when the accumulator is the element type — F32 native, F64) merging
/// its `splits` partial sums incorrectly, or a chunk boundary that drops
/// elements.
///
/// Shape `[1, 4096]` reduced over dim 1: `outer=1`, `inner=1`, `reduce=4096`.
/// `outputs=1` is far below `compute_units * 2` on any real GPU, so the shape
/// gate passes. `max_splits = 4096 / 256 = 16`. `target_splits =
/// (compute_units*2).div_ceil(1).min(16) = 16` on any device with
/// `compute_units >= 8` (every real CUDA GPU), and 16 divides 4096 exactly
/// (`4096 / 16 = 256 = BLOCK_SIZE`), so the scan picks `splits = 16` on the
/// first try. Reassociating a 4096-term sum into 16 chunks of 256 moves the
/// last bits of an F32/F64 result, so this uses `gemm_long_k_tolerance`
/// rather than exact comparison.
#[test]
fn reduce_split_sum_parity() {
    let shape = [1usize, 4096];
    let input = varying_signed(shape.iter().product());
    let scale = input.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for dtype in [DType::F32, DType::F64] {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        let (rtol, atol) = gemm_long_k_tolerance(dtype, shape[1], scale);
        assert_reduce_dim_parity(
            "reduce_split_sum",
            "sum",
            &input,
            &shape,
            1,
            dtype,
            rtol,
            atol,
        );
    }
}

/// Catches: `max`/`min` splitting for every dtype (unlike `sum`/`prod`, they
/// merge exactly regardless of accumulator) with a wrong stage-1/stage-2
/// merge. Every partial `max`/`min` writes back an input element verbatim, so
/// the two-stage result must be bit-exact against CPU — this uses zero
/// tolerance, not `gemm_long_k_tolerance`.
///
/// Same `[1, 4096]` shape and `splits = 16` as `reduce_split_sum_parity`
/// (the shape gate does not depend on the op). Covers every dtype the domain
/// admits (`AllNumeric` minus `Bool`, which every backend rejects for
/// reductions), not only floats.
#[test]
fn reduce_split_max_min_parity() {
    let shape = [1usize, 4096];
    let input = unique_extrema(shape.iter().product());
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
        assert_reduce_dim_parity(
            "reduce_split_max",
            "max",
            &input,
            &shape,
            1,
            dtype,
            0.0,
            0.0,
        );
        assert_reduce_dim_parity(
            "reduce_split_min",
            "min",
            &input,
            &shape,
            1,
            dtype,
            0.0,
            0.0,
        );
    }
}

/// Catches: `prod`'s two-stage split (same `accumulates_in_element_type` gate
/// as `sum`: F32 native, F64) reassociating the running product incorrectly,
/// or a chunk that reads a wrong slice and shifts the product's magnitude.
///
/// Same `[1, 4096]` shape and `splits = 16`. Factors sit in
/// `[1 - 4*0.0003, 1 + 4*0.0003]` so the full 4096-term product stays within
/// roughly `[e^-4.9, e^4.9]` — nowhere near F32 overflow or flush-to-zero —
/// while still varying with the flat index so a dropped chunk is visible.
#[test]
fn reduce_split_prod_parity() {
    let shape = [1usize, 4096];
    let input = varying_near_one(shape.iter().product());
    let scale = input.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for dtype in [DType::F32, DType::F64] {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        let (rtol, atol) = gemm_long_k_tolerance(dtype, shape[1], scale);
        assert_reduce_dim_parity(
            "reduce_split_prod",
            "prod",
            &input,
            &shape,
            1,
            dtype,
            rtol,
            atol,
        );
    }
}

/// Catches: `any`/`all` splitting (every dtype, like `max`/`min`) losing the
/// single truthy/falsy element when it sits in a LATE chunk. With
/// `splits = 16` and `chunk = 256`, index 4090 sits in chunk 15 (the last of
/// 16, elements 3840..4096) — a stage that drops or reads only the first N
/// chunks of stage 1 would still report the identity value and this catches
/// it. Result must be bit-exact against CPU, so zero tolerance.
#[test]
fn reduce_split_any_all_parity() {
    let shape = [1usize, 4096];
    let n = shape.iter().product();

    // `any`: all zero except one true element deep in the last chunk.
    let mut any_input = vec![0.0f64; n];
    any_input[4090] = 1.0;

    // `all`: all one except one false element at the same late position.
    let mut all_input = vec![1.0f64; n];
    all_input[4090] = 0.0;

    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
        assert_reduce_dim_parity(
            "reduce_split_any",
            "any",
            &any_input,
            &shape,
            1,
            dtype,
            0.0,
            0.0,
        );
        assert_reduce_dim_parity(
            "reduce_split_all",
            "all",
            &all_input,
            &shape,
            1,
            dtype,
            0.0,
            0.0,
        );
    }
}

/// Catches: a wrong stage-1 index when `outer > 1`. `outer == 1` collapses
/// `outer * splits` down to `splits`, which cannot distinguish an `outer`
/// stride bug from a `splits` stride bug — this shape can.
///
/// Shape `[2, 8192]` reduced over dim 1: `outer=2`, `inner=1`, `reduce=8192`.
/// `outputs=2` clears the `outputs < compute_units * 2` gate on any GPU with
/// more than one compute unit. `max_splits = 8192 / 256 = 32`.
/// `reduce = 8192 = 2^13` is a pure power of two, so whatever `splits` the
/// scan lands on (`(compute_units*2).div_ceil(2)` clamped to `32`, then
/// walked down to the nearest divisor) is itself a power of two `>= 2` and
/// therefore always divides 8192 exactly — the split path is reached
/// independent of the actual `compute_units` on the test machine. Row 0 and
/// row 1 use different value sequences so a stage-1 index that reads row 1's
/// data into row 0's output (or vice versa) is visible.
#[test]
fn reduce_split_2d_few_rows_parity() {
    let shape = [2usize, 8192];
    let row_len = shape[1];
    let mut input = varying_signed(row_len);
    // Row 1 gets a disjoint value range from row 0.
    input.extend(varying_signed(row_len).into_iter().map(|v| v + 100.0));
    let scale = input.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);

    for dtype in [DType::F32, DType::F64] {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        let (rtol, atol) = gemm_long_k_tolerance(dtype, row_len, scale);
        assert_reduce_dim_parity(
            "reduce_split_2d_few_rows_sum",
            "sum",
            &input,
            &shape,
            1,
            dtype,
            rtol,
            atol,
        );
    }

    // max/min are bit-exact through the split regardless of dtype.
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
        assert_reduce_dim_parity(
            "reduce_split_2d_few_rows_max",
            "max",
            &input,
            &shape,
            1,
            dtype,
            0.0,
            0.0,
        );
    }
}

/// Catches: a fallback bug in the single-stage path when the split rule finds
/// no usable divisor. `reduce = 4099` is prime, so no `splits` in the scanned
/// range `[2, 16]` divides it and `reduce_split_for_units` returns `None`
/// (`outer=1`, `inner=1`, `outputs=1`; `max_splits = 4099 / 256 = 16`;
/// `target_splits = min(compute_units*2, 16) = 16` on any real GPU; the
/// downward scan from 16 to 2 finds no divisor of the prime 4099). The op
/// must still be correct on the single-stage launch this falls back to.
#[test]
fn reduce_split_non_divisible_falls_back_parity() {
    let shape = [1usize, 4099];
    let input = varying_signed(shape.iter().product());
    let scale = input.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1.0);
    for dtype in [DType::F32, DType::F64] {
        if !is_dtype_supported("cuda", dtype) {
            continue;
        }
        let (rtol, atol) = gemm_long_k_tolerance(dtype, shape[1], scale);
        assert_reduce_dim_parity(
            "reduce_split_non_divisible_sum",
            "sum",
            &input,
            &shape,
            1,
            dtype,
            rtol,
            atol,
        );
    }
}
