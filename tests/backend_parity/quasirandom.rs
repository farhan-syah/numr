//! Backend parity for `QuasiRandomOps`: `sobol`, `halton`, `latin_hypercube`.
//!
//! CPU is the reference throughout, but the three ops do NOT share one notion of
//! parity, so they are not asserted the same way:
//!
//! - `sobol` and `halton` are deterministic functions of `(n_points, dimension,
//!   skip)`. Every backend builds them from the same inputs — the Joe & Kuo
//!   direction numbers via `ops::common::quasirandom` for Sobol, the same first
//!   100 primes for Halton — and indexes them the same way (`point_index = skip +
//!   i`, Gray code `i ^ (i >> 1)`, scale `1 / 2^32`). Both are therefore asserted
//!   against KNOWN REFERENCE VALUES on each backend, not merely backend to
//!   backend: a shared wrong answer would pass a backend-to-backend check.
//! - `latin_hypercube` takes no seed. CPU draws from `rng::thread_rng()`, CUDA
//!   from a clock-derived seed, WebGPU from `generate_wgpu_seed()`. Values do not
//!   reproduce across backends and do not reproduce across CALLS on one backend,
//!   which matches the contract the `random` parity tests established (seeds
//!   reproduce per backend, never across them). Asserting values would assert
//!   something no backend promises, so what is asserted instead is the property
//!   that defines LHS and must hold everywhere: each of the N strata of `[0, 1)`
//!   is occupied exactly once per dimension, and every value lies in `[0, 1)`.
//!
//! Both sequences are 0-indexed and do NOT skip their first point: index 0 has
//! Gray code 0 and every van der Corput digit 0, so the first row is the origin.
//! That is pinned explicitly — it is the convention most likely to drift.

use numr::dtype::DType;
use numr::ops::QuasiRandomOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{assert_tensor_allclose, create_cpu_client};

/// Dtypes CPU and CUDA accept for quasi-random generation.
///
/// Not `parity_dtypes`: these ops take a dtype argument rather than inheriting
/// one from an input, and both backends declare F32 and F64 only — F16, BF16 and
/// FP8 are rejected outright.
const FULL_DTYPES: &[DType] = &[DType::F32, DType::F64];

/// WebGPU declares F32 only, matching its 32-bit-only scope.
#[cfg(feature = "wgpu")]
const WGPU_DTYPES: &[DType] = &[DType::F32];

// ============================================================================
// Reference sequences
// ============================================================================

/// Sobol dimension 0, indices 0..8. Direction vectors are the powers of two, so
/// the point at index `i` is the bit-reversal of `i ^ (i >> 1)` over 32 bits —
/// the classic first column of the Sobol sequence.
const SOBOL_DIM0: [f64; 8] = [0.0, 0.5, 0.75, 0.25, 0.375, 0.875, 0.625, 0.125];

/// Halton dimension 0: van der Corput in base 2.
const HALTON_BASE2: [f64; 8] = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];

/// Halton dimension 1: van der Corput in base 3.
const HALTON_BASE3: [f64; 8] = [
    0.0,
    1.0 / 3.0,
    2.0 / 3.0,
    1.0 / 9.0,
    4.0 / 9.0,
    7.0 / 9.0,
    2.0 / 9.0,
    5.0 / 9.0,
];

/// Halton dimension 2: van der Corput in base 5.
const HALTON_BASE5: [f64; 8] = [0.0, 0.2, 0.4, 0.6, 0.8, 0.04, 0.24, 0.44];

// ============================================================================
// Helpers
// ============================================================================

/// Read a generated tensor back in ITS OWN dtype, widened to f64 for comparison
/// against the reference constants.
fn read_as_f64<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<f64> {
    match t.dtype() {
        DType::F32 => t.to_vec::<f32>().into_iter().map(|v| v as f64).collect(),
        DType::F64 => t.to_vec::<f64>(),
        d => panic!("quasirandom parity: unexpected result dtype {d:?}"),
    }
}

/// Absolute tolerance for a reference-value comparison.
///
/// The references are exact rationals; the only error is the round to the output
/// dtype, so an absolute bound suffices — every value lies in `[0, 1)`.
fn tol_for(dtype: DType) -> f64 {
    match dtype {
        DType::F32 => 1e-6,
        _ => 1e-12,
    }
}

/// Column `d` of a row-major `[n, dimension]` buffer.
fn column(vals: &[f64], dimension: usize, d: usize) -> Vec<f64> {
    vals.iter().skip(d).step_by(dimension).copied().collect()
}

fn assert_column(actual: &[f64], expected: &[f64], dtype: DType, what: &str) {
    assert_eq!(actual.len(), expected.len(), "{what}: length mismatch");
    let tol = tol_for(dtype);
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{what}[{dtype:?}] at index {i}: {a} vs expected {e} (tol {tol})"
        );
    }
}

fn assert_unit_hypercube(vals: &[f64], dtype: DType, what: &str) {
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (0.0..1.0).contains(&v),
            "{what}[{dtype:?}] value {i} outside [0, 1): {v}"
        );
    }
}

// ============================================================================
// Per-backend invariants
// ============================================================================

/// Sobol against known values, plus the two structural facts that pin the
/// construction: the sequence starts at index 0, and 2^k consecutive points from
/// index 0 are balanced.
fn check_sobol<R, C>(client: &C, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    let dimension = 3;
    let points = client
        .sobol(8, dimension, 0, dtype)
        .unwrap_or_else(|e| panic!("{backend} sobol(8, {dimension}, 0, {dtype:?}): {e:?}"));
    assert_eq!(points.shape(), &[8, dimension], "{backend} sobol shape");
    assert_eq!(points.dtype(), dtype, "{backend} sobol dtype");

    let vals = read_as_f64(&points);
    assert_unit_hypercube(&vals, dtype, &format!("{backend} sobol"));
    assert_column(
        &column(&vals, dimension, 0),
        &SOBOL_DIM0,
        dtype,
        &format!("{backend} sobol dim 0"),
    );

    // Index 0 has Gray code 0, so no direction vector is XORed in and EVERY
    // dimension starts at the origin. This is what distinguishes the 0-indexed
    // convention from one that discards the first point.
    for (d, &v) in vals.iter().take(dimension).enumerate() {
        assert_eq!(
            v, 0.0,
            "{backend} sobol[{dtype:?}] first point, dim {d}: expected the origin, got {v}"
        );
    }

    // Balance: for 2^k points from index 0, each dimension's values are a
    // permutation of {j / 2^k}. Only the top k direction-vector bits participate
    // and their matrix is unit upper-triangular, so this holds for every
    // dimension with genuine Joe & Kuo direction numbers — a degenerate or
    // misindexed dimension fails it.
    let k_points = 16;
    let block = client
        .sobol(k_points, dimension, 0, dtype)
        .unwrap_or_else(|e| {
            panic!("{backend} sobol({k_points}, {dimension}, 0, {dtype:?}): {e:?}")
        });
    let block_vals = read_as_f64(&block);
    for d in 0..dimension {
        let mut col = column(&block_vals, dimension, d);
        col.sort_by(|a, b| a.partial_cmp(b).expect("sobol produced NaN"));
        let expected: Vec<f64> = (0..k_points).map(|j| j as f64 / k_points as f64).collect();
        assert_column(
            &col,
            &expected,
            dtype,
            &format!("{backend} sobol balance over {k_points} points, dim {d}"),
        );
    }
}

/// Halton against van der Corput in bases 2, 3 and 5.
fn check_halton<R, C>(client: &C, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    let dimension = 3;
    let points = client
        .halton(8, dimension, 0, dtype)
        .unwrap_or_else(|e| panic!("{backend} halton(8, {dimension}, 0, {dtype:?}): {e:?}"));
    assert_eq!(points.shape(), &[8, dimension], "{backend} halton shape");
    assert_eq!(points.dtype(), dtype, "{backend} halton dtype");

    let vals = read_as_f64(&points);
    assert_unit_hypercube(&vals, dtype, &format!("{backend} halton"));
    for (d, expected) in [HALTON_BASE2, HALTON_BASE3, HALTON_BASE5]
        .iter()
        .enumerate()
    {
        assert_column(
            &column(&vals, dimension, d),
            expected,
            dtype,
            &format!("{backend} halton dim {d}"),
        );
    }
}

/// `skip` advances the index, it does not reseed: row `i` of a run with `skip`
/// equals row `skip + i` of a run without it.
fn check_skip_is_an_offset<R, C>(client: &C, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    let dimension = 3;
    let skip = 4;
    let tail = 8;

    let full = client
        .sobol(skip + tail, dimension, 0, dtype)
        .unwrap_or_else(|e| panic!("{backend} sobol full run: {e:?}"));
    let skipped = client
        .sobol(tail, dimension, skip, dtype)
        .unwrap_or_else(|e| panic!("{backend} sobol skipped run: {e:?}"));
    assert_column(
        &read_as_f64(&skipped),
        &read_as_f64(&full)[skip * dimension..],
        dtype,
        &format!("{backend} sobol skip={skip} vs offset into a full run"),
    );

    let full = client
        .halton(skip + tail, dimension, 0, dtype)
        .unwrap_or_else(|e| panic!("{backend} halton full run: {e:?}"));
    let skipped = client
        .halton(tail, dimension, skip, dtype)
        .unwrap_or_else(|e| panic!("{backend} halton skipped run: {e:?}"));
    assert_column(
        &read_as_f64(&skipped),
        &read_as_f64(&full)[skip * dimension..],
        dtype,
        &format!("{backend} halton skip={skip} vs offset into a full run"),
    );
}

/// The LHS property every backend must satisfy whatever its seed: each of the N
/// strata of `[0, 1)` holds exactly one sample, per dimension.
///
/// N is a power of two so the stratum boundaries `j / N` are exact in both F32
/// and F64 and the bin index cannot be decided by a rounding artefact.
fn check_latin_hypercube<R, C>(client: &C, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    let n_samples = 16;
    let dimension = 4;
    let samples = client
        .latin_hypercube(n_samples, dimension, dtype)
        .unwrap_or_else(|e| {
            panic!("{backend} latin_hypercube({n_samples}, {dimension}, {dtype:?}): {e:?}")
        });
    assert_eq!(
        samples.shape(),
        &[n_samples, dimension],
        "{backend} latin_hypercube shape"
    );
    assert_eq!(samples.dtype(), dtype, "{backend} latin_hypercube dtype");

    let vals = read_as_f64(&samples);
    assert_unit_hypercube(&vals, dtype, &format!("{backend} latin_hypercube"));

    for d in 0..dimension {
        let mut occupancy = vec![0usize; n_samples];
        for v in column(&vals, dimension, d) {
            let bin = ((v * n_samples as f64).floor() as usize).min(n_samples - 1);
            occupancy[bin] += 1;
        }
        for (bin, &count) in occupancy.iter().enumerate() {
            assert_eq!(
                count, 1,
                "{backend} latin_hypercube[{dtype:?}] dim {d}: stratum {bin} holds {count} \
                 samples, expected exactly 1 (occupancy: {occupancy:?})"
            );
        }
    }
}

fn check_all<R, C>(client: &C, dtypes: &[DType], backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    for &dtype in dtypes {
        check_sobol::<R, C>(client, dtype, backend);
        check_halton::<R, C>(client, dtype, backend);
        check_skip_is_an_offset::<R, C>(client, dtype, backend);
        check_latin_hypercube::<R, C>(client, dtype, backend);
    }
}

/// Cross-backend value parity, for the two DETERMINISTIC sequences only.
///
/// `latin_hypercube` is absent by design: it has no seed argument and each
/// backend draws its own, so a value comparison would be asserting a
/// reproducibility the crate never promised across backends.
fn check_deterministic_parity<R, C>(client: &C, dtypes: &[DType], backend: &str)
where
    R: Runtime<DType = DType>,
    C: QuasiRandomOps<R>,
{
    let (cpu_client, _cpu_device) = create_cpu_client();
    let dimension = 6;

    for &dtype in dtypes {
        for skip in [0usize, 5] {
            let expected = cpu_client
                .sobol(32, dimension, skip, dtype)
                .expect("cpu sobol");
            let actual = client
                .sobol(32, dimension, skip, dtype)
                .unwrap_or_else(|e| panic!("{backend} sobol skip={skip} {dtype:?}: {e:?}"));
            assert_tensor_allclose(
                &actual,
                &expected,
                dtype,
                &format!("sobol skip={skip} {backend} vs cpu"),
            );

            let expected = cpu_client
                .halton(32, dimension, skip, dtype)
                .expect("cpu halton");
            let actual = client
                .halton(32, dimension, skip, dtype)
                .unwrap_or_else(|e| panic!("{backend} halton skip={skip} {dtype:?}: {e:?}"));
            assert_tensor_allclose(
                &actual,
                &expected,
                dtype,
                &format!("halton skip={skip} {backend} vs cpu"),
            );
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_quasirandom_cpu_reference() {
    let (client, _device) = create_cpu_client();
    check_all::<CpuRuntime, _>(&client, FULL_DTYPES, "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn test_quasirandom_cuda_matches_cpu() {
    with_cuda_backend(|client, _device| {
        check_all::<CudaRuntime, _>(&client, FULL_DTYPES, "cuda");
        check_deterministic_parity::<CudaRuntime, _>(&client, FULL_DTYPES, "cuda");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_quasirandom_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, _device| {
        check_all::<WgpuRuntime, _>(&client, WGPU_DTYPES, "wgpu");
        check_deterministic_parity::<WgpuRuntime, _>(&client, WGPU_DTYPES, "wgpu");
    });
}
