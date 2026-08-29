//! CUDA indexing, sorting, and scatter-reduce kernels must cover the integer
//! dtypes the CPU backend covers, with the same numerical semantics.
//!
//! Each family looks its kernel up by NAME — `masked_select_broadcast_{suffix}`,
//! `gather_nd_{suffix}`, `count_unique_{suffix}`, `scatter_reduce_int_mean_{suffix}`
//! — so a dtype with no `.cu` instantiation compiles fine and then fails at
//! launch with `named symbol not found`, or is refused by a dtype gate that
//! never grew past F32/F64/I32. U32 and the narrow integers had neither the
//! instantiation nor the gate.
//!
//! Semantics come from the CPU reference:
//!
//! * `masked_select_kernel` / `masked_fill_kernel` in
//!   `src/runtime/cpu/kernels/index.rs`: a mask byte is true when non-zero, and
//!   the fill value is `T::from_f64(value)`, which saturates.
//! * `gather_nd_kernel` in the same file: an out-of-range coordinate yields 0.
//! * `count_nonzero_kernel` and `extract_unique_kernel` in
//!   `src/runtime/cpu/kernels/sort.rs`: "nonzero" is `!= 0`, and "unique"
//!   compares each sorted element against its predecessor.
//! * `scatter_reduce_int_kernel` in
//!   `src/runtime/cpu/kernels/scatter_reduce_int.rs`: an integer reduction
//!   accumulates in i128 and `mean` divides that total exactly ONCE, so a set
//!   whose sum leaves the dtype but whose mean does not reports the true mean.
//!
//! Every case asserts CPU and CUDA against the same literal expectation. A
//! CUDA-only check would pass a kernel that agrees with a wrong expectation.
//!
//! Run: cargo test --features cuda --test cuda_indexing_int_coverage

#![cfg(feature = "cuda")]

mod common;

use common::{create_cpu_client, create_cuda_client};
use numr::dtype::DType;
use numr::ops::{IndexingOps, ScatterReduceOp, SortingOps, TypeConversionOps};
use numr::runtime::Runtime;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

// ============================================================================
// Case definitions, each generic over the runtime so one body pins both
// backends
// ============================================================================

/// `masked_select` with a same-shape mask over a `[2, 3]` U32 tensor.
fn masked_select_same_shape<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let a = Tensor::<R>::from_slice(&[10u32, 20, 30, 40, 50, 60], &[2, 3], device)
        .expect("staging the input must succeed");
    let mask = Tensor::<R>::from_slice(&[1u8, 0, 1, 0, 1, 1], &[2, 3], device)
        .expect("staging the mask must succeed");
    client
        .masked_select(&a, &mask)
        .expect("masked_select must succeed")
        .to_vec::<u32>()
}

/// `masked_select` with a `[1, 3]` mask broadcast over a `[2, 3]` tensor. This
/// is the `masked_select_broadcast_u32` kernel, a separate instantiation.
fn masked_select_broadcast<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let a = Tensor::<R>::from_slice(&[10u32, 20, 30, 40, 50, 60], &[2, 3], device)
        .expect("staging the input must succeed");
    let mask = Tensor::<R>::from_slice(&[1u8, 0, 1], &[1, 3], device)
        .expect("staging the mask must succeed");
    client
        .masked_select(&a, &mask)
        .expect("broadcast masked_select must succeed")
        .to_vec::<u32>()
}

fn masked_fill_same_shape<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let a = Tensor::<R>::from_slice(&[10u32, 20, 30, 40, 50, 60], &[2, 3], device)
        .expect("staging the input must succeed");
    let mask = Tensor::<R>::from_slice(&[1u8, 0, 1, 0, 1, 1], &[2, 3], device)
        .expect("staging the mask must succeed");
    client
        .masked_fill(&a, &mask, 7.0)
        .expect("masked_fill must succeed")
        .to_vec::<u32>()
}

/// A negative fill on an unsigned dtype: `T::from_f64(-1.0)` saturates to 0,
/// and the mask broadcasts, so this covers `masked_fill_broadcast_u32` and the
/// saturation rule at once.
fn masked_fill_broadcast_saturating<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let a = Tensor::<R>::from_slice(&[10u32, 20, 30, 40, 50, 60], &[2, 3], device)
        .expect("staging the input must succeed");
    let mask = Tensor::<R>::from_slice(&[0u8, 1, 0], &[1, 3], device)
        .expect("staging the mask must succeed");
    client
        .masked_fill(&a, &mask, -1.0)
        .expect("broadcast masked_fill must succeed")
        .to_vec::<u32>()
}

/// `gather_nd` over a `[3, 2]` table with coordinate pairs, one of them
/// repeated so a kernel that ignored the index tensor could not pass.
fn gather_nd_pairs<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let table = Tensor::<R>::from_slice(&[1u32, 2, 3, 4, 5, 6], &[3, 2], device)
        .expect("staging the table must succeed");
    let idx = Tensor::<R>::from_slice(&[2i64, 1, 0, 0, 2, 1], &[3, 2], device)
        .expect("staging the coordinates must succeed");
    client
        .gather_nd(&table, &idx)
        .expect("gather_nd must succeed")
        .to_vec::<u32>()
}

/// `nonzero` runs `count_nonzero_{suffix}` then `gather_nonzero_{suffix}`.
/// Returns the number of reported positions; the input holds two zeros.
fn nonzero_count<R: Runtime<DType = DType>, C: SortingOps<R>>(
    client: &C,
    device: &R::Device,
) -> usize {
    let a = Tensor::<R>::from_slice(&[5u32, 0, 7, 0, 5, 9], &[6], device)
        .expect("staging the input must succeed");
    client
        .nonzero(&a)
        .expect("nonzero must succeed")
        .shape()
        .first()
        .copied()
        .expect("nonzero output must be 2D")
}

/// `unique` runs `count_unique_{suffix}` then `extract_unique_{suffix}`. The
/// input repeats 5 and contains 0, so a kernel that skipped zeros or kept
/// duplicates would report a different set.
fn unique_values<R: Runtime<DType = DType>, C: SortingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let a = Tensor::<R>::from_slice(&[5u32, 0, 7, 0, 5, 9], &[6], device)
        .expect("staging the input must succeed");
    client
        .unique(&a, true)
        .expect("unique must succeed")
        .to_vec::<u32>()
}

/// `scatter_reduce` with `mean` over I32, where the two values scattered into
/// slot 0 sum to 4e9 — past `i32::MAX` — while their mean, 2e9, is
/// representable. A kernel that summed in the element type would saturate to
/// `i32::MAX` and report a mean near 1.07e9.
fn scatter_reduce_mean_i32<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<i32> {
    let dst = Tensor::<R>::from_slice(&[0i32, 0], &[2], device).expect("staging dst must succeed");
    let idx =
        Tensor::<R>::from_slice(&[0i64, 0, 1], &[3], device).expect("staging indices must succeed");
    let src = Tensor::<R>::from_slice(&[2_000_000_000i32, 2_000_000_000, 7], &[3], device)
        .expect("staging src must succeed");
    client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Mean, false)
        .expect("scatter_reduce mean must succeed")
        .to_vec::<i32>()
}

/// The same overflow shape on U32: 3e9 + 3e9 is 6e9, past `u32::MAX`, and the
/// mean 3e9 is not.
fn scatter_reduce_mean_u32<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<u32> {
    let dst = Tensor::<R>::from_slice(&[0u32, 0], &[2], device).expect("staging dst must succeed");
    let idx =
        Tensor::<R>::from_slice(&[0i64, 0, 1], &[3], device).expect("staging indices must succeed");
    let src = Tensor::<R>::from_slice(&[3_000_000_000u32, 3_000_000_000, 11], &[3], device)
        .expect("staging src must succeed");
    client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Mean, false)
        .expect("scatter_reduce mean must succeed")
        .to_vec::<u32>()
}

/// `include_self` makes the destination's own value one of the averaged
/// contributions, so slot 0 averages three values and slot 1 keeps its own.
fn scatter_reduce_mean_include_self<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<i32> {
    let dst =
        Tensor::<R>::from_slice(&[10i32, 40], &[2], device).expect("staging dst must succeed");
    let idx =
        Tensor::<R>::from_slice(&[0i64, 0], &[2], device).expect("staging indices must succeed");
    let src =
        Tensor::<R>::from_slice(&[20i32, 30], &[2], device).expect("staging src must succeed");
    client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Mean, true)
        .expect("scatter_reduce mean must succeed")
        .to_vec::<i32>()
}

/// `mean` truncates toward zero, the same as the CPU epilogue's i128 division.
fn scatter_reduce_mean_truncates<R: Runtime<DType = DType>, C: IndexingOps<R>>(
    client: &C,
    device: &R::Device,
) -> Vec<i32> {
    let dst = Tensor::<R>::from_slice(&[0i32, 0], &[2], device).expect("staging dst must succeed");
    let idx = Tensor::<R>::from_slice(&[0i64, 0, 1, 1], &[4], device)
        .expect("staging indices must succeed");
    let src = Tensor::<R>::from_slice(&[3i32, 4, -3, -4], &[4], device)
        .expect("staging src must succeed");
    client
        .scatter_reduce(&dst, 0, &idx, &src, ScatterReduceOp::Mean, false)
        .expect("scatter_reduce mean must succeed")
        .to_vec::<i32>()
}

// ============================================================================
// Paired CPU / CUDA assertions
// ============================================================================

/// Run one case on CPU and on CUDA, asserting both equal `expected`.
///
/// `$case` is a generic function above. The CPU assertion pins the reference
/// semantics; the CUDA assertion pins parity with it.
macro_rules! check_case {
    ($case:ident, $expected:expr) => {{
        let label = stringify!($case);

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_out = $case::<CpuRuntime, _>(&cpu_client, &cpu_device);
        assert_eq!(cpu_out, $expected, "{label}: CPU reference");

        if let Some((cuda_client, cuda_device)) = create_cuda_client() {
            let cuda_out = $case::<CudaRuntime, _>(&cuda_client, &cuda_device);
            assert_eq!(cuda_out, $expected, "{label}: CUDA vs CPU");
        }
    }};
}

#[test]
fn masked_select_covers_u32() {
    check_case!(masked_select_same_shape, vec![10u32, 30, 50, 60]);
}

#[test]
fn masked_select_broadcast_covers_u32() {
    check_case!(masked_select_broadcast, vec![10u32, 30, 40, 60]);
}

#[test]
fn masked_fill_covers_u32() {
    check_case!(masked_fill_same_shape, vec![7u32, 20, 7, 40, 7, 7]);
}

#[test]
fn masked_fill_broadcast_covers_u32_and_saturates_a_negative_value() {
    check_case!(
        masked_fill_broadcast_saturating,
        vec![10u32, 0, 30, 40, 0, 60]
    );
}

#[test]
fn gather_nd_covers_u32() {
    check_case!(gather_nd_pairs, vec![6u32, 1, 6]);
}

#[test]
fn nonzero_covers_u32_including_zeros() {
    check_case!(nonzero_count, 4usize);
}

#[test]
fn unique_covers_u32_with_repeats_and_a_zero() {
    check_case!(unique_values, vec![0u32, 5, 7, 9]);
}

#[test]
fn scatter_reduce_mean_on_i32_divides_once() {
    check_case!(scatter_reduce_mean_i32, vec![2_000_000_000i32, 7]);
}

#[test]
fn scatter_reduce_mean_on_u32_divides_once() {
    check_case!(scatter_reduce_mean_u32, vec![3_000_000_000u32, 11]);
}

#[test]
fn scatter_reduce_mean_counts_the_destination_when_include_self() {
    check_case!(scatter_reduce_mean_include_self, vec![20i32, 40]);
}

#[test]
fn scatter_reduce_mean_truncates_toward_zero() {
    check_case!(scatter_reduce_mean_truncates, vec![3i32, -3]);
}

// ============================================================================
// Kernel-resolution sweep
// ============================================================================

/// The integer dtypes every family below is instantiated for. F16, BF16, and
/// FP8 are feature-gated and covered elsewhere; Bool is rejected by the CPU
/// backend's own `dispatch_dtype!`, so there is no reference to match.
const INT_DTYPES: [DType; 8] = [
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
];

/// Resolve every op against every instantiated dtype.
///
/// The point is the lookup, not the values: a missing `.cu` row fails here with
/// `named symbol not found`, and a dtype gate that forgot a dtype fails with
/// `UnsupportedDType`. Either way the sweep names the op and the dtype.
#[test]
fn every_integer_dtype_resolves_every_indexing_kernel() {
    let Some((client, device)) = create_cuda_client() else {
        return;
    };

    // Values stay small so every dtype, U8 included, represents them exactly.
    let base =
        Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
            .expect("staging the base tensor must succeed");
    let base_flat =
        Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], &device)
            .expect("staging the flat base tensor must succeed");
    let base_src = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 3.0], &[3], &device)
        .expect("staging the scatter source must succeed");
    let mask_full = Tensor::<CudaRuntime>::from_slice(&[1u8, 0, 1, 0, 1, 1], &[3, 2], &device)
        .expect("staging the mask must succeed");
    let mask_row = Tensor::<CudaRuntime>::from_slice(&[1u8, 0], &[1, 2], &device)
        .expect("staging the broadcast mask must succeed");
    let coords = Tensor::<CudaRuntime>::from_slice(&[2i64, 1, 0, 0], &[2, 2], &device)
        .expect("staging the coordinates must succeed");
    let row_idx = Tensor::<CudaRuntime>::from_slice(&[2i64, 0, 1], &[3], &device)
        .expect("staging the row indices must succeed");
    let scatter_idx = Tensor::<CudaRuntime>::from_slice(&[0i64, 0, 2], &[3], &device)
        .expect("staging the scatter indices must succeed");

    for &dtype in INT_DTYPES.iter() {
        let a = client
            .cast(&base, dtype)
            .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));
        let flat = client
            .cast(&base_flat, dtype)
            .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));
        let dst = client
            .cast(&base_flat, dtype)
            .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));
        let src = client
            .cast(&base_src, dtype)
            .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));

        for (name, out) in [
            ("masked_select", client.masked_select(&a, &mask_full)),
            (
                "masked_select_broadcast",
                client.masked_select(&a, &mask_row),
            ),
            ("masked_fill", client.masked_fill(&a, &mask_full, 1.0)),
            (
                "masked_fill_broadcast",
                client.masked_fill(&a, &mask_row, 1.0),
            ),
            ("gather_nd", client.gather_nd(&a, &coords)),
            ("index_select", client.index_select(&a, 0, &row_idx)),
            ("gather", client.gather(&a, 0, &coords)),
            ("embedding_lookup", client.embedding_lookup(&a, &row_idx)),
            ("unique", client.unique(&flat, true)),
            ("nonzero", client.nonzero(&flat)),
            (
                "scatter_reduce_sum",
                client.scatter_reduce(&dst, 0, &scatter_idx, &src, ScatterReduceOp::Sum, true),
            ),
            (
                "scatter_reduce_prod",
                client.scatter_reduce(&dst, 0, &scatter_idx, &src, ScatterReduceOp::Prod, true),
            ),
            (
                "scatter_reduce_max",
                client.scatter_reduce(&dst, 0, &scatter_idx, &src, ScatterReduceOp::Max, true),
            ),
            (
                "scatter_reduce_min",
                client.scatter_reduce(&dst, 0, &scatter_idx, &src, ScatterReduceOp::Min, true),
            ),
            (
                "scatter_reduce_mean",
                client.scatter_reduce(&dst, 0, &scatter_idx, &src, ScatterReduceOp::Mean, true),
            ),
        ] {
            out.unwrap_or_else(|e| panic!("{name} on {dtype:?} failed: {e:?}"));
        }
    }
    client.synchronize();
}
