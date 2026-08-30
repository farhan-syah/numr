//! Backend parity for integer `sum` and `prod` at the saturation boundary.
//!
//! Integer accumulators in numr SATURATE at the output dtype's bounds — the
//! convention documented in `src/runtime/cpu/kernels/wide_acc.rs`, where
//! accumulators clamp and element-wise ops wrap. CPU has unit tests for it in
//! `src/runtime/cpu/helpers/reduce/mod.rs`; this file is the cross-backend half.
//!
//! It matters because the two GPU backends reach the same answer by different
//! routes: CUDA keeps a `Numr128` accumulator and narrows once at the store
//! (`src/runtime/cuda/kernels/reduce_int.cu`), while WebGPU has no 128-bit type
//! and carries a magnitude plus three flags — `NUMR_PROD_ZERO`,
//! `NUMR_PROD_SAT`, `NUMR_PROD_NEG` — so a clamped product's sign stays
//! recoverable (`src/runtime/wgpu/shaders/reduce_int_acc_i32.wgsl`). A zero
//! factor arriving AFTER the product has already saturated is where that scheme
//! can go wrong: the true product is 0, not the clamped bound.
//!
//! The existing `reduce.rs` sweep uses factors 2..7 over three elements, six
//! orders of magnitude short of any bound, so none of this was covered.

use numr::dtype::DType;
use numr::ops::ReduceOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

// ============================================================================
// Exact integer construction
// ============================================================================

/// The dtype's representable range, exactly.
///
/// `DType::min_value`/`max_value` return f64, which cannot represent
/// `i64::MAX` or `u64::MAX`, so the bounds a saturation test compares against
/// are carried in i128 instead.
fn int_bounds(dtype: DType) -> (i128, i128) {
    match dtype {
        DType::I8 => (i8::MIN as i128, i8::MAX as i128),
        DType::I16 => (i16::MIN as i128, i16::MAX as i128),
        DType::I32 => (i32::MIN as i128, i32::MAX as i128),
        DType::I64 => (i64::MIN as i128, i64::MAX as i128),
        DType::U8 => (0, u8::MAX as i128),
        DType::U16 => (0, u16::MAX as i128),
        DType::U32 => (0, u32::MAX as i128),
        DType::U64 => (0, u64::MAX as i128),
        other => panic!("int_bounds: {other:?} is not an integer dtype"),
    }
}

/// Build a tensor of `dtype` from exact i128 values.
///
/// Every value goes to the device in the dtype's own native Rust type. Routing
/// through f64 (as `tensor_from_f64` does) would round any value above 2^53,
/// which is exactly the region these tests live in.
fn int_tensor<R: Runtime<DType = DType>>(
    vals: &[i128],
    shape: &[usize],
    dtype: DType,
    device: &R::Device,
) -> Tensor<R> {
    macro_rules! build {
        ($T:ty) => {{
            let native: Vec<$T> = vals.iter().map(|&v| v as $T).collect();
            Tensor::<R>::from_slice(&native, shape, device).expect("int tensor allocation")
        }};
    }
    match dtype {
        DType::I8 => build!(i8),
        DType::I16 => build!(i16),
        DType::I32 => build!(i32),
        DType::I64 => build!(i64),
        DType::U8 => build!(u8),
        DType::U16 => build!(u16),
        DType::U32 => build!(u32),
        DType::U64 => build!(u64),
        other => panic!("int_tensor: {other:?} is not an integer dtype"),
    }
}

/// Read a tensor back as exact i128 values.
fn to_i128<R: Runtime<DType = DType>>(t: &Tensor<R>) -> Vec<i128> {
    match t.dtype() {
        DType::I8 => t.to_vec::<i8>().into_iter().map(|v| v as i128).collect(),
        DType::I16 => t.to_vec::<i16>().into_iter().map(|v| v as i128).collect(),
        DType::I32 => t.to_vec::<i32>().into_iter().map(|v| v as i128).collect(),
        DType::I64 => t.to_vec::<i64>().into_iter().map(|v| v as i128).collect(),
        DType::U8 => t.to_vec::<u8>().into_iter().map(|v| v as i128).collect(),
        DType::U16 => t.to_vec::<u16>().into_iter().map(|v| v as i128).collect(),
        DType::U32 => t.to_vec::<u32>().into_iter().map(|v| v as i128).collect(),
        DType::U64 => t.to_vec::<u64>().into_iter().map(|v| v as i128).collect(),
        other => panic!("to_i128: {other:?} is not an integer dtype"),
    }
}

// ============================================================================
// Case table
// ============================================================================

/// Which reduction a case exercises.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Sum,
    Prod,
}

/// The expected result, written against the dtype's bounds rather than a
/// literal, so one case covers every integer width.
#[derive(Clone, Copy, Debug)]
enum Expect {
    /// The dtype's maximum: the accumulator clamped upward.
    Max,
    /// The dtype's minimum: the accumulator clamped downward.
    Min,
    /// An exact value the reduction does not saturate to.
    Exact(i128),
}

impl Expect {
    fn value(self, dtype: DType) -> i128 {
        let (min, max) = int_bounds(dtype);
        match self {
            Self::Max => max,
            Self::Min => min,
            Self::Exact(v) => v,
        }
    }
}

/// One reduction, its input written in terms of the dtype's bounds.
struct Case {
    name: &'static str,
    op: Op,
    /// Input values as `f(min, max) -> Vec<i128>`, so a case can place the
    /// dtype's own boundary value in the data.
    build: fn(i128, i128) -> Vec<i128>,
    shape: &'static [usize],
    dims: &'static [usize],
    expected: Expect,
    /// Cases needing a negative value skip the unsigned dtypes.
    signed_only: bool,
}

fn cases() -> Vec<Case> {
    vec![
        // Sum past the top of the range. Two maxima plus one already exceeds
        // any width, so the accumulator must clamp rather than wrap to a
        // negative (signed) or small (unsigned) value.
        Case {
            name: "sum saturates at MAX",
            op: Op::Sum,
            build: |_min, max| vec![max, max, 1],
            shape: &[3],
            dims: &[0],
            expected: Expect::Max,
            signed_only: false,
        },
        // Sum past the bottom. Unsigned dtypes have no value below zero, so
        // this one is signed-only.
        Case {
            name: "sum saturates at MIN",
            op: Op::Sum,
            build: |min, _max| vec![min, min, -1],
            shape: &[3],
            dims: &[0],
            expected: Expect::Min,
            signed_only: true,
        },
        // Product past the top: max * 2 * 2 overflows every width.
        Case {
            name: "prod saturates at MAX",
            op: Op::Prod,
            build: |_min, max| vec![max, 2, 2],
            shape: &[3],
            dims: &[0],
            expected: Expect::Max,
            signed_only: false,
        },
        // Product past the bottom, with an ODD count of negative factors so
        // the clamped result must keep its sign. This is what WebGPU's
        // NUMR_PROD_NEG flag exists to preserve.
        Case {
            name: "prod saturates at MIN (odd negative count)",
            op: Op::Prod,
            build: |min, _max| vec![min, 2, 2],
            shape: &[3],
            dims: &[0],
            expected: Expect::Min,
            signed_only: true,
        },
        // The highest-value case: the zero arrives AFTER saturation. The true
        // product is 0, not the clamped bound, so a scheme that latches a
        // saturation flag and ignores later factors answers MAX here.
        Case {
            name: "prod zero factor after saturation",
            op: Op::Prod,
            build: |_min, max| vec![max, 2, 2, 0],
            shape: &[4],
            dims: &[0],
            expected: Expect::Exact(0),
            signed_only: false,
        },
        // Same, with the sign flag also set: negative, saturated, then zeroed.
        Case {
            name: "prod zero factor after negative saturation",
            op: Op::Prod,
            build: |min, _max| vec![min, 2, 2, 0],
            shape: &[4],
            dims: &[0],
            expected: Expect::Exact(0),
            signed_only: true,
        },
        // Sum with a zero after saturation still saturates: zero is the
        // additive identity, so the clamp stands.
        Case {
            name: "sum zero term after saturation",
            op: Op::Sum,
            build: |_min, max| vec![max, max, 0],
            shape: &[3],
            dims: &[0],
            expected: Expect::Max,
            signed_only: false,
        },
        // A reduction over exactly one element: no accumulation happens, and
        // the boundary value must survive the round trip unchanged.
        Case {
            name: "sum over one element at MAX",
            op: Op::Sum,
            build: |_min, max| vec![max],
            shape: &[1],
            dims: &[0],
            expected: Expect::Max,
            signed_only: false,
        },
        Case {
            name: "prod over one element at MIN",
            op: Op::Prod,
            build: |min, _max| vec![min],
            shape: &[1],
            dims: &[0],
            expected: Expect::Min,
            signed_only: true,
        },
        // Multi-dimension reduction that saturates. Chaining one reduction per
        // dimension would narrow the accumulator once per dimension; the whole
        // reduced set has to fold into a single wide accumulator.
        Case {
            name: "multi-dim sum saturates at MAX",
            op: Op::Sum,
            build: |_min, max| vec![max, max, max, max],
            shape: &[2, 2],
            dims: &[0, 1],
            expected: Expect::Max,
            signed_only: false,
        },
        Case {
            name: "multi-dim prod saturates at MAX",
            op: Op::Prod,
            build: |_min, max| vec![max, 2, 2, 2],
            shape: &[2, 2],
            dims: &[0, 1],
            expected: Expect::Max,
            signed_only: false,
        },
        // Multi-dim with the zero arriving in the LAST reduced position.
        Case {
            name: "multi-dim prod zero after saturation",
            op: Op::Prod,
            build: |_min, max| vec![max, 2, 2, 0],
            shape: &[2, 2],
            dims: &[0, 1],
            expected: Expect::Exact(0),
            signed_only: false,
        },
        // Controls: nothing here comes near a bound, so an implementation that
        // saturates too eagerly fails these.
        Case {
            name: "sum does not saturate",
            op: Op::Sum,
            build: |_min, _max| vec![2, 3, 4],
            shape: &[3],
            dims: &[0],
            expected: Expect::Exact(9),
            signed_only: false,
        },
        Case {
            name: "prod does not saturate",
            op: Op::Prod,
            build: |_min, _max| vec![2, 3, 4],
            shape: &[3],
            dims: &[0],
            expected: Expect::Exact(24),
            signed_only: false,
        },
        Case {
            name: "multi-dim sum does not saturate",
            op: Op::Sum,
            build: |_min, _max| vec![1, 2, 3, 4],
            shape: &[2, 2],
            dims: &[0, 1],
            expected: Expect::Exact(10),
            signed_only: false,
        },
    ]
}

/// True when `case` fits inside `dtype`'s range, both for its inputs and for
/// its non-saturating expectation.
///
/// U8's maximum is 255, so `Exact(24)` fits but a hypothetical `Exact(300)`
/// would not. Skipping such a pair keeps one case table usable for every width
/// instead of forcing a table per dtype.
fn case_applies(case: &Case, dtype: DType) -> bool {
    if case.signed_only && !dtype.is_signed_int() {
        return false;
    }
    let (min, max) = int_bounds(dtype);
    let vals = (case.build)(min, max);
    if vals.iter().any(|&v| v < min || v > max) {
        return false;
    }
    let expected = case.expected.value(dtype);
    expected >= min && expected <= max
}

fn reduce<R: Runtime<DType = DType>>(
    client: &impl ReduceOps<R>,
    op: Op,
    a: &Tensor<R>,
    dims: &[usize],
) -> Tensor<R> {
    match op {
        Op::Sum => client.sum(a, dims, false).expect("sum"),
        Op::Prod => client.prod(a, dims, false).expect("prod"),
    }
}

// ============================================================================
// CPU reference
// ============================================================================

/// CPU is the reference, so its answers are pinned against the written
/// expectation before any backend is compared to it.
#[test]
fn test_int_reduce_saturation_cpu_matches_expected() {
    let (client, device) = create_cpu_client();

    for dtype in parity_dtypes(DTypeDomain::IntsOnly, "cpu") {
        for case in cases() {
            if !case_applies(&case, dtype) {
                continue;
            }
            let (min, max) = int_bounds(dtype);
            let vals = (case.build)(min, max);
            let a = int_tensor::<CpuRuntime>(&vals, case.shape, dtype, &device);
            let out = reduce(&client, case.op, &a, case.dims);

            assert_eq!(
                out.dtype(),
                dtype,
                "{}: dtype={dtype:?}: reduction must not change dtype",
                case.name
            );
            assert_eq!(
                to_i128(&out),
                vec![case.expected.value(dtype)],
                "{}: dtype={dtype:?}: CPU reference value",
                case.name
            );
        }
    }
}

// ============================================================================
// CUDA vs CPU
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_int_reduce_saturation_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        for dtype in parity_dtypes(DTypeDomain::IntsOnly, "cuda") {
            for case in cases() {
                if !case_applies(&case, dtype) {
                    continue;
                }
                let (min, max) = int_bounds(dtype);
                let vals = (case.build)(min, max);

                let a_cpu = int_tensor::<CpuRuntime>(&vals, case.shape, dtype, &cpu_device);
                let expected = reduce(&cpu_client, case.op, &a_cpu, case.dims);

                let a = int_tensor(&vals, case.shape, dtype, &device);
                let actual = reduce(&client, case.op, &a, case.dims);

                assert_tensor_allclose(
                    &actual,
                    &expected,
                    dtype,
                    &format!("{} [{:?}] cuda vs cpu", case.name, case.op),
                );
            }
        }
    });
}

// ============================================================================
// WebGPU vs CPU
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_int_reduce_saturation_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        for dtype in parity_dtypes(DTypeDomain::IntsOnly, "wgpu") {
            for case in cases() {
                if !case_applies(&case, dtype) {
                    continue;
                }
                let (min, max) = int_bounds(dtype);
                let vals = (case.build)(min, max);

                let a_cpu = int_tensor::<CpuRuntime>(&vals, case.shape, dtype, &cpu_device);
                let expected = reduce(&cpu_client, case.op, &a_cpu, case.dims);

                let a = int_tensor(&vals, case.shape, dtype, &device);
                let actual = reduce(&client, case.op, &a, case.dims);

                assert_tensor_allclose(
                    &actual,
                    &expected,
                    dtype,
                    &format!("{} [{:?}] wgpu vs cpu", case.name, case.op),
                );
            }
        }
    });
}
