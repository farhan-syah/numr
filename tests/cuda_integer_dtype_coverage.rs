//! CUDA scan, creation, and semiring kernels must cover every integer dtype the
//! CPU backend covers, with the same numerical semantics.
//!
//! The CUDA path looks its kernel up by NAME — `cumsum_{suffix}`,
//! `arange_{suffix}`, `semiring_matmul_{suffix}` — so a dtype with no `.cu`
//! instantiation compiles fine and fails at launch with a missing-kernel error.
//! `cumsum`, `cumprod`, `arange`, `eye` and `linspace` had instantiations for
//! I32/I64/U32/U64 only, and `semiring_matmul` had none for I64.
//!
//! Semantics come from the CPU reference:
//!
//! * `cumsum` and `cumprod` are ACCUMULATORS, so they SATURATE at the element
//!   dtype's bound — see `runtime/cpu/kernels/wide_acc.rs`. The accumulator is
//!   128 bits wide and only the store clamps, so a running total that leaves
//!   the range and comes back reports the true value.
//! * `arange`, `eye` and `linspace` build every value in f64 and store it
//!   through `Element::from_f64`, which is Rust's saturating `as` cast: a
//!   negative value on an unsigned dtype becomes 0, and anything past the
//!   maximum clamps to the maximum.
//! * A semiring's `combine` is elementwise, so it WRAPS — `SemiringOp::combine`
//!   uses Rust's plain `+`. No case below overflows, because the CPU reference
//!   panics rather than wraps in a debug build.
//!
//! Run: cargo test --features cuda --test cuda_integer_dtype_coverage

#![cfg(feature = "cuda")]

mod common;

use common::{create_cpu_client, create_cuda_client};
use numr::dtype::DType;
use numr::ops::{CumulativeOps, SemiringMatmulOps, SemiringOp, UtilityOps};
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Run one scan on CPU and on CUDA, asserting both equal `expected`.
///
/// The CPU assertion pins the reference semantics; the CUDA assertion pins
/// parity with it. A CUDA-only check would pass a kernel that agrees with a
/// wrong expectation.
macro_rules! check_scan {
    ($ty:ty, $method:ident, $input:expr, $shape:expr, $dim:expr, $expected:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_in = Tensor::<CpuRuntime>::from_slice($input, $shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU input failed: {e:?}"));
        let cpu_out = cpu_client
            .$method(&cpu_in, $dim)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), $expected, "{label}: CPU reference");

        if let Some((cuda_client, cuda_device)) = create_cuda_client() {
            let input = Tensor::<CudaRuntime>::from_slice($input, $shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
            let out = cuda_client
                .$method(&input, $dim)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), $expected, "{label}: CUDA vs CPU");
        }
    }};
}

/// Run one tensor-creation call on both backends, asserting both equal
/// `expected`. The dtype travels in the call's own arguments.
macro_rules! check_creation {
    ($ty:ty, $method:ident ( $($arg:expr),* $(,)? ), $expected:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));

        let (cpu_client, _cpu_device) = create_cpu_client();
        let cpu_out = cpu_client
            .$method($($arg),*)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), $expected, "{label}: CPU reference");

        if let Some((cuda_client, _cuda_device)) = create_cuda_client() {
            let out = cuda_client
                .$method($($arg),*)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), $expected, "{label}: CUDA vs CPU");
        }
    }};
}

// ============================================================================
// cumsum saturates instead of wrapping
// ============================================================================

/// The running total is 100, 200, 100, 150. Only the store clamps, so index 2
/// reports the true 100 after the total left I8's range at index 1. A wrapping
/// accumulator answers [100, -56, 100, -106].
#[test]
fn i8_cumsum_saturates_and_recovers() {
    check_scan!(
        i8,
        cumsum,
        &[100i8, 100, -100, 50],
        &[4],
        -1,
        &[100i8, 127, 100, 127]
    );
}

/// Unsigned running totals only grow, so once the total passes U16's maximum
/// every later element clamps too. A wrapping accumulator answers
/// [60000, 4464, 4469].
#[test]
fn u16_cumsum_saturates() {
    check_scan!(
        u16,
        cumsum,
        &[60000u16, 10000, 5],
        &[3],
        -1,
        &[60000u16, 65535, 65535]
    );
}

/// The strided path — a scan along a non-last dimension — is a separate kernel,
/// so it needs its own saturation check. This scans the columns of
/// [[30000, 1], [30000, 2], [30000, 3]].
#[test]
fn i16_cumsum_strided_saturates() {
    check_scan!(
        i16,
        cumsum,
        &[30000i16, 1, 30000, 2, 30000, 3],
        &[3, 2],
        0,
        &[30000i16, 1, 32767, 3, 32767, 6]
    );
}

// ============================================================================
// cumprod saturates instead of wrapping
// ============================================================================

/// The true products are 200, -40000, -80000. Saturation keeps the SIGN right:
/// a product that overflows negative clamps to I16::MIN, not to I16::MAX.
#[test]
fn i16_cumprod_saturates_with_the_right_sign() {
    check_scan!(
        i16,
        cumprod,
        &[200i16, -200, 2],
        &[3],
        -1,
        &[200i16, -32768, -32768]
    );
}

/// 16 * 16 = 256 leaves U8 at index 1, and an integer product never shrinks, so
/// it never comes back.
#[test]
fn u8_cumprod_saturates() {
    check_scan!(u8, cumprod, &[16u8, 16, 2], &[3], -1, &[16u8, 255, 255]);
}

/// A zero factor pins the true product at 0 from there on, even though the
/// running product had already saturated.
#[test]
fn i8_cumprod_zero_beats_saturation() {
    check_scan!(
        i8,
        cumprod,
        &[100i8, 100, 0, 5],
        &[4],
        -1,
        &[100i8, 127, 0, 0]
    );
}

// ============================================================================
// arange
// ============================================================================

#[test]
fn i16_arange_counts_up() {
    check_creation!(i16, arange(0.0, 5.0, 1.0, DType::I16), &[0i16, 1, 2, 3, 4]);
}

/// 128 and 130 are past I8's maximum, so they clamp instead of wrapping to
/// -128 and -126.
#[test]
fn i8_arange_saturates_at_the_upper_bound() {
    check_creation!(
        i8,
        arange(120.0, 132.0, 2.0, DType::I8),
        &[120i8, 122, 124, 126, 127, 127]
    );
}

/// A negative start on an unsigned dtype clamps to 0. Wrapping the f64 through
/// the element type instead would answer [253, 254, 255, 0, 1].
#[test]
fn u8_arange_clamps_negatives_to_zero() {
    check_creation!(u8, arange(-3.0, 2.0, 1.0, DType::U8), &[0u8, 0, 0, 0, 1]);
}

// ============================================================================
// linspace
// ============================================================================

/// The endpoint 300 and the third value 200 are both past I8's maximum.
#[test]
fn i8_linspace_saturates_at_the_upper_bound() {
    check_creation!(
        i8,
        linspace(0.0, 300.0, 4, DType::I8),
        &[0i8, 100, 127, 127]
    );
}

/// Everything below zero clamps to 0 on an unsigned dtype.
#[test]
fn u16_linspace_clamps_negatives_to_zero() {
    check_creation!(
        u16,
        linspace(-10.0, 10.0, 5, DType::U16),
        &[0u16, 0, 0, 5, 10]
    );
}

// ============================================================================
// eye
// ============================================================================

#[test]
fn i8_eye_is_rectangular() {
    check_creation!(i8, eye(2, Some(3), DType::I8), &[1i8, 0, 0, 0, 1, 0]);
}

#[test]
fn u16_eye_is_square_by_default() {
    check_creation!(
        u16,
        eye(3, None, DType::U16),
        &[1u16, 0, 0, 0, 1, 0, 0, 0, 1]
    );
}

// ============================================================================
// semiring_matmul on I64
// ============================================================================
//
// `SemiringOp::validate_dtype` admits I64 for every semiring except OrAnd,
// which is Bool and U8 only. All five are covered below, sharing one pair of
// operands:
//
//   A = [[0, 3],   B = [[0, 2],
//        [7, 1]]        [5, 0]]

const SEMIRING_A: [i64; 4] = [0, 3, 7, 1];
const SEMIRING_B: [i64; 4] = [0, 2, 5, 0];

/// Run one semiring product on both backends, asserting both equal `expected`.
fn check_semiring_i64(op: SemiringOp, expected: &[i64; 4]) {
    let label = format!("semiring_matmul i64 {op}");

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_a = Tensor::<CpuRuntime>::from_slice(&SEMIRING_A, &[2, 2], &cpu_device)
        .unwrap_or_else(|e| panic!("{label}: staging the CPU lhs failed: {e:?}"));
    let cpu_b = Tensor::<CpuRuntime>::from_slice(&SEMIRING_B, &[2, 2], &cpu_device)
        .unwrap_or_else(|e| panic!("{label}: staging the CPU rhs failed: {e:?}"));
    let cpu_out = cpu_client
        .semiring_matmul(&cpu_a, &cpu_b, op)
        .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
    assert_eq!(
        cpu_out.to_vec::<i64>().as_slice(),
        expected,
        "{label}: CPU reference"
    );

    if let Some((cuda_client, cuda_device)) = create_cuda_client() {
        let a = Tensor::<CudaRuntime>::from_slice(&SEMIRING_A, &[2, 2], &cuda_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CUDA lhs failed: {e:?}"));
        let b = Tensor::<CudaRuntime>::from_slice(&SEMIRING_B, &[2, 2], &cuda_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CUDA rhs failed: {e:?}"));
        let out = cuda_client
            .semiring_matmul(&a, &b, op)
            .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
        assert_eq!(
            out.to_vec::<i64>().as_slice(),
            expected,
            "{label}: CUDA vs CPU"
        );
    }
}

/// Shortest paths: min(0+0, 3+5) = 0, min(0+2, 3+0) = 2, and so on.
#[test]
fn i64_semiring_min_plus() {
    check_semiring_i64(SemiringOp::MinPlus, &[0, 2, 6, 1]);
}

/// Longest paths. The identity is I64::MIN, the saturating narrowing of -inf.
#[test]
fn i64_semiring_max_plus() {
    check_semiring_i64(SemiringOp::MaxPlus, &[8, 3, 7, 9]);
}

/// Bottleneck capacity: reduce=max, combine=min.
#[test]
fn i64_semiring_max_min() {
    check_semiring_i64(SemiringOp::MaxMin, &[3, 0, 1, 2]);
}

/// Fuzzy relations: reduce=min, combine=max. The identity is I64::MAX.
#[test]
fn i64_semiring_min_max() {
    check_semiring_i64(SemiringOp::MinMax, &[0, 2, 5, 1]);
}

/// reduce=+, combine=max.
#[test]
fn i64_semiring_plus_max() {
    check_semiring_i64(SemiringOp::PlusMax, &[5, 5, 12, 8]);
}

// ============================================================================
// Instantiation sweep
// ============================================================================
//
// One resolution of every op against every integer dtype instantiated in
// `cumulative_int.cu` and `utility.cu`. A missing instantiation fails here at
// kernel lookup, and the panic message names the op and the dtype.
//
// The values are deliberately tiny — the semantics are pinned by the tests
// above — so that one expectation fits every width, signed and unsigned.

macro_rules! sweep_dtype {
    ($ty:ty, $dt:expr) => {{
        check_scan!(
            $ty,
            cumsum,
            &[1 as $ty, 2 as $ty, 3 as $ty],
            &[3],
            -1,
            &[1 as $ty, 3 as $ty, 6 as $ty]
        );
        check_scan!(
            $ty,
            cumsum,
            &[1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty],
            &[2, 2],
            0,
            &[1 as $ty, 2 as $ty, 4 as $ty, 6 as $ty]
        );
        check_scan!(
            $ty,
            cumprod,
            &[1 as $ty, 2 as $ty, 3 as $ty],
            &[3],
            -1,
            &[1 as $ty, 2 as $ty, 6 as $ty]
        );
        check_scan!(
            $ty,
            cumprod,
            &[1 as $ty, 2 as $ty, 3 as $ty, 4 as $ty],
            &[2, 2],
            0,
            &[1 as $ty, 2 as $ty, 3 as $ty, 8 as $ty]
        );
        check_creation!(
            $ty,
            arange(0.0, 3.0, 1.0, $dt),
            &[0 as $ty, 1 as $ty, 2 as $ty]
        );
        check_creation!(
            $ty,
            linspace(0.0, 2.0, 3, $dt),
            &[0 as $ty, 1 as $ty, 2 as $ty]
        );
        check_creation!(
            $ty,
            eye(2, None, $dt),
            &[1 as $ty, 0 as $ty, 0 as $ty, 1 as $ty]
        );
    }};
}

#[test]
fn every_integer_dtype_resolves_every_kernel() {
    sweep_dtype!(i64, DType::I64);
    sweep_dtype!(i32, DType::I32);
    sweep_dtype!(i16, DType::I16);
    sweep_dtype!(i8, DType::I8);
    sweep_dtype!(u64, DType::U64);
    sweep_dtype!(u32, DType::U32);
    sweep_dtype!(u16, DType::U16);
    sweep_dtype!(u8, DType::U8);
}
