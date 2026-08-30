//! CUDA unary kernels must cover every integer dtype the CPU backend covers,
//! with the same numerical semantics.
//!
//! The CUDA path looks its kernel up by NAME — `{op}_{suffix}` — so a dtype
//! with no `.cu` instantiation compiles fine and fails at launch with
//! `named symbol not found`. Before `unary_int.cu` there was no integer unary
//! kernel at all for `u8`, `u16`, `u32` or `u64`, and `i8`/`i16` carried only
//! `neg`, `abs` and `sign`.
//!
//! # The integer-defined unary ops
//!
//! `UnaryOps` (`src/ops/traits/unary.rs`) is mostly transcendental and rounding
//! ops, which are float-only in this crate. Four members are defined on an
//! integer dtype and have integer kernels:
//!
//! * `neg` — every integer dtype. Wrapping on the unsigned ones.
//! * `abs` — every integer dtype, the identity on the unsigned ones.
//! * `sign` — every integer dtype, -1/0/1 signed and 0/1 unsigned.
//! * `square` — every integer dtype.
//!
//! # Semantics, from the CPU reference
//!
//! `src/runtime/cpu/kernels/unary/int.rs`, following the convention
//! `wide_acc.rs` documents: element-wise integer ops WRAP, accumulators
//! saturate. So `neg(i32::MIN)`, `abs(i32::MIN)` and `square(u8: 16)` answer
//! `i32::MIN`, `i32::MIN` and `0`.
//!
//! `neg` on an UNSIGNED dtype wraps too: it is `0 - a` in modular arithmetic,
//! so `neg(1u32)` is `u32::MAX` and `neg(0)` is 0. That is what this crate's
//! unsigned `sub` answers and what NumPy answers, so rejecting it would put
//! `neg` at odds with the very next operation.
//!
//! Every expectation below is an explicit literal asserted against BOTH
//! backends, so a wrong kernel cannot agree with a wrong expectation.
//!
//! Run: cargo test --features cuda --test cuda_unary_int_coverage

#![cfg(feature = "cuda")]

mod common;

use common::backend_lock::with_cuda_backend;
use common::create_cpu_client;
use numr::dtype::DType;
use numr::ops::UnaryOps;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Run one unary op on CPU and on CUDA, asserting both equal `expected`.
///
/// The CPU assertion pins the reference semantics; the CUDA assertion pins
/// parity with it. A CUDA-only check would pass a kernel that agrees with a
/// wrong expectation.
macro_rules! check_unary {
    ($ty:ty, $method:ident, $a:expr, $expected:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));
        // Bind both slices at `$ty` first. Untyped literals otherwise infer
        // `i32` from `from_slice`, so the tensor carries the WRONG dtype and
        // `to_vec::<$ty>()` reinterprets its bytes — `100i32` squared reads
        // back as `[16, 39, 0]` for `i8`.
        let a_typed: &[$ty] = $a;
        let expected_typed: &[$ty] = $expected;
        let shape = [a_typed.len()];

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_a = Tensor::<CpuRuntime>::from_slice(a_typed, &shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU input failed: {e:?}"));
        let cpu_out = cpu_client
            .$method(&cpu_a)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), expected_typed, "{label}: CPU reference");

        with_cuda_backend(|cuda_client, cuda_device| {
            let a = Tensor::<CudaRuntime>::from_slice(a_typed, &shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
            let out = cuda_client
                .$method(&a)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), expected_typed, "{label}: CUDA vs CPU");
        });
    }};
}

// ============================================================================
// abs and sign on the unsigned widths
//
// `abs` is the identity and `sign` is 0-or-1. Both had no CUDA kernel at all
// for any unsigned dtype.
// ============================================================================

const U8_A: [u8; 4] = [0, 1, 200, u8::MAX];
const U16_A: [u16; 4] = [0, 1, 40_000, u16::MAX];
const U32_A: [u32; 4] = [0, 1, 4_000_000_000, u32::MAX];
const U64_A: [u64; 4] = [0, 1, 18_000_000_000_000_000_000, u64::MAX];

#[test]
fn u8_abs_is_the_identity() {
    check_unary!(u8, abs, &U8_A, &[0, 1, 200, u8::MAX]);
}

#[test]
fn u8_sign_is_zero_or_one() {
    check_unary!(u8, sign, &U8_A, &[0, 1, 1, 1]);
}

#[test]
fn u16_abs_is_the_identity() {
    check_unary!(u16, abs, &U16_A, &[0, 1, 40_000, u16::MAX]);
}

#[test]
fn u16_sign_is_zero_or_one() {
    check_unary!(u16, sign, &U16_A, &[0, 1, 1, 1]);
}

#[test]
fn u32_abs_is_the_identity() {
    check_unary!(u32, abs, &U32_A, &[0, 1, 4_000_000_000, u32::MAX]);
}

#[test]
fn u32_sign_is_zero_or_one() {
    check_unary!(u32, sign, &U32_A, &[0, 1, 1, 1]);
}

/// The two largest inputs are past 2^53, where the old `f64` round trip on CPU
/// lost the low bits outright.
#[test]
fn u64_abs_is_the_identity() {
    check_unary!(
        u64,
        abs,
        &U64_A,
        &[0, 1, 18_000_000_000_000_000_000, u64::MAX]
    );
}

#[test]
fn u64_sign_is_zero_or_one() {
    check_unary!(u64, sign, &U64_A, &[0, 1, 1, 1]);
}

// ============================================================================
// abs and sign on the narrow signed widths
//
// CUDA had `neg`, `abs` and `sign` here already; these pin them at the extremes
// alongside the new widths so one sweep covers every integer dtype.
// ============================================================================

const I8_A: [i8; 5] = [i8::MIN, -1, 0, 1, i8::MAX];
const I16_A: [i16; 5] = [i16::MIN, -1, 0, 1, i16::MAX];

#[test]
fn i8_abs_wraps_at_the_minimum() {
    check_unary!(i8, abs, &I8_A, &[i8::MIN, 1, 0, 1, i8::MAX]);
}

#[test]
fn i8_sign_reports_three_values() {
    check_unary!(i8, sign, &I8_A, &[-1, -1, 0, 1, 1]);
}

#[test]
fn i16_abs_wraps_at_the_minimum() {
    check_unary!(i16, abs, &I16_A, &[i16::MIN, 1, 0, 1, i16::MAX]);
}

#[test]
fn i16_sign_reports_three_values() {
    check_unary!(i16, sign, &I16_A, &[-1, -1, 0, 1, 1]);
}

// ============================================================================
// neg and abs wrap at every signed minimum
//
// A saturating kernel answers MAX for both. The wrap is the contract.
// ============================================================================

#[test]
fn i8_neg_wraps_at_the_minimum() {
    check_unary!(i8, neg, &I8_A, &[i8::MIN, 1, 0, -1, -127]);
}

#[test]
fn i16_neg_wraps_at_the_minimum() {
    check_unary!(i16, neg, &I16_A, &[i16::MIN, 1, 0, -1, -32_767]);
}

#[test]
fn i32_neg_wraps_at_the_minimum() {
    check_unary!(
        i32,
        neg,
        &[i32::MIN, -1, 0, 1, i32::MAX],
        &[i32::MIN, 1, 0, -1, -2_147_483_647]
    );
}

#[test]
fn i32_abs_wraps_at_the_minimum() {
    check_unary!(
        i32,
        abs,
        &[i32::MIN, -1, 0, 1, i32::MAX],
        &[i32::MIN, 1, 0, 1, i32::MAX]
    );
}

/// `i64::MIN` and `i64::MAX` are both past 2^53, so this also pins the widths
/// an `f64` round trip cannot carry.
#[test]
fn i64_neg_wraps_at_the_minimum() {
    check_unary!(
        i64,
        neg,
        &[i64::MIN, -1, 0, 1, i64::MAX],
        &[i64::MIN, 1, 0, -1, -9_223_372_036_854_775_807]
    );
}

#[test]
fn i64_abs_wraps_at_the_minimum() {
    check_unary!(
        i64,
        abs,
        &[i64::MIN, -1, 0, 1, i64::MAX],
        &[i64::MIN, 1, 0, 1, i64::MAX]
    );
}

#[test]
fn i32_sign_reports_three_values() {
    check_unary!(
        i32,
        sign,
        &[i32::MIN, -1, 0, 1, i32::MAX],
        &[-1, -1, 0, 1, 1]
    );
}

#[test]
fn i64_sign_reports_three_values() {
    check_unary!(
        i64,
        sign,
        &[i64::MIN, -1, 0, 1, i64::MAX],
        &[-1, -1, 0, 1, 1]
    );
}

// ============================================================================
// square wraps rather than saturating
//
// Each input below leaves its dtype's range when squared. A saturating kernel
// answers MAX instead.
// ============================================================================

/// `100 * 100` is 10000, which is `16` modulo 256.
#[test]
fn i8_square_wraps() {
    check_unary!(i8, square, &[100, -100, 3], &[16, 16, 9]);
}

/// `16 * 16` is exactly 256, which is 0 modulo 256; `255 * 255` is 65025.
#[test]
fn u8_square_wraps() {
    check_unary!(u8, square, &[16, u8::MAX, 3], &[0, 1, 9]);
}

/// `300 * 300` is 90000, which is 24464 modulo 65536.
#[test]
fn i16_square_wraps() {
    check_unary!(i16, square, &[300, -300, 3], &[24_464, 24_464, 9]);
}

#[test]
fn u16_square_wraps() {
    check_unary!(u16, square, &[300, 40_000, 3], &[24_464, 4_096, 9]);
}

/// `100_000 * 100_000` is 1e10, which is 1_410_065_408 modulo 2^32.
#[test]
fn i32_square_wraps() {
    check_unary!(
        i32,
        square,
        &[100_000, -100_000, 3],
        &[1_410_065_408, 1_410_065_408, 9]
    );
}

#[test]
fn u32_square_wraps() {
    check_unary!(u32, square, &[100_000, 3], &[1_410_065_408, 9]);
}

/// `2^32` squared is exactly `2^64`, which is 0 modulo 2^64.
#[test]
fn i64_square_wraps() {
    check_unary!(i64, square, &[4_294_967_296i64, 3], &[0i64, 9]);
}

#[test]
fn u64_square_wraps() {
    check_unary!(u64, square, &[4_294_967_296u64, 3], &[0u64, 9]);
}

// ============================================================================
// neg on an unsigned dtype wraps, identically, on both backends
//
// `neg(0)` is 0 and `neg(1)` is the dtype's MAX. A kernel that saturated, or a
// backend that refused, disagrees at the second element.
// ============================================================================

#[test]
fn u8_neg_wraps() {
    check_unary!(u8, neg, &U8_A, &[0, u8::MAX, 56, 1]);
}

#[test]
fn u16_neg_wraps() {
    check_unary!(u16, neg, &U16_A, &[0, u16::MAX, 25_536, 1]);
}

#[test]
fn u32_neg_wraps() {
    check_unary!(u32, neg, &U32_A, &[0, u32::MAX, 294_967_296, 1]);
}

/// The two largest inputs are past 2^53, where an `f64` round trip loses the
/// low bits outright.
#[test]
fn u64_neg_wraps() {
    check_unary!(u64, neg, &U64_A, &[0, u64::MAX, 446_744_073_709_551_616, 1]);
}

// ============================================================================
// Resolution sweep
//
// Every integer-defined op against every integer dtype. A missing `.cu`
// instantiation fails here at kernel lookup, and the panic names the op and the
// dtype that has no kernel.
// ============================================================================

/// Every integer dtype, signed and unsigned.
const INT_DTYPES: [DType; 8] = [
    DType::I8,
    DType::I16,
    DType::I32,
    DType::I64,
    DType::U8,
    DType::U16,
    DType::U32,
    DType::U64,
];

/// The unary ops defined on an integer dtype. All four apply to every one of
/// them, signed and unsigned alike.
const INT_UNARY_OPS: [&str; 4] = ["neg", "abs", "sign", "square"];

/// Apply `op` by name, so the sweep can drive the whole matrix.
fn apply<R: numr::runtime::Runtime>(
    client: &impl UnaryOps<R>,
    op: &str,
    x: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "neg" => client.neg(x),
        "abs" => client.abs(x),
        "sign" => client.sign(x),
        "square" => client.square(x),
        other => panic!("unknown op: {other}"),
    }
}

#[test]
fn every_integer_defined_op_resolves_on_every_integer_dtype() {
    // Ones, not an uninitialised buffer: the value is inside every dtype's
    // range and every op's answer for it is 1, so the sweep also catches a
    // kernel that resolves and then computes nothing.
    with_cuda_backend(|cuda_client, cuda_device| {
        let (cpu_client, cpu_device) = create_cpu_client();

        for dtype in INT_DTYPES {
            for op in INT_UNARY_OPS {
                let cpu_x = Tensor::<CpuRuntime>::ones(&[8], dtype, &cpu_device)
                    .unwrap_or_else(|e| panic!("{op} {dtype:?}: CPU input failed: {e:?}"));
                apply(&cpu_client, op, &cpu_x)
                    .unwrap_or_else(|e| panic!("{op} {dtype:?}: CPU has no kernel: {e:?}"));

                let x = Tensor::<CudaRuntime>::ones(&[8], dtype, &cuda_device)
                    .unwrap_or_else(|e| panic!("{op} {dtype:?}: CUDA input failed: {e:?}"));
                apply(&cuda_client, op, &x)
                    .unwrap_or_else(|e| panic!("{op} {dtype:?}: CUDA has no kernel: {e:?}"));
            }
        }
    });
}
