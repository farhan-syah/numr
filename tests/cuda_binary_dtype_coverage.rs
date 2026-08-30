//! CUDA binary and comparison kernels must cover every dtype the CPU backend
//! covers, with the same numerical semantics.
//!
//! The CUDA path looks its kernel up by NAME — `{op}_{suffix}`,
//! `{op}_broadcast_{suffix}_inline`, `{op}_broadcast_fast_trailing_{suffix}` —
//! so a dtype with no `.cu` instantiation compiles fine and fails at launch
//! with `named symbol not found`. U32 and the narrow integers had no
//! instantiation at all.
//!
//! Semantics come from the CPU reference (`src/runtime/cpu/kernels/binary.rs`
//! and `binary_int.rs`):
//!
//! * add, sub, mul wrap on overflow.
//! * div by zero yields 0; `i32::MIN / -1` yields `i32::MIN`.
//! * pow is exact and saturating (`ipow.rs` / `ipow.cuh`).
//! * a comparison returns the INPUT dtype, 1 for true and 0 for false.
//!
//! Run: cargo test --features cuda --test cuda_binary_dtype_coverage

#![cfg(feature = "cuda")]

mod common;

use common::backend_lock::with_cuda_backend;
use common::create_cpu_client;
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, TypeConversionOps};
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Run one operation on CPU and on CUDA, asserting both equal `expected`.
///
/// The CPU assertion pins the reference semantics; the CUDA assertion pins
/// parity with it. A CUDA-only check would pass a kernel that agrees with a
/// wrong expectation.
macro_rules! check_binary {
    ($ty:ty, $method:ident, $a:expr, $a_shape:expr, $b:expr, $b_shape:expr, $expected:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_a = Tensor::<CpuRuntime>::from_slice($a, $a_shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU lhs failed: {e:?}"));
        let cpu_b = Tensor::<CpuRuntime>::from_slice($b, $b_shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU rhs failed: {e:?}"));
        let cpu_out = cpu_client
            .$method(&cpu_a, &cpu_b)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), $expected, "{label}: CPU reference");

        with_cuda_backend(|cuda_client, cuda_device| {
            let a = Tensor::<CudaRuntime>::from_slice($a, $a_shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA lhs failed: {e:?}"));
            let b = Tensor::<CudaRuntime>::from_slice($b, $b_shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA rhs failed: {e:?}"));
            let out = cuda_client
                .$method(&a, &b)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), $expected, "{label}: CUDA vs CPU");
        });
    }};
}

/// Same-shape operands, so both backends take the element-wise kernel.
macro_rules! check_elementwise {
    ($ty:ty, $method:ident, $a:expr, $b:expr, $expected:expr) => {{
        let shape = [$a.len()];
        check_binary!($ty, $method, $a, &shape, $b, &shape, $expected)
    }};
}

// ============================================================================
// U32 arithmetic
// ============================================================================

/// The operands every U32 arithmetic test below shares.
///
/// Index 2 overflows under mul, index 3 divides by zero, and index 4 goes
/// below zero under sub.
const U32_A: [u32; 5] = [10, 5, 4_000_000_000, 7, 1];
const U32_B: [u32; 5] = [3, 5, 2, 0, 4];

#[test]
fn u32_add_matches_cpu() {
    check_elementwise!(u32, add, &U32_A, &U32_B, &[13, 10, 4_000_000_002, 7, 5]);
}

/// `1 - 4` wraps to `u32::MAX - 2`. A saturating kernel would answer 0.
#[test]
fn u32_sub_wraps_below_zero() {
    check_elementwise!(
        u32,
        sub,
        &U32_A,
        &U32_B,
        &[7, 0, 3_999_999_998, 7, 4_294_967_293]
    );
}

/// `4_000_000_000 * 2` is 8e9, which wraps to `8e9 - 2^32`.
#[test]
fn u32_mul_wraps_past_u32_max() {
    check_elementwise!(u32, mul, &U32_A, &U32_B, &[30, 25, 3_705_032_704, 0, 4]);
}

/// Index 3 divides by zero, which yields 0 rather than trapping.
#[test]
fn u32_div_matches_cpu() {
    check_elementwise!(u32, div, &U32_A, &U32_B, &[3, 1, 2_000_000_000, 0, 0]);
}

/// `4_000_000_000 ** 2` leaves the dtype, and pow SATURATES where the other
/// arithmetic wraps: its result is an accumulator.
#[test]
fn u32_pow_saturates_on_overflow() {
    check_elementwise!(u32, pow, &U32_A, &U32_B, &[1000, 3125, u32::MAX, 1, 1]);
}

#[test]
fn u32_maximum_matches_cpu() {
    check_elementwise!(u32, maximum, &U32_A, &U32_B, &[10, 5, 4_000_000_000, 7, 4]);
}

#[test]
fn u32_minimum_matches_cpu() {
    check_elementwise!(u32, minimum, &U32_A, &U32_B, &[3, 5, 2, 0, 1]);
}

// ============================================================================
// U32 comparison — output keeps the input dtype
// ============================================================================

#[test]
fn u32_eq_matches_cpu() {
    check_elementwise!(u32, eq, &U32_A, &U32_B, &[0, 1, 0, 0, 0]);
}

#[test]
fn u32_ne_matches_cpu() {
    check_elementwise!(u32, ne, &U32_A, &U32_B, &[1, 0, 1, 1, 1]);
}

/// Index 2 holds a value above `i32::MAX`: a kernel that compared as signed
/// would call it less than 2.
#[test]
fn u32_lt_matches_cpu() {
    check_elementwise!(u32, lt, &U32_A, &U32_B, &[0, 0, 0, 0, 1]);
}

#[test]
fn u32_le_matches_cpu() {
    check_elementwise!(u32, le, &U32_A, &U32_B, &[0, 1, 0, 0, 1]);
}

#[test]
fn u32_gt_matches_cpu() {
    check_elementwise!(u32, gt, &U32_A, &U32_B, &[1, 0, 1, 1, 0]);
}

#[test]
fn u32_ge_matches_cpu() {
    check_elementwise!(u32, ge, &U32_A, &U32_B, &[1, 1, 1, 1, 0]);
}

// ============================================================================
// Signed wrapping
// ============================================================================

/// `i32::MAX + 1` wraps to `i32::MIN`. Plain `a + b` on `int` is undefined
/// behaviour in C++ on overflow, so the CUDA kernel does the arithmetic in the
/// unsigned counterpart; this pins the result the compiler is free to break.
#[test]
fn i32_add_wraps_at_the_signed_bound() {
    check_elementwise!(
        i32,
        add,
        &[i32::MAX, i32::MIN, 5],
        &[1, -1, 3],
        &[i32::MIN, i32::MAX, 8]
    );
}

#[test]
fn i32_mul_wraps_at_the_signed_bound() {
    check_elementwise!(i32, mul, &[i32::MAX, -7, 6], &[2, 3, 7], &[-2, -21, 42]);
}

/// `i32::MIN / -1` overflows the type; CPU's `wrapping_div` answers
/// `i32::MIN`. A zero divisor yields 0.
#[test]
fn i32_div_defines_its_overflow_cases() {
    check_elementwise!(
        i32,
        div,
        &[i32::MIN, 7, -9],
        &[-1, 0, 2],
        &[i32::MIN, 0, -4]
    );
}

#[test]
fn i16_sub_wraps_at_the_signed_bound() {
    check_elementwise!(
        i16,
        sub,
        &[i16::MIN, 100i16],
        &[1i16, 40i16],
        &[i16::MAX, 60i16]
    );
}

#[test]
fn u8_add_wraps_at_the_byte_bound() {
    check_elementwise!(u8, add, &[250u8, 3u8], &[10u8, 4u8], &[4u8, 7u8]);
}

// ============================================================================
// Broadcasting
// ============================================================================

/// `[2,3] + [1,3]` is the contiguous trailing broadcast, which has its own
/// `{op}_broadcast_fast_trailing_{suffix}` kernel.
#[test]
fn u32_add_broadcasts_over_rows() {
    check_binary!(
        u32,
        add,
        &[1u32, 2, 3, 4, 5, 6],
        &[2, 3],
        &[10u32, 0, 3],
        &[1, 3],
        &[11, 2, 6, 14, 5, 9]
    );
}

/// `[2,3] + [2,1]` is not a trailing broadcast, so it takes the general
/// `{op}_broadcast_{suffix}_inline` kernel instead.
#[test]
fn u32_add_broadcasts_over_columns() {
    check_binary!(
        u32,
        add,
        &[1u32, 2, 3, 4, 5, 6],
        &[2, 3],
        &[10u32, 100],
        &[2, 1],
        &[11, 12, 13, 104, 105, 106]
    );
}

/// Comparison has only the pointer-strided broadcast kernel
/// (`{op}_broadcast_{suffix}`), a different instantiation from the arithmetic
/// one above.
#[test]
fn u32_lt_broadcasts_over_rows() {
    check_binary!(
        u32,
        lt,
        &[1u32, 2, 3, 4, 5, 6],
        &[2, 3],
        &[10u32, 0, 3],
        &[1, 3],
        &[1, 0, 0, 1, 0, 0]
    );
}

// ============================================================================
// Whole-matrix kernel resolution
// ============================================================================

/// Every dtype the binary and compare instantiation matrices claim to cover.
/// F16, BF16 and the FP8 types are feature-gated, so they are exercised by the
/// backend-parity suite rather than here.
const BINARY_DTYPES: [DType; 10] = [
    DType::F32,
    DType::F64,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
];

/// A missing instantiation fails here at module lookup, not months later in a
/// caller. Divisors are non-zero and exponents small so the check is about
/// kernel resolution, not about values.
#[test]
fn every_dtype_resolves_a_kernel_for_every_op() {
    with_cuda_backend(|client, device| {
        let base_a = Tensor::<CudaRuntime>::from_slice(&[4.0f64, 3.0, 2.0, 1.0], &[4], &device)
            .expect("staging the lhs must succeed");
        let base_b = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0, 1.0, 2.0], &[4], &device)
            .expect("staging the rhs must succeed");

        for &dtype in BINARY_DTYPES.iter() {
            let a = client
                .cast(&base_a, dtype)
                .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));
            let b = client
                .cast(&base_b, dtype)
                .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));

            for (name, out) in [
                ("add", client.add(&a, &b)),
                ("sub", client.sub(&a, &b)),
                ("mul", client.mul(&a, &b)),
                ("div", client.div(&a, &b)),
                ("pow", client.pow(&a, &b)),
                ("maximum", client.maximum(&a, &b)),
                ("minimum", client.minimum(&a, &b)),
                ("eq", client.eq(&a, &b)),
                ("ne", client.ne(&a, &b)),
                ("lt", client.lt(&a, &b)),
                ("le", client.le(&a, &b)),
                ("gt", client.gt(&a, &b)),
                ("ge", client.ge(&a, &b)),
            ] {
                out.unwrap_or_else(|e| panic!("{name} on {dtype:?} failed: {e:?}"));
            }
        }
        client.synchronize();
    });
}

/// The broadcast kernels are separate instantiations from the element-wise
/// ones, so they need their own resolution sweep.
#[test]
fn every_dtype_resolves_a_broadcast_kernel() {
    with_cuda_backend(|client, device| {
        let base_a = Tensor::<CudaRuntime>::from_slice(&[4.0f64, 3.0, 2.0, 1.0], &[2, 2], &device)
            .expect("staging the lhs must succeed");
        let base_b = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 2.0], &[1, 2], &device)
            .expect("staging the rhs must succeed");

        for &dtype in BINARY_DTYPES.iter() {
            let a = client
                .cast(&base_a, dtype)
                .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));
            let b = client
                .cast(&base_b, dtype)
                .unwrap_or_else(|e| panic!("cast f64 -> {dtype:?} failed: {e:?}"));

            for (name, out) in [
                ("add", client.add(&a, &b)),
                ("sub", client.sub(&a, &b)),
                ("mul", client.mul(&a, &b)),
                ("div", client.div(&a, &b)),
                ("pow", client.pow(&a, &b)),
                ("maximum", client.maximum(&a, &b)),
                ("minimum", client.minimum(&a, &b)),
                ("eq", client.eq(&a, &b)),
                ("ne", client.ne(&a, &b)),
                ("lt", client.lt(&a, &b)),
                ("le", client.le(&a, &b)),
                ("gt", client.gt(&a, &b)),
                ("ge", client.ge(&a, &b)),
            ] {
                out.unwrap_or_else(|e| panic!("broadcast {name} on {dtype:?} failed: {e:?}"));
            }
        }
        client.synchronize();
    });
}
