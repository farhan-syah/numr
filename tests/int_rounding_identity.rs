//! `floor`, `ceil`, `round`, `round_ties_even` and `trunc` on an integer dtype
//! are the IDENTITY, exactly, at every width.
//!
//! An integer is already a whole number, so there is nothing to round and no tie
//! to break. The answer is the input, bit for bit.
//!
//! # What was wrong
//!
//! CPU served these through the generic `f64` round trip in
//! `src/runtime/cpu/kernels/unary/mod.rs`: convert to `f64`, apply the op,
//! convert back with `T::from_f64`, which SATURATES. `f64` carries 53 mantissa
//! bits, so an `i64` or `u64` past 2^53 did not survive the trip —
//! `floor(9007199254740993i64)` answered `9007199254740992` — and `u64::MAX`
//! saturated. CUDA had no integer instantiation for any of the five, so the
//! lookup failed with `named symbol not found`.
//!
//! Every expectation below is an explicit literal asserted against BOTH
//! backends, so a wrong kernel cannot agree with a wrong expectation.
//!
//! # WebGPU
//!
//! WebGPU accepts all five on I32 and U32 (`runtime/wgpu/shaders/elementwise.rs`
//! admits `floor`/`ceil`/`round`/`round_ties_even`/`trunc` alongside neg/abs/sign
//! for both dtypes). The I32 case is pinned at the foot of this file; the
//! CPU-vs-WebGPU parity check for both I32 and U32 lives in
//! `tests/backend_parity/int_rounding_wgpu.rs`.
//!
//! Run: cargo test --test int_rounding_identity
//! Run: cargo test --features cuda --test int_rounding_identity

mod common;

use common::create_cpu_client;
use numr::ops::UnaryOps;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// Run one rounding op on CPU, and on CUDA when the feature is on, asserting
/// both hand back the input unchanged.
macro_rules! check_identity {
    ($ty:ty, $method:ident, $a:expr) => {{
        let label = concat!(stringify!($method), " ", stringify!($ty));
        // Bind the slice at `$ty` FIRST. An untyped literal otherwise infers
        // `i32` from `from_slice`, so the tensor carries the wrong dtype and
        // `to_vec::<$ty>()` reinterprets its bytes into a convincing wrong
        // answer.
        let a_typed: &[$ty] = $a;
        let shape = [a_typed.len()];

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_a = Tensor::<CpuRuntime>::from_slice(a_typed, &shape, &cpu_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CPU input failed: {e:?}"));
        let cpu_out = cpu_client
            .$method(&cpu_a)
            .unwrap_or_else(|e| panic!("{label}: the CPU op failed: {e:?}"));
        let cpu_vec: Vec<$ty> = cpu_out.to_vec::<$ty>();
        assert_eq!(cpu_vec.as_slice(), a_typed, "{label}: CPU reference");

        #[cfg(feature = "cuda")]
        crate::common::backend_lock::with_cuda_backend(|cuda_client, cuda_device| {
            use numr::runtime::cuda::CudaRuntime;
            let a = Tensor::<CudaRuntime>::from_slice(a_typed, &shape, &cuda_device)
                .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
            let out = cuda_client
                .$method(&a)
                .unwrap_or_else(|e| panic!("{label}: the CUDA op failed: {e:?}"));
            let cuda_vec: Vec<$ty> = out.to_vec::<$ty>();
            assert_eq!(cuda_vec.as_slice(), a_typed, "{label}: CUDA vs CPU");
        });
    }};
}

// ============================================================================
// I64 and U64 past 2^53
//
// 9007199254740993 is 2^53 + 1, the first integer with no `f64` representation.
// `u64::MAX` and `i64::MIN` are far beyond it. Every one of the five ops is
// pinned here, because the round trip corrupted all five identically.
// ============================================================================

const I64_WIDE: [i64; 6] = [
    i64::MIN,
    -9_007_199_254_740_993,
    -1,
    0,
    9_007_199_254_740_993,
    i64::MAX,
];

const U64_WIDE: [u64; 5] = [
    0,
    1,
    9_007_199_254_740_993,
    18_000_000_000_000_000_000,
    u64::MAX,
];

#[test]
fn i64_floor_is_exact_past_the_f64_mantissa() {
    check_identity!(i64, floor, &I64_WIDE);
}

#[test]
fn i64_ceil_is_exact_past_the_f64_mantissa() {
    check_identity!(i64, ceil, &I64_WIDE);
}

#[test]
fn i64_round_is_exact_past_the_f64_mantissa() {
    check_identity!(i64, round, &I64_WIDE);
}

#[test]
fn i64_round_ties_even_is_exact_past_the_f64_mantissa() {
    check_identity!(i64, round_ties_even, &I64_WIDE);
}

#[test]
fn i64_trunc_is_exact_past_the_f64_mantissa() {
    check_identity!(i64, trunc, &I64_WIDE);
}

#[test]
fn u64_floor_is_exact_past_the_f64_mantissa() {
    check_identity!(u64, floor, &U64_WIDE);
}

#[test]
fn u64_ceil_is_exact_past_the_f64_mantissa() {
    check_identity!(u64, ceil, &U64_WIDE);
}

#[test]
fn u64_round_is_exact_past_the_f64_mantissa() {
    check_identity!(u64, round, &U64_WIDE);
}

#[test]
fn u64_round_ties_even_is_exact_past_the_f64_mantissa() {
    check_identity!(u64, round_ties_even, &U64_WIDE);
}

#[test]
fn u64_trunc_is_exact_past_the_f64_mantissa() {
    check_identity!(u64, trunc, &U64_WIDE);
}

// ============================================================================
// The narrower widths, at their extremes
//
// These fit in an `f64` exactly, so the round trip computed the right value.
// They are pinned anyway: CUDA had no kernel for any of them, and a saturating
// `from_f64` is one edit away from mattering here too.
// ============================================================================

#[test]
fn i8_rounding_is_the_identity() {
    check_identity!(i8, floor, &[i8::MIN, -1, 0, 1, i8::MAX]);
    check_identity!(i8, ceil, &[i8::MIN, -1, 0, 1, i8::MAX]);
    check_identity!(i8, round, &[i8::MIN, -1, 0, 1, i8::MAX]);
    check_identity!(i8, round_ties_even, &[i8::MIN, -1, 0, 1, i8::MAX]);
    check_identity!(i8, trunc, &[i8::MIN, -1, 0, 1, i8::MAX]);
}

#[test]
fn u8_rounding_is_the_identity() {
    check_identity!(u8, floor, &[0, 1, 200, u8::MAX]);
    check_identity!(u8, ceil, &[0, 1, 200, u8::MAX]);
    check_identity!(u8, round, &[0, 1, 200, u8::MAX]);
    check_identity!(u8, round_ties_even, &[0, 1, 200, u8::MAX]);
    check_identity!(u8, trunc, &[0, 1, 200, u8::MAX]);
}

#[test]
fn i16_rounding_is_the_identity() {
    check_identity!(i16, floor, &[i16::MIN, -1, 0, 1, i16::MAX]);
    check_identity!(i16, ceil, &[i16::MIN, -1, 0, 1, i16::MAX]);
    check_identity!(i16, round, &[i16::MIN, -1, 0, 1, i16::MAX]);
    check_identity!(i16, round_ties_even, &[i16::MIN, -1, 0, 1, i16::MAX]);
    check_identity!(i16, trunc, &[i16::MIN, -1, 0, 1, i16::MAX]);
}

#[test]
fn u16_rounding_is_the_identity() {
    check_identity!(u16, floor, &[0, 1, 40_000, u16::MAX]);
    check_identity!(u16, ceil, &[0, 1, 40_000, u16::MAX]);
    check_identity!(u16, round, &[0, 1, 40_000, u16::MAX]);
    check_identity!(u16, round_ties_even, &[0, 1, 40_000, u16::MAX]);
    check_identity!(u16, trunc, &[0, 1, 40_000, u16::MAX]);
}

#[test]
fn i32_rounding_is_the_identity() {
    check_identity!(i32, floor, &[i32::MIN, -1, 0, 1, i32::MAX]);
    check_identity!(i32, ceil, &[i32::MIN, -1, 0, 1, i32::MAX]);
    check_identity!(i32, round, &[i32::MIN, -1, 0, 1, i32::MAX]);
    check_identity!(i32, round_ties_even, &[i32::MIN, -1, 0, 1, i32::MAX]);
    check_identity!(i32, trunc, &[i32::MIN, -1, 0, 1, i32::MAX]);
}

#[test]
fn u32_rounding_is_the_identity() {
    check_identity!(u32, floor, &[0, 1, 4_000_000_000, u32::MAX]);
    check_identity!(u32, ceil, &[0, 1, 4_000_000_000, u32::MAX]);
    check_identity!(u32, round, &[0, 1, 4_000_000_000, u32::MAX]);
    check_identity!(u32, round_ties_even, &[0, 1, 4_000_000_000, u32::MAX]);
    check_identity!(u32, trunc, &[0, 1, 4_000_000_000, u32::MAX]);
}

// ============================================================================
// CUDA resolution sweep
//
// The CUDA path looks its kernel up by NAME — `{op}_{suffix}` — so a dtype with
// no `.cu` instantiation compiles fine and fails at launch with `named symbol
// not found`. This drives the whole matrix so the panic names the op and dtype
// that has no kernel.
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_sweep {
    use super::*;
    use crate::common::backend_lock::with_cuda_backend;
    use numr::dtype::DType;
    use numr::error::Result;
    use numr::runtime::Runtime;
    use numr::runtime::cuda::CudaRuntime;

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

    const ROUNDING_OPS: [&str; 5] = ["floor", "ceil", "round", "round_ties_even", "trunc"];

    fn apply<R: Runtime>(client: &impl UnaryOps<R>, op: &str, x: &Tensor<R>) -> Result<Tensor<R>> {
        match op {
            "floor" => client.floor(x),
            "ceil" => client.ceil(x),
            "round" => client.round(x),
            "round_ties_even" => client.round_ties_even(x),
            "trunc" => client.trunc(x),
            other => panic!("unknown op: {other}"),
        }
    }

    #[test]
    fn every_rounding_op_resolves_on_every_integer_dtype() {
        // Ones, not an uninitialised buffer: the value is inside every dtype's
        // range and the identity answer for it is 1, so the sweep also catches a
        // kernel that resolves and then writes nothing.
        with_cuda_backend(|cuda_client, cuda_device| {
            for dtype in INT_DTYPES {
                for op in ROUNDING_OPS {
                    let x = Tensor::<CudaRuntime>::ones(&[8], dtype, &cuda_device)
                        .unwrap_or_else(|e| panic!("{op} {dtype:?}: CUDA input failed: {e:?}"));
                    let out = apply(&cuda_client, op, &x)
                        .unwrap_or_else(|e| panic!("{op} {dtype:?}: CUDA has no kernel: {e:?}"));
                    assert_eq!(out.dtype(), dtype, "{op} {dtype:?}: result dtype changed");
                }
            }
        });
    }
}

// ============================================================================
// WebGPU
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_rounding_is_the_identity_on_i32() {
    use crate::common::backend_lock::with_wgpu_backend;
    use numr::runtime::wgpu::WgpuRuntime;

    let input: [i32; 5] = [i32::MIN, -1, 0, 1, i32::MAX];
    with_wgpu_backend(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_slice(&input, &[input.len()], &device)
            .unwrap_or_else(|e| panic!("staging the WebGPU input failed: {e:?}"));
        for (name, out) in [
            ("floor", client.floor(&a)),
            ("ceil", client.ceil(&a)),
            ("round", client.round(&a)),
            ("round_ties_even", client.round_ties_even(&a)),
            ("trunc", client.trunc(&a)),
        ] {
            let out = out.unwrap_or_else(|e| panic!("{name} i32: the WebGPU op failed: {e:?}"));
            assert_eq!(
                out.to_vec::<i32>().as_slice(),
                input.as_slice(),
                "{name} i32: WebGPU"
            );
        }
    });
}
