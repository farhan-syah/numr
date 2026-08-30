//! CUDA `cast` must cover every dtype pair the CPU backend covers, with the
//! same numerical semantics.
//!
//! The CUDA path looks its kernel up by NAME — `cast_{src}_{dst}` — so a pair
//! with no `.cu` instantiation compiles fine and fails at launch. Worse, the
//! Rust-side dtype gate used to reject U32 and the narrow integers outright,
//! which blocked every backend-parity test that stages tensors by casting from
//! F64.
//!
//! Semantics come from the CPU reference (`src/runtime/cpu/kernels/memory.rs`),
//! which routes every conversion through f64 and then applies a Rust `as` cast.
//! That makes float -> int saturating (NaN to 0, out of range to the nearest
//! bound) AND int -> int saturating, because the integer also passes through
//! f64 on the way.
//!
//! Run: cargo test --features cuda --test cuda_cast_dtype_coverage

#![cfg(feature = "cuda")]

mod common;

use common::backend_lock::with_cuda_backend;
use common::create_cpu_client;
use numr::dtype::{DType, Element};
use numr::ops::TypeConversionOps;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::CpuRuntime;
use numr::runtime::cuda::CudaRuntime;
use numr::tensor::Tensor;

/// Every dtype the `cast.cu` instantiation matrix covers.
const CAST_DTYPES: [DType; 15] = [
    DType::F32,
    DType::F64,
    DType::F16,
    DType::BF16,
    DType::FP8E4M3,
    DType::FP8E5M2,
    DType::I64,
    DType::I32,
    DType::I16,
    DType::I8,
    DType::U64,
    DType::U32,
    DType::U16,
    DType::U8,
    DType::Bool,
];

/// Cast `input` on CPU and on CUDA, asserting both match `expected`.
///
/// The CPU assertion pins the reference semantics; the CUDA assertion pins
/// parity with it. A CUDA-only check would pass a kernel that agrees with a
/// wrong expectation.
fn check_cast<S, D>(input: &[S], expected: &[D], label: &str)
where
    S: Element + bytemuck::Pod,
    D: Element + bytemuck::Pod + PartialEq + std::fmt::Debug,
{
    let shape = [input.len()];

    let (cpu_client, cpu_device) = create_cpu_client();
    let cpu_in = Tensor::<CpuRuntime>::from_slice(input, &shape, &cpu_device)
        .unwrap_or_else(|e| panic!("{label}: staging the CPU input failed: {e:?}"));
    let cpu_out = cpu_client
        .cast(&cpu_in, D::DTYPE)
        .unwrap_or_else(|e| panic!("{label}: CPU cast failed: {e:?}"));
    assert_eq!(cpu_out.to_vec::<D>(), expected, "{label}: CPU reference");

    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_in = Tensor::<CudaRuntime>::from_slice(input, &shape, &cuda_device)
            .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
        let cuda_out = cuda_client
            .cast(&cuda_in, D::DTYPE)
            .unwrap_or_else(|e| panic!("{label}: CUDA cast failed: {e:?}"));
        assert_eq!(cuda_out.to_vec::<D>(), expected, "{label}: CUDA vs CPU");
    });
}

/// The pair that blocked the parity suite: the harness stages U32 tensors by
/// casting from F64.
#[test]
fn cast_f64_to_u32_saturates() {
    check_cast::<f64, u32>(
        &[42.0, -1.0, 5.0e9, f64::NAN, 0.0],
        &[42, 0, u32::MAX, 0, 0],
        "f64 -> u32",
    );
}

/// A value above `i32::MAX` proves U32 is not being routed through a signed
/// type on the way back out.
#[test]
fn cast_u32_to_f64_keeps_the_high_range() {
    check_cast::<u32, f64>(
        &[3_000_000_000, 0, u32::MAX],
        &[3_000_000_000.0, 0.0, 4_294_967_295.0],
        "u32 -> f64",
    );
}

/// Integer to integer goes through f64 on the CPU, so it SATURATES rather than
/// wrapping: -1i32 becomes 0u32, not `u32::MAX`.
#[test]
fn cast_i32_to_u32_saturates_negatives() {
    check_cast::<i32, u32>(&[-1, i32::MIN, 7], &[0, 0, 7], "i32 -> u32");
}

#[test]
fn cast_u32_to_i32_saturates_high_values() {
    check_cast::<u32, i32>(
        &[3_000_000_000, 5, u32::MAX],
        &[i32::MAX, 5, i32::MAX],
        "u32 -> i32",
    );
}

#[test]
fn cast_f32_to_i8_saturates() {
    check_cast::<f32, i8>(
        &[200.0, -200.0, 3.7, -3.7, f32::NAN],
        &[127, -128, 3, -3, 0],
        "f32 -> i8",
    );
}

#[test]
fn cast_i32_to_i8_saturates() {
    check_cast::<i32, i8>(&[300, -300, 5, -5], &[127, -128, 5, -5], "i32 -> i8");
}

#[test]
fn cast_i64_to_u16_saturates() {
    check_cast::<i64, u16>(&[70_000, -3, 1234], &[65535, 0, 1234], "i64 -> u16");
}

/// f64 -> F16 narrows the way `half::f16::from_f64` narrows, which on x86-64
/// with F16C is f64 -> f32 -> f16: a DOUBLE rounding, not a single one.
///
/// 1 + 2^-11 is the exact midpoint between the F16 values 0x3c00 (1.0) and
/// 0x3c01 (1.0009765625). The input adds 2^-30 on top, which puts it above the
/// midpoint, so the two candidates are:
///
/// - 0x3c01 - a single IEEE rounding of the f64 sees the 2^-30 and rounds up.
/// - 0x3c00 - CPU. 2^-30 is below half an f32 ulp at 1.0 (2^-24), so the f32
///   stage drops it; the value is then exactly on the F16 midpoint and
///   ties-to-even rounds down.
#[cfg(feature = "f16")]
#[test]
fn cast_f64_to_f16_double_rounds_through_f32() {
    check_cast::<f64, half::f16>(
        &[1.0 + 1.0 / 2048.0 + 1.0 / 1_073_741_824.0],
        &[half::f16::from_bits(0x3c00)],
        "f64 -> f16 with bits below half an f32 ulp",
    );
}

/// f64 -> BF16 narrows the way `half::bf16::from_f64` narrows, which is neither
/// a single rounding nor an f32 stage: it discards the low 32 mantissa bits of
/// the f64 outright, then rounds the remaining 20 bits to 7, half-to-even.
///
/// 1 + 2^-8 is the exact midpoint between the BF16 values 0x3f80 (1.0) and
/// 0x3f81 (1.0078125). Both inputs add a bit on top of it, and both are 0x3f80
/// on CPU because both added bits land in the discarded low 32:
///
/// - `+ 2^-30` - candidates 0x3f81 (single rounding, which sees the bit) and
///   0x3f80 (CPU). An f32 stage would also give 0x3f80, since 2^-30 is below
///   half an f32 ulp at 1.0.
/// - `+ 2^-23` - candidates 0x3f81 (single rounding AND an f32 stage: 2^-23 is
///   above half an f32 ulp, so f32 keeps it and the sticky bits are non-zero)
///   and 0x3f80 (CPU). This is the value that separates BF16's rule from F16's.
#[cfg(feature = "f16")]
#[test]
fn cast_f64_to_bf16_drops_the_low_32_mantissa_bits() {
    check_cast::<f64, half::bf16>(
        &[
            1.0 + 1.0 / 256.0 + 1.0 / 1_073_741_824.0,
            1.0 + 1.0 / 256.0 + 1.0 / 8_388_608.0,
        ],
        &[half::bf16::from_bits(0x3f80), half::bf16::from_bits(0x3f80)],
        "f64 -> bf16 with bits below the discarded mantissa window",
    );
}

/// Every ordered pair must resolve to a real kernel. A missing instantiation
/// fails here at module lookup, not months later in a caller.
#[test]
fn cast_matrix_covers_every_ordered_pair() {
    with_cuda_backend(|client, device| {
        let base = Tensor::<CudaRuntime>::from_slice(&[1.0f64, 0.0, 2.0, 3.0], &[4], &device)
            .expect("staging the base tensor must succeed");

        for &src in CAST_DTYPES.iter() {
            let src_tensor = client
                .cast(&base, src)
                .unwrap_or_else(|e| panic!("cast f64 -> {src:?} failed: {e:?}"));
            for &dst in CAST_DTYPES.iter() {
                if src == dst {
                    continue;
                }
                client
                    .cast(&src_tensor, dst)
                    .unwrap_or_else(|e| panic!("cast {src:?} -> {dst:?} failed: {e:?}"));
            }
        }
        client.synchronize();
    });
}
