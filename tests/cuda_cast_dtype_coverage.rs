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

use common::{create_cpu_client, create_cuda_client};
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

    let Some((cuda_client, cuda_device)) = create_cuda_client() else {
        return;
    };
    let cuda_in = Tensor::<CudaRuntime>::from_slice(input, &shape, &cuda_device)
        .unwrap_or_else(|e| panic!("{label}: staging the CUDA input failed: {e:?}"));
    let cuda_out = cuda_client
        .cast(&cuda_in, D::DTYPE)
        .unwrap_or_else(|e| panic!("{label}: CUDA cast failed: {e:?}"));
    assert_eq!(cuda_out.to_vec::<D>(), expected, "{label}: CUDA vs CPU");
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

/// Every ordered pair must resolve to a real kernel. A missing instantiation
/// fails here at module lookup, not months later in a caller.
#[test]
fn cast_matrix_covers_every_ordered_pair() {
    let Some((client, device)) = create_cuda_client() else {
        return;
    };
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
}
