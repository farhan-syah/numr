//! Backend parity coverage for `DType::Bool`.
//!
//! `Bool` is listed as a supported dtype on CPU, CUDA, and WebGPU (see
//! numr/CLAUDE.md's dtype matrix), but almost no operation actually computes
//! on it: `dispatch_dtype!` has no Bool arm, so unary/binary/reduce/compare
//! all reject it, and the "boolean tensor" operations (logical ops, masking)
//! use `U8` (CPU/CUDA) or `U32` (WebGPU) by design instead. What actually
//! touches Bool is narrow: `cast`, `fill`/`zeros`/`ones`, and
//! `semiring_matmul`'s `OrAnd` op. This file pins:
//!
//! - Real computation on Bool (cast, fill, zeros/ones, OrAnd matmul) agreeing
//!   bit-for-bit across CPU, CUDA, and WebGPU.
//! - Every op that does NOT support Bool rejecting it with the SAME error
//!   variant and payload on every backend, so a future change that silently
//!   starts accepting it (or crashes uglier on one backend) fails here.
//!
//! CPU is the reference; CUDA and WebGPU must match it exactly (Bool results
//! are asserted exactly, never with a tolerance).

mod common;

#[cfg(feature = "cuda")]
use common::backend_lock::with_cuda_backend;
#[cfg(feature = "wgpu")]
use common::backend_lock::with_wgpu_backend_or_skip;
use common::create_cpu_client;

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{CompareOps, IndexingOps, LogicalOps, ReduceOps, SemiringOp, UnaryOps};
use numr::ops::{SemiringMatmulOps, TypeConversionOps, UtilityOps};
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

// ============================================================================
// cast: Bool -> numeric, numeric -> Bool
// ============================================================================

/// `Bool -> F32` on the CPU reference: the stored byte feeds the cast
/// directly (not a collapsed 0/1), matching CUDA's `numr_bool` cast and the
/// WebGPU boundary conversion after the fix in
/// `src/ops/wgpu/type_conversion.rs`. A byte holding something other than 0
/// or 1 (possible only via `Tensor::from_bytes`, never via `from_slice::<bool>`
/// or any real op) still casts identically everywhere.
#[test]
fn cast_bool_raw_byte_to_f32_cpu_reference() {
    let (client, device) = create_cpu_client();
    let raw: [u8; 4] = [0, 1, 5, 200];
    let t = Tensor::<CpuRuntime>::from_bytes(&raw, &[4], DType::Bool, &device).unwrap();
    let out: Vec<f32> = client.cast(&t, DType::F32).unwrap().to_vec();
    assert_eq!(out, vec![0.0, 1.0, 5.0, 200.0]);
}

#[cfg(feature = "cuda")]
#[test]
fn cast_bool_raw_byte_to_f32_cuda_matches_cpu() {
    let (cpu, cpu_dev) = create_cpu_client();
    let raw: [u8; 4] = [0, 1, 5, 200];
    let cpu_t = Tensor::<CpuRuntime>::from_bytes(&raw, &[4], DType::Bool, &cpu_dev).unwrap();
    let expected: Vec<f32> = cpu.cast(&cpu_t, DType::F32).unwrap().to_vec();

    with_cuda_backend(|client, device| {
        let t = Tensor::<CudaRuntime>::from_bytes(&raw, &[4], DType::Bool, &device).unwrap();
        let out: Vec<f32> = client.cast(&t, DType::F32).unwrap().to_vec();
        assert_eq!(out, expected);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn cast_bool_raw_byte_to_f32_wgpu_matches_cpu() {
    let (cpu, cpu_dev) = create_cpu_client();
    let raw: [u8; 4] = [0, 1, 5, 200];
    let cpu_t = Tensor::<CpuRuntime>::from_bytes(&raw, &[4], DType::Bool, &cpu_dev).unwrap();
    let expected: Vec<f32> = cpu.cast(&cpu_t, DType::F32).unwrap().to_vec();

    with_wgpu_backend_or_skip(|client, device| {
        let t = Tensor::<WgpuRuntime>::from_bytes(&raw, &[4], DType::Bool, &device).unwrap();
        let out: Vec<f32> = client.cast(&t, DType::F32).unwrap().to_vec();
        assert_eq!(out, expected);
    });
}

/// WebGPU pads every buffer to a 4-byte boundary
/// (`src/runtime/wgpu/client.rs`'s allocator), so a Bool tensor whose byte
/// count is NOT a multiple of 4 (3 elements here) exercises the padding path
/// directly: the readback must return exactly 3 elements, not 4 (an
/// over-read of the padding) and not fewer (a size miscalculation).
#[cfg(feature = "wgpu")]
#[test]
fn cast_bool_odd_length_wgpu_matches_cpu() {
    let (cpu, cpu_dev) = create_cpu_client();
    let raw: [u8; 3] = [1, 0, 1];
    let cpu_t = Tensor::<CpuRuntime>::from_bytes(&raw, &[3], DType::Bool, &cpu_dev).unwrap();
    let expected: Vec<f32> = cpu.cast(&cpu_t, DType::F32).unwrap().to_vec();
    assert_eq!(expected.len(), 3);

    with_wgpu_backend_or_skip(|client, device| {
        let t = Tensor::<WgpuRuntime>::from_bytes(&raw, &[3], DType::Bool, &device).unwrap();
        assert_eq!(t.numel(), 3);
        let out: Vec<f32> = client.cast(&t, DType::F32).unwrap().to_vec();
        assert_eq!(out, expected);
    });
}

/// `F32 -> Bool`: NaN is truthy (nonzero), `-0.0` is falsy (`-0.0 == 0.0`).
/// This is the crate-wide "any nonzero, NaN included, is true" convention
/// documented on `ReduceOps::any`/`all`, applied consistently by cast too.
#[test]
fn cast_f32_to_bool_nan_and_negzero_cpu_reference() {
    let (client, device) = create_cpu_client();
    let a: &[f32] = &[0.0, f32::NAN, 5.0, -0.0, -1.0];
    let t = Tensor::<CpuRuntime>::from_slice(a, &[5], &device).unwrap();
    let out: Vec<u8> = client.cast(&t, DType::Bool).unwrap().to_vec();
    assert_eq!(out, vec![0u8, 1, 1, 0, 1]);
}

#[cfg(feature = "cuda")]
#[test]
fn cast_f32_to_bool_nan_and_negzero_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let a: &[f32] = &[0.0, f32::NAN, 5.0, -0.0, -1.0];
        let t = Tensor::<CudaRuntime>::from_slice(a, &[5], &device).unwrap();
        let out: Vec<u8> = client.cast(&t, DType::Bool).unwrap().to_vec();
        assert_eq!(out, vec![0u8, 1, 1, 0, 1]);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn cast_f32_to_bool_nan_and_negzero_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let a: &[f32] = &[0.0, f32::NAN, 5.0, -0.0, -1.0];
        let t = Tensor::<WgpuRuntime>::from_slice(a, &[5], &device).unwrap();
        let out: Vec<u8> = client.cast(&t, DType::Bool).unwrap().to_vec();
        assert_eq!(out, vec![0u8, 1, 1, 0, 1]);
    });
}

/// Empty and single-element edge cases for cast round-trips through Bool.
#[test]
fn cast_bool_edge_cases_cpu() {
    // `bool` implements neither `Element` nor `bytemuck::Pod` in this crate
    // (see `src/dtype/element.rs`): Bool tensors are built and read back as
    // raw `u8` bytes (0/1), never `Vec<bool>`.
    let (client, device) = create_cpu_client();

    let empty: [u8; 0] = [];
    let t = Tensor::<CpuRuntime>::from_bytes(&empty, &[0], DType::Bool, &device).unwrap();
    let out: Vec<f32> = client.cast(&t, DType::F32).unwrap().to_vec();
    assert!(out.is_empty());

    let t2 = Tensor::<CpuRuntime>::from_bytes(&empty, &[0, 5], DType::Bool, &device).unwrap();
    let out2: Vec<f32> = client.cast(&t2, DType::F32).unwrap().to_vec();
    assert!(out2.is_empty());

    let single: [u8; 1] = [1];
    let t3 = Tensor::<CpuRuntime>::from_bytes(&single, &[1], DType::Bool, &device).unwrap();
    let out3: Vec<f32> = client.cast(&t3, DType::F32).unwrap().to_vec();
    assert_eq!(out3, vec![1.0]);

    // 0-dim scalar.
    let t4 = Tensor::<CpuRuntime>::from_bytes(&single, &[], DType::Bool, &device).unwrap();
    let out4: Vec<f32> = client.cast(&t4, DType::F32).unwrap().to_vec();
    assert_eq!(out4, vec![1.0]);
}

// ============================================================================
// fill / zeros / ones
// ============================================================================

/// `fill` with a non-canonical value must collapse to the boolean convention
/// (nonzero -> true), not write the raw value. Before the fix, CUDA wrote
/// `value as u8` verbatim (byte 2 for `fill(_, 2.0, Bool)`), CPU rejected
/// Bool outright, and WebGPU rejected it too (`add_scalar` has no Bool path).
fn assert_fill_bool_matches<R: numr::runtime::Runtime<DType = DType>>(
    client: &impl UtilityOps<R>,
    device: &R::Device,
) {
    let zero = client.fill(&[3], 0.0, DType::Bool).unwrap();
    assert_eq!(zero.to_vec::<u8>(), vec![0u8, 0, 0]);

    let one = client.fill(&[3], 1.0, DType::Bool).unwrap();
    assert_eq!(one.to_vec::<u8>(), vec![1u8, 1, 1]);

    let two = client.fill(&[3], 2.0, DType::Bool).unwrap();
    assert_eq!(two.to_vec::<u8>(), vec![1u8, 1, 1]);

    let neg = client.fill(&[3], -1.0, DType::Bool).unwrap();
    assert_eq!(neg.to_vec::<u8>(), vec![1u8, 1, 1]);

    // Empty shape.
    let empty = client.fill(&[0, 4], 1.0, DType::Bool).unwrap();
    assert_eq!(empty.numel(), 0);

    let _ = device;
}

#[test]
fn fill_bool_collapses_nonzero_cpu() {
    let (client, device) = create_cpu_client();
    assert_fill_bool_matches::<CpuRuntime>(&client, &device);
}

#[cfg(feature = "cuda")]
#[test]
fn fill_bool_collapses_nonzero_cuda() {
    with_cuda_backend(|client, device| {
        assert_fill_bool_matches::<CudaRuntime>(&client, &device);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn fill_bool_collapses_nonzero_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        assert_fill_bool_matches::<WgpuRuntime>(&client, &device);
    });
}

/// `Tensor::zeros`/`ones` for Bool bypass the `RuntimeClient::fill` trait
/// method entirely (dtype-generic constructor), so they already worked
/// before the `fill` fix. Pinned here so a future refactor that routes them
/// through `fill` doesn't regress.
fn assert_zeros_ones_bool<R: numr::runtime::Runtime<DType = DType>>(device: &R::Device) {
    let z = Tensor::<R>::zeros(&[4], DType::Bool, device).unwrap();
    assert_eq!(z.to_vec::<u8>(), vec![0u8, 0, 0, 0]);
    let o = Tensor::<R>::ones(&[4], DType::Bool, device).unwrap();
    assert_eq!(o.to_vec::<u8>(), vec![1u8, 1, 1, 1]);
}

#[test]
fn zeros_ones_bool_cpu() {
    let (_client, device) = create_cpu_client();
    assert_zeros_ones_bool::<CpuRuntime>(&device);
}

#[cfg(feature = "cuda")]
#[test]
fn zeros_ones_bool_cuda() {
    with_cuda_backend(|_client, device| {
        assert_zeros_ones_bool::<CudaRuntime>(&device);
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn zeros_ones_bool_wgpu() {
    with_wgpu_backend_or_skip(|_client, device| {
        assert_zeros_ones_bool::<WgpuRuntime>(&device);
    });
}

// ============================================================================
// semiring_matmul OrAnd: the one op that computes on Bool directly
// ============================================================================

/// 2x2 boolean matmul (OR-AND / transitive-closure step): `[[T,F],[F,T]] *
/// [[T,T],[F,F]] = [[T,T],[F,F]]`.
fn orand_inputs() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let a: Vec<u8> = vec![1, 0, 0, 1];
    let b: Vec<u8> = vec![1, 1, 0, 0];
    let expected: Vec<u8> = vec![1, 1, 0, 0];
    (a, b, expected)
}

#[test]
fn semiring_matmul_orand_bool_cpu_reference() {
    let (client, device) = create_cpu_client();
    let (a, b, expected) = orand_inputs();
    let ta = Tensor::<CpuRuntime>::from_bytes(&a, &[2, 2], DType::Bool, &device).unwrap();
    let tb = Tensor::<CpuRuntime>::from_bytes(&b, &[2, 2], DType::Bool, &device).unwrap();
    let out = client.semiring_matmul(&ta, &tb, SemiringOp::OrAnd).unwrap();
    assert_eq!(out.dtype(), DType::Bool);
    assert_eq!(out.to_vec::<u8>(), expected);
}

#[cfg(feature = "cuda")]
#[test]
fn semiring_matmul_orand_bool_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (a, b, expected) = orand_inputs();
        let ta = Tensor::<CudaRuntime>::from_bytes(&a, &[2, 2], DType::Bool, &device).unwrap();
        let tb = Tensor::<CudaRuntime>::from_bytes(&b, &[2, 2], DType::Bool, &device).unwrap();
        let out = client.semiring_matmul(&ta, &tb, SemiringOp::OrAnd).unwrap();
        assert_eq!(out.dtype(), DType::Bool);
        assert_eq!(out.to_vec::<u8>(), expected);
    });
}

/// WebGPU refuses `OrAnd` by design (no U8/Bool compute), not by omission:
/// see `src/runtime/wgpu/shaders/semiring_matmul.rs`. Pin the clean,
/// consistent-variant rejection instead of a silent wrong answer.
#[cfg(feature = "wgpu")]
#[test]
fn semiring_matmul_orand_bool_wgpu_rejects_cleanly() {
    with_wgpu_backend_or_skip(|client, device| {
        let (a, b, _expected) = orand_inputs();
        let ta = Tensor::<WgpuRuntime>::from_bytes(&a, &[2, 2], DType::Bool, &device).unwrap();
        let tb = Tensor::<WgpuRuntime>::from_bytes(&b, &[2, 2], DType::Bool, &device).unwrap();
        let err = client
            .semiring_matmul(&ta, &tb, SemiringOp::OrAnd)
            .unwrap_err();
        match err {
            Error::UnsupportedDType { dtype, .. } => assert_eq!(dtype, DType::Bool),
            other => panic!("expected UnsupportedDType, got {other:?}"),
        }
    });
}

// ============================================================================
// Ops that reject Bool: same error variant + payload on every backend
// ============================================================================

/// `neg`/`abs` on Bool: `dispatch_dtype!` has no Bool arm on CPU, so both
/// reject with `UnsupportedDType`. Before the CUDA fix, `neg`/`abs` on Bool
/// crashed with an opaque `Error::Internal` (kernel symbol not found)
/// instead, because `unary.cu` instantiates no `bool` row.
fn assert_unsupported_dtype(err: Error, op: &str) {
    match err {
        Error::UnsupportedDType { dtype, op: got_op } => {
            assert_eq!(dtype, DType::Bool);
            assert_eq!(got_op, op);
        }
        other => panic!("expected UnsupportedDType for '{op}', got {other:?}"),
    }
}

#[test]
fn unary_neg_abs_reject_bool_cpu() {
    let (client, device) = create_cpu_client();
    let t = Tensor::<CpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
    assert_unsupported_dtype(client.neg(&t).unwrap_err(), "neg");
    assert_unsupported_dtype(client.abs(&t).unwrap_err(), "abs");
}

#[cfg(feature = "cuda")]
#[test]
fn unary_neg_abs_reject_bool_cuda() {
    with_cuda_backend(|client, device| {
        let t = Tensor::<CudaRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.neg(&t).unwrap_err(), "neg");
        assert_unsupported_dtype(client.abs(&t).unwrap_err(), "abs");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn unary_neg_abs_reject_bool_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let t = Tensor::<WgpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.neg(&t).unwrap_err(), "neg");
        assert_unsupported_dtype(client.abs(&t).unwrap_err(), "abs");
    });
}

/// `sum`/`any`/`all` on Bool: same story as unary. Before the CUDA fix, these
/// crashed with `Error::Internal` (`reduce.cu`/`reduce_int.cu` instantiate no
/// `bool` row) instead of the clean `UnsupportedDType` CPU already returns.
#[test]
fn reduce_sum_any_all_reject_bool_cpu() {
    let (client, device) = create_cpu_client();
    let t = Tensor::<CpuRuntime>::from_bytes(&[1u8, 0, 1, 0], &[4], DType::Bool, &device).unwrap();
    assert_unsupported_dtype(client.sum(&t, &[0], false).unwrap_err(), "sum");
    assert_unsupported_dtype(client.any(&t, &[0], false).unwrap_err(), "any");
    assert_unsupported_dtype(client.all(&t, &[0], false).unwrap_err(), "all");
}

#[cfg(feature = "cuda")]
#[test]
fn reduce_sum_any_all_reject_bool_cuda() {
    with_cuda_backend(|client, device| {
        let t =
            Tensor::<CudaRuntime>::from_bytes(&[1u8, 0, 1, 0], &[4], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.sum(&t, &[0], false).unwrap_err(), "sum");
        assert_unsupported_dtype(client.any(&t, &[0], false).unwrap_err(), "any");
        assert_unsupported_dtype(client.all(&t, &[0], false).unwrap_err(), "all");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn reduce_sum_any_all_reject_bool_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let t =
            Tensor::<WgpuRuntime>::from_bytes(&[1u8, 0, 1, 0], &[4], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.sum(&t, &[0], false).unwrap_err(), "sum");
        assert_unsupported_dtype(client.any(&t, &[0], false).unwrap_err(), "any");
        assert_unsupported_dtype(client.all(&t, &[0], false).unwrap_err(), "all");
    });
}

/// `eq` on two Bool tensors: same story again. Before the CUDA fix this
/// crashed with `Error::Internal` (`compare.cu` instantiates no `bool` row).
#[test]
fn compare_eq_rejects_bool_cpu() {
    let (client, device) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
    assert_unsupported_dtype(client.eq(&a, &b).unwrap_err(), "eq");
}

#[cfg(feature = "cuda")]
#[test]
fn compare_eq_rejects_bool_cuda() {
    with_cuda_backend(|client, device| {
        let a = Tensor::<CudaRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        let b = Tensor::<CudaRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.eq(&a, &b).unwrap_err(), "eq");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn compare_eq_rejects_bool_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
        assert_unsupported_dtype(client.eq(&a, &b).unwrap_err(), "eq");
    });
}

// ============================================================================
// "Boolean tensor" ops that use U8/U32 by design, and reject Bool for it
// ============================================================================

/// Logical ops require U8 (CPU/CUDA) or U32 (WebGPU) masks by design (see
/// `LogicalOps`'s trait doc), not `DType::Bool`. Pin the exact payload so a
/// future change doesn't silently start accepting Bool with different
/// semantics, or start rejecting it with a different error variant.
#[test]
fn logical_and_rejects_bool_wants_u8_cpu() {
    let (client, device) = create_cpu_client();
    let a = Tensor::<CpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
    match client.logical_and(&a, &b).unwrap_err() {
        Error::DTypeMismatch { lhs, rhs } => {
            assert_eq!(lhs, DType::U8);
            assert_eq!(rhs, DType::Bool);
        }
        other => panic!("expected DTypeMismatch, got {other:?}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn logical_and_rejects_bool_wants_u8_cuda() {
    with_cuda_backend(|client, device| {
        let a = Tensor::<CudaRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        let b = Tensor::<CudaRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
        match client.logical_and(&a, &b).unwrap_err() {
            Error::DTypeMismatch { lhs, rhs } => {
                assert_eq!(lhs, DType::U8);
                assert_eq!(rhs, DType::Bool);
            }
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn logical_and_rejects_bool_wants_u32_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let a = Tensor::<WgpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        let b = Tensor::<WgpuRuntime>::from_bytes(&[1u8, 1], &[2], DType::Bool, &device).unwrap();
        match client.logical_and(&a, &b).unwrap_err() {
            Error::DTypeMismatch { lhs, rhs } => {
                assert_eq!(lhs, DType::U32);
                assert_eq!(rhs, DType::Bool);
            }
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
    });
}

/// `masked_select`/`masked_fill` require a U8 (CPU/CUDA) or U32 (WebGPU)
/// mask, same convention as logical ops.
#[test]
fn masked_select_rejects_bool_mask_wants_u8_cpu() {
    let (client, device) = create_cpu_client();
    let a: &[f32] = &[1.0, 2.0];
    let a = Tensor::<CpuRuntime>::from_slice(a, &[2], &device).unwrap();
    let mask = Tensor::<CpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
    match client.masked_select(&a, &mask).unwrap_err() {
        Error::DTypeMismatch { lhs, rhs } => {
            assert_eq!(lhs, DType::U8);
            assert_eq!(rhs, DType::Bool);
        }
        other => panic!("expected DTypeMismatch, got {other:?}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
fn masked_select_rejects_bool_mask_wants_u8_cuda() {
    with_cuda_backend(|client, device| {
        let a: &[f32] = &[1.0, 2.0];
        let a = Tensor::<CudaRuntime>::from_slice(a, &[2], &device).unwrap();
        let mask =
            Tensor::<CudaRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        match client.masked_select(&a, &mask).unwrap_err() {
            Error::DTypeMismatch { lhs, rhs } => {
                assert_eq!(lhs, DType::U8);
                assert_eq!(rhs, DType::Bool);
            }
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn masked_select_rejects_bool_mask_wants_u32_wgpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let a: &[f32] = &[1.0, 2.0];
        let a = Tensor::<WgpuRuntime>::from_slice(a, &[2], &device).unwrap();
        let mask =
            Tensor::<WgpuRuntime>::from_bytes(&[1u8, 0], &[2], DType::Bool, &device).unwrap();
        match client.masked_select(&a, &mask).unwrap_err() {
            Error::DTypeMismatch { lhs, rhs } => {
                assert_eq!(lhs, DType::U32);
                assert_eq!(rhs, DType::Bool);
            }
            other => panic!("expected DTypeMismatch, got {other:?}"),
        }
    });
}
