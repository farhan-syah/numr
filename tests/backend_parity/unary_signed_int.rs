// Backend parity for the integer-domain unary ops: neg, abs and sign.
//
// The sweep in `unary.rs` runs `DTypeDomain::FloatsOnly`, so no integer dtype
// ever reached `neg`, `abs` or `sign`. This file wires up
// `DTypeDomain::SignedOnly` and drives every signed integer width at its
// minimum, which is where the three backends can disagree.
//
// The contract, stated in `src/runtime/cpu/kernels/wide_acc.rs` and implemented
// for the binary ops in `src/runtime/cpu/kernels/binary_int.rs`: element-wise
// integer ops WRAP, accumulators saturate. `neg` and `abs` are element-wise, so
// `neg(i32::MIN)` and `abs(i32::MIN)` both answer `i32::MIN`. Each backend is
// checked against that contract in absolute terms as well as against CPU, so a
// build with no GPU feature still fails when CPU alone is wrong.

use numr::dtype::DType;
use numr::ops::UnaryOps;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(any(feature = "cuda", feature = "wgpu"))]
use crate::common::is_dtype_supported;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

// ============================================================================
// DType Selection
// ============================================================================

/// The signed integer members of `DTypeDomain::SignedOnly`.
///
/// `SignedOnly` is the domain of an operation that needs a representable
/// negation, which is floats plus signed integers. The extreme values below are
/// integer-specific, so the float members stay with the `FloatsOnly` sweep in
/// `unary.rs`.
fn signed_int_dtypes() -> Vec<DType> {
    parity_dtypes(DTypeDomain::SignedOnly, "cpu")
        .into_iter()
        .filter(|dtype| dtype.is_signed_int())
        .collect()
}

/// Bit width of a signed integer dtype.
fn bits(dtype: DType) -> u32 {
    match dtype {
        DType::I8 => 8,
        DType::I16 => 16,
        DType::I32 => 32,
        DType::I64 => 64,
        _ => panic!("not a signed integer dtype: {dtype:?}"),
    }
}

/// Inputs that pin the wrap boundary: the minimum, its neighbour, the two signs
/// around zero, zero itself, and the maximum.
fn extremes(dtype: DType) -> Vec<f64> {
    let min = -(1i128 << (bits(dtype) - 1));
    let max = (1i128 << (bits(dtype) - 1)) - 1;
    vec![min as f64, (min + 1) as f64, -1.0, 0.0, 1.0, max as f64]
}

/// Reduce `v` into the dtype's range the way a wrapping op does.
fn wrap(v: i128, dtype: DType) -> i128 {
    let modulus = 1i128 << bits(dtype);
    let r = v.rem_euclid(modulus);
    if r >= modulus / 2 { r - modulus } else { r }
}

/// The contract's answer for one element.
fn expected_elem(op: &str, v: i128, dtype: DType) -> i128 {
    match op {
        "neg" => wrap(-v, dtype),
        "abs" => wrap(v.abs(), dtype),
        "sign" => v.signum(),
        _ => panic!("unknown op: {op}"),
    }
}

/// The value the tensor actually stores for `v`.
///
/// The extremes travel through `f64`, which cannot represent `i64::MAX` — it
/// rounds to `2^63`. `Element::from_f64` then SATURATES that back to
/// `i64::MAX`, so the contract has to be evaluated against the saturated value.
/// Wrapping `2^63` instead answers `i64::MIN` for `abs(i64::MAX)`.
fn stored(v: f64, dtype: DType) -> i128 {
    let min = -(1i128 << (bits(dtype) - 1));
    let max = (1i128 << (bits(dtype) - 1)) - 1;
    (v as i128).clamp(min, max)
}

/// The contract's answer for every element, as exact integers.
///
/// These deliberately do NOT travel through `f64`: `-i64::MAX` is not
/// representable there and rounds to `i64::MIN`, which would assert the wrong
/// answer for `neg(i64::MAX)`.
fn expected_ints(op: &str, dtype: DType) -> Vec<i128> {
    extremes(dtype)
        .into_iter()
        .map(|v| expected_elem(op, stored(v, dtype), dtype))
        .collect()
}

/// Read a signed-integer tensor back as exact `i128` values.
fn read_ints<R: Runtime>(t: &Tensor<R>, dtype: DType) -> Vec<i128> {
    match dtype {
        DType::I8 => t.to_vec::<i8>().into_iter().map(i128::from).collect(),
        DType::I16 => t.to_vec::<i16>().into_iter().map(i128::from).collect(),
        DType::I32 => t.to_vec::<i32>().into_iter().map(i128::from).collect(),
        DType::I64 => t.to_vec::<i64>().into_iter().map(i128::from).collect(),
        other => panic!("read_ints: not a signed integer dtype: {other:?}"),
    }
}

fn apply<R: Runtime>(
    client: &impl UnaryOps<R>,
    op: &str,
    x: &Tensor<R>,
) -> numr::error::Result<Tensor<R>> {
    match op {
        "neg" => client.neg(x),
        "abs" => client.abs(x),
        "sign" => client.sign(x),
        _ => panic!("unknown op: {op}"),
    }
}

// ============================================================================
// Parity Driver
// ============================================================================

fn test_signed_int_parity(op: &str) {
    for dtype in signed_int_dtypes() {
        let data = extremes(dtype);
        let shape = vec![data.len()];
        let want = expected_ints(op, dtype);

        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_input = tensor_from_f64(&data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU tensor_from_f64 failed for {dtype:?}: {e}"));
        let cpu_result = apply(&cpu_client, op, &cpu_input)
            .unwrap_or_else(|e| panic!("CPU {op} failed for {dtype:?}: {e}"));

        assert_eq!(
            read_ints(&cpu_result, dtype),
            want,
            "{op} CPU vs wrapping contract [{dtype:?}]"
        );

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|cuda_client, cuda_device| {
                let input = tensor_from_f64(&data, &shape, dtype, &cuda_device, &cuda_client)
                    .unwrap_or_else(|e| panic!("CUDA tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply(&cuda_client, op, &input)
                    .unwrap_or_else(|e| panic!("CUDA {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{op} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend_or_skip(|wgpu_client, wgpu_device| {
                let input = tensor_from_f64(&data, &shape, dtype, &wgpu_device, &wgpu_client)
                    .unwrap_or_else(|e| panic!("WebGPU tensor_from_f64 failed for {dtype:?}: {e}"));
                let result = apply(&wgpu_client, op, &input)
                    .unwrap_or_else(|e| panic!("WebGPU {op} failed for {dtype:?}: {e}"));
                assert_tensor_allclose(
                    &result,
                    &cpu_result,
                    dtype,
                    &format!("{op} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_neg_signed_int_parity() {
    test_signed_int_parity("neg");
}

#[test]
fn test_abs_signed_int_parity() {
    test_signed_int_parity("abs");
}

#[test]
fn test_sign_signed_int_parity() {
    test_signed_int_parity("sign");
}
