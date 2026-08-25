//! Regression tests for kernels that must accumulate in a type wider than their
//! elements.
//!
//! A summing kernel that keeps its accumulator in the element type is wrong for
//! every dtype narrower than the total it builds. Narrow floats stall or
//! saturate; integers wrap (release) or panic (debug). These tests drive the
//! public API so they cover the dispatch path a caller actually takes, not just
//! the kernel in isolation.
//!
//! Every case here pins a value. The comment on each test names the sabotage it
//! catches.

use numr::dtype::DType;
use numr::ops::{CumulativeOps, MatmulOps, ReduceOps, StatisticalOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Catches an i32 `cumsum` accumulator.
///
/// The running total leaves i32's range at element 1 and returns at element 2.
/// An i32 accumulator panics on that overflow in a debug build; in a release
/// build it stores the wrapped -294_967_296 where the documented answer is the
/// saturated `i32::MAX`.
#[test]
fn cumsum_i32_accumulates_in_a_wider_integer() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[2_000_000_000i32, 2_000_000_000, -2_000_000_000],
        &[3],
        &device,
    )
    .unwrap();

    let out = client.cumsum(&a, 0).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [2_000_000_000, i32::MAX, 2_000_000_000]);
    assert_eq!(out.dtype(), DType::I32);
}

/// Same defect on the strided `cumsum` path (scan over a non-last dimension),
/// which no SIMD path covers for integers.
#[test]
fn cumsum_strided_i32_accumulates_in_a_wider_integer() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [3, 2]: column 0 overflows i32, column 1 stays small and pins that
    // ordinary columns are untouched.
    let a = Tensor::<CpuRuntime>::from_slice(
        &[2_000_000_000i32, 1, 2_000_000_000, 2, -2_000_000_000, 3],
        &[3, 2],
        &device,
    )
    .unwrap();

    let out = client.cumsum(&a, 0).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [2_000_000_000, 1, i32::MAX, 3, 2_000_000_000, 6]);
}

/// Catches an i32 matmul accumulator.
///
/// Column 0's dot product is 4_000_000_000, which i32 cannot hold: an i32
/// accumulator panics in a debug build and wraps to -294_967_296 in a release
/// build, where the documented answer is the saturated `i32::MAX`. Column 1
/// stays in range and pins that ordinary results are unchanged.
#[test]
fn matmul_i32_saturates_instead_of_wrapping() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[2_000_000_000i32, 2_000_000_000], &[1, 2], &device)
        .unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&[1i32, 1, 1, -1], &[2, 2], &device).unwrap();

    let out = client.matmul(&a, &b).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [i32::MAX, 0]);
    assert_eq!(out.dtype(), DType::I32);
}

/// Ordinary i32 matmul results must be identical to what the removed AVX2 i32
/// microkernel produced. This is the guard against the accumulator change
/// disturbing dtypes and magnitudes that were already correct.
#[test]
fn matmul_i32_in_range_results_are_unchanged() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let (m, n, k) = (2usize, 16usize, 3usize);
    let a_data: Vec<i32> = (0..m * k).map(|i| (i + 1) as i32).collect();
    let b_data: Vec<i32> = (0..k * n).map(|i| (i + 1) as i32).collect();

    let a = Tensor::<CpuRuntime>::from_slice(&a_data, &[m, k], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&b_data, &[k, n], &device).unwrap();

    let out = client.matmul(&a, &b).unwrap();

    let mut expected = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            for kk in 0..k {
                expected[i * n + j] += a_data[i * k + kk] * b_data[kk * n + j];
            }
        }
    }

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, expected);
}

/// Catches an integer `mean` that sums in the element type, on the contiguous
/// last-dimension path.
///
/// The sum needs 33 bits but the mean is exactly 2_000_000_000, which i32
/// holds. An i32 sum panics in debug and wraps to -294_967_296 in release, and
/// the division then reports -147_483_648. Unlike a plain sum, a division does
/// not recover from a wrapped total.
#[test]
fn mean_i32_last_dim_sums_in_a_wider_integer() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[2_000_000_000i32, 2_000_000_000, 4, 6],
        &[2, 2],
        &device,
    )
    .unwrap();

    let out = client.mean(&a, &[1], false).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [2_000_000_000, 5]);
}

/// Same defect on the non-last-dimension `mean` path, which has its own
/// accumulator loop.
#[test]
fn mean_i32_non_last_dim_sums_in_a_wider_integer() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Reducing dim 0 of [2, 2]: column 0 overflows i32, column 1 does not.
    let a = Tensor::<CpuRuntime>::from_slice(
        &[2_000_000_000i32, 4, 2_000_000_000, 6],
        &[2, 2],
        &device,
    )
    .unwrap();

    let out = client.mean(&a, &[0], false).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [2_000_000_000, 5]);
}

/// Same defect on the fused multi-dimension `mean` path, which keeps one
/// accumulator per output bucket.
#[test]
fn mean_i32_fused_multi_dim_sums_in_a_wider_integer() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    // Shape [2, 2, 2], reducing dims 0 and 1: each output bucket sums four
    // elements. Bucket 0 overflows i32; bucket 1 does not.
    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            2_000_000_000i32,
            1,
            2_000_000_000,
            2,
            2_000_000_000,
            3,
            2_000_000_000,
            4,
        ],
        &[2, 2, 2],
        &device,
    )
    .unwrap();

    let out = client.mean(&a, &[0, 1], false).unwrap();

    let result: Vec<i32> = out.to_vec();
    // Bucket 0: 8_000_000_000 / 4 = 2_000_000_000. Bucket 1: 10 / 4 = 2.
    assert_eq!(result, [2_000_000_000, 2]);
}

/// Integer `mean` truncates toward zero, which is what the previous
/// float-division epilogue did for every sum it could represent. Pinned so the
/// wider accumulator cannot silently change the rounding convention.
#[test]
fn mean_i32_truncates_toward_zero() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[7i32, 0, -7, 0], &[2, 2], &device).unwrap();

    let out = client.mean(&a, &[1], false).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [3, -3]);
}

/// `sum` is deliberately left accumulating in the element type: its output
/// dtype is the element type, so a total that overflows has nowhere to go.
/// Pinned so the `mean` fix is not mistaken for a `sum` fix.
#[test]
fn sum_i32_in_range_is_unchanged() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(&[1i32, 2, 3, 4], &[2, 2], &device).unwrap();

    let out = client.sum(&a, &[1], false).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [3, 7]);
}

/// `var` on an integer dtype already accumulates in f64 on CPU, so it is not
/// part of this defect class. Pinned so a later "consistency" edit cannot
/// quietly narrow it: the intermediate sum here is 4_000_000_000, well past
/// i32's range, and the variance is still exact.
#[test]
fn var_i32_already_accumulates_in_f64() {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let a = Tensor::<CpuRuntime>::from_slice(
        &[
            2_000_000_000i32,
            2_000_000_000,
            2_000_000_000,
            2_000_000_000,
        ],
        &[4],
        &device,
    )
    .unwrap();

    let out = client.var(&a, &[], false, 0).unwrap();

    let result: Vec<i32> = out.to_vec();
    assert_eq!(result, [0]);
}

/// Catches an F16 `cumsum` accumulator.
///
/// F16 has ten mantissa bits, so above 2048 its spacing is 2 and `2048 + 1`
/// rounds back to 2048. An F16 accumulator stalls there and every later output
/// reads 2048.
#[cfg(feature = "f16")]
#[test]
fn cumsum_f16_accumulates_in_f32() {
    use half::f16;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let data = vec![f16::from_f32(1.0); 3000];
    let a = Tensor::<CpuRuntime>::from_slice(&data, &[3000], &device).unwrap();

    let out = client.cumsum(&a, 0).unwrap();

    let result: Vec<f16> = out.to_vec();
    // 2500 and 3000 are even, so both are exact in F16.
    assert_eq!(result[2499].to_f32(), 2500.0);
    assert_eq!(result[2999].to_f32(), 3000.0);
}

/// Catches an FP8 matmul accumulator.
///
/// A length-32 dot product of ones is 32. Accumulated in FP8E4M3 the running
/// sum stalls at 16, because above 16 the format's spacing is 2 and `16 + 1`
/// rounds back to 16.
#[cfg(feature = "fp8")]
#[test]
fn matmul_fp8_accumulates_in_f32() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let ones = vec![FP8E4M3::from_f32(1.0); 32];
    let a = Tensor::<CpuRuntime>::from_slice(&ones, &[1, 32], &device).unwrap();
    let b = Tensor::<CpuRuntime>::from_slice(&ones, &[32, 1], &device).unwrap();

    let out = client.matmul(&a, &b).unwrap();

    let result: Vec<FP8E4M3> = out.to_vec();
    assert_eq!(result[0].to_f32(), 32.0);
}

/// Catches an FP8 `cumsum` accumulator, same stall as the matmul case above.
#[cfg(feature = "fp8")]
#[test]
fn cumsum_fp8_accumulates_in_f32() {
    use numr::dtype::FP8E4M3;

    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);

    let ones = vec![FP8E4M3::from_f32(1.0); 32];
    let a = Tensor::<CpuRuntime>::from_slice(&ones, &[32], &device).unwrap();

    let out = client.cumsum(&a, 0).unwrap();

    let result: Vec<FP8E4M3> = out.to_vec();
    // 24 and 32 are both exactly representable in FP8E4M3.
    assert_eq!(result[23].to_f32(), 24.0);
    assert_eq!(result[31].to_f32(), 32.0);
}
