// Backend parity tests for I32 / I64 `matmul` - CUDA vs CPU.
//
// The CPU backend scope in `../../common/mod.rs` stops at 32-bit integers, so the
// macro-driven tests in `float.rs` never reach I64 matmul. CUDA used to answer
// integer matmul by copying both operands to the host, so this file is the
// coverage that keeps the native kernel honest across both widths.
//
// Both backends accumulate in a 128-bit accumulator and saturate exactly once,
// at the narrow-back store: CPU in `matmul_scalar_acc::<T, i128>`, CUDA in
// `Numr128` (see `runtime/cuda/kernels/numr128.cuh`). The consequence the tests
// below pin down is that a partial sum which leaves the output dtype's range and
// returns to it reports the true value, not a clamped one.
//
// Every test is `#[cfg(feature = "cuda")]`, so these imports are too - otherwise
// a non-CUDA build would warn on all of them as unused.
#[cfg(feature = "cuda")]
use numr::dtype::{DType, Element};
#[cfg(feature = "cuda")]
use numr::ops::MatmulOps;
#[cfg(feature = "cuda")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "cuda")]
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

/// Run one integer matmul on CPU and CUDA and require an exact element-wise
/// match. `expected`, when given, is the hand-checked answer, which pins the CPU
/// reference down as well.
#[cfg(feature = "cuda")]
fn assert_int_matmul_parity<T>(
    label: &str,
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    expected: Option<&[T]>,
) where
    T: Element + PartialEq + std::fmt::Debug,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let cpu_result = cpu_client
        .matmul(&a_cpu, &b_cpu)
        .expect("CPU matmul failed");
    let cpu_vec = cpu_result.to_vec::<T>();

    if let Some(want) = expected {
        assert_eq!(
            cpu_vec, want,
            "{label}: CPU matmul disagrees with hand value"
        );
    }

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(a, a_shape, &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(b, b_shape, &device).expect("CUDA B");
        let result = client
            .matmul(&a_gpu, &b_gpu)
            .expect("CUDA integer matmul must be native, not unsupported");
        assert_eq!(result.dtype(), T::DTYPE, "{label}: output dtype changed");
        assert_eq!(
            result.shape(),
            cpu_result.shape(),
            "{label}: output shape differs from CPU"
        );
        assert_eq!(
            result.to_vec::<T>(),
            cpu_vec,
            "{label}: CUDA must match CPU element for element"
        );
    });
}

/// Deterministic operand values that stay small enough not to overflow.
#[cfg(feature = "cuda")]
fn ramp_i32(len: usize, offset: i32) -> Vec<i32> {
    (0..len).map(|i| (i as i32 % 11) - 5 + offset).collect()
}

#[cfg(feature = "cuda")]
fn ramp_i64(len: usize, offset: i64) -> Vec<i64> {
    (0..len).map(|i| (i as i64 % 11) - 5 + offset).collect()
}

// ============================================================================
// I32 - small hand-checkable product (m <= 16, so the GEMV exclusion applies)
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_small_hand_checked_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 2x2 @ 2x2",
        &[1i32, 2, 3, 4],
        &[2, 2],
        &[5i32, 6, 7, 8],
        &[2, 2],
        Some(&[19i32, 22, 43, 50]),
    );
}

// ============================================================================
// I32 - negative values and a result that is exactly zero
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_negatives_and_zero_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 negatives with exact zero",
        &[1i32, -1, -3, 4],
        &[2, 2],
        &[7i32, -2, 7, 5],
        &[2, 2],
        // Row 0: 1*7 + (-1)*7 = 0, 1*(-2) + (-1)*5 = -7.
        // Row 1: (-3)*7 + 4*7 = 7, (-3)*(-2) + 4*5 = 26.
        Some(&[0i32, -7, 7, 26]),
    );
}

// ============================================================================
// I32 - a partial sum leaves i32's range and comes back
// ============================================================================
//
// The running total reaches 4_000_000_000, past i32::MAX, then returns to
// 2_000_000_000. A per-step clamp would answer i32::MAX; a 128-bit accumulator
// answers the true value.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_overflow_and_recovers_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 overflow and recovers",
        &[2_000_000_000i32, 2_000_000_000, -2_000_000_000],
        &[1, 3],
        &[1i32, 1, 1],
        &[3, 1],
        Some(&[2_000_000_000i32]),
    );
}

// ============================================================================
// I32 - a final value genuinely out of range saturates both ways
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_saturates_both_directions_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 saturates to MAX",
        &[2_000_000_000i32, 2_000_000_000],
        &[1, 2],
        &[1i32, 1],
        &[2, 1],
        Some(&[i32::MAX]),
    );
    assert_int_matmul_parity(
        "i32 saturates to MIN",
        &[-2_000_000_000i32, -2_000_000_000],
        &[1, 2],
        &[1i32, 1],
        &[2, 1],
        Some(&[i32::MIN]),
    );
}

// ============================================================================
// I32 - one product alone exceeds the element type
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_single_product_overflows_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 single product overflows",
        &[100_000i32],
        &[1, 1],
        &[100_000i32],
        &[1, 1],
        Some(&[i32::MAX]),
    );
}

// ============================================================================
// I32 - batched
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_batched_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 batched 2x[2x3] @ 2x[3x2]",
        &[1i32, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6],
        &[2, 2, 3],
        &[1i32, 0, 0, 1, 1, 1, 2, 1, 1, 2, 0, 1],
        &[2, 3, 2],
        // B batch 1 is [[2,1],[1,2],[0,1]], so its columns are [2,1,0] and [1,2,1].
        // Batch 0: [[1+3, 2+3],[4+6, 5+6]] = [[4,5],[10,11]].
        // Batch 1: [[-2-2+0, -1-4-3],[-8-5+0, -4-10-6]] = [[-4,-8],[-13,-20]].
        Some(&[4i32, 5, 10, 11, -4, -8, -13, -20]),
    );
}

// ============================================================================
// I32 - non-square shapes, including one large enough for the tiled path
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_non_square_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i32 3x5 @ 5x2",
        &ramp_i32(15, 0),
        &[3, 5],
        &ramp_i32(10, 2),
        &[5, 2],
        None,
    );
    // m > 16 leaves the small-M shortcut behind, and none of the dims is a
    // multiple of the 64x64 block tile, so every ragged edge is exercised.
    assert_int_matmul_parity(
        "i32 130x70 @ 70x37",
        &ramp_i32(130 * 70, 0),
        &[130, 70],
        &ramp_i32(70 * 37, 3),
        &[70, 37],
        None,
    );
}

// ============================================================================
// I64 - small hand-checkable product
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_small_hand_checked_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i64 2x2 @ 2x2",
        &[1i64, 2, 3, 4],
        &[2, 2],
        &[5i64, 6, 7, 8],
        &[2, 2],
        Some(&[19i64, 22, 43, 50]),
    );
}

// ============================================================================
// I64 - negative values and a result that is exactly zero
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_negatives_and_zero_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i64 negatives with exact zero",
        &[1i64, -1, -3, 4],
        &[2, 2],
        &[7i64, -2, 7, 5],
        &[2, 2],
        Some(&[0i64, -7, 7, 26]),
    );
}

// ============================================================================
// I64 - a partial sum leaves i64's range and comes back
// ============================================================================
//
// 2^62 + 2^62 is 2^63, one past i64::MAX; subtracting 2^62 returns the total to
// 2^62. Only an accumulator wider than 64 bits reports 2^62 here.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_overflow_and_recovers_cuda_matches_cpu() {
    const HALF: i64 = 1i64 << 62;
    assert_int_matmul_parity(
        "i64 overflow and recovers",
        &[HALF, HALF, -HALF],
        &[1, 3],
        &[1i64, 1, 1],
        &[3, 1],
        Some(&[HALF]),
    );
}

// ============================================================================
// I64 - a final value genuinely out of range saturates both ways
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_saturates_both_directions_cuda_matches_cpu() {
    const HALF: i64 = 1i64 << 62;
    assert_int_matmul_parity(
        "i64 saturates to MAX",
        &[HALF, HALF],
        &[1, 2],
        &[1i64, 1],
        &[2, 1],
        Some(&[i64::MAX]),
    );
    assert_int_matmul_parity(
        "i64 saturates to MIN",
        &[-HALF, -HALF, -HALF],
        &[1, 3],
        &[1i64, 1, 1],
        &[3, 1],
        Some(&[i64::MIN]),
    );
}

// ============================================================================
// I64 - one product alone exceeds 64 bits
// ============================================================================
//
// 3_037_000_500^2 is 9_223_372_037_000_250_000, just past i64::MAX. This is the
// case a 64-bit multiply gets wrong before any accumulation happens, so it tests
// the 64x64 -> 128 multiply rather than the running sum.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_single_product_overflows_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i64 single product overflows",
        &[3_037_000_500i64],
        &[1, 1],
        &[3_037_000_500i64],
        &[1, 1],
        Some(&[i64::MAX]),
    );
    assert_int_matmul_parity(
        "i64 single negative product overflows",
        &[-3_037_000_500i64],
        &[1, 1],
        &[3_037_000_500i64],
        &[1, 1],
        Some(&[i64::MIN]),
    );
}

// ============================================================================
// I64 - the sign path of the 128-bit multiply, all four combinations
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_mul_sign_combinations_cuda_matches_cpu() {
    // 3e9 squared is 9e18, inside i64 but far outside 32 bits, so every partial
    // product of the 64x64 -> 128 multiply is exercised without saturating.
    const BIG: i64 = 3_000_000_000;
    const SQ: i64 = 9_000_000_000_000_000_000;
    assert_int_matmul_parity(
        "i64 mul sign combinations",
        &[BIG, -BIG],
        &[2, 1],
        &[BIG, -BIG],
        &[1, 2],
        Some(&[SQ, -SQ, -SQ, SQ]),
    );
}

// ============================================================================
// I64 - batched
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_batched_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i64 batched 2x[2x3] @ 2x[3x2]",
        &[1i64, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6],
        &[2, 2, 3],
        &[1i64, 0, 0, 1, 1, 1, 2, 1, 1, 2, 0, 1],
        &[2, 3, 2],
        Some(&[4i64, 5, 10, 11, -4, -8, -13, -20]),
    );
}

// ============================================================================
// I64 - non-square shapes, including one large enough for the tiled path
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_non_square_cuda_matches_cpu() {
    assert_int_matmul_parity(
        "i64 3x5 @ 5x2",
        &ramp_i64(15, 0),
        &[3, 5],
        &ramp_i64(10, 2),
        &[5, 2],
        None,
    );
    assert_int_matmul_parity(
        "i64 130x70 @ 70x37",
        &ramp_i64(130 * 70, 0),
        &[130, 70],
        &ramp_i64(70 * 37, 3),
        &[70, 37],
        None,
    );
}

// ============================================================================
// The dtypes with no CUDA integer arithmetic stay unsupported
// ============================================================================
//
// I32 and I64 are the only integers with a CUDA arithmetic pipeline, so matmul
// is deliberately not extended to the others. This pins that boundary so a
// future dtype addition is a deliberate change, not a silent one.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_stays_unsupported_on_cuda() {
    with_cuda_backend(|client, device| {
        let a = Tensor::<CudaRuntime>::from_slice(&[1u32, 2, 3, 4], &[2, 2], &device).expect("A");
        let b = Tensor::<CudaRuntime>::from_slice(&[1u32, 0, 0, 1], &[2, 2], &device).expect("B");
        let err = client
            .matmul(&a, &b)
            .expect_err("U32 matmul has no CUDA kernel and must not fall back to the host");
        assert!(
            matches!(
                err,
                numr::error::Error::UnsupportedDType {
                    dtype: DType::U32,
                    ..
                }
            ),
            "expected UnsupportedDType, got {err:?}"
        );
    });
}
