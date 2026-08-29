// Backend parity tests for integer `matmul` / `matmul_bias` - WebGPU vs CPU.
//
// The macro-driven tests in `float.rs` run I32 and U32, but their operands stay
// small, so they say nothing about the accumulator's width. WGSL has no 64-bit
// integer, so the WebGPU kernels build one: I32 accumulates in the 96-bit
// `NumrI96` (`runtime/wgpu/shaders/int_matmul_acc.wgsl`) and narrows once at the
// store, U32 uses a per-step saturating add that its own monotonicity makes
// exact. This file pins both against CPU's i128 accumulator
// (`matmul_scalar_acc` in `runtime/cpu/kernels/matmul/kernel.rs`).
//
// Every test below is `#[cfg(feature = "wgpu")]`, so these imports are too -
// otherwise a non-WebGPU build would warn on all of them as unused.
#[cfg(feature = "wgpu")]
use numr::dtype::Element;
#[cfg(feature = "wgpu")]
use numr::ops::MatmulOps;
#[cfg(feature = "wgpu")]
use numr::runtime::cpu::CpuRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
#[cfg(feature = "wgpu")]
use numr::tensor::Tensor;

#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
#[cfg(feature = "wgpu")]
use crate::common::create_cpu_client;

// ============================================================================
// Test Utilities
// ============================================================================

/// Run one integer matmul on CPU and WebGPU and require an exact element-wise
/// match. `expected`, when given, is the hand-checked answer, which pins the CPU
/// reference down as well.
#[cfg(feature = "wgpu")]
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

    with_wgpu_backend_or_skip(|client, device| {
        let a_gpu = Tensor::<WgpuRuntime>::from_slice(a, a_shape, &device).expect("WGPU A");
        let b_gpu = Tensor::<WgpuRuntime>::from_slice(b, b_shape, &device).expect("WGPU B");
        let result = client
            .matmul(&a_gpu, &b_gpu)
            .expect("WebGPU integer matmul must be native, not unsupported");
        assert_eq!(result.dtype(), T::DTYPE, "{label}: output dtype changed");
        assert_eq!(
            result.shape(),
            cpu_result.shape(),
            "{label}: output shape differs from CPU"
        );
        assert_eq!(
            result.to_vec::<T>(),
            cpu_vec,
            "{label}: WebGPU must match CPU element for element"
        );
    });
}

/// Same contract as [`assert_int_matmul_parity`], for the fused bias form.
#[cfg(feature = "wgpu")]
#[allow(clippy::too_many_arguments)]
fn assert_int_matmul_bias_parity<T>(
    label: &str,
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    bias: &[T],
    expected: Option<&[T]>,
) where
    T: Element + PartialEq + std::fmt::Debug,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let bias_cpu =
        Tensor::<CpuRuntime>::from_slice(bias, &[bias.len()], &cpu_device).expect("CPU bias");
    let cpu_result = cpu_client
        .matmul_bias(&a_cpu, &b_cpu, &bias_cpu)
        .expect("CPU matmul_bias failed");
    let cpu_vec = cpu_result.to_vec::<T>();

    if let Some(want) = expected {
        assert_eq!(
            cpu_vec, want,
            "{label}: CPU matmul_bias disagrees with hand value"
        );
    }

    with_wgpu_backend_or_skip(|client, device| {
        let a_gpu = Tensor::<WgpuRuntime>::from_slice(a, a_shape, &device).expect("WGPU A");
        let b_gpu = Tensor::<WgpuRuntime>::from_slice(b, b_shape, &device).expect("WGPU B");
        let bias_gpu =
            Tensor::<WgpuRuntime>::from_slice(bias, &[bias.len()], &device).expect("WGPU bias");
        let result = client
            .matmul_bias(&a_gpu, &b_gpu, &bias_gpu)
            .expect("WebGPU integer matmul_bias must be native, not unsupported");
        assert_eq!(result.dtype(), T::DTYPE, "{label}: output dtype changed");
        assert_eq!(
            result.shape(),
            cpu_result.shape(),
            "{label}: output shape differs from CPU"
        );
        assert_eq!(
            result.to_vec::<T>(),
            cpu_vec,
            "{label}: WebGPU must match CPU element for element"
        );
    });
}

/// Deterministic operand values small enough that no product or sum overflows.
#[cfg(feature = "wgpu")]
fn ramp_i32(len: usize, period: i32, shift: i32) -> Vec<i32> {
    (0..len).map(|i| (i as i32 % period) - shift).collect()
}

#[cfg(feature = "wgpu")]
fn ramp_u32(len: usize, period: u32) -> Vec<u32> {
    (0..len).map(|i| i as u32 % period).collect()
}

// ============================================================================
// I32 - exact small products
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_small_hand_checked_wgpu_matches_cpu() {
    assert_int_matmul_parity(
        "i32 2x2 @ 2x2",
        &[1i32, 2, 3, 4],
        &[2, 2],
        &[5i32, 6, 7, 8],
        &[2, 2],
        Some(&[19i32, 22, 43, 50]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_negatives_and_zero_wgpu_matches_cpu() {
    // C[0,0] = 3*4 + (-2)*6 = 0, so a sign error cannot hide behind a magnitude.
    assert_int_matmul_parity(
        "i32 negatives",
        &[3i32, -2, -5, 7],
        &[2, 2],
        &[4i32, -1, 6, 2],
        &[2, 2],
        Some(&[0i32, -7, 22, 19]),
    );
}

// ============================================================================
// I32 - saturation at the store
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_saturates_positive_wgpu_matches_cpu() {
    // Eight products of 4e18 sum to 3.2e19, past i32's range by ten orders of
    // magnitude. Saturation is the answer; wraparound would report a small,
    // arbitrarily signed number.
    let a = vec![2_000_000_000i32; 8];
    let b = vec![2_000_000_000i32; 8];
    assert_int_matmul_parity(
        "i32 saturates positive",
        &a,
        &[1, 8],
        &b,
        &[8, 1],
        Some(&[i32::MAX]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_saturates_negative_wgpu_matches_cpu() {
    let a = vec![2_000_000_000i32; 8];
    let b = vec![-2_000_000_000i32; 8];
    assert_int_matmul_parity(
        "i32 saturates negative",
        &a,
        &[1, 8],
        &b,
        &[8, 1],
        Some(&[i32::MIN]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_partial_sum_leaves_and_returns_wgpu_matches_cpu() {
    // The discriminating case for accumulator WIDTH. Five products of +4e18 come
    // first, so the running total peaks at 2e19 - past i64::MAX, let alone i32 -
    // before five products of -4e18 bring it back to exactly 0. A 64-bit
    // accumulator overflows here, and a per-step saturating add clamps and never
    // recovers; only a genuinely wider accumulator reports 0.
    let a = vec![2_000_000_000i32; 10];
    let mut b = vec![2_000_000_000i32; 5];
    b.extend_from_slice(&[-2_000_000_000i32; 5]);
    assert_int_matmul_parity(
        "i32 partial sum leaves i64 and returns",
        &a,
        &[1, 10],
        &b,
        &[10, 1],
        Some(&[0i32]),
    );
}

// ============================================================================
// I32 - shapes: non-square, K tail, and the tiled kernel
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_non_square_k_tail_wgpu_matches_cpu() {
    // K = 5 is not a multiple of the 16-wide tile, and M != N != K.
    let a = ramp_i32(3 * 5, 7, 3);
    let b = ramp_i32(5 * 2, 5, 2);
    assert_int_matmul_parity("i32 3x5 @ 5x2", &a, &[3, 5], &b, &[5, 2], None);
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_i32_tiled_kernel_wgpu_matches_cpu() {
    // M*N above 256*256 routes to the tiled kernel rather than the simple one,
    // and K = 37 leaves a 5-wide tail in the last tile.
    let (m, k, n) = (200usize, 37usize, 400usize);
    let a = ramp_i32(m * k, 7, 3);
    let b = ramp_i32(k * n, 5, 2);
    assert_int_matmul_parity("i32 200x37 @ 37x400 tiled", &a, &[m, k], &b, &[k, n], None);
}

// ============================================================================
// I32 - batched
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_batched_matmul_i32_wgpu_matches_cpu() {
    let a = ramp_i32(2 * 3 * 4, 7, 3);
    let b = ramp_i32(2 * 4 * 5, 5, 2);
    assert_int_matmul_parity(
        "i32 batched 2x3x4 @ 2x4x5",
        &a,
        &[2, 3, 4],
        &b,
        &[2, 4, 5],
        None,
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_batched_matmul_i32_saturates_wgpu_matches_cpu() {
    // Batch 0 saturates high, batch 1 low, so a shared accumulator or a dropped
    // batch offset shows up as a sign error rather than a near miss.
    let a = vec![2_000_000_000i32; 16];
    let mut b = vec![2_000_000_000i32; 8];
    b.extend_from_slice(&[-2_000_000_000i32; 8]);
    assert_int_matmul_parity(
        "i32 batched saturation",
        &a,
        &[2, 1, 8],
        &b,
        &[2, 8, 1],
        Some(&[i32::MAX, i32::MIN]),
    );
}

// ============================================================================
// U32
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_u32_small_hand_checked_wgpu_matches_cpu() {
    assert_int_matmul_parity(
        "u32 2x2 @ 2x2",
        &[1u32, 2, 3, 4],
        &[2, 2],
        &[5u32, 6, 7, 8],
        &[2, 2],
        Some(&[19u32, 22, 43, 50]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_u32_saturates_on_overflowing_product_wgpu_matches_cpu() {
    // A single product of 8e9 already exceeds u32. The shader has to detect that
    // before the multiply wraps, which is what `numr_u32_mul_overflows` is for.
    assert_int_matmul_parity(
        "u32 overflowing product",
        &[4_000_000_000u32],
        &[1, 1],
        &[2u32],
        &[1, 1],
        Some(&[u32::MAX]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_u32_saturates_on_overflowing_sum_wgpu_matches_cpu() {
    // Each product fits; their sum of 6e9 does not.
    assert_int_matmul_parity(
        "u32 overflowing sum",
        &[3_000_000_000u32, 3_000_000_000],
        &[1, 2],
        &[1u32, 1],
        &[2, 1],
        Some(&[u32::MAX]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_u32_non_square_k_tail_wgpu_matches_cpu() {
    let a = ramp_u32(3 * 5, 7);
    let b = ramp_u32(5 * 2, 5);
    assert_int_matmul_parity("u32 3x5 @ 5x2", &a, &[3, 5], &b, &[5, 2], None);
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_u32_tiled_kernel_wgpu_matches_cpu() {
    let (m, k, n) = (200usize, 37usize, 400usize);
    let a = ramp_u32(m * k, 7);
    let b = ramp_u32(k * n, 5);
    assert_int_matmul_parity("u32 200x37 @ 37x400 tiled", &a, &[m, k], &b, &[k, n], None);
}

#[cfg(feature = "wgpu")]
#[test]
fn test_batched_matmul_u32_wgpu_matches_cpu() {
    let a = ramp_u32(2 * 3 * 4, 7);
    let b = ramp_u32(2 * 4 * 5, 5);
    assert_int_matmul_parity(
        "u32 batched 2x3x4 @ 2x4x5",
        &a,
        &[2, 3, 4],
        &b,
        &[2, 4, 5],
        None,
    );
}

// ============================================================================
// Fused bias
// ============================================================================

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_bias_i32_wgpu_matches_cpu() {
    // C[0] = 1*1 + 2*3 + 10 = 17, C[1] = 1*2 + 2*4 - 5 = 5.
    assert_int_matmul_bias_parity(
        "i32 bias 1x2 @ 2x2",
        &[1i32, 2],
        &[1, 2],
        &[1i32, 2, 3, 4],
        &[2, 2],
        &[10i32, -5],
        Some(&[17i32, 5]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_bias_i32_seeds_the_wide_accumulator_wgpu_matches_cpu() {
    // The dot product alone is 4e9, past i32's range; the bias pulls it back to
    // 2e9. CPU seeds its i128 accumulator with the bias and narrows once, so the
    // answer is exact. Adding the bias to a narrowed product instead would report
    // i32::MAX - 2e9 = 147_483_647.
    assert_int_matmul_bias_parity(
        "i32 bias seeds the accumulator",
        &[2_000_000_000i32, 2_000_000_000],
        &[1, 2],
        &[1i32, 1],
        &[2, 1],
        &[-2_000_000_000i32],
        Some(&[2_000_000_000i32]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_bias_i32_non_square_k_tail_wgpu_matches_cpu() {
    let a = ramp_i32(3 * 5, 7, 3);
    let b = ramp_i32(5 * 2, 5, 2);
    assert_int_matmul_bias_parity(
        "i32 bias 3x5 @ 5x2",
        &a,
        &[3, 5],
        &b,
        &[5, 2],
        &[7i32, -9],
        None,
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_batched_matmul_bias_i32_wgpu_matches_cpu() {
    // One bias, shared by every batch.
    let a = ramp_i32(2 * 3 * 4, 7, 3);
    let b = ramp_i32(2 * 4 * 5, 5, 2);
    assert_int_matmul_bias_parity(
        "i32 batched bias 2x3x4 @ 2x4x5",
        &a,
        &[2, 3, 4],
        &b,
        &[2, 4, 5],
        &[1i32, -2, 3, -4, 5],
        None,
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_bias_u32_wgpu_matches_cpu() {
    assert_int_matmul_bias_parity(
        "u32 bias 1x2 @ 2x2",
        &[1u32, 2],
        &[1, 2],
        &[1u32, 2, 3, 4],
        &[2, 2],
        &[10u32, 5],
        Some(&[17u32, 15]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_matmul_bias_u32_saturates_wgpu_matches_cpu() {
    // Bias plus product reaches 6e9, so the store clamps.
    assert_int_matmul_bias_parity(
        "u32 bias saturates",
        &[3_000_000_000u32],
        &[1, 1],
        &[1u32],
        &[1, 1],
        &[3_000_000_000u32],
        Some(&[u32::MAX]),
    );
}

#[cfg(feature = "wgpu")]
#[test]
fn test_batched_matmul_bias_u32_wgpu_matches_cpu() {
    let a = ramp_u32(2 * 3 * 4, 7);
    let b = ramp_u32(2 * 4 * 5, 5);
    assert_int_matmul_bias_parity(
        "u32 batched bias 2x3x4 @ 2x4x5",
        &a,
        &[2, 3, 4],
        &b,
        &[2, 4, 5],
        &[1u32, 2, 3, 4, 5],
        None,
    );
}
