// Backend parity tests for the integer widths added to CUDA `matmul` after
// I32 / I64 - CUDA vs CPU.
//
// `integer_cuda.rs` and `integer_gemv_cuda.rs` cover I32 and I64. This file
// covers U8, U16, U32 and U64, with U32 in depth because it is the width the
// rest of numr uses for index and count tensors. I16 sits with the other signed
// widths in `integer_cuda.rs`, and the fused-bias epilogue is checked here for
// both signs because every integer width shares it.
//
// Two things separate these dtypes from the signed ones already covered:
//
// 1. Widening. `numr128_from_i64` sign-extends, so an unsigned element above
//    LLONG_MAX routed through it would enter the accumulator negative and
//    narrow back to 0. `Numr128From` / `Numr128MulAdd` in
//    `runtime/cuda/kernels/numr128.cuh` zero-extend instead, and the U64 tests
//    below use operands past LLONG_MAX precisely to catch a regression there.
// 2. Saturation. Matmul is an accumulator, so it saturates at the narrow-back
//    store instead of wrapping (`runtime/cpu/kernels/wide_acc.rs`). The tests
//    that overflow the element type assert the clamped value, which is the one
//    place a wrapping kernel is visibly wrong.
//
// I8 is absent by design: CPU `matmul` on I8 returns an I32 tensor (quantized
// accumulation), so there is no I8-in/I8-out reference to match. See the
// boundary test in `integer_cuda.rs`.
//
// Every test is `#[cfg(feature = "cuda")]`, so these imports are too.
#[cfg(feature = "cuda")]
use numr::dtype::Element;
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

/// How the B operand is handed to `matmul`.
#[cfg(feature = "cuda")]
#[derive(Clone, Copy, PartialEq)]
enum BLayout {
    /// `b_shape` is the operand shape, used as stored.
    AsStored,
    /// `b_shape` is the stored `[N, K]` shape and the operand is its transpose.
    /// This is the layout that selects the `gemv_bt_mr_*` kernels, which read
    /// the stored buffer by raw pointer with 16-byte vector loads.
    Transposed,
}

/// Run one integer matmul on CPU and CUDA and require an exact element-wise
/// match. `expected`, when given, is the hand-checked answer, which pins the
/// CPU reference down as well.
#[cfg(feature = "cuda")]
fn assert_parity<T>(
    label: &str,
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    layout: BLayout,
    expected: Option<&[T]>,
) where
    T: Element + PartialEq + std::fmt::Debug,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_stored = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let b_cpu = match layout {
        BLayout::AsStored => b_stored,
        BLayout::Transposed => b_stored.t().expect("CPU B transpose"),
    };
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
        let b_gpu_stored = Tensor::<CudaRuntime>::from_slice(b, b_shape, &device).expect("CUDA B");
        let b_gpu = match layout {
            BLayout::AsStored => b_gpu_stored,
            BLayout::Transposed => b_gpu_stored.t().expect("CUDA B transpose"),
        };
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

/// The same check for the fused `matmul_bias` entry point.
#[cfg(feature = "cuda")]
fn assert_bias_parity<T>(
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
    let bias_shape = [bias.len()];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let bias_cpu =
        Tensor::<CpuRuntime>::from_slice(bias, &bias_shape, &cpu_device).expect("CPU bias");
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

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(a, a_shape, &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(b, b_shape, &device).expect("CUDA B");
        let bias_gpu =
            Tensor::<CudaRuntime>::from_slice(bias, &bias_shape, &device).expect("CUDA bias");
        let result = client
            .matmul_bias(&a_gpu, &b_gpu, &bias_gpu)
            .expect("CUDA integer matmul_bias must be native, not unsupported");
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

/// Deterministic operand values, small enough that no dtype here overflows.
#[cfg(feature = "cuda")]
fn ramp<T>(len: usize, f: impl Fn(usize) -> T) -> Vec<T> {
    (0..len).map(f).collect()
}

// ============================================================================
// U32 - small hand-checkable product, both B layouts
// ============================================================================
//
// m = 2, so both cases take the small-M shortcut: `gemv_u32` for the stored
// layout and `gemv_bt_mr_u32` for the transposed one, which is also the only
// path with 16-byte vector loads.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_small_hand_checked_cuda_matches_cpu() {
    assert_parity(
        "u32 2x2 @ 2x2",
        &[1u32, 2, 3, 4],
        &[2, 2],
        &[5u32, 6, 7, 8],
        &[2, 2],
        BLayout::AsStored,
        Some(&[19u32, 22, 43, 50]),
    );
    // B stored as [N, K] = [[5, 7], [6, 8]] is the transpose of [[5, 6], [7, 8]],
    // so the product is the same one hand-checked above.
    assert_parity(
        "u32 2x2 @ transposed 2x2",
        &[1u32, 2, 3, 4],
        &[2, 2],
        &[5u32, 7, 6, 8],
        &[2, 2],
        BLayout::Transposed,
        Some(&[19u32, 22, 43, 50]),
    );
}

// ============================================================================
// U32 - a matrix times a vector, the GEMV shape
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_matrix_times_vector_cuda_matches_cpu() {
    // Rows of A dotted with [2, 3, 4]:
    //   [1,2,3] -> 2 + 6 + 12 = 20
    //   [4,5,6] -> 8 + 15 + 24 = 47
    //   [7,8,9] -> 14 + 24 + 36 = 74
    assert_parity(
        "u32 3x3 @ 3x1",
        &[1u32, 2, 3, 4, 5, 6, 7, 8, 9],
        &[3, 3],
        &[2u32, 3, 4],
        &[3, 1],
        BLayout::AsStored,
        Some(&[20u32, 47, 74]),
    );
}

// ============================================================================
// U32 - the true product exceeds u32::MAX and must SATURATE, not wrap
// ============================================================================
//
// This is the case that separates a correct wide accumulator from a naive one.
// 3_000_000_000 + 3_000_000_000 = 6_000_000_000, which is 1_705_032_704 modulo
// 2^32: a wrapping kernel reports that, a saturating one reports u32::MAX.
// The single-product case is the same test one step earlier, before any
// accumulation: 100_000 * 100_000 = 10^10 does not fit a u32 element.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_saturates_past_u32_max_cuda_matches_cpu() {
    assert_parity(
        "u32 sum saturates",
        &[3_000_000_000u32, 3_000_000_000],
        &[1, 2],
        &[1u32, 1],
        &[2, 1],
        BLayout::AsStored,
        Some(&[u32::MAX]),
    );
    assert_parity(
        "u32 single product saturates",
        &[100_000u32],
        &[1, 1],
        &[100_000u32],
        &[1, 1],
        BLayout::AsStored,
        Some(&[u32::MAX]),
    );
}

// ============================================================================
// U32 - a saturating column beside an exact one
// ============================================================================
//
// Every unsigned term is non-negative, so a column total can only climb: the
// "overflows and comes back" case that the signed tests use has no unsigned
// counterpart. What this pins instead is that the whole 8_000_000_000 total is
// carried in the wide accumulator and clamped once at the store - a wrapping
// accumulator answers 3_705_032_704 - and that the neighbouring column, whose
// total is in range, is untouched by its neighbour saturating.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_saturating_and_exact_columns_cuda_matches_cpu() {
    assert_parity(
        "u32 saturating column beside an exact one",
        &[3_000_000_000u32, 3_000_000_000, 2_000_000_000],
        &[1, 3],
        // B rows are [1,0], [1,0] and [1,1]: column 0 sums all three terms,
        // column 1 takes only the last.
        &[1u32, 0, 1, 0, 1, 1],
        &[3, 2],
        BLayout::AsStored,
        Some(&[u32::MAX, 2_000_000_000]),
    );
}

// ============================================================================
// U32 - batched
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_batched_cuda_matches_cpu() {
    // Batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,1]]
    //   row 0: [1+0+3, 0+2+3] = [4, 5]
    //   row 1: [4+0+6, 0+5+6] = [10, 11]
    // Batch 1: [[6,5,4],[3,2,1]] @ [[2,1],[1,2],[0,1]]
    //   row 0: [12+5+0, 6+10+4] = [17, 20]
    //   row 1: [6+2+0, 3+4+1] = [8, 8]
    assert_parity(
        "u32 batched 2x[2x3] @ 2x[3x2]",
        &[1u32, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
        &[2, 2, 3],
        &[1u32, 0, 0, 1, 1, 1, 2, 1, 1, 2, 0, 1],
        &[2, 3, 2],
        BLayout::AsStored,
        Some(&[4u32, 5, 10, 11, 17, 20, 8, 8]),
    );
}

// ============================================================================
// U32 - bias, plain and batched
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_u32_cuda_matches_cpu() {
    // [[19,22],[43,50]] + [10, 100] broadcast over rows.
    assert_bias_parity(
        "u32 2x2 @ 2x2 + bias",
        &[1u32, 2, 3, 4],
        &[2, 2],
        &[5u32, 6, 7, 8],
        &[2, 2],
        &[10u32, 100],
        Some(&[29u32, 122, 53, 150]),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_u32_batched_cuda_matches_cpu() {
    // The batched product above plus [10, 100] on every row of every batch.
    assert_bias_parity(
        "u32 batched 2x[2x3] @ 2x[3x2] + bias",
        &[1u32, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1],
        &[2, 2, 3],
        &[1u32, 0, 0, 1, 1, 1, 2, 1, 1, 2, 0, 1],
        &[2, 3, 2],
        &[10u32, 100],
        Some(&[14u32, 105, 20, 111, 27, 120, 18, 108]),
    );
}

// ============================================================================
// The bias joins the wide accumulator, it is not added to a saturated product
// ============================================================================
//
// 3e9 + 3e9 = 6e9 saturates u32 on its own, and the true total with the bias is
// 6_000_000_010, which saturates to the same bound. A backend that ran a plain
// matmul and then an elementwise add would compute u32::MAX + 10 in the element
// type, which wraps to 9. Both widths share the epilogue, so the signed case is
// checked here beside the unsigned one.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_int_seeds_accumulator_cuda_matches_cpu() {
    assert_bias_parity(
        "u32 bias on a saturating product",
        &[3_000_000_000u32, 3_000_000_000],
        &[1, 2],
        &[1u32, 1],
        &[2, 1],
        &[10u32],
        Some(&[u32::MAX]),
    );
    assert_bias_parity(
        "i32 bias on a saturating product",
        &[2_000_000_000i32, 2_000_000_000],
        &[1, 2],
        &[1i32, 1],
        &[2, 1],
        &[10i32],
        Some(&[i32::MAX]),
    );
}

// ============================================================================
// U32 - the tiled path, with a K that is not a multiple of the tile width
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u32_tiled_ragged_shapes_cuda_matches_cpu() {
    // m > 16 leaves the small-M shortcut behind. K = 13 is not a multiple of the
    // kernel's BK = 8, and neither M = 33 nor N = 7 is a multiple of the 64x64
    // block tile, so every ragged edge is exercised.
    assert_parity(
        "u32 33x13 @ 13x7",
        &ramp(33 * 13, |i| (i % 7) as u32 + 1),
        &[33, 13],
        &ramp(13 * 7, |i| (i % 5) as u32 + 2),
        &[13, 7],
        BLayout::AsStored,
        None,
    );
    // The same shape with B stored transposed keeps m > 16, so this one takes
    // the tiled kernel over a materialised B rather than the GEMV fast path.
    assert_parity(
        "u32 33x13 @ transposed 7x13",
        &ramp(33 * 13, |i| (i % 7) as u32 + 1),
        &[33, 13],
        &ramp(7 * 13, |i| (i % 5) as u32 + 2),
        &[7, 13],
        BLayout::Transposed,
        None,
    );
}

// ============================================================================
// U64 - the sign-extension trap
// ============================================================================
//
// 18e18 is above LLONG_MAX (about 9.22e18) and below u64::MAX (about 1.84e19).
// Widening it with a sign-extending conversion makes the accumulator negative,
// and the unsigned narrow-back clamps a negative accumulator to 0. So a kernel
// that takes the signed path answers 0 here while the true answer is the input.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u64_above_i64_max_zero_extends_cuda_matches_cpu() {
    const BIG: u64 = 18_000_000_000_000_000_000;
    assert_parity(
        "u64 operand past LLONG_MAX",
        &[BIG],
        &[1, 1],
        &[1u64],
        &[1, 1],
        BLayout::AsStored,
        Some(&[BIG]),
    );
    // A sum that also lands above LLONG_MAX, so the accumulator - not just the
    // widening of one element - has to stay unsigned-correct.
    assert_parity(
        "u64 sum past LLONG_MAX",
        &[10_000_000_000_000_000_000u64, 8_000_000_000_000_000_000],
        &[1, 2],
        &[1u64, 1],
        &[2, 1],
        BLayout::AsStored,
        Some(&[18_000_000_000_000_000_000u64]),
    );
}

// ============================================================================
// U64 - saturation at both ends of the 128-bit accumulator
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u64_saturates_cuda_matches_cpu() {
    // 2e19 is past u64::MAX, so the sum saturates at the narrow-back store.
    assert_parity(
        "u64 sum saturates",
        &[10_000_000_000_000_000_000u64, 10_000_000_000_000_000_000],
        &[1, 2],
        &[1u64, 1],
        &[2, 1],
        BLayout::AsStored,
        Some(&[u64::MAX]),
    );
    // u64::MAX squared is nearly 2^128, past the signed 128-bit accumulator
    // itself. CPU saturates the i128 multiply and then the narrow; CUDA does the
    // same through `numr128_mul_sat`, so both land on u64::MAX.
    assert_parity(
        "u64 product exceeds the 128-bit accumulator",
        &[u64::MAX],
        &[1, 1],
        &[u64::MAX],
        &[1, 1],
        BLayout::AsStored,
        Some(&[u64::MAX]),
    );
}

// ============================================================================
// U64 - hand-checked product on both paths
// ============================================================================

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u64_hand_checked_cuda_matches_cpu() {
    assert_parity(
        "u64 2x2 @ 2x2",
        &[1u64, 2, 3, 4],
        &[2, 2],
        &[5u64, 6, 7, 8],
        &[2, 2],
        BLayout::AsStored,
        Some(&[19u64, 22, 43, 50]),
    );
    assert_parity(
        "u64 2x2 @ transposed 2x2",
        &[1u64, 2, 3, 4],
        &[2, 2],
        &[5u64, 7, 6, 8],
        &[2, 2],
        BLayout::Transposed,
        Some(&[19u64, 22, 43, 50]),
    );
    assert_parity(
        "u64 20x9 @ 9x5",
        &ramp(20 * 9, |i| (i % 6) as u64 + 1),
        &[20, 9],
        &ramp(9 * 5, |i| (i % 4) as u64 + 1),
        &[9, 5],
        BLayout::AsStored,
        None,
    );
}

// ============================================================================
// U16, U8 - each resolved on both the GEMV and the tiled path
// ============================================================================
//
// A missing instantiation fails at kernel lookup, so one small case (m <= 16,
// GEMV) and one m > 16 case (tiled) per dtype is what proves both modules were
// built for it. The hand value is the same 2x2 product throughout, which every
// one of these ranges holds.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u16_cuda_matches_cpu() {
    assert_parity(
        "u16 2x2 @ 2x2",
        &[1u16, 2, 3, 4],
        &[2, 2],
        &[5u16, 6, 7, 8],
        &[2, 2],
        BLayout::AsStored,
        Some(&[19u16, 22, 43, 50]),
    );
    // 1000 * 1000 = 1_000_000, past u16::MAX.
    assert_parity(
        "u16 single product saturates",
        &[1_000u16],
        &[1, 1],
        &[1_000u16],
        &[1, 1],
        BLayout::AsStored,
        Some(&[u16::MAX]),
    );
    assert_parity(
        "u16 20x9 @ 9x5",
        &ramp(20 * 9, |i| (i % 6) as u16 + 1),
        &[20, 9],
        &ramp(9 * 5, |i| (i % 4) as u16 + 1),
        &[9, 5],
        BLayout::AsStored,
        None,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_u8_cuda_matches_cpu() {
    assert_parity(
        "u8 2x2 @ 2x2",
        &[1u8, 2, 3, 4],
        &[2, 2],
        &[5u8, 6, 7, 8],
        &[2, 2],
        BLayout::AsStored,
        Some(&[19u8, 22, 43, 50]),
    );
    // 100 + 100 + 100 + 100 = 400, past u8::MAX, so the sum clamps to 255.
    assert_parity(
        "u8 sum saturates",
        &[100u8, 100, 100, 100],
        &[1, 4],
        &[1u8, 1, 1, 1],
        &[4, 1],
        BLayout::AsStored,
        Some(&[u8::MAX]),
    );
    // Values kept small so the reference product stays inside u8 and the test
    // measures the kernel rather than the clamp: 9 * 3 * 3 = 81 at most.
    assert_parity(
        "u8 20x9 @ 9x5",
        &ramp(20 * 9, |i| (i % 3) as u8 + 1),
        &[20, 9],
        &ramp(9 * 5, |i| (i % 3) as u8 + 1),
        &[9, 5],
        BLayout::AsStored,
        None,
    );
}
