// Backend parity tests for the I32 / I64 CUDA GEMV path - CUDA vs CPU.
//
// `integer_cuda.rs` covers the tiled integer GEMM. This file covers the small-M
// shortcut: `m <= 16` routes to `gemv_i32` / `gemv_i64`, and a transposed B
// operand routes to `gemv_bt_mr_i32` / `gemv_bt_mr_i64` (see
// `runtime/cuda/kernels/gemv_int.cu`). Those are separate implementations of the
// same product, so three things must hold and are tested here:
//
// 1. Each GEMV kernel matches CPU, which accumulates in i128.
// 2. The GEMV path and the tiled path agree on the same operands. Both
//    accumulate in a 128-bit accumulator and saturate once at the store, and
//    128-bit integer addition is exact, so "agree" means bit for bit.
// 3. The accumulator survives a partial sum that leaves the output dtype's range
//    and returns, on the GEMV path as well as the tiled one.
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
    /// `b_shape` is the stored `[.., N, K]` shape, and the operand is its
    /// last-two transpose. This is the layout that selects the `gemv_bt_mr_*`
    /// kernels: a transposed view is passed to the kernel by raw pointer instead
    /// of being materialised.
    TransposedLast2,
}

/// Run one integer matmul on CPU and CUDA and require an exact element-wise
/// match. Returns the shared result so a caller can compare two paths.
#[cfg(feature = "cuda")]
fn assert_gemv_parity<T>(
    label: &str,
    a: &[T],
    a_shape: &[usize],
    b: &[T],
    b_shape: &[usize],
    layout: BLayout,
    expected: Option<&[T]>,
) -> Vec<T>
where
    T: Element + PartialEq + std::fmt::Debug,
{
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_stored = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let b_cpu = match layout {
        BLayout::AsStored => b_stored,
        BLayout::TransposedLast2 => b_stored.t().expect("CPU B transpose"),
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
            BLayout::TransposedLast2 => b_gpu_stored.t().expect("CUDA B transpose"),
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

    cpu_vec
}

/// Deterministic operand values that stay small enough not to overflow.
#[cfg(feature = "cuda")]
fn ramp_i32(len: usize, offset: i32) -> Vec<i32> {
    (0..len).map(|i| (i as i32 % 13) - 6 + offset).collect()
}

#[cfg(feature = "cuda")]
fn ramp_i64(len: usize, offset: i64) -> Vec<i64> {
    (0..len).map(|i| (i as i64 % 13) - 6 + offset).collect()
}

// ============================================================================
// Non-transposed B: gemv_i32 / gemv_i64
// ============================================================================
//
// M of 1, 8 and 16 are all at or under the fast-path threshold, so all three
// take the GEMV kernel. K is not a multiple of the tiled kernel's block_k, so
// the two paths cannot accidentally share a code path.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_gemv_path_cuda_matches_cpu() {
    for m in [1usize, 8, 16] {
        assert_gemv_parity(
            &format!("i32 gemv {m}x37 @ 37x45"),
            &ramp_i32(m * 37, 0),
            &[m, 37],
            &ramp_i32(37 * 45, 2),
            &[37, 45],
            BLayout::AsStored,
            None,
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_gemv_path_cuda_matches_cpu() {
    for m in [1usize, 8, 16] {
        assert_gemv_parity(
            &format!("i64 gemv {m}x37 @ 37x45"),
            &ramp_i64(m * 37, 0),
            &[m, 37],
            &ramp_i64(37 * 45, 2),
            &[37, 45],
            BLayout::AsStored,
            None,
        );
    }
}

// A GEMV accumulator that clamped per step would answer i32::MAX / i64::MAX
// here. A 128-bit one recovers the true value, and saturates only when the
// final total is genuinely out of range.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_gemv_overflow_and_recovers_cuda_matches_cpu() {
    assert_gemv_parity(
        "i32 gemv overflow and recovers",
        &[2_000_000_000i32, 2_000_000_000, -2_000_000_000],
        &[1, 3],
        &[1i32, 1, 1],
        &[3, 1],
        BLayout::AsStored,
        Some(&[2_000_000_000i32]),
    );
    assert_gemv_parity(
        "i32 gemv saturates to MIN",
        &[-2_000_000_000i32, -2_000_000_000],
        &[1, 2],
        &[1i32, 1],
        &[2, 1],
        BLayout::AsStored,
        Some(&[i32::MIN]),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_gemv_overflow_and_recovers_cuda_matches_cpu() {
    const HALF: i64 = 1i64 << 62;
    assert_gemv_parity(
        "i64 gemv overflow and recovers",
        &[HALF, HALF, -HALF],
        &[1, 3],
        &[1i64, 1, 1],
        &[3, 1],
        BLayout::AsStored,
        Some(&[HALF]),
    );
    assert_gemv_parity(
        "i64 gemv saturates to MAX",
        &[HALF, HALF],
        &[1, 2],
        &[1i64, 1],
        &[2, 1],
        BLayout::AsStored,
        Some(&[i64::MAX]),
    );
}

// ============================================================================
// Transposed B: gemv_bt_mr_i32 / gemv_bt_mr_i64
// ============================================================================
//
// The multi-row kernel takes a 16-byte vector load when K is a multiple of the
// vector width (4 for I32, 2 for I64) and falls back to a scalar loop otherwise,
// so both K values below are needed to cover it. N=17 is odd, which leaves the
// last warp with only one of its two output columns in range.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_gemv_transposed_b_cuda_matches_cpu() {
    for k in [64usize, 37] {
        for m in [1usize, 5] {
            assert_gemv_parity(
                &format!("i32 gemv_bt {m}x{k} @ ({k}x17 transposed)"),
                &ramp_i32(m * k, 0),
                &[m, k],
                // Stored [N,K]; the operand is its transpose.
                &ramp_i32(17 * k, 3),
                &[17, k],
                BLayout::TransposedLast2,
                None,
            );
        }
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_gemv_transposed_b_cuda_matches_cpu() {
    for k in [64usize, 37] {
        for m in [1usize, 5] {
            assert_gemv_parity(
                &format!("i64 gemv_bt {m}x{k} @ ({k}x17 transposed)"),
                &ramp_i64(m * k, 0),
                &[m, k],
                &ramp_i64(17 * k, 3),
                &[17, k],
                BLayout::TransposedLast2,
                None,
            );
        }
    }
}

// The transposed kernel reduces along K across a warp, so its accumulator is a
// tree rather than a running sum. Exact 128-bit addition makes that irrelevant,
// which this pins: K=48 spans two lanes' worth of work per column.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_gemv_transposed_b_overflow_and_recovers_cuda_matches_cpu() {
    const HALF: i64 = 1i64 << 62;
    let mut a = vec![0i64; 48];
    a[0] = HALF;
    a[40] = HALF;
    a[47] = -HALF;
    assert_gemv_parity(
        "i64 gemv_bt overflow and recovers",
        &a,
        &[1, 48],
        &vec![1i64; 48],
        &[1, 48],
        BLayout::TransposedLast2,
        Some(&[HALF]),
    );
}

// Batched transposed B keeps the same shortcut, with one grid z slice per batch.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_batched_gemv_transposed_b_cuda_matches_cpu() {
    assert_gemv_parity(
        "i32 batched gemv_bt 3x[4x20] @ 3x(20x9 transposed)",
        &ramp_i32(3 * 4 * 20, 0),
        &[3, 4, 20],
        &ramp_i32(3 * 9 * 20, 5),
        &[3, 9, 20],
        BLayout::TransposedLast2,
        None,
    );
}

// ============================================================================
// The GEMV path and the tiled path must agree
// ============================================================================
//
// Same operand rows, run twice: once at M=4, which takes the GEMV kernel, and
// once at M=20, which is past the `m <= 16` threshold and takes the tiled
// kernel. The first four rows of the tiled result must equal the GEMV result
// exactly - the two kernels share the accumulator and the saturation rule, so
// there is no tolerance to spend.

#[cfg(feature = "cuda")]
fn assert_gemv_and_tiled_agree<T>(label: &str, a_full: &[T], k: usize, n: usize, b: &[T])
where
    T: Element + PartialEq + Copy + std::fmt::Debug,
{
    const GEMV_ROWS: usize = 4;
    const TILED_ROWS: usize = 20;
    assert_eq!(
        a_full.len(),
        TILED_ROWS * k,
        "{label}: A has the wrong length"
    );

    let gemv = assert_gemv_parity(
        &format!("{label} (gemv, m={GEMV_ROWS})"),
        &a_full[..GEMV_ROWS * k],
        &[GEMV_ROWS, k],
        b,
        &[k, n],
        BLayout::AsStored,
        None,
    );
    let tiled = assert_gemv_parity(
        &format!("{label} (tiled, m={TILED_ROWS})"),
        a_full,
        &[TILED_ROWS, k],
        b,
        &[k, n],
        BLayout::AsStored,
        None,
    );

    assert_eq!(
        gemv,
        tiled[..GEMV_ROWS * n].to_vec(),
        "{label}: GEMV and tiled paths disagree on the same rows"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_gemv_and_tiled_paths_agree() {
    assert_gemv_and_tiled_agree(
        "i32 paths",
        &ramp_i32(20 * 37, 0),
        37,
        45,
        &ramp_i32(37 * 45, 2),
    );
    // Operands large enough that a 32-bit accumulator would wrap and a clamping
    // one would saturate, so the two paths only agree if both stay 128-bit.
    let big_a: Vec<i32> = (0..20 * 33)
        .map(|i| {
            if i % 2 == 0 {
                2_000_000_000
            } else {
                -1_999_999_999
            }
        })
        .collect();
    assert_gemv_and_tiled_agree("i32 paths near range", &big_a, 33, 21, &vec![1i32; 33 * 21]);
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_gemv_and_tiled_paths_agree() {
    assert_gemv_and_tiled_agree(
        "i64 paths",
        &ramp_i64(20 * 37, 0),
        37,
        45,
        &ramp_i64(37 * 45, 2),
    );
    const HALF: i64 = 1i64 << 62;
    let big_a: Vec<i64> = (0..20 * 33)
        .map(|i| if i % 2 == 0 { HALF } else { -HALF + 1 })
        .collect();
    assert_gemv_and_tiled_agree("i64 paths near range", &big_a, 33, 21, &vec![1i64; 33 * 21]);
}

// ============================================================================
// The compile-time tile instantiation, at both of its entry points
// ============================================================================
//
// `matmul_int.cu` instantiates one shape per dtype, BM=BN=64 BK=8 TM=TN=4, with
// a 2-D entry point and a batched one. These shapes select each: dims that are
// exact multiples of the 64x64 block tile (no ragged edge), dims that are not
// (every edge guard), and a batched call with a broadcast B operand so the
// batched kernel's `b_batch_count` modulo is exercised.

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i32_tiled_instantiation_shapes_cuda_matches_cpu() {
    assert_gemv_parity(
        "i32 tiled 128x64 @ 64x128 (exact block multiples)",
        &ramp_i32(128 * 64, 0),
        &[128, 64],
        &ramp_i32(64 * 128, 1),
        &[64, 128],
        BLayout::AsStored,
        None,
    );
    assert_gemv_parity(
        "i32 tiled batched 2x[40x30] @ 2x[30x20]",
        &ramp_i32(2 * 40 * 30, 0),
        &[2, 40, 30],
        &ramp_i32(2 * 30 * 20, 4),
        &[2, 30, 20],
        BLayout::AsStored,
        None,
    );
    assert_gemv_parity(
        "i32 tiled batched 3x[40x30] @ broadcast [30x20]",
        &ramp_i32(3 * 40 * 30, 0),
        &[3, 40, 30],
        &ramp_i32(30 * 20, 4),
        &[30, 20],
        BLayout::AsStored,
        None,
    );
}

#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i64_tiled_instantiation_shapes_cuda_matches_cpu() {
    assert_gemv_parity(
        "i64 tiled 128x64 @ 64x128 (exact block multiples)",
        &ramp_i64(128 * 64, 0),
        &[128, 64],
        &ramp_i64(64 * 128, 1),
        &[64, 128],
        BLayout::AsStored,
        None,
    );
    assert_gemv_parity(
        "i64 tiled batched 2x[40x30] @ 2x[30x20]",
        &ramp_i64(2 * 40 * 30, 0),
        &[2, 40, 30],
        &ramp_i64(2 * 30 * 20, 4),
        &[2, 30, 20],
        BLayout::AsStored,
        None,
    );
    assert_gemv_parity(
        "i64 tiled batched 3x[40x30] @ broadcast [30x20]",
        &ramp_i64(3 * 40 * 30, 0),
        &[3, 40, 30],
        &ramp_i64(30 * 20, 4),
        &[30, 20],
        BLayout::AsStored,
        None,
    );
}
