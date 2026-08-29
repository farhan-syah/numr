// Backend parity tests for I8 `matmul` and `matmul_bias` on CUDA vs CPU.
//
// I8 is the one element type whose matmul does not write its own dtype, which
// is why it gets a file rather than a row in `integer_dtypes_cuda.rs`:
//
// - `matmul` on I8 returns an I32 tensor. CPU allocates the output as I32 and
//   runs `matmul_i8_to_i32_kernel` (quantized accumulation, see the I8 branch in
//   `ops/cpu/matmul.rs`), which sums in i64 and clamps once at the store
//   (`i8xi8_dot_scalar` / `saturate_i64_to_i32`). CUDA's
//   `matmul_i8_i32_tiled_64x64x8_4x4` does the same.
// - `matmul_bias` on I8 returns an I32 tensor as well, and takes an I32 bias.
//   The bias seeds the accumulator rather than being added to a narrowed
//   product, so a sum that leaves I8's range is reported, not clamped.
//
// The tests below assert the dtype as well as the values, because a kernel that
// returned the element type would still match on values for small operands.
//
// Every test is `#[cfg(feature = "cuda")]`, so these imports are too - otherwise
// a non-CUDA build would warn on all of them as unused.
#[cfg(feature = "cuda")]
use numr::dtype::DType;
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

/// Run one I8 `matmul` on CPU and CUDA and require an exact element-wise match.
///
/// `expected` is the hand-checked answer in I32, which pins the CPU reference
/// down as well. Both backends must report I32, not I8.
#[cfg(feature = "cuda")]
fn assert_i8_matmul_parity(
    label: &str,
    a: &[i8],
    a_shape: &[usize],
    b: &[i8],
    b_shape: &[usize],
    expected: Option<&[i32]>,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let cpu_result = cpu_client
        .matmul(&a_cpu, &b_cpu)
        .expect("CPU I8 matmul failed");
    assert_eq!(
        cpu_result.dtype(),
        DType::I32,
        "{label}: CPU I8 matmul must widen to I32"
    );
    let cpu_vec = cpu_result.to_vec::<i32>();

    if let Some(want) = expected {
        assert_eq!(
            cpu_vec, want,
            "{label}: CPU I8 matmul disagrees with hand value"
        );
    }

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(a, a_shape, &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(b, b_shape, &device).expect("CUDA B");
        let result = client
            .matmul(&a_gpu, &b_gpu)
            .expect("CUDA I8 matmul must be native, not unsupported");
        assert_eq!(
            result.dtype(),
            DType::I32,
            "{label}: CUDA I8 matmul must widen to I32 like CPU"
        );
        assert_eq!(
            result.shape(),
            cpu_result.shape(),
            "{label}: output shape differs from CPU"
        );
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_vec,
            "{label}: CUDA must match CPU element for element"
        );
    });
}

/// The same check for the fused `matmul_bias` entry point, where I8 operands
/// mean an I32 bias and an I32 result.
#[cfg(feature = "cuda")]
fn assert_i8_bias_parity(
    label: &str,
    a: &[i8],
    a_shape: &[usize],
    b: &[i8],
    b_shape: &[usize],
    bias: &[i32],
    expected: Option<&[i32]>,
) {
    let bias_shape = [bias.len()];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(a, a_shape, &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(b, b_shape, &cpu_device).expect("CPU B");
    let bias_cpu =
        Tensor::<CpuRuntime>::from_slice(bias, &bias_shape, &cpu_device).expect("CPU bias");
    let cpu_result = cpu_client
        .matmul_bias(&a_cpu, &b_cpu, &bias_cpu)
        .expect("CPU I8 matmul_bias failed");
    assert_eq!(
        cpu_result.dtype(),
        DType::I32,
        "{label}: CPU I8 matmul_bias must widen to I32"
    );
    let cpu_vec = cpu_result.to_vec::<i32>();

    if let Some(want) = expected {
        assert_eq!(
            cpu_vec, want,
            "{label}: CPU I8 matmul_bias disagrees with hand value"
        );
    }

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(a, a_shape, &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(b, b_shape, &device).expect("CUDA B");
        let bias_gpu =
            Tensor::<CudaRuntime>::from_slice(bias, &bias_shape, &device).expect("CUDA bias");
        let result = client
            .matmul_bias(&a_gpu, &b_gpu, &bias_gpu)
            .expect("CUDA I8 matmul_bias must be native, not unsupported");
        assert_eq!(
            result.dtype(),
            DType::I32,
            "{label}: CUDA I8 matmul_bias must widen to I32 like CPU"
        );
        assert_eq!(
            result.shape(),
            cpu_result.shape(),
            "{label}: output shape differs from CPU"
        );
        assert_eq!(
            result.to_vec::<i32>(),
            cpu_vec,
            "{label}: CUDA must match CPU element for element"
        );
    });
}

// ============================================================================
// Plain matmul: I8 x I8 -> I32
// ============================================================================

/// [[1,2],[3,4]] @ [[5,6],[7,8]]:
///   row 0 = [1*5 + 2*7, 1*6 + 2*8] = [19, 22]
///   row 1 = [3*5 + 4*7, 3*6 + 4*8] = [43, 50]
/// B is read column-wise here; a prior test in this directory shipped a wrong
/// expectation by reading it row-wise, so both columns are written out above.
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_small_hand_checked_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 2x2 @ 2x2",
        &[1i8, 2, 3, 4],
        &[2, 2],
        &[5i8, 6, 7, 8],
        &[2, 2],
        Some(&[19i32, 22, 43, 50]),
    );
}

/// Negative operands on both sides, with one exact zero:
///   [[-1,2],[3,-4]] @ [[5,-6],[-7,8]]
///   row 0 = [-1*5 + 2*-7, -1*-6 + 2*8] = [-19, 22]
///   row 1 = [3*5 + -4*-7, 3*-6 + -4*8] = [43, -50]
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_negatives_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 negatives",
        &[-1i8, 2, 3, -4],
        &[2, 2],
        &[5i8, -6, -7, 8],
        &[2, 2],
        Some(&[-19i32, 22, 43, -50]),
    );
}

/// The widening is the point of this file: every value below leaves I8's range,
/// and three of the four leave I16's as well, so a kernel that narrowed to the
/// element type could not report any of them.
///
/// A = [[127 x4], [-128 x4]], every row of B = [127, -128], K = 4:
///   C[0][0] = 4 * 127 * 127   =  64_516
///   C[0][1] = 4 * 127 * -128  = -65_024
///   C[1][0] = 4 * -128 * 127  = -65_024
///   C[1][1] = 4 * -128 * -128 =  65_536
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_accumulates_past_i8_range_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 2x4 @ 4x2 past i8 range",
        &[127i8, 127, 127, 127, -128, -128, -128, -128],
        &[2, 4],
        &[127i8, -128, 127, -128, 127, -128, 127, -128],
        &[4, 2],
        Some(&[64_516i32, -65_024, -65_024, 65_536]),
    );
}

/// Non-square with K = 3, not a multiple of the tiled kernel's BK = 8:
///   [[1,2,3],[4,5,6]] @ [[7],[8],[9]]
///   = [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_non_square_ragged_k_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 2x3 @ 3x1",
        &[1i8, 2, 3, 4, 5, 6],
        &[2, 3],
        &[7i8, 8, 9],
        &[3, 1],
        Some(&[50i32, 122]),
    );
}

/// A larger ragged shape: M = 20 is past the small-M cutoff the other integer
/// widths use for their GEMV shortcut, and K = 9 leaves the last tile of the
/// BK = 8 loop partly out of range. Too large to hand-check, so CPU is the
/// reference.
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_tiled_ragged_shape_cuda_matches_cpu() {
    let a: Vec<i8> = (0..20 * 9).map(|i| (i % 61) as i8 - 30).collect();
    let b: Vec<i8> = (0..9 * 5).map(|i| (i * 7 % 59) as i8 - 29).collect();
    assert_i8_matmul_parity("i8 20x9 @ 9x5", &a, &[20, 9], &b, &[9, 5], None);
}

/// Batched, with a second slice that scales the first by 2:
///   batch 0 = [[1,2],[3,4]] @ I           = [[1,2],[3,4]]
///   batch 1 = [[5,6],[7,8]] @ 2I          = [[10,12],[14,16]]
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_batched_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 batched 2x[2x2] @ 2x[2x2]",
        &[1i8, 2, 3, 4, 5, 6, 7, 8],
        &[2, 2, 2],
        &[1i8, 0, 0, 1, 2, 0, 0, 2],
        &[2, 2, 2],
        Some(&[1i32, 2, 3, 4, 10, 12, 14, 16]),
    );
}

/// Batched at the same magnitudes as the widening test above, so the widening
/// holds on the batched entry point too:
///   every slice is [[127 x4]] @ [[127, -128] x4] = [64_516, -65_024]
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_i8_batched_past_i8_range_cuda_matches_cpu() {
    assert_i8_matmul_parity(
        "i8 batched 2x[1x4] @ 2x[4x2] past i8 range",
        &[127i8; 8],
        &[2, 1, 4],
        &[
            127i8, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127, -128, 127,
            -128,
        ],
        &[2, 4, 2],
        Some(&[64_516i32, -65_024, 64_516, -65_024]),
    );
}

// ============================================================================
// Fused bias: I8 x I8 + I32 -> I32
// ============================================================================

/// The 2x2 product above plus a bias of [1, -2] on every row:
///   [[19,22],[43,50]] + [1,-2] = [[20,20],[44,48]]
/// B is read column-wise: column 0 is [5,7], column 1 is [6,8].
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_cuda_matches_cpu() {
    assert_i8_bias_parity(
        "i8 2x2 @ 2x2 + bias",
        &[1i8, 2, 3, 4],
        &[2, 2],
        &[5i8, 6, 7, 8],
        &[2, 2],
        &[1i32, -2],
        Some(&[20i32, 20, 44, 48]),
    );
}

/// 12 * 11 = 132, past `i8::MAX`. The bias form widens exactly as the plain one
/// does, so the store reports 132 rather than clamping to 127. One value, so
/// the contract cannot be confused with a narrowing one.
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_past_i8_range_at_the_store_cuda_matches_cpu() {
    assert_i8_bias_parity(
        "i8 1x1 @ 1x1 + zero bias past i8 range",
        &[12i8],
        &[1, 1],
        &[11i8],
        &[1, 1],
        &[0i32],
        Some(&[132i32]),
    );
}

/// The widening reaches the bias, not only the product:
///   A = [[127, 127, 127, 127]], every row of B = [127], K = 4
///   product = 4 * 127 * 127 = 64_516, bias = 1_000, sum = 65_516
/// Both the product and the sum leave I8's range and fit I32, so a bias added
/// after a narrowing store could not report this.
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_bias_past_i8_range_cuda_matches_cpu() {
    assert_i8_bias_parity(
        "i8 1x4 @ 4x1 + bias past i8 range",
        &[127i8; 4],
        &[1, 4],
        &[127i8; 4],
        &[4, 1],
        &[1_000i32],
        Some(&[65_516i32]),
    );
}

/// Batched fused bias: the batched product above plus [1, -2] on every row of
/// every slice.
///   batch 0 = [[1,2],[3,4]]     + [1,-2] = [[2,0],[4,2]]
///   batch 1 = [[10,12],[14,16]] + [1,-2] = [[11,10],[15,14]]
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_batched_cuda_matches_cpu() {
    assert_i8_bias_parity(
        "i8 batched 2x[2x2] @ 2x[2x2] + bias",
        &[1i8, 2, 3, 4, 5, 6, 7, 8],
        &[2, 2, 2],
        &[1i8, 0, 0, 1, 2, 0, 0, 2],
        &[2, 2, 2],
        &[1i32, -2],
        Some(&[2i32, 0, 4, 2, 11, 10, 15, 14]),
    );
}

/// Batched, at magnitudes that leave I8's range on both the product and the
/// bias, so the batched entry point is pinned to the same widening:
///   every slice = [[127, 127, 127, 127]] @ [[127] x4] + 1_000 = 65_516
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_batched_past_i8_range_cuda_matches_cpu() {
    assert_i8_bias_parity(
        "i8 batched 2x[1x4] @ 2x[4x1] + bias past i8 range",
        &[127i8; 8],
        &[2, 1, 4],
        &[127i8; 8],
        &[2, 4, 1],
        &[1_000i32],
        Some(&[65_516i32, 65_516]),
    );
}

/// The bias dtype is part of the contract: an I8 bias is refused on both
/// backends, and the I32 one every test above uses is accepted.
#[cfg(feature = "cuda")]
#[test]
fn test_matmul_bias_i8_rejects_i8_bias_cuda_matches_cpu() {
    let a: [i8; 4] = [1, 2, 3, 4];
    let b: [i8; 4] = [5, 6, 7, 8];

    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&a, &[2, 2], &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &[2, 2], &cpu_device).expect("CPU B");
    let bias_i8 =
        Tensor::<CpuRuntime>::from_slice(&[1i8, -2], &[2], &cpu_device).expect("CPU i8 bias");
    assert!(
        cpu_client.matmul_bias(&a_cpu, &b_cpu, &bias_i8).is_err(),
        "CPU must refuse an I8 bias for I8 operands"
    );

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(&a, &[2, 2], &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(&b, &[2, 2], &device).expect("CUDA B");
        let bias_i8 =
            Tensor::<CudaRuntime>::from_slice(&[1i8, -2], &[2], &device).expect("CUDA i8 bias");
        assert!(
            client.matmul_bias(&a_gpu, &b_gpu, &bias_i8).is_err(),
            "CUDA must refuse an I8 bias for I8 operands"
        );

        let bias_i32 =
            Tensor::<CudaRuntime>::from_slice(&[1i32, -2], &[2], &device).expect("CUDA i32 bias");
        let out = client
            .matmul_bias(&a_gpu, &b_gpu, &bias_i32)
            .expect("CUDA must accept an I32 bias for I8 operands");
        assert_eq!(out.dtype(), DType::I32);
        assert_eq!(out.to_vec::<i32>(), vec![20i32, 20, 44, 48]);
    });
}

// ============================================================================
// einsum follows matmul
// ============================================================================

/// `einsum` maps "ij,jk->ik" onto `matmul`, so the widened output dtype has to
/// arrive through it unchanged. This is the delegation checked, not einsum's
/// own contraction path.
#[cfg(feature = "cuda")]
#[test]
fn test_einsum_i8_delegates_to_matmul_cuda_matches_cpu() {
    use numr::ops::EinsumOps;

    let a: [i8; 4] = [1, 2, 3, 4];
    let b: [i8; 4] = [5, 6, 7, 8];
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_cpu = Tensor::<CpuRuntime>::from_slice(&a, &[2, 2], &cpu_device).expect("CPU A");
    let b_cpu = Tensor::<CpuRuntime>::from_slice(&b, &[2, 2], &cpu_device).expect("CPU B");
    let cpu_result = cpu_client
        .einsum("ij,jk->ik", &[&a_cpu, &b_cpu])
        .expect("CPU I8 einsum failed");
    assert_eq!(
        cpu_result.dtype(),
        DType::I32,
        "CPU einsum must widen to I32"
    );
    assert_eq!(cpu_result.to_vec::<i32>(), vec![19i32, 22, 43, 50]);

    with_cuda_backend(|client, device| {
        let a_gpu = Tensor::<CudaRuntime>::from_slice(&a, &[2, 2], &device).expect("CUDA A");
        let b_gpu = Tensor::<CudaRuntime>::from_slice(&b, &[2, 2], &device).expect("CUDA B");
        let result = client
            .einsum("ij,jk->ik", &[&a_gpu, &b_gpu])
            .expect("CUDA I8 einsum must reach the native matmul");
        assert_eq!(result.dtype(), DType::I32, "CUDA einsum must widen to I32");
        assert_eq!(result.to_vec::<i32>(), cpu_result.to_vec::<i32>());
    });
}
