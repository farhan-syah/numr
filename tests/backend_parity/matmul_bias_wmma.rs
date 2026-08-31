// Backend parity tests for the WMMA tensor-core epilogue in
// MatmulOps::matmul_bias (F16/BF16).
//
// `use_wmma` (src/ops/cuda/matmul.rs) dispatches the fused bias-in-epilogue
// WMMA kernel only when `caps.f16_mma`/`caps.bf16`, `m > 16`, and M/N/K are
// all 16-multiples; unaligned shapes are padded up to 16-multiples (A, B, and
// the bias vector) and sliced back afterward. `matmul_bias.rs`'s existing
// tests all use 2x2 shapes, so `m > 16` never held and none of them ever
// reached this path. These cases do: aligned sizes that dispatch straight to
// WMMA, a ragged block edge that stays 16-aligned, sizes below and at
// 16-multiples that force the padding path, sizes below the `m > 16` gate
// that must still fall back to the generic kernel, batched broadcast at a
// WMMA-eligible size, and a bias-dominant case where a dropped or
// mis-indexed bias fails loudly.

#[cfg(feature = "f16")]
use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(all(feature = "f16", feature = "cuda"))]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "f16")]
use crate::common::create_cpu_client;
#[cfg(all(feature = "f16", feature = "cuda"))]
use crate::common::{assert_tensor_allclose, is_dtype_supported};
#[cfg(feature = "f16")]
use numr::dtype::DType;
#[cfg(feature = "f16")]
use numr::ops::MatmulOps;

#[cfg(feature = "f16")]
fn deterministic_f64(n: usize, phase: f64) -> Vec<f64> {
    (0..n)
        .map(|i| {
            ((i as f64 * 0.013 + phase).sin() * 0.5) + ((i as f64 * 0.0047 + phase).cos() * 0.3)
        })
        .collect()
}

/// Build CPU (reference) and, when CUDA is available, CUDA `matmul_bias`
/// results for the given dtype/shapes and assert they agree within
/// `tolerance_for_dtype`. Shared by every WMMA-epilogue case below.
#[cfg(feature = "f16")]
#[allow(clippy::too_many_arguments)]
fn assert_matmul_bias_wmma_parity(
    dtype: DType,
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
    bias_data: &[f64],
    bias_shape: &[usize],
    test_name: &str,
) {
    let (cpu_client, cpu_device) = create_cpu_client();
    let a_t = tensor_from_f64(a_data, a_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let b_t = tensor_from_f64(b_data, b_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let bias_t = tensor_from_f64(bias_data, bias_shape, dtype, &cpu_device, &cpu_client).unwrap();
    let cpu_result = cpu_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();

    #[cfg(feature = "cuda")]
    if is_dtype_supported("cuda", dtype) {
        with_cuda_backend(|cuda_client, cuda_device| {
            let a_t = tensor_from_f64(a_data, a_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let b_t = tensor_from_f64(b_data, b_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let bias_t =
                tensor_from_f64(bias_data, bias_shape, dtype, &cuda_device, &cuda_client).unwrap();
            let result = cuda_client.matmul_bias(&a_t, &b_t, &bias_t).unwrap();
            assert_tensor_allclose(&result, &cpu_result, dtype, test_name);
        });
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = test_name;
    }
}

/// Plain 2D case: deterministic A/B/bias at the given shape.
#[cfg(feature = "f16")]
fn assert_matmul_bias_wmma_2d(dtype: DType, m: usize, k: usize, n: usize, test_name: &str) {
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        dtype,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        test_name,
    );
}

// --- Case 1: aligned, reaches WMMA directly (m > 16, M/N/K all 16-multiples) ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_aligned_128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        128,
        128,
        128,
        "matmul_bias_f16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_aligned_256x512x128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        256,
        128,
        512,
        "matmul_bias_f16_wmma_aligned_256x512x128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_aligned_128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        128,
        128,
        128,
        "matmul_bias_bf16_wmma_aligned_128 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_aligned_256x512x128_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        256,
        128,
        512,
        "matmul_bias_bf16_wmma_aligned_256x512x128 CUDA vs CPU",
    );
}

// --- Case 2: 16-aligned but ragged against the 128x128 block tile: catches ---
// --- an out-of-range bias[col] read or a mishandled partial tile.         ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_partial_tile_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        144,
        144,
        144,
        "matmul_bias_f16_wmma_partial_tile CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_partial_tile_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        144,
        144,
        144,
        "matmul_bias_bf16_wmma_partial_tile CUDA vs CPU",
    );
}

// --- Case 3: unaligned, exercises the pad-to-16-multiples-then-slice path ---
// --- for A, B, AND the bias vector.                                       ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_padded_100_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        100,
        100,
        100,
        "matmul_bias_f16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_padded_130x70x50_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        130,
        50,
        70,
        "matmul_bias_f16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_padded_100_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        100,
        100,
        100,
        "matmul_bias_bf16_wmma_padded_100 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_padded_130x70x50_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        130,
        50,
        70,
        "matmul_bias_bf16_wmma_padded_130x70x50 CUDA vs CPU",
    );
}

// --- Case 4: below the `m > 16` WMMA gate: must still route through the  ---
// --- generic kernel and match CPU. A regression here means the dispatch  ---
// --- guard itself is wrong, not the WMMA kernel.                         ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_below_wmma_gate_m16_match_cpu() {
    // m == 16 exactly: use_wmma requires m > 16, so this must NOT reach WMMA
    // even though 16 is itself a 16-multiple.
    assert_matmul_bias_wmma_2d(
        DType::F16,
        16,
        64,
        64,
        "matmul_bias_f16_below_wmma_gate_m16 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_below_wmma_gate_m8_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::F16,
        8,
        64,
        64,
        "matmul_bias_f16_below_wmma_gate_m8 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_below_wmma_gate_m16_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        16,
        64,
        64,
        "matmul_bias_bf16_below_wmma_gate_m16 CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_below_wmma_gate_m8_match_cpu() {
    assert_matmul_bias_wmma_2d(
        DType::BF16,
        8,
        64,
        64,
        "matmul_bias_bf16_below_wmma_gate_m8 CUDA vs CPU",
    );
}

// --- Case 5: batched with broadcast, at a WMMA-eligible size (m=64>16,    ---
// --- M/N/K all 16-multiples). Bias is indexed by global column only and  ---
// --- must broadcast across rows AND batch slices.                       ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_batched_a_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[1, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_batched_a_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_batched_b_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[1, k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_batched_b_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_batched_a_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(m * k, 0.0);
    let b_data = deterministic_f64(batch * k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[1, m, k],
        &b_data,
        &[batch, k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_batched_a_broadcast CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_batched_b_broadcast_match_cpu() {
    let (batch, m, k, n) = (4usize, 64usize, 128usize, 128usize);
    let a_data = deterministic_f64(batch * m * k, 0.0);
    let b_data = deterministic_f64(k * n, 1.7);
    let bias_data = deterministic_f64(n, 3.1);
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[batch, m, k],
        &b_data,
        &[1, k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_batched_b_broadcast CUDA vs CPU",
    );
}

// --- Case 6: bias dominates the result at a WMMA-eligible size (A/B tiny, ---
// --- bias large), so a dropped or mis-indexed bias fails loudly instead  ---
// --- of hiding under matmul noise.                                       ---

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_f16_wmma_bias_dominant_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f64> = deterministic_f64(m * k, 0.0)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let b_data: Vec<f64> = deterministic_f64(k * n, 1.7)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let bias_data: Vec<f64> = deterministic_f64(n, 3.1)
        .iter()
        .map(|v| v * 50.0 + 100.0)
        .collect();
    assert_matmul_bias_wmma_parity(
        DType::F16,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        "matmul_bias_f16_wmma_bias_dominant CUDA vs CPU",
    );
}

#[cfg(feature = "f16")]
#[test]
fn matmul_bias_bf16_wmma_bias_dominant_match_cpu() {
    let (m, k, n) = (128usize, 128usize, 128usize);
    let a_data: Vec<f64> = deterministic_f64(m * k, 0.0)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let b_data: Vec<f64> = deterministic_f64(k * n, 1.7)
        .iter()
        .map(|v| v * 1e-3)
        .collect();
    let bias_data: Vec<f64> = deterministic_f64(n, 3.1)
        .iter()
        .map(|v| v * 50.0 + 100.0)
        .collect();
    assert_matmul_bias_wmma_parity(
        DType::BF16,
        &a_data,
        &[m, k],
        &b_data,
        &[k, n],
        &bias_data,
        &[n],
        "matmul_bias_bf16_wmma_bias_dominant CUDA vs CPU",
    );
}
