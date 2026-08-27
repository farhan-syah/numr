//! Matmul against a transposed B operand (CPU).
//!
//! A `[K, N]` operand with strides `[1, K]` is the transposed view of a
//! contiguous `[N, K]` weight matrix, which is what every `Linear` weight is.
//! The CPU backend reads that buffer directly instead of materializing the
//! view. These tests pin the property that makes the optimization safe: the
//! transposed operand and the same logical matrix made contiguous produce the
//! same result.

use numr::ops::MatmulOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

/// Above this element count the CPU backend uses the tiled kernel rather than
/// its small-matrix or dot-product kernels.
const TILED_THRESHOLD: usize = 128 * 128 * 128 + 1;

fn client_and_device() -> (<CpuRuntime as Runtime>::Client, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

fn f32_values(len: usize, seed: usize) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f32) * 0.03 - 3.5)
        .collect()
}

fn f64_values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| (((i * 37 + seed * 11) % 251) as f64) * 0.03 - 3.5)
        .collect()
}

/// Multiply `A[batch.., m, k]` by the transpose of `W[batch.., n, k]`, both as
/// the transposed view and as the same matrix made contiguous.
fn matmul_both_ways_f32(a_shape: &[usize], w_shape: &[usize]) -> (Vec<f32>, Vec<f32>) {
    let (client, device) = client_and_device();
    let a_len: usize = a_shape.iter().product();
    let w_len: usize = w_shape.iter().product();

    let a = Tensor::<CpuRuntime>::from_slice(&f32_values(a_len, 1), a_shape, &device).unwrap();
    let w = Tensor::<CpuRuntime>::from_slice(&f32_values(w_len, 2), w_shape, &device).unwrap();

    let last = w_shape.len() - 1;
    let b_view = w.transpose((last - 1) as isize, last as isize).unwrap();
    let b_contig = b_view.contiguous().unwrap();

    let from_view = client.matmul(&a, &b_view).unwrap();
    let from_contig = client.matmul(&a, &b_contig).unwrap();
    (from_view.to_vec::<f32>(), from_contig.to_vec::<f32>())
}

fn matmul_both_ways_f64(a_shape: &[usize], w_shape: &[usize]) -> (Vec<f64>, Vec<f64>) {
    let (client, device) = client_and_device();
    let a_len: usize = a_shape.iter().product();
    let w_len: usize = w_shape.iter().product();

    let a = Tensor::<CpuRuntime>::from_slice(&f64_values(a_len, 1), a_shape, &device).unwrap();
    let w = Tensor::<CpuRuntime>::from_slice(&f64_values(w_len, 2), w_shape, &device).unwrap();

    let last = w_shape.len() - 1;
    let b_view = w.transpose((last - 1) as isize, last as isize).unwrap();
    let b_contig = b_view.contiguous().unwrap();

    let from_view = client.matmul(&a, &b_view).unwrap();
    let from_contig = client.matmul(&a, &b_contig).unwrap();
    (from_view.to_vec::<f64>(), from_contig.to_vec::<f64>())
}

fn assert_bit_identical_f32(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "{label}: element {i} differs ({g} vs {w})"
        );
    }
}

fn assert_close_f32(got: &[f32], want: &[f32], label: &str) {
    assert_eq!(got.len(), want.len(), "{label}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-3 + 1e-4 * w.abs(),
            "{label}: element {i} differs ({g} vs {w})"
        );
    }
}

/// The shape that exposed the cost: VoxCPM2's local DiT runs 11 positions × 2
/// CFG branches, so M is 22 — just past the M ≤ 16 decode gate, which used to
/// send it down the materializing path.
#[test]
fn test_transposed_b_m22_matches_contiguous_bitwise() {
    let (m, k, n) = (22, 257, 513);
    assert!(
        m * n * k >= TILED_THRESHOLD,
        "shape must reach the tiled kernel"
    );
    let (from_view, from_contig) = matmul_both_ways_f32(&[m, k], &[n, k]);
    assert_bit_identical_f32(&from_view, &from_contig, "m=22");
}

/// Non-square, non-power-of-two K and N: the packed panels end in partial
/// blocks, so the remainder handling is exercised on both sides.
#[test]
fn test_transposed_b_odd_shapes_match_contiguous_bitwise() {
    for &(m, k, n) in &[
        (17usize, 411usize, 301usize),
        (37, 173, 349),
        (64, 199, 421),
    ] {
        assert!(
            m * n * k >= TILED_THRESHOLD,
            "shape must reach the tiled kernel"
        );
        let (from_view, from_contig) = matmul_both_ways_f32(&[m, k], &[n, k]);
        assert_bit_identical_f32(&from_view, &from_contig, &format!("m={m} k={k} n={n}"));
    }
}

#[test]
fn test_transposed_b_f64_matches_contiguous_bitwise() {
    let (m, k, n) = (22, 257, 513);
    let (from_view, from_contig) = matmul_both_ways_f64(&[m, k], &[n, k]);
    assert_eq!(from_view.len(), from_contig.len());
    for (i, (g, w)) in from_view.iter().zip(&from_contig).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "f64 element {i} differs");
    }
}

/// M ≤ 16 keeps the dot-product decode kernel, which accumulates in its own
/// order, so it matches within tolerance rather than bit for bit.
#[test]
fn test_transposed_b_decode_shapes_match_contiguous() {
    for &(m, k, n) in &[(1usize, 257usize, 513usize), (16, 411, 301)] {
        let (from_view, from_contig) = matmul_both_ways_f32(&[m, k], &[n, k]);
        assert_close_f32(&from_view, &from_contig, &format!("m={m}"));
    }
}

/// Small products stay below the tiled threshold, where the transposed operand
/// keeps being materialized, so the two agree trivially. The test pins that the
/// gate does not change results at those shapes.
#[test]
fn test_transposed_b_small_shapes_match_contiguous() {
    for &(m, k, n) in &[(22usize, 5usize, 7usize), (33, 64, 48)] {
        let (from_view, from_contig) = matmul_both_ways_f32(&[m, k], &[n, k]);
        assert_close_f32(&from_view, &from_contig, &format!("m={m} k={k} n={n}"));
    }
}

/// Batched: the transposed operand's batch stride is N*K, the same element
/// count a contiguous [K, N] batch spans.
#[test]
fn test_transposed_b_batched_matches_contiguous_bitwise() {
    let (batch, m, k, n) = (3usize, 22usize, 257usize, 513usize);
    let (from_view, from_contig) = matmul_both_ways_f32(&[batch, m, k], &[batch, n, k]);
    assert_bit_identical_f32(&from_view, &from_contig, "batched");
}

/// Broadcast batch dims: one weight matrix shared across every output batch.
#[test]
fn test_transposed_b_broadcast_batch_matches_contiguous_bitwise() {
    let (m, k, n) = (22usize, 257usize, 513usize);
    let (from_view, from_contig) = matmul_both_ways_f32(&[2, 3, m, k], &[1, n, k]);
    assert_bit_identical_f32(&from_view, &from_contig, "broadcast batch");
}
