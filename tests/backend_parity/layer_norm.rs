// Backend parity tests for layer_norm (NormalizationOps trait)
//
// Two CUDA launcher defects motivate the shapes here:
//   - eps: the F64 kernel takes a `double`, so a launcher pushing an f32 leaves
//     the upper half of the parameter unwritten. Only visible where the variance
//     is small enough that eps is not negligible, hence the tiny-spread cases.
//   - shared memory: the Welford merge stores one triple PER WARP, so it indexes
//     `3 * ceil(blockDim.x / 32)` elements. A per-thread estimate is short of
//     that for a block of 1 or 2 threads, hence the hidden_size 1 and 2 cases.
//
// Dtype-parameterized: each test runs for all supported dtypes across all backends.

use numr::ops::NormalizationOps;

use crate::backend_parity::dtype_helpers::tensor_from_f64;
#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use crate::common::{
    DTypeDomain, assert_tensor_allclose, create_cpu_client, is_dtype_supported, parity_dtypes,
};

// ============================================================================
// Test Data
// ============================================================================

/// Row values centered on 1.0 with a controllable spread.
///
/// `spread` sets the variance: a small spread puts the variance near eps, which
/// is the only regime where a wrong eps changes the output.
fn ln_input(n: usize, spread: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i as f64) * 0.017).sin() * spread)
        .collect()
}

fn ln_weight(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.75 + ((i as f64) * 0.011).cos() * 0.25)
        .collect()
}

fn ln_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.013).sin() * 0.1).collect()
}

/// Runs `layer_norm` on CPU and each GPU backend and asserts they agree.
///
/// The CUDA block is `min(256, hidden)`, so `hidden` also picks the block size
/// and with it the warp count the Welford merge allocates for.
fn assert_layer_norm_parity(label: &str, batch: usize, hidden: usize, spread: f64) {
    let shape = [batch, hidden];
    let x_data = ln_input(batch * hidden, spread);
    let w_data = ln_weight(hidden);
    let b_data = ln_bias(hidden);
    let eps = 1e-5f32;

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_x = tensor_from_f64(&x_data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(&w_data, &[hidden], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = tensor_from_f64(&b_data, &[hidden], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU bias failed for {label} [{dtype:?}]: {e}"));
        let cpu_out = cpu_client
            .layer_norm(&cpu_x, &cpu_w, &cpu_b, eps)
            .unwrap_or_else(|e| panic!("CPU layer_norm failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let x = tensor_from_f64(&x_data, &shape, dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA input failed for {label} [{dtype:?}]: {e}"));
                let w = tensor_from_f64(&w_data, &[hidden], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA weight failed for {label} [{dtype:?}]: {e}"));
                let b = tensor_from_f64(&b_data, &[hidden], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA bias failed for {label} [{dtype:?}]: {e}"));
                let out = client.layer_norm(&x, &w, &b, eps).unwrap_or_else(|e| {
                    panic!("CUDA layer_norm failed for {label} [{dtype:?}]: {e}")
                });
                assert_tensor_allclose(
                    &out,
                    &cpu_out,
                    dtype,
                    &format!("{label} CUDA vs CPU [{dtype:?}]"),
                );
            });
        }

        #[cfg(feature = "wgpu")]
        if is_dtype_supported("wgpu", dtype) {
            with_wgpu_backend(|client, device| {
                let x = tensor_from_f64(&x_data, &shape, dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("WGPU input failed for {label} [{dtype:?}]: {e}"));
                let w = tensor_from_f64(&w_data, &[hidden], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("WGPU weight failed for {label} [{dtype:?}]: {e}"));
                let b = tensor_from_f64(&b_data, &[hidden], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("WGPU bias failed for {label} [{dtype:?}]: {e}"));
                let out = client.layer_norm(&x, &w, &b, eps).unwrap_or_else(|e| {
                    panic!("WGPU layer_norm failed for {label} [{dtype:?}]: {e}")
                });
                assert_tensor_allclose(
                    &out,
                    &cpu_out,
                    dtype,
                    &format!("{label} WebGPU vs CPU [{dtype:?}]"),
                );
            });
        }
    }
}

/// Ordinary variance, so eps is negligible. This is the case that must stay
/// green regardless of how eps is pushed; it pins the normal path.
#[test]
fn layer_norm_ordinary_variance_parity() {
    assert_layer_norm_parity("layer_norm_ordinary_variance", 4, 256, 1.0);
}

/// Variance near eps, single warp. A wrong eps changes `1 / sqrt(var + eps)`
/// here by orders of magnitude.
#[test]
fn layer_norm_tiny_variance_parity() {
    assert_layer_norm_parity("layer_norm_tiny_variance", 2, 32, 2e-3);
}

/// Variance near eps across several warps, so the block-level Welford merge
/// runs before eps is applied.
#[test]
fn layer_norm_tiny_variance_wide_row_parity() {
    assert_layer_norm_parity("layer_norm_tiny_variance_wide_row", 2, 1024, 2e-3);
}

/// Two-thread block: the merge still needs its three per-warp slots, which is
/// more shared memory than a per-thread estimate reserves.
#[test]
fn layer_norm_two_thread_block_parity() {
    assert_layer_norm_parity("layer_norm_two_thread_block", 3, 2, 2e-3);
}

/// One-thread block, the largest gap between the per-warp requirement and a
/// per-thread estimate. Variance is zero, so the output is the bias unless the
/// reduction reads memory it does not own.
#[test]
fn layer_norm_one_thread_block_parity() {
    assert_layer_norm_parity("layer_norm_one_thread_block", 3, 1, 2e-3);
}
