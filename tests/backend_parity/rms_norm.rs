// Backend parity tests for rms_norm (NormalizationOps trait)
//
// The shapes here straddle the CUDA launcher's register-cached gate: rows at or
// under `NORM_MAX_REGS_PER_THREAD * block_size` take the single-pass kernel,
// wider rows fall back to the two-pass kernel. Both are compared against CPU.
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

/// Deterministic, non-repeating input values, so a mis-strided read or a
/// dropped grid-stride iteration changes the result.
fn rms_input(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.017).sin() + 0.5).collect()
}

fn rms_weight(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.75 + ((i as f64) * 0.011).cos() * 0.25)
        .collect()
}

/// Runs `rms_norm` on CPU and each GPU backend and asserts they agree.
///
/// The CUDA block is `min(256, hidden)` and the register-cached kernel holds 16
/// elements per thread, so the dispatch gate sits at `hidden == 4096`. Each
/// test below states which side of that gate it exercises; a shape on the wrong
/// side silently covers the other kernel.
fn assert_rms_norm_parity(label: &str, batch: usize, hidden: usize) {
    let shape = [batch, hidden];
    let x_data = rms_input(batch * hidden);
    let w_data = rms_weight(hidden);
    let eps = 1e-5f32;

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_x = tensor_from_f64(&x_data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(&w_data, &[hidden], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight failed for {label} [{dtype:?}]: {e}"));
        let cpu_out = cpu_client
            .rms_norm(&cpu_x, &cpu_w, eps)
            .unwrap_or_else(|e| panic!("CPU rms_norm failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let x = tensor_from_f64(&x_data, &shape, dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA input failed for {label} [{dtype:?}]: {e}"));
                let w = tensor_from_f64(&w_data, &[hidden], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA weight failed for {label} [{dtype:?}]: {e}"));
                let out = client.rms_norm(&x, &w, eps).unwrap_or_else(|e| {
                    panic!("CUDA rms_norm failed for {label} [{dtype:?}]: {e}")
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
                let out = client.rms_norm(&x, &w, eps).unwrap_or_else(|e| {
                    panic!("WGPU rms_norm failed for {label} [{dtype:?}]: {e}")
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

/// Register path with a block narrower than a warp.
#[test]
fn rms_norm_narrow_block_parity() {
    assert_rms_norm_parity("rms_norm_narrow_block", 3, 24);
}

/// Register path with a block size that is NOT a power of two, so the
/// reduction runs its ragged branch. The two-pass kernel drops an element for
/// such block sizes; every shape this small now takes the register path.
#[test]
fn rms_norm_non_power_of_two_block_parity() {
    assert_rms_norm_parity("rms_norm_non_power_of_two_block", 2, 100);
}

/// Register path with a partial last grid-stride iteration, so the bound on
/// the register loop is exercised rather than a whole number of passes.
#[test]
fn rms_norm_partial_stride_parity() {
    assert_rms_norm_parity("rms_norm_partial_stride", 2, 300);
}

/// Register path exactly at the gate: 16 elements per thread, the maximum the
/// register array holds. One element wider falls back.
#[test]
fn rms_norm_at_gate_parity() {
    assert_rms_norm_parity("rms_norm_at_gate", 2, 4096);
}

/// One block-stride past the gate, so the two-pass fallback runs. This is the
/// case that must NOT change when the register kernel does.
#[test]
fn rms_norm_past_gate_falls_back_parity() {
    assert_rms_norm_parity("rms_norm_past_gate_falls_back", 2, 4352);
}
