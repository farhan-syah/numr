// Backend parity tests for group_norm (NormalizationOps trait)
//
// Two CUDA launcher defects motivate the shapes here:
//   - eps: the F64 kernel takes a `double`, so a launcher pushing an f32 leaves
//     the upper half of the parameter unwritten. Only visible where the variance
//     is small enough that eps is not negligible, hence the tiny-spread cases.
//   - shared memory: the kernel splits its dynamic shared memory into two
//     per-thread arrays and the F64 kernel indexes doubles, so it needs twice
//     what an f32 estimate reserves. The block is `min(256, group_size)`, so the
//     group sizes below span a narrow block and a full one.
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

/// Values centered on 1.0 with a controllable spread.
///
/// `spread` sets the within-group variance: a small spread puts it near eps,
/// which is the only regime where a wrong eps changes the output.
fn gn_input(n: usize, spread: f64) -> Vec<f64> {
    (0..n)
        .map(|i| 1.0 + ((i as f64) * 0.017).sin() * spread)
        .collect()
}

fn gn_weight(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.75 + ((i as f64) * 0.011).cos() * 0.25)
        .collect()
}

fn gn_bias(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i as f64) * 0.013).sin() * 0.1).collect()
}

/// Runs `group_norm` on CPU and each GPU backend and asserts they agree.
///
/// The CUDA block is `min(256, channels_per_group * spatial)`, so those two
/// dimensions together pick the block size the shared arrays are sized from.
fn assert_group_norm_parity(
    label: &str,
    batch: usize,
    channels: usize,
    spatial: usize,
    num_groups: usize,
    spread: f64,
) {
    let shape = [batch, channels, spatial];
    let x_data = gn_input(batch * channels * spatial, spread);
    let w_data = gn_weight(channels);
    let b_data = gn_bias(channels);
    let eps = 1e-5f32;

    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        let (cpu_client, cpu_device) = create_cpu_client();
        let cpu_x = tensor_from_f64(&x_data, &shape, dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU input failed for {label} [{dtype:?}]: {e}"));
        let cpu_w = tensor_from_f64(&w_data, &[channels], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU weight failed for {label} [{dtype:?}]: {e}"));
        let cpu_b = tensor_from_f64(&b_data, &[channels], dtype, &cpu_device, &cpu_client)
            .unwrap_or_else(|e| panic!("CPU bias failed for {label} [{dtype:?}]: {e}"));
        let cpu_out = cpu_client
            .group_norm(&cpu_x, &cpu_w, &cpu_b, num_groups, eps)
            .unwrap_or_else(|e| panic!("CPU group_norm failed for {label} [{dtype:?}]: {e}"));

        #[cfg(feature = "cuda")]
        if is_dtype_supported("cuda", dtype) {
            with_cuda_backend(|client, device| {
                let x = tensor_from_f64(&x_data, &shape, dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA input failed for {label} [{dtype:?}]: {e}"));
                let w = tensor_from_f64(&w_data, &[channels], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA weight failed for {label} [{dtype:?}]: {e}"));
                let b = tensor_from_f64(&b_data, &[channels], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("CUDA bias failed for {label} [{dtype:?}]: {e}"));
                let out = client
                    .group_norm(&x, &w, &b, num_groups, eps)
                    .unwrap_or_else(|e| {
                        panic!("CUDA group_norm failed for {label} [{dtype:?}]: {e}")
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
                let w = tensor_from_f64(&w_data, &[channels], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("WGPU weight failed for {label} [{dtype:?}]: {e}"));
                let b = tensor_from_f64(&b_data, &[channels], dtype, &device, &client)
                    .unwrap_or_else(|e| panic!("WGPU bias failed for {label} [{dtype:?}]: {e}"));
                let out = client
                    .group_norm(&x, &w, &b, num_groups, eps)
                    .unwrap_or_else(|e| {
                        panic!("WGPU group_norm failed for {label} [{dtype:?}]: {e}")
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
fn group_norm_ordinary_variance_parity() {
    assert_group_norm_parity("group_norm_ordinary_variance", 2, 8, 16, 2, 1.0);
}

/// Variance near eps. A wrong eps changes `1 / sqrt(var + eps)` here by orders
/// of magnitude.
#[test]
fn group_norm_tiny_variance_parity() {
    assert_group_norm_parity("group_norm_tiny_variance", 2, 8, 16, 4, 2e-3);
}

/// Four-element group, so the block is far narrower than a warp and the two
/// shared arrays are at their smallest.
#[test]
fn group_norm_narrow_block_parity() {
    assert_group_norm_parity("group_norm_narrow_block", 2, 4, 2, 2, 2e-3);
}

/// Group wider than the maximum block, so the reduction runs a full block and
/// the two shared arrays are at their largest.
#[test]
fn group_norm_full_block_parity() {
    assert_group_norm_parity("group_norm_full_block", 2, 16, 64, 2, 1.0);
}

/// Full block AND variance near eps, so both defects are in play at once.
#[test]
fn group_norm_full_block_tiny_variance_parity() {
    assert_group_norm_parity("group_norm_full_block_tiny_variance", 2, 16, 64, 2, 2e-3);
}

/// Group size 9, so the CUDA block is not a power of two. A tree reduction that
/// starts at `blockDim.x / 2` never folds in the entries above the largest power
/// of two below the block size, and every group size above is a power of two, so
/// nothing else here would notice.
#[test]
fn group_norm_odd_block_parity() {
    assert_group_norm_parity("group_norm_odd_block", 2, 6, 3, 2, 2e-3);
}

/// Group size 48: not a power of two either, but wide enough that the dropped
/// entries are a whole tail of the reduction rather than a single element.
#[test]
fn group_norm_wide_odd_block_parity() {
    assert_group_norm_parity("group_norm_wide_odd_block", 2, 12, 8, 2, 1.0);
}
