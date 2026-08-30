//! CPU kernels must not fabricate work for a zero-sized dimension.
//!
//! Every extent below (`outer_size`, `inner_size`, `batch_size`, `spatial`)
//! used to be clamped with `.max(1)`. `product()` of an empty slice is already
//! 1, so the clamp never helped the rank-boundary case it looked like it was
//! defending — it only fired when a dimension was genuinely 0, and then the
//! kernel indexed a row the allocation does not contain.
//!
//! Since a zero-byte allocation now returns a dangling (non-null) address, such
//! a read is a silent out-of-bounds access rather than a loud null dereference.
//! One test per kernel family that carried the clamp.

use numr::prelude::*;

fn client() -> (CpuClient, CpuDevice) {
    let device = CpuDevice::new();
    let client = CpuRuntime::default_client(&device);
    (client, device)
}

// ============================================================================
// index / select
// ============================================================================

#[test]
fn index_select_with_empty_inner_dim_writes_nothing() {
    let (client, device) = client();

    // `inner_size` is 0 here. Clamped to 1 it indexed `outer * dim_size + idx`
    // into a zero-element allocation, both to read and to write.
    let a = Tensor::<CpuRuntime>::zeros(&[2, 3, 0], DType::F32, &device).unwrap();
    let idx = Tensor::<CpuRuntime>::from_slice(&[0i64, 2i64], &[2], &device).unwrap();

    let out = client.index_select(&a, 1, &idx).unwrap();

    assert_eq!(out.shape(), &[2, 2, 0]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn index_select_with_empty_outer_dim_writes_nothing() {
    let (client, device) = client();

    let a = Tensor::<CpuRuntime>::zeros(&[0, 3, 2], DType::F32, &device).unwrap();
    let idx = Tensor::<CpuRuntime>::from_slice(&[1i64], &[1], &device).unwrap();

    let out = client.index_select(&a, 1, &idx).unwrap();

    assert_eq!(out.shape(), &[0, 1, 2]);
    assert_eq!(out.numel(), 0);
}

// ============================================================================
// cumulative / logsumexp
// ============================================================================

#[test]
fn logsumexp_over_empty_batch_returns_empty() {
    let (client, device) = client();

    // `outer_size` is 0. Clamped to 1 the kernel read `reduce_size` elements
    // from a zero-element allocation and wrote one result past the output.
    let a = Tensor::<CpuRuntime>::zeros(&[0, 4], DType::F32, &device).unwrap();

    let out = client.logsumexp(&a, &[1], false).unwrap();

    assert_eq!(out.shape(), &[0]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn cumsum_over_empty_batch_returns_empty() {
    let (client, device) = client();

    let a = Tensor::<CpuRuntime>::zeros(&[0, 4], DType::F32, &device).unwrap();

    let out = client.cumsum(&a, 1).unwrap();

    assert_eq!(out.shape(), &[0, 4]);
    assert_eq!(out.numel(), 0);
}

// ============================================================================
// normalization
// ============================================================================

#[test]
fn rms_norm_over_empty_batch_returns_empty() {
    let (client, device) = client();

    // `batch_size` is 0. Clamped to 1 the kernel normalized one fabricated row
    // of `hidden_size` elements that the allocation does not contain.
    let input = Tensor::<CpuRuntime>::zeros(&[0, 4], DType::F32, &device).unwrap();
    let weight = Tensor::<CpuRuntime>::ones(&[4], DType::F32, &device).unwrap();

    let out = client.rms_norm(&input, &weight, 1e-5).unwrap();

    assert_eq!(out.shape(), &[0, 4]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn layer_norm_over_empty_batch_returns_empty() {
    let (client, device) = client();

    let input = Tensor::<CpuRuntime>::zeros(&[0, 4], DType::F32, &device).unwrap();
    let weight = Tensor::<CpuRuntime>::ones(&[4], DType::F32, &device).unwrap();
    let bias = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();

    let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();

    assert_eq!(out.shape(), &[0, 4]);
    assert_eq!(out.numel(), 0);
}

#[test]
fn group_norm_with_empty_spatial_dim_returns_empty() {
    let (client, device) = client();

    // `spatial` is 0. Clamped to 1 the kernel read and wrote `batch * channels`
    // elements from a zero-element allocation.
    let input = Tensor::<CpuRuntime>::zeros(&[2, 4, 0], DType::F32, &device).unwrap();
    let weight = Tensor::<CpuRuntime>::ones(&[4], DType::F32, &device).unwrap();
    let bias = Tensor::<CpuRuntime>::zeros(&[4], DType::F32, &device).unwrap();

    let out = client.group_norm(&input, &weight, &bias, 2, 1e-5).unwrap();

    assert_eq!(out.shape(), &[2, 4, 0]);
    assert_eq!(out.numel(), 0);
}

// ============================================================================
// matmul
// ============================================================================

#[test]
fn batched_matmul_with_empty_batch_dim_returns_empty() {
    let (client, device) = client();

    // `batch_size` is 0. Clamped to 1 it read one full `[3, 4] x [4, 5]` pair
    // from empty allocations; `matmul_batch_indices` also panicked on `rem % 0`
    // while enumerating the fabricated batch.
    let a = Tensor::<CpuRuntime>::zeros(&[0, 3, 4], DType::F32, &device).unwrap();
    let b = Tensor::<CpuRuntime>::zeros(&[0, 4, 5], DType::F32, &device).unwrap();

    let out = client.matmul(&a, &b).unwrap();

    assert_eq!(out.shape(), &[0, 3, 5]);
    assert_eq!(out.numel(), 0);
}

// ============================================================================
// fft
// ============================================================================

#[test]
fn rfft_over_empty_batch_returns_empty() {
    let (client, device) = client();

    // `batch_size` is 0. Clamped to 1 it built a `from_raw_parts` slice of
    // `n` elements over a zero-element allocation.
    let a = Tensor::<CpuRuntime>::zeros(&[0, 4], DType::F32, &device).unwrap();

    let out = client.rfft(&a, FftNormalization::None).unwrap();

    assert_eq!(out.shape(), &[0, 3]);
    assert_eq!(out.numel(), 0);
}
