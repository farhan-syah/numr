//! `bincount` on the two degenerate inputs: empty, and containing a negative.
//!
//! # Empty input
//!
//! NumPy returns `zeros(minlength)`. CPU used to ERROR with "bincount requires
//! non-negative values": its `max_i64_kernel` answers -1 for an empty input and
//! the sign test then fired. An empty input holds no value, so it holds no
//! negative one — the error was plainly wrong, and CPU was the only backend
//! that raised it.
//!
//! # Negative input
//!
//! CPU rejects any negative value, and that is the reference: a negative index
//! has no bin, and NumPy raises `ValueError`. The GPU backends checked only
//! `max < 0`, so a negative sitting beside a POSITIVE maximum passed the check
//! and was then silently dropped by the kernel's bounds test. `[-1, 5]` counted
//! as if the -1 were not there. That case is pinned below explicitly, because a
//! test using only all-negative input passes against the broken check.
//!
//! `bincount_with_len` is deliberately NOT covered here: its documented
//! contract is that values outside `[0, len)` are ignored, which is the whole
//! reason it exists (no sizing sync). It is a different operation, not a laxer
//! one.
//!
//! Run: cargo test --test bincount_degenerate
//! Run: cargo test --features cuda --test bincount_degenerate
//! Run: cargo test --features wgpu --test bincount_degenerate

mod common;

use common::create_cpu_client;
use numr::dtype::DType;
use numr::error::Error;
use numr::ops::IndexingOps;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;
use numr::tensor::Tensor;

/// The exact payload every backend must produce for a negative input. CPU is
/// the reference, so its variant and its string are the contract.
const NEGATIVE_REASON: &str = "bincount requires non-negative values";

// ============================================================================
// Shared checks, run against each backend
// ============================================================================

/// An empty input yields `minlength` zeroed I64 bins, never an error.
fn check_empty_input<R, C>(client: &C, device: &R::Device, backend: &str)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    let empty: [i32; 0] = [];
    let input = Tensor::<R>::from_slice(&empty, &[0], device)
        .unwrap_or_else(|e| panic!("{backend}: staging the empty input failed: {e:?}"));

    let out = client
        .bincount(&input, None, 3)
        .unwrap_or_else(|e| panic!("{backend}: bincount on an empty input failed: {e:?}"));
    assert_eq!(
        out.shape(),
        &[3],
        "{backend}: empty input, minlength 3: shape"
    );
    assert_eq!(
        out.dtype(),
        DType::I64,
        "{backend}: empty input, minlength 3: dtype"
    );
    assert_eq!(
        out.to_vec::<i64>(),
        vec![0i64; 3],
        "{backend}: empty input, minlength 3: bins must be zero"
    );

    // minlength 0 leaves nothing to size the output with either. The answer is
    // the empty histogram, not an error.
    let out = client
        .bincount(&input, None, 0)
        .unwrap_or_else(|e| panic!("{backend}: bincount on an empty input, minlength 0: {e:?}"));
    assert_eq!(
        out.shape(),
        &[0],
        "{backend}: empty input, minlength 0: shape"
    );
}

/// Assert one negative-bearing input is rejected with CPU's exact error.
fn assert_rejects<R, C>(client: &C, device: &R::Device, backend: &str, case: &str, values: &[i32])
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    let input = Tensor::<R>::from_slice(values, &[values.len()], device)
        .unwrap_or_else(|e| panic!("{backend} {case}: staging the input failed: {e:?}"));

    let result = client.bincount(&input, None, 0);
    match result {
        Err(Error::InvalidArgument { arg, reason }) => {
            assert_eq!(arg, "input", "{backend} {case}: error names the wrong arg");
            assert_eq!(
                reason, NEGATIVE_REASON,
                "{backend} {case}: error reason diverges from CPU"
            );
        }
        Err(other) => panic!("{backend} {case}: expected InvalidArgument, got {other:?}"),
        Ok(out) => panic!(
            "{backend} {case}: negative value accepted, bins {:?}",
            out.to_vec::<i64>()
        ),
    }
}

/// Every negative-input shape, including the one a `max < 0` check misses.
fn check_negative_input<R, C>(client: &C, device: &R::Device, backend: &str)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    // The case the GPU backends silently dropped: the maximum is positive, so a
    // `max < 0` check passes and the kernel's bounds test discards the -1.
    assert_rejects::<R, C>(
        client,
        device,
        backend,
        "negative beside positive max",
        &[-1, 5],
    );
    assert_rejects::<R, C>(
        client,
        device,
        backend,
        "negative in the middle",
        &[5, -3, 2],
    );
    // All-negative: the only shape a `max < 0` check ever caught.
    assert_rejects::<R, C>(client, device, backend, "all negative", &[-5, -1]);
    // A lone -1 is also the value CPU's empty-input max scan reports, so this
    // pins that a real -1 is still rejected after the empty-input fix.
    assert_rejects::<R, C>(client, device, backend, "lone -1", &[-1]);
}

// ============================================================================
// CPU — the reference
// ============================================================================

#[test]
fn cpu_empty_input_returns_zeroed_bins() {
    let (client, device) = create_cpu_client();
    check_empty_input::<CpuRuntime, _>(&client, &device, "cpu");
}

#[test]
fn cpu_rejects_a_negative_input() {
    let (client, device) = create_cpu_client();
    check_negative_input::<CpuRuntime, _>(&client, &device, "cpu");
}

/// I64 input takes a different widening path than I32 on every backend, so the
/// rejection is pinned there too.
#[test]
fn cpu_rejects_a_negative_i64_input() {
    let (client, device) = create_cpu_client();
    let values: [i64; 2] = [-1, 5];
    let input = Tensor::<CpuRuntime>::from_slice(&values, &[2], &device).expect("cpu i64 input");
    match client.bincount(&input, None, 0) {
        Err(Error::InvalidArgument { arg, reason }) => {
            assert_eq!(arg, "input");
            assert_eq!(reason, NEGATIVE_REASON);
        }
        Err(other) => panic!("cpu i64: expected InvalidArgument, got {other:?}"),
        Ok(out) => panic!("cpu i64: negative accepted, bins {:?}", out.to_vec::<i64>()),
    }
}

// ============================================================================
// CUDA
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use crate::common::backend_lock::with_cuda_backend;
    use numr::runtime::cuda::CudaRuntime;

    #[test]
    fn empty_input_returns_zeroed_bins() {
        with_cuda_backend(|client, device| {
            check_empty_input::<CudaRuntime, _>(&client, &device, "cuda");
        });
    }

    #[test]
    fn rejects_a_negative_input() {
        with_cuda_backend(|client, device| {
            check_negative_input::<CudaRuntime, _>(&client, &device, "cuda");
        });
    }

    #[test]
    fn rejects_a_negative_i64_input() {
        with_cuda_backend(|client, device| {
            let values: [i64; 2] = [-1, 5];
            let input =
                Tensor::<CudaRuntime>::from_slice(&values, &[2], &device).expect("cuda i64 input");
            match client.bincount(&input, None, 0) {
                Err(Error::InvalidArgument { arg, reason }) => {
                    assert_eq!(arg, "input");
                    assert_eq!(reason, NEGATIVE_REASON);
                }
                Err(other) => panic!("cuda i64: expected InvalidArgument, got {other:?}"),
                Ok(out) => {
                    panic!(
                        "cuda i64: negative accepted, bins {:?}",
                        out.to_vec::<i64>()
                    )
                }
            }
        });
    }

    /// The min/max readback must not disturb a valid call: a positive input
    /// still sizes its histogram from the maximum.
    #[test]
    fn a_valid_input_still_counts() {
        with_cuda_backend(|client, device| {
            let values: [i32; 7] = [0, 1, 1, 3, 2, 1, 3];
            let input = Tensor::<CudaRuntime>::from_slice(&values, &[7], &device)
                .expect("cuda valid input");
            let out = client.bincount(&input, None, 0).expect("cuda bincount");
            assert_eq!(out.to_vec::<i64>(), vec![1i64, 3, 1, 2]);
        });
    }
}

// ============================================================================
// WebGPU
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu {
    use super::*;
    use crate::common::backend_lock::with_wgpu_backend;
    use numr::runtime::wgpu::WgpuRuntime;

    #[test]
    fn empty_input_returns_zeroed_bins() {
        with_wgpu_backend(|client, device| {
            check_empty_input::<WgpuRuntime, _>(&client, &device, "wgpu");
        });
    }

    #[test]
    fn rejects_a_negative_input() {
        with_wgpu_backend(|client, device| {
            check_negative_input::<WgpuRuntime, _>(&client, &device, "wgpu");
        });
    }
}
