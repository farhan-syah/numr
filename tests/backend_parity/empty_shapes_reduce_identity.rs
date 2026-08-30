//! Backend parity for a reduction whose reduce dimension is empty but whose
//! output is not.
//!
//! Two shapes of the same problem, and the whole point of this file is that they
//! behave differently:
//!
//! - `[0, 5]` reduced over dim 1 leaves `[0]`: the OUTPUT is empty. Nothing to
//!   compute, and every backend returns the empty tensor.
//! - `[3, 0]` reduced over dim 1 leaves `[3]`: the output is NOT empty, and each
//!   element folds over zero terms. The value is the reduction's identity —
//!   `-/+inf` for a float wide enough to hold it, the dtype's own extreme for
//!   FP8, integers and bool (which is what the CUDA kernels seed with).
//!
//! `argmax`/`argmin` are the exception: no index names an element of an empty
//! dimension, so all three backends must reject it rather than invent one.
//!
//! CPU is the reference. `assert_tensor_allclose` compares integer dtypes
//! exactly, and compares infinities by identity, so an infinite expectation is
//! a real assertion here.

use numr::dtype::DType;
use numr::error::Error;
use numr::ops::{CumulativeOps, IndexingOps, ReduceOps};
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuRuntime};
#[cfg(feature = "cuda")]
use numr::runtime::cuda::CudaRuntime;
#[cfg(feature = "wgpu")]
use numr::runtime::wgpu::WgpuRuntime;
use numr::tensor::Tensor;

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend_or_skip;
use crate::common::{DTypeDomain, assert_tensor_allclose, create_cpu_client, parity_dtypes};

/// Reduced over dim 1 this leaves `[3]`: three elements, each folding over
/// nothing.
const FOLD_OVER_NOTHING: [usize; 2] = [3, 0];

/// Reduced over dim 1 this leaves `[0]`: the output is empty too.
const EMPTY_OUTPUT: [usize; 2] = [0, 5];

/// Dtypes whose `max`/`min` identity is an infinity rather than the dtype's own
/// bound. FP8 is excluded on purpose: the CUDA accumulator traits seed it from
/// `-/+FP8_*_MAX`, so infinity would be an answer no backend produces.
fn has_infinite_identity(dtype: DType) -> bool {
    matches!(dtype, DType::F64 | DType::F32 | DType::F16 | DType::BF16)
}

fn max_identity(dtype: DType) -> f64 {
    if has_infinite_identity(dtype) {
        f64::NEG_INFINITY
    } else {
        dtype.min_value()
    }
}

fn min_identity(dtype: DType) -> f64 {
    if has_infinite_identity(dtype) {
        f64::INFINITY
    } else {
        dtype.max_value()
    }
}

/// The `[3]` tensor every non-empty case below is compared against.
fn identity_tensor(
    dtype: DType,
    value: f64,
    cpu_device: &<CpuRuntime as Runtime>::Device,
) -> Tensor<CpuRuntime> {
    Tensor::<CpuRuntime>::full_scalar(&[3], dtype, value, cpu_device).expect("identity tensor")
}

// ============================================================================
// max / min / sum / prod over a zero-length dim
// ============================================================================

fn check_identity_reduce<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: ReduceOps<R>,
{
    let shape = &FOLD_OVER_NOTHING[..];
    let a = Tensor::<R>::empty(shape, dtype, device).expect("a");
    let a_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu a");

    // max: the reduction's identity, not the dtype's uninitialized allocation.
    let want = identity_tensor(dtype, max_identity(dtype), cpu_device);
    let cpu_got = cpu_client.max(&a_cpu, &[1], false).expect("cpu max");
    assert_tensor_allclose(&cpu_got, &want, dtype, "max [3, 0] cpu vs identity");
    let got = client
        .max(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} max {shape:?} {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[3], "max [3, 0] {backend} output shape");
    assert_tensor_allclose(&got, &want, dtype, &format!("max [3, 0] {backend} vs cpu"));

    // min: the mirror identity.
    let want = identity_tensor(dtype, min_identity(dtype), cpu_device);
    let cpu_got = cpu_client.min(&a_cpu, &[1], false).expect("cpu min");
    assert_tensor_allclose(&cpu_got, &want, dtype, "min [3, 0] cpu vs identity");
    let got = client
        .min(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} min {shape:?} {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[3], "min [3, 0] {backend} output shape");
    assert_tensor_allclose(&got, &want, dtype, &format!("min [3, 0] {backend} vs cpu"));

    // Controls: sum and prod already answered their identities.
    let want = identity_tensor(dtype, 0.0, cpu_device);
    let got = client
        .sum(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} sum {shape:?} {dtype:?}: {e:?}"));
    assert_tensor_allclose(&got, &want, dtype, &format!("sum [3, 0] {backend} vs 0"));

    let want = identity_tensor(dtype, 1.0, cpu_device);
    let got = client
        .prod(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} prod {shape:?} {dtype:?}: {e:?}"));
    assert_tensor_allclose(&got, &want, dtype, &format!("prod [3, 0] {backend} vs 1"));
}

/// The `[0, 5]` counterpart: the output is empty, so there is no identity to
/// write and every op returns the empty tensor.
fn check_empty_output_reduce<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: ReduceOps<R>,
{
    let shape = &EMPTY_OUTPUT[..];
    let a = Tensor::<R>::empty(shape, dtype, device).expect("a");
    let a_cpu = Tensor::<CpuRuntime>::empty(shape, dtype, cpu_device).expect("cpu a");

    let cpu_max = cpu_client.max(&a_cpu, &[1], false).expect("cpu max");
    assert_eq!(cpu_max.shape(), &[0], "max [0, 5] cpu output shape");

    let got = client
        .max(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} max {shape:?} {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[0], "max [0, 5] {backend} output shape");
    assert_tensor_allclose(&got, &cpu_max, dtype, &format!("max [0, 5] {backend}"));

    let cpu_min = cpu_client.min(&a_cpu, &[1], false).expect("cpu min");
    let got = client
        .min(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} min {shape:?} {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[0], "min [0, 5] {backend} output shape");
    assert_tensor_allclose(&got, &cpu_min, dtype, &format!("min [0, 5] {backend}"));
}

// ============================================================================
// logsumexp
// ============================================================================

fn check_logsumexp<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &<CpuRuntime as Runtime>::Device,
    dtype: DType,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: CumulativeOps<R>,
{
    // `log(sum of nothing)` is `log(0)`. FP8E4M3 has no infinity, so the same
    // conversion that CPU and CUDA both use lands on its lowest value.
    let want_value = f64::NEG_INFINITY;

    let a = Tensor::<R>::empty(&FOLD_OVER_NOTHING[..], dtype, device).expect("a");
    let a_cpu =
        Tensor::<CpuRuntime>::empty(&FOLD_OVER_NOTHING[..], dtype, cpu_device).expect("cpu a");

    let want = identity_tensor(dtype, want_value, cpu_device);
    let cpu_got = cpu_client
        .logsumexp(&a_cpu, &[1], false)
        .expect("cpu logsumexp over a zero-length dim");
    assert_tensor_allclose(&cpu_got, &want, dtype, "logsumexp [3, 0] cpu vs -inf");

    let got = client
        .logsumexp(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} logsumexp [3, 0] {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[3], "logsumexp [3, 0] {backend} output shape");
    assert_tensor_allclose(
        &got,
        &want,
        dtype,
        &format!("logsumexp [3, 0] {backend} vs cpu"),
    );

    // The empty-output counterpart stays empty.
    let a = Tensor::<R>::empty(&EMPTY_OUTPUT[..], dtype, device).expect("a");
    let a_cpu = Tensor::<CpuRuntime>::empty(&EMPTY_OUTPUT[..], dtype, cpu_device).expect("cpu a");
    let cpu_got = cpu_client
        .logsumexp(&a_cpu, &[1], false)
        .expect("cpu logsumexp with an empty output");
    assert_eq!(cpu_got.shape(), &[0], "logsumexp [0, 5] cpu output shape");
    let got = client
        .logsumexp(&a, &[1], false)
        .unwrap_or_else(|e| panic!("{backend} logsumexp [0, 5] {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[0], "logsumexp [0, 5] {backend} output shape");
}

// ============================================================================
// argmax / argmin
// ============================================================================

fn check_arg_reduce<R, C>(client: &C, device: &R::Device, dtype: DType, backend: &str)
where
    R: Runtime<DType = DType>,
    C: IndexingOps<R>,
{
    let a = Tensor::<R>::empty(&FOLD_OVER_NOTHING[..], dtype, device).expect("a");

    let err = client.argmax(&a, 1, false).expect_err(&format!(
        "{backend} argmax [3, 0] {dtype:?} must be rejected"
    ));
    assert!(
        matches!(err, Error::InvalidArgument { .. }),
        "{backend} argmax [3, 0] {dtype:?}: want InvalidArgument, got {err:?}"
    );

    let err = client.argmin(&a, 1, false).expect_err(&format!(
        "{backend} argmin [3, 0] {dtype:?} must be rejected"
    ));
    assert!(
        matches!(err, Error::InvalidArgument { .. }),
        "{backend} argmin [3, 0] {dtype:?}: want InvalidArgument, got {err:?}"
    );

    // `[0, 5]` over dim 1 reduces over FIVE elements into an empty output. That
    // is not the rejected case, and it must still return the empty tensor.
    let a = Tensor::<R>::empty(&EMPTY_OUTPUT[..], dtype, device).expect("a");
    let got = client
        .argmax(&a, 1, false)
        .unwrap_or_else(|e| panic!("{backend} argmax [0, 5] {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[0], "argmax [0, 5] {backend} output shape");
    let got = client
        .argmin(&a, 1, false)
        .unwrap_or_else(|e| panic!("{backend} argmin [0, 5] {dtype:?}: {e:?}"));
    assert_eq!(got.shape(), &[0], "argmin [0, 5] {backend} output shape");
}

// ============================================================================
// Per-backend entry points
// ============================================================================

#[test]
fn test_empty_reduce_identity_cpu() {
    let (client, device) = create_cpu_client();
    let (cpu_client, cpu_device) = create_cpu_client();
    for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cpu") {
        check_identity_reduce::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
        check_empty_output_reduce::<CpuRuntime, _>(
            &client,
            &device,
            &cpu_client,
            &cpu_device,
            dtype,
            "cpu",
        );
        check_arg_reduce::<CpuRuntime, _>(&client, &device, dtype, "cpu");
    }
    for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cpu") {
        check_logsumexp::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, dtype, "cpu");
    }
}

#[cfg(feature = "cuda")]
#[test]
fn test_empty_reduce_identity_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "cuda") {
            check_identity_reduce::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_empty_output_reduce::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
            check_arg_reduce::<CudaRuntime, _>(&client, &device, dtype, "cuda");
        }
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "cuda") {
            check_logsumexp::<CudaRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "cuda",
            );
        }
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_empty_reduce_identity_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        for dtype in parity_dtypes(DTypeDomain::AllNumeric, "wgpu") {
            check_identity_reduce::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_empty_output_reduce::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
            check_arg_reduce::<WgpuRuntime, _>(&client, &device, dtype, "wgpu");
        }
        for dtype in parity_dtypes(DTypeDomain::FloatsOnly, "wgpu") {
            check_logsumexp::<WgpuRuntime, _>(
                &client,
                &device,
                &cpu_client,
                &cpu_device,
                dtype,
                "wgpu",
            );
        }
    });
}
