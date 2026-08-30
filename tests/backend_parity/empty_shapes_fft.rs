//! Backend parity for zero-element inputs in the FFT family.
//!
//! Companion to the other `empty_shapes_*` files. Every transform here takes its
//! launch geometry from a `batch_size` — the product of every dimension but the
//! last — and a transform length `n`. A zero batch used to be floored to 1, so
//! the kernels transformed one full row out of an allocation holding none.
//!
//! The zero sits in the batch rather than in `n` because every backend rejects
//! `n == 0` up front with an `InvalidArgument`.
//!
//! CPU is the reference. Each result is empty, so the check is that the call
//! succeeds and reports the shape CPU reports, rather than a value comparison.

use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::dtype::DType;
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
use crate::common::create_cpu_client;

type CpuDev = <CpuRuntime as Runtime>::Device;

const NORM: FftNormalization = FftNormalization::None;

fn check_fft<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R>,
{
    // Complex forward transform over an empty batch of 4-point rows.
    let c_cpu =
        Tensor::<CpuRuntime>::empty(&[0, 4], DType::Complex64, cpu_device).expect("cpu complex");
    let c = Tensor::<R>::empty(&[0, 4], DType::Complex64, device).expect("complex");

    let expected = cpu_client
        .fft(&c_cpu, FftDirection::Forward, NORM)
        .expect("cpu fft over an empty batch");
    let actual = client
        .fft(&c, FftDirection::Forward, NORM)
        .unwrap_or_else(|e| panic!("{backend} fft: {e:?}"));
    assert_eq!(actual.shape(), expected.shape(), "{backend} fft shape");
    assert_eq!(actual.numel(), 0, "{backend} fft element count");

    // fftshift / ifftshift keep the input shape.
    let expected = cpu_client
        .fftshift(&c_cpu)
        .expect("cpu fftshift over an empty batch");
    let actual = client
        .fftshift(&c)
        .unwrap_or_else(|e| panic!("{backend} fftshift: {e:?}"));
    assert_eq!(actual.shape(), expected.shape(), "{backend} fftshift shape");

    let expected = cpu_client
        .ifftshift(&c_cpu)
        .expect("cpu ifftshift over an empty batch");
    let actual = client
        .ifftshift(&c)
        .unwrap_or_else(|e| panic!("{backend} ifftshift: {e:?}"));
    assert_eq!(
        actual.shape(),
        expected.shape(),
        "{backend} ifftshift shape"
    );
}

fn check_real_fft<R, C>(
    client: &C,
    device: &R::Device,
    cpu_client: &CpuClient,
    cpu_device: &CpuDev,
    backend: &str,
) where
    R: Runtime<DType = DType>,
    C: FftAlgorithms<R>,
{
    // rfft: real `[0, 4]` in, complex `[0, 3]` out.
    let r_cpu = Tensor::<CpuRuntime>::empty(&[0, 4], DType::F32, cpu_device).expect("cpu real");
    let r = Tensor::<R>::empty(&[0, 4], DType::F32, device).expect("real");

    let expected = cpu_client
        .rfft(&r_cpu, NORM)
        .expect("cpu rfft over an empty batch");
    let actual = client
        .rfft(&r, NORM)
        .unwrap_or_else(|e| panic!("{backend} rfft: {e:?}"));
    assert_eq!(actual.shape(), expected.shape(), "{backend} rfft shape");
    assert_eq!(actual.numel(), 0, "{backend} rfft element count");

    // irfft: complex `[0, 3]` in, real `[0, 4]` out.
    let h_cpu =
        Tensor::<CpuRuntime>::empty(&[0, 3], DType::Complex64, cpu_device).expect("cpu half");
    let h = Tensor::<R>::empty(&[0, 3], DType::Complex64, device).expect("half");

    let expected = cpu_client
        .irfft(&h_cpu, Some(4), NORM)
        .expect("cpu irfft over an empty batch");
    let actual = client
        .irfft(&h, Some(4), NORM)
        .unwrap_or_else(|e| panic!("{backend} irfft: {e:?}"));
    assert_eq!(actual.shape(), expected.shape(), "{backend} irfft shape");
    assert_eq!(actual.numel(), 0, "{backend} irfft element count");
}

// ============================================================================
// Per-backend entry points
// ============================================================================

#[test]
fn test_empty_fft_cpu_is_self_consistent() {
    let (client, device) = create_cpu_client();
    let (cpu_client, cpu_device) = create_cpu_client();
    check_fft::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cpu");
    check_real_fft::<CpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cpu");
}

#[cfg(feature = "cuda")]
#[test]
fn test_empty_fft_cuda_matches_cpu() {
    with_cuda_backend(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        check_fft::<CudaRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cuda");
        check_real_fft::<CudaRuntime, _>(&client, &device, &cpu_client, &cpu_device, "cuda");
    });
}

#[cfg(feature = "wgpu")]
#[test]
fn test_empty_fft_wgpu_matches_cpu() {
    with_wgpu_backend_or_skip(|client, device| {
        let (cpu_client, cpu_device) = create_cpu_client();
        check_fft::<WgpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "wgpu");
        check_real_fft::<WgpuRuntime, _>(&client, &device, &cpu_client, &cpu_device, "wgpu");
    });
}
