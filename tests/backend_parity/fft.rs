// Backend parity tests migrated from tests/fft_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::algorithm::fft::{FftAlgorithms, FftDirection, FftNormalization};
use numr::dtype::Complex64;
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

fn get_cpu_client() -> CpuClient {
    let device = CpuDevice::new();
    CpuClient::new(device)
}

fn assert_complex_close(cpu: &[Complex64], other: &[Complex64], tol: f32, label: &str) {
    assert_eq!(cpu.len(), other.len(), "{} length mismatch", label);
    for (i, (c, g)) in cpu.iter().zip(other.iter()).enumerate() {
        assert!((c.re - g.re).abs() < tol, "{} re idx {}", label, i);
        assert!((c.im - g.im).abs() < tol, "{} im idx {}", label, i);
    }
}

fn assert_f32_close(cpu: &[f32], other: &[f32], tol: f32, label: &str) {
    assert_eq!(cpu.len(), other.len(), "{} length mismatch", label);
    for (i, (c, g)) in cpu.iter().zip(other.iter()).enumerate() {
        assert!((c - g).abs() < tol, "{} idx {}", label, i);
    }
}

#[test]
fn test_fft_forward_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    for size in [4, 8, 16, 64, 128, 256] {
        let input_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.1).cos()))
            .collect();

        let cpu_input =
            Tensor::<CpuRuntime>::try_from_slice(&input_data, &[size], &cpu_device).unwrap();
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_data: Vec<Complex64> = cpu_result.to_vec();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
                &input_data,
                &[size],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, 1e-4, "fft cuda");
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
                &input_data,
                &[size],
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, 1e-4, "fft wgpu");
        });
    }
}

#[test]
fn test_fft_roundtrip_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    let input_data: Vec<Complex64> = (0..64)
        .map(|i| Complex64::new(i as f32, -(i as f32) * 0.5))
        .collect();

    let cpu_input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &[64], &cpu_device).unwrap();
    let cpu_fft = cpu_client
        .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
        .unwrap();
    let cpu_result = cpu_client
        .fft(&cpu_fft, FftDirection::Inverse, FftNormalization::Backward)
        .unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &input_data,
            &[64],
            &cuda_device,
        )
        .unwrap();
        let fft = cuda_client
            .fft(&input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let result = cuda_client
            .fft(&fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-4, "roundtrip cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &input_data,
            &[64],
            &wgpu_device,
        )
        .unwrap();
        let fft = wgpu_client
            .fft(&input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let result = wgpu_client
            .fft(&fft, FftDirection::Inverse, FftNormalization::Backward)
            .unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-3, "roundtrip wgpu");
    });
}

#[test]
fn test_rfft_irfft_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();
    let n = 64;
    let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).cos()).collect();

    let cpu_real = Tensor::<CpuRuntime>::try_from_slice(&input_data, &[n], &cpu_device).unwrap();
    let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
    let cpu_ir = cpu_client
        .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
        .unwrap();
    let cpu_ir_data: Vec<f32> = cpu_ir.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let real = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &input_data,
            &[n],
            &cuda_device,
        )
        .unwrap();
        let freq = cuda_client.rfft(&real, FftNormalization::None).unwrap();
        let ir = cuda_client
            .irfft(&freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let data: Vec<f32> = ir.to_vec();
        for (c, g) in cpu_ir_data.iter().zip(data.iter()) {
            assert!((c - g).abs() < 1e-4);
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let real = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &input_data,
            &[n],
            &wgpu_device,
        )
        .unwrap();
        let freq = wgpu_client.rfft(&real, FftNormalization::None).unwrap();
        let ir = wgpu_client
            .irfft(&freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let data: Vec<f32> = ir.to_vec();
        for (c, g) in cpu_ir_data.iter().zip(data.iter()) {
            assert!((c - g).abs() < 1e-4);
        }
    });
}

#[test]
fn test_fftshift_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    let input_data: Vec<Complex64> = (0..16)
        .map(|i| Complex64::new(i as f32, -i as f32))
        .collect();
    let cpu_input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &[16], &cpu_device).unwrap();
    let cpu_result = cpu_client.fftshift(&cpu_input).unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &input_data,
            &[16],
            &cuda_device,
        )
        .unwrap();
        let result = cuda_client.fftshift(&input).unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-5, "fftshift cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &input_data,
            &[16],
            &wgpu_device,
        )
        .unwrap();
        let result = wgpu_client.fftshift(&input).unwrap();
        let data: Vec<Complex64> = result.to_vec();
        assert_complex_close(&cpu_data, &data, 1e-5, "fftshift wgpu");
    });
}

#[test]
fn test_cpu_fft_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    // Use a batched shape so CPU FFT goes through the batched kernel path.
    let shape = [8, 128];
    let numel: usize = shape.iter().product();
    let input_data: Vec<Complex64> = (0..numel)
        .map(|i| Complex64::new((i as f32 * 0.031).sin(), (i as f32 * 0.017).cos()))
        .collect();

    let input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &shape, &device).unwrap();
    let base = default_client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();
    let configured = configured_client
        .fft(&input, FftDirection::Forward, FftNormalization::None)
        .unwrap();

    let base_data: Vec<Complex64> = base.to_vec();
    let configured_data: Vec<Complex64> = configured.to_vec();
    assert_complex_close(
        &base_data,
        &configured_data,
        1e-5,
        "cpu fft parallelism config",
    );
}

#[test]
fn test_cpu_rfft_irfft_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    let shape = [6, 64];
    let numel: usize = shape.iter().product();
    let input_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.023).sin()).collect();

    let input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &shape, &device).unwrap();
    let base_freq = default_client.rfft(&input, FftNormalization::None).unwrap();
    let cfg_freq = configured_client
        .rfft(&input, FftNormalization::None)
        .unwrap();
    let base_freq_data: Vec<Complex64> = base_freq.to_vec();
    let cfg_freq_data: Vec<Complex64> = cfg_freq.to_vec();
    assert_complex_close(
        &base_freq_data,
        &cfg_freq_data,
        1e-5,
        "cpu rfft parallelism config",
    );

    let base_rec = default_client
        .irfft(&base_freq, Some(shape[1]), FftNormalization::Backward)
        .unwrap();
    let cfg_rec = configured_client
        .irfft(&cfg_freq, Some(shape[1]), FftNormalization::Backward)
        .unwrap();
    let base_rec_data: Vec<f32> = base_rec.to_vec();
    let cfg_rec_data: Vec<f32> = cfg_rec.to_vec();
    assert_f32_close(
        &base_rec_data,
        &cfg_rec_data,
        1e-5,
        "cpu irfft parallelism config",
    );
}

#[test]
fn test_cpu_fftshift_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(1), Some(1024)));

    let shape = [5, 32];
    let numel: usize = shape.iter().product();
    let input_data: Vec<Complex64> = (0..numel)
        .map(|i| Complex64::new((i as f32 * 0.013).cos(), (i as f32 * 0.019).sin()))
        .collect();

    let input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &shape, &device).unwrap();
    let base_shift = default_client.fftshift(&input).unwrap();
    let cfg_shift = configured_client.fftshift(&input).unwrap();
    let base_shift_data: Vec<Complex64> = base_shift.to_vec();
    let cfg_shift_data: Vec<Complex64> = cfg_shift.to_vec();
    assert_complex_close(
        &base_shift_data,
        &cfg_shift_data,
        1e-5,
        "cpu fftshift parallelism config",
    );

    let base_unshift = default_client.ifftshift(&base_shift).unwrap();
    let cfg_unshift = configured_client.ifftshift(&cfg_shift).unwrap();
    let base_unshift_data: Vec<Complex64> = base_unshift.to_vec();
    let cfg_unshift_data: Vec<Complex64> = cfg_unshift.to_vec();
    assert_complex_close(
        &base_unshift_data,
        &cfg_unshift_data,
        1e-5,
        "cpu ifftshift parallelism config",
    );
}
