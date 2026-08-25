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
            Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device).unwrap();
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_data: Vec<Complex64> = cpu_result.to_vec();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
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
            let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
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

    let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[64], &cpu_device).unwrap();
    let cpu_fft = cpu_client
        .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
        .unwrap();
    let cpu_result = cpu_client
        .fft(&cpu_fft, FftDirection::Inverse, FftNormalization::Backward)
        .unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
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
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
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

    let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device).unwrap();
    let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
    let cpu_ir = cpu_client
        .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
        .unwrap();
    let cpu_ir_data: Vec<f32> = cpu_ir.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let real =
            Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(&input_data, &[n], &cuda_device)
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
        let real =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(&input_data, &[n], &wgpu_device)
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
    let cpu_input = Tensor::<CpuRuntime>::from_slice(&input_data, &[16], &cpu_device).unwrap();
    let cpu_result = cpu_client.fftshift(&cpu_input).unwrap();
    let cpu_data: Vec<Complex64> = cpu_result.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
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
        let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
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

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &shape, &device).unwrap();
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

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &shape, &device).unwrap();
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

    let input = Tensor::<CpuRuntime>::from_slice(&input_data, &shape, &device).unwrap();
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

// ---------------------------------------------------------------------------
// Arbitrary (non-power-of-two) transform sizes — the Bluestein path.
//
// Every parity test above uses a power of two, which is exactly the set of
// sizes the Stockham kernels already handled. These pin the sizes that used to
// be rejected outright, including the two that real vocoders use: NeuCodec's
// n_fft = 1920 and Kokoro's 20.
// ---------------------------------------------------------------------------

/// Sizes chosen to cover the distinct shapes the algorithm can hit: 1 (M == 1
/// degenerate), a prime, an odd composite, Kokoro's 20, one just past a power
/// of two, one just under, and NeuCodec's 1920.
const ARBITRARY_SIZES: [usize; 8] = [1, 3, 5, 7, 20, 100, 1000, 1920];

#[test]
fn test_fft_arbitrary_size_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    for size in ARBITRARY_SIZES {
        assert!(
            !size.is_power_of_two() || size == 1,
            "size {size} is a power of two; it would not exercise Bluestein"
        );
        let input_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f32 * 0.1).sin(), (i as f32 * 0.07).cos()))
            .collect();

        let cpu_input =
            Tensor::<CpuRuntime>::from_slice(&input_data, &[size], &cpu_device).unwrap();
        let cpu_result = cpu_client
            .fft(&cpu_input, FftDirection::Forward, FftNormalization::None)
            .unwrap();
        let cpu_data: Vec<Complex64> = cpu_result.to_vec();

        // Tolerance scales with size: an unnormalized transform of N terms has
        // magnitude up to N, so a fixed absolute bound would be far tighter at
        // N = 1920 than at N = 3 while testing less.
        let tol = 1e-4 * (size as f32).max(1.0);

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &input_data,
                &[size],
                &cuda_device,
            )
            .unwrap();
            let result = cuda_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, tol, &format!("fft cuda n={size}"));
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &input_data,
                &[size],
                &wgpu_device,
            )
            .unwrap();
            let result = wgpu_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let data: Vec<Complex64> = result.to_vec();
            assert_complex_close(&cpu_data, &data, tol, &format!("fft wgpu n={size}"));
        });
    }
}

#[test]
fn test_fft_arbitrary_size_roundtrip_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    for size in ARBITRARY_SIZES {
        let input_data: Vec<Complex64> = (0..size)
            .map(|i| Complex64::new((i as f32 * 0.3).cos(), (i as f32 * 0.11).sin()))
            .collect();

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let input = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &input_data,
                &[size],
                &cuda_device,
            )
            .unwrap();
            let fwd = cuda_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let back = cuda_client
                .fft(&fwd, FftDirection::Inverse, FftNormalization::Backward)
                .unwrap();
            let data: Vec<Complex64> = back.to_vec();
            // Round trip returns the ORIGINAL input, so this checks the forward
            // and inverse chirps against each other, not just against CPU.
            assert_complex_close(
                &input_data,
                &data,
                1e-3,
                &format!("fft roundtrip cuda n={size}"),
            );
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let input = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &input_data,
                &[size],
                &wgpu_device,
            )
            .unwrap();
            let fwd = wgpu_client
                .fft(&input, FftDirection::Forward, FftNormalization::None)
                .unwrap();
            let back = wgpu_client
                .fft(&fwd, FftDirection::Inverse, FftNormalization::Backward)
                .unwrap();
            let data: Vec<Complex64> = back.to_vec();
            assert_complex_close(
                &input_data,
                &data,
                1e-3,
                &format!("fft roundtrip wgpu n={size}"),
            );
        });

        // `input_data` is only read inside the backend arms; without either
        // feature this test compiles to a no-op loop rather than an unused
        // binding.
        let _ = (&input_data, &cpu_device, &cpu_client);
    }
}

#[test]
fn test_rfft_irfft_arbitrary_size_parity() {
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();

    for n in ARBITRARY_SIZES {
        let input_data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.13).cos()).collect();

        let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[n], &cpu_device).unwrap();
        let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
        let cpu_freq_data: Vec<Complex64> = cpu_freq.to_vec();
        assert_eq!(cpu_freq_data.len(), n / 2 + 1, "rfft bin count n={n}");

        let cpu_ir = cpu_client
            .irfft(&cpu_freq, Some(n), FftNormalization::Backward)
            .unwrap();
        let cpu_ir_data: Vec<f32> = cpu_ir.to_vec();

        let tol = 1e-4 * (n as f32).max(1.0);

        #[cfg(feature = "cuda")]
        with_cuda_backend(|cuda_client, cuda_device| {
            let real = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
                &input_data,
                &[n],
                &cuda_device,
            )
            .unwrap();
            let freq = cuda_client.rfft(&real, FftNormalization::None).unwrap();
            let freq_data: Vec<Complex64> = freq.to_vec();
            assert_complex_close(&cpu_freq_data, &freq_data, tol, &format!("rfft cuda n={n}"));

            let ir = cuda_client
                .irfft(&freq, Some(n), FftNormalization::Backward)
                .unwrap();
            let ir_data: Vec<f32> = ir.to_vec();
            assert_f32_close(&cpu_ir_data, &ir_data, 1e-3, &format!("irfft cuda n={n}"));
            // The round trip must return the samples we started from.
            assert_f32_close(
                &input_data,
                &ir_data,
                1e-3,
                &format!("rfft/irfft roundtrip cuda n={n}"),
            );
        });

        #[cfg(feature = "wgpu")]
        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let real = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
                &input_data,
                &[n],
                &wgpu_device,
            )
            .unwrap();
            let freq = wgpu_client.rfft(&real, FftNormalization::None).unwrap();
            let freq_data: Vec<Complex64> = freq.to_vec();
            assert_complex_close(&cpu_freq_data, &freq_data, tol, &format!("rfft wgpu n={n}"));

            let ir = wgpu_client
                .irfft(&freq, Some(n), FftNormalization::Backward)
                .unwrap();
            let ir_data: Vec<f32> = ir.to_vec();
            assert_f32_close(&cpu_ir_data, &ir_data, 1e-3, &format!("irfft wgpu n={n}"));
            assert_f32_close(
                &input_data,
                &ir_data,
                1e-3,
                &format!("rfft/irfft roundtrip wgpu n={n}"),
            );
        });
    }
}

#[test]
fn test_batched_arbitrary_size_parity() {
    // A batch dimension exercises the per-row indexing in the Bluestein
    // kernels; a single row would let a batch-stride bug pass.
    let cpu_client = get_cpu_client();
    let cpu_device = cpu_client.device().clone();
    let (batch, n) = (4usize, 20usize);

    let input_data: Vec<f32> = (0..batch * n)
        .map(|i| ((i as f32) * 0.17).sin() + (i % 7) as f32 * 0.05)
        .collect();

    let cpu_real = Tensor::<CpuRuntime>::from_slice(&input_data, &[batch, n], &cpu_device).unwrap();
    let cpu_freq = cpu_client.rfft(&cpu_real, FftNormalization::None).unwrap();
    assert_eq!(cpu_freq.shape(), &[batch, n / 2 + 1]);
    let cpu_freq_data: Vec<Complex64> = cpu_freq.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let real = Tensor::<numr::runtime::cuda::CudaRuntime>::from_slice(
            &input_data,
            &[batch, n],
            &cuda_device,
        )
        .unwrap();
        let freq = cuda_client.rfft(&real, FftNormalization::None).unwrap();
        assert_eq!(freq.shape(), &[batch, n / 2 + 1]);
        let data: Vec<Complex64> = freq.to_vec();
        assert_complex_close(&cpu_freq_data, &data, 1e-3, "batched rfft cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let real = Tensor::<numr::runtime::wgpu::WgpuRuntime>::from_slice(
            &input_data,
            &[batch, n],
            &wgpu_device,
        )
        .unwrap();
        let freq = wgpu_client.rfft(&real, FftNormalization::None).unwrap();
        assert_eq!(freq.shape(), &[batch, n / 2 + 1]);
        let data: Vec<Complex64> = freq.to_vec();
        assert_complex_close(&cpu_freq_data, &data, 1e-3, "batched rfft wgpu");
    });
}
