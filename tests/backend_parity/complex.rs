// Backend parity tests migrated from tests/complex_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::dtype::Complex64;
#[cfg(feature = "wgpu")]
use numr::error::Error;
use numr::ops::{BinaryOps, ComplexOps, UnaryOps};
#[cfg(feature = "wgpu")]
use numr::prelude::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

fn assert_complex_close(cpu: &[Complex64], other: &[Complex64], tol: f32, label: &str) {
    assert_eq!(
        cpu.len(),
        other.len(),
        "{label}: length mismatch {} vs {}",
        cpu.len(),
        other.len()
    );
    for (i, (a, b)) in cpu.iter().zip(other.iter()).enumerate() {
        assert!(
            (a.re - b.re).abs() <= tol,
            "{label}: real mismatch at {i}: {} vs {}",
            a.re,
            b.re
        );
        assert!(
            (a.im - b.im).abs() <= tol,
            "{label}: imag mismatch at {i}: {} vs {}",
            a.im,
            b.im
        );
    }
}

#[test]
fn test_cpu_complex_parallelism_config_matches_default() {
    let device = CpuDevice::new();
    let default_client = CpuClient::new(device.clone());
    let configured_client =
        default_client.with_parallelism(ParallelismConfig::new(Some(2), Some(257)));

    // Keep this above CPU complex kernel parallel threshold (4096) to exercise
    // the Rayon chunking path with custom chunk_size.
    let shape = [128, 64];
    let numel: usize = shape.iter().product();
    let a_data: Vec<Complex64> = (0..numel)
        .map(|i| Complex64::new((i as f32 * 0.011).sin(), (i as f32 * 0.017).cos()))
        .collect();
    let real_data: Vec<f32> = (0..numel)
        .map(|i| 1.1 + (i as f32 * 0.019).sin().abs())
        .collect();
    let imag_data: Vec<f32> = (0..numel).map(|i| (i as f32 * 0.023).cos()).collect();

    let a = Tensor::<CpuRuntime>::try_from_slice(&a_data, &shape, &device).unwrap();
    let real = Tensor::<CpuRuntime>::try_from_slice(&real_data, &shape, &device).unwrap();
    let imag = Tensor::<CpuRuntime>::try_from_slice(&imag_data, &shape, &device).unwrap();

    let base_conj: Vec<Complex64> = default_client.conj(&a).unwrap().to_vec();
    let cfg_conj: Vec<Complex64> = configured_client.conj(&a).unwrap().to_vec();
    assert_complex_close(
        &base_conj,
        &cfg_conj,
        1e-6,
        "cpu complex conj parallelism config",
    );

    let base_real: Vec<f32> = default_client.real(&a).unwrap().to_vec();
    let cfg_real: Vec<f32> = configured_client.real(&a).unwrap().to_vec();
    for (idx, (b, c)) in base_real.iter().zip(cfg_real.iter()).enumerate() {
        assert!((b - c).abs() <= 1e-6, "cpu complex real mismatch at {idx}");
    }

    let base_imag: Vec<f32> = default_client.imag(&a).unwrap().to_vec();
    let cfg_imag: Vec<f32> = configured_client.imag(&a).unwrap().to_vec();
    for (idx, (b, c)) in base_imag.iter().zip(cfg_imag.iter()).enumerate() {
        assert!((b - c).abs() <= 1e-6, "cpu complex imag mismatch at {idx}");
    }

    let base_angle: Vec<f32> = default_client.angle(&a).unwrap().to_vec();
    let cfg_angle: Vec<f32> = configured_client.angle(&a).unwrap().to_vec();
    for (idx, (b, c)) in base_angle.iter().zip(cfg_angle.iter()).enumerate() {
        assert!((b - c).abs() <= 1e-6, "cpu complex angle mismatch at {idx}");
    }

    let base_make: Vec<Complex64> = default_client.make_complex(&real, &imag).unwrap().to_vec();
    let cfg_make: Vec<Complex64> = configured_client
        .make_complex(&real, &imag)
        .unwrap()
        .to_vec();
    assert_complex_close(
        &base_make,
        &cfg_make,
        1e-6,
        "cpu make_complex parallelism config",
    );

    let made = default_client.make_complex(&real, &imag).unwrap();
    let base_mul: Vec<Complex64> = default_client
        .complex_mul_real(&made, &real)
        .unwrap()
        .to_vec();
    let cfg_mul: Vec<Complex64> = configured_client
        .complex_mul_real(&made, &real)
        .unwrap()
        .to_vec();
    assert_complex_close(
        &base_mul,
        &cfg_mul,
        1e-6,
        "cpu complex_mul_real parallelism config",
    );

    let base_div: Vec<Complex64> = default_client
        .complex_div_real(&made, &real)
        .unwrap()
        .to_vec();
    let cfg_div: Vec<Complex64> = configured_client
        .complex_div_real(&made, &real)
        .unwrap()
        .to_vec();
    assert_complex_close(
        &base_div,
        &cfg_div,
        1e-6,
        "cpu complex_div_real parallelism config",
    );
}

#[test]
fn test_complex_angle_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let complex_data = vec![
        Complex64::new(1.0, 1.0),
        Complex64::new(-1.0, 1.0),
        Complex64::new(0.0, -1.0),
    ];
    let real_data = vec![1.0f32, -2.0, 3.0, -5.0, 0.0];

    let cpu_complex =
        Tensor::<CpuRuntime>::try_from_slice(&complex_data, &[3], &cpu_device).unwrap();
    let cpu_real = Tensor::<CpuRuntime>::try_from_slice(&real_data, &[5], &cpu_device).unwrap();
    let cpu_angle_complex: Vec<f32> = cpu_client.angle(&cpu_complex).unwrap().to_vec();
    let cpu_angle_real: Vec<f32> = cpu_client.angle(&cpu_real).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_complex = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &complex_data,
            &[3],
            &cuda_device,
        )
        .unwrap();
        let cuda_real = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &real_data,
            &[5],
            &cuda_device,
        )
        .unwrap();
        let cuda_angle_complex: Vec<f32> = cuda_client.angle(&cuda_complex).unwrap().to_vec();
        let cuda_angle_real: Vec<f32> = cuda_client.angle(&cuda_real).unwrap().to_vec();

        for (c, g) in cpu_angle_complex.iter().zip(cuda_angle_complex.iter()) {
            assert!((c - g).abs() < 1e-6, "CPU {} CUDA {}", c, g);
        }
        for (c, g) in cpu_angle_real.iter().zip(cuda_angle_real.iter()) {
            assert!((c - g).abs() < 1e-6, "CPU {} CUDA {}", c, g);
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_complex = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &complex_data,
            &[3],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_real = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &real_data,
            &[5],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_angle_complex: Vec<f32> = wgpu_client.angle(&wgpu_complex).unwrap().to_vec();
        let wgpu_angle_real: Vec<f32> = wgpu_client.angle(&wgpu_real).unwrap().to_vec();

        for (c, g) in cpu_angle_complex.iter().zip(wgpu_angle_complex.iter()) {
            assert!((c - g).abs() < 1e-4, "CPU {} WGPU {}", c, g);
        }
        for (c, g) in cpu_angle_real.iter().zip(wgpu_angle_real.iter()) {
            assert!((c - g).abs() < 1e-4, "CPU {} WGPU {}", c, g);
        }
    });
}

#[test]
fn test_complex_make_mul_div_real_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let real_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
    let imag_data = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];

    let complex_data = vec![
        Complex64::new(4.0, 6.0),
        Complex64::new(3.0, 9.0),
        Complex64::new(-10.0, 5.0),
    ];
    let mul_real = vec![2.0f32, 0.5, -1.0];
    let div_real = vec![2.0f32, 3.0, 5.0];

    let cpu_real = Tensor::<CpuRuntime>::try_from_slice(&real_data, &[5], &cpu_device).unwrap();
    let cpu_imag = Tensor::<CpuRuntime>::try_from_slice(&imag_data, &[5], &cpu_device).unwrap();
    let cpu_make: Vec<Complex64> = cpu_client
        .make_complex(&cpu_real, &cpu_imag)
        .unwrap()
        .to_vec();

    let cpu_complex =
        Tensor::<CpuRuntime>::try_from_slice(&complex_data, &[3], &cpu_device).unwrap();
    let cpu_mul_r = Tensor::<CpuRuntime>::try_from_slice(&mul_real, &[3], &cpu_device).unwrap();
    let cpu_div_r = Tensor::<CpuRuntime>::try_from_slice(&div_real, &[3], &cpu_device).unwrap();
    let cpu_mul: Vec<Complex64> = cpu_client
        .complex_mul_real(&cpu_complex, &cpu_mul_r)
        .unwrap()
        .to_vec();
    let cpu_div: Vec<Complex64> = cpu_client
        .complex_div_real(&cpu_complex, &cpu_div_r)
        .unwrap()
        .to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_real = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &real_data,
            &[5],
            &cuda_device,
        )
        .unwrap();
        let cuda_imag = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &imag_data,
            &[5],
            &cuda_device,
        )
        .unwrap();
        let cuda_make: Vec<Complex64> = cuda_client
            .make_complex(&cuda_real, &cuda_imag)
            .unwrap()
            .to_vec();
        for (c, g) in cpu_make.iter().zip(cuda_make.iter()) {
            assert!((c.re - g.re).abs() < 1e-6 && (c.im - g.im).abs() < 1e-6);
        }

        let cuda_complex = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &complex_data,
            &[3],
            &cuda_device,
        )
        .unwrap();
        let cuda_mul_r = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &mul_real,
            &[3],
            &cuda_device,
        )
        .unwrap();
        let cuda_div_r = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &div_real,
            &[3],
            &cuda_device,
        )
        .unwrap();
        let cuda_mul: Vec<Complex64> = cuda_client
            .complex_mul_real(&cuda_complex, &cuda_mul_r)
            .unwrap()
            .to_vec();
        let cuda_div: Vec<Complex64> = cuda_client
            .complex_div_real(&cuda_complex, &cuda_div_r)
            .unwrap()
            .to_vec();
        for (c, g) in cpu_mul.iter().zip(cuda_mul.iter()) {
            assert!((c.re - g.re).abs() < 1e-6 && (c.im - g.im).abs() < 1e-6);
        }
        for (c, g) in cpu_div.iter().zip(cuda_div.iter()) {
            assert!((c.re - g.re).abs() < 1e-6 && (c.im - g.im).abs() < 1e-6);
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_real = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &real_data,
            &[5],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_imag = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &imag_data,
            &[5],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_make: Vec<Complex64> = wgpu_client
            .make_complex(&wgpu_real, &wgpu_imag)
            .unwrap()
            .to_vec();
        for (c, g) in cpu_make.iter().zip(wgpu_make.iter()) {
            assert!((c.re - g.re).abs() < 1e-4 && (c.im - g.im).abs() < 1e-4);
        }

        let wgpu_complex = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &complex_data,
            &[3],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_mul_r = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &mul_real,
            &[3],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_div_r = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &div_real,
            &[3],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_mul: Vec<Complex64> = wgpu_client
            .complex_mul_real(&wgpu_complex, &wgpu_mul_r)
            .unwrap()
            .to_vec();
        let wgpu_div: Vec<Complex64> = wgpu_client
            .complex_div_real(&wgpu_complex, &wgpu_div_r)
            .unwrap()
            .to_vec();
        for (c, g) in cpu_mul.iter().zip(wgpu_mul.iter()) {
            assert!((c.re - g.re).abs() < 1e-4 && (c.im - g.im).abs() < 1e-4);
        }
        for (c, g) in cpu_div.iter().zip(wgpu_div.iter()) {
            assert!((c.re - g.re).abs() < 1e-4 && (c.im - g.im).abs() < 1e-4);
        }
    });
}

#[test]
fn test_complex64_binary_ops_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let a_data = vec![Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)];
    let b_data = vec![Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)];
    let b_div_data = vec![Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)];

    let cpu_a = Tensor::<CpuRuntime>::try_from_slice(&a_data, &[2], &cpu_device).unwrap();
    let cpu_b = Tensor::<CpuRuntime>::try_from_slice(&b_data, &[2], &cpu_device).unwrap();
    let cpu_b_div = Tensor::<CpuRuntime>::try_from_slice(&b_div_data, &[2], &cpu_device).unwrap();

    let cpu_add: Vec<Complex64> = cpu_client.add(&cpu_a, &cpu_b).unwrap().to_vec();
    let cpu_mul: Vec<Complex64> = cpu_client.mul(&cpu_a, &cpu_b).unwrap().to_vec();
    let cpu_div: Vec<Complex64> = cpu_client.div(&cpu_a, &cpu_b_div).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_a =
            Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(&a_data, &[2], &cuda_device)
                .unwrap();
        let cuda_b =
            Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(&b_data, &[2], &cuda_device)
                .unwrap();
        let cuda_b_div = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &b_div_data,
            &[2],
            &cuda_device,
        )
        .unwrap();

        let cuda_add: Vec<Complex64> = cuda_client.add(&cuda_a, &cuda_b).unwrap().to_vec();
        let cuda_mul: Vec<Complex64> = cuda_client.mul(&cuda_a, &cuda_b).unwrap().to_vec();
        let cuda_div: Vec<Complex64> = cuda_client.div(&cuda_a, &cuda_b_div).unwrap().to_vec();

        assert_complex_close(&cpu_add, &cuda_add, 1e-6, "complex_add_cuda");
        assert_complex_close(&cpu_mul, &cuda_mul, 1e-6, "complex_mul_cuda");
        assert_complex_close(&cpu_div, &cuda_div, 1e-6, "complex_div_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_a =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(&a_data, &[2], &wgpu_device)
                .unwrap();
        let wgpu_b =
            Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(&b_data, &[2], &wgpu_device)
                .unwrap();
        let wgpu_b_div = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &b_div_data,
            &[2],
            &wgpu_device,
        )
        .unwrap();

        let wgpu_add: Option<Vec<Complex64>> = match wgpu_client.add(&wgpu_a, &wgpu_b) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                // Intentional exception: backend does not implement Complex64 path yet.
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU add error: {e}"),
        };
        let wgpu_mul: Option<Vec<Complex64>> = match wgpu_client.mul(&wgpu_a, &wgpu_b) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU mul error: {e}"),
        };
        let wgpu_div: Option<Vec<Complex64>> = match wgpu_client.div(&wgpu_a, &wgpu_b_div) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU div error: {e}"),
        };

        if let Some(wgpu_add) = &wgpu_add {
            assert_complex_close(&cpu_add, wgpu_add, 1e-4, "complex_add_wgpu");
        }
        if let Some(wgpu_mul) = &wgpu_mul {
            assert_complex_close(&cpu_mul, wgpu_mul, 1e-4, "complex_mul_wgpu");
        }
        if let Some(wgpu_div) = &wgpu_div {
            assert_complex_close(&cpu_div, wgpu_div, 1e-4, "complex_div_wgpu");
        }
    });
}

#[test]
fn test_complex64_unary_ops_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);

    let neg_data = vec![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0)];
    let exp_data = vec![
        Complex64::new(0.0, std::f32::consts::PI),
        Complex64::new(1.0, 0.0),
    ];
    let sqrt_data = vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 2.0)];

    let cpu_neg_in = Tensor::<CpuRuntime>::try_from_slice(&neg_data, &[2], &cpu_device).unwrap();
    let cpu_exp_in = Tensor::<CpuRuntime>::try_from_slice(&exp_data, &[2], &cpu_device).unwrap();
    let cpu_sqrt_in = Tensor::<CpuRuntime>::try_from_slice(&sqrt_data, &[2], &cpu_device).unwrap();

    let cpu_neg: Vec<Complex64> = cpu_client.neg(&cpu_neg_in).unwrap().to_vec();
    let cpu_exp: Vec<Complex64> = cpu_client.exp(&cpu_exp_in).unwrap().to_vec();
    let cpu_sqrt: Vec<Complex64> = cpu_client.sqrt(&cpu_sqrt_in).unwrap().to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let cuda_neg_in = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &neg_data,
            &[2],
            &cuda_device,
        )
        .unwrap();
        let cuda_exp_in = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &exp_data,
            &[2],
            &cuda_device,
        )
        .unwrap();
        let cuda_sqrt_in = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &sqrt_data,
            &[2],
            &cuda_device,
        )
        .unwrap();

        let cuda_neg: Vec<Complex64> = cuda_client.neg(&cuda_neg_in).unwrap().to_vec();
        let cuda_exp: Vec<Complex64> = cuda_client.exp(&cuda_exp_in).unwrap().to_vec();
        let cuda_sqrt: Vec<Complex64> = cuda_client.sqrt(&cuda_sqrt_in).unwrap().to_vec();

        assert_complex_close(&cpu_neg, &cuda_neg, 1e-6, "complex_neg_cuda");
        assert_complex_close(&cpu_exp, &cuda_exp, 1e-5, "complex_exp_cuda");
        assert_complex_close(&cpu_sqrt, &cuda_sqrt, 1e-5, "complex_sqrt_cuda");
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let wgpu_neg_in = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &neg_data,
            &[2],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_exp_in = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &exp_data,
            &[2],
            &wgpu_device,
        )
        .unwrap();
        let wgpu_sqrt_in = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &sqrt_data,
            &[2],
            &wgpu_device,
        )
        .unwrap();

        let wgpu_neg: Option<Vec<Complex64>> = match wgpu_client.neg(&wgpu_neg_in) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                // Intentional exception: backend does not implement Complex64 path yet.
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU neg error: {e}"),
        };
        let wgpu_exp: Option<Vec<Complex64>> = match wgpu_client.exp(&wgpu_exp_in) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU exp error: {e}"),
        };
        let wgpu_sqrt: Option<Vec<Complex64>> = match wgpu_client.sqrt(&wgpu_sqrt_in) {
            Ok(t) => Some(t.to_vec()),
            Err(Error::UnsupportedDType { dtype, op }) => {
                assert_eq!(
                    dtype,
                    DType::Complex64,
                    "unexpected unsupported dtype for op `{op}`"
                );
                None
            }
            Err(e) => panic!("unexpected WGPU sqrt error: {e}"),
        };

        if let Some(wgpu_neg) = &wgpu_neg {
            assert_complex_close(&cpu_neg, wgpu_neg, 1e-4, "complex_neg_wgpu");
        }
        if let Some(wgpu_exp) = &wgpu_exp {
            assert_complex_close(&cpu_exp, wgpu_exp, 1e-4, "complex_exp_wgpu");
        }
        if let Some(wgpu_sqrt) = &wgpu_sqrt {
            assert_complex_close(&cpu_sqrt, wgpu_sqrt, 1e-4, "complex_sqrt_wgpu");
        }
    });
}
