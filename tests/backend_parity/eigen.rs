// Backend parity tests migrated from tests/eigendecomposition_ops.rs

#[cfg(feature = "cuda")]
use crate::backend_parity::helpers::with_cuda_backend;
#[cfg(feature = "wgpu")]
use crate::backend_parity::helpers::with_wgpu_backend;
use numr::algorithm::linalg::LinearAlgebraAlgorithms;
use numr::runtime::Runtime;
use numr::runtime::cpu::{CpuDevice, CpuRuntime};
use numr::tensor::Tensor;

#[test]
fn test_eig_decompose_symmetric_parity() {
    let cpu_device = CpuDevice::new();
    let cpu_client = CpuRuntime::default_client(&cpu_device);
    let data = vec![3.0f32, 1.0, 1.0, 3.0];

    let a_cpu = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 2], &cpu_device).unwrap();
    let eig_cpu = cpu_client.eig_decompose_symmetric(&a_cpu).unwrap();
    let cpu_eigenvalues: Vec<f32> = eig_cpu.eigenvalues.to_vec();

    #[cfg(feature = "cuda")]
    with_cuda_backend(|cuda_client, cuda_device| {
        let a_cuda = Tensor::<numr::runtime::cuda::CudaRuntime>::try_from_slice(
            &data,
            &[2, 2],
            &cuda_device,
        )
        .unwrap();
        let eig_cuda = cuda_client.eig_decompose_symmetric(&a_cuda).unwrap();
        let cuda_eigenvalues: Vec<f32> = eig_cuda.eigenvalues.to_vec();
        for (i, (c, g)) in cpu_eigenvalues
            .iter()
            .zip(cuda_eigenvalues.iter())
            .enumerate()
        {
            let diff = (c - g).abs();
            assert!(
                diff < 1e-5,
                "CPU/CUDA eigenvalue {} mismatch: {} vs {}",
                i,
                c,
                g
            );
        }
    });

    #[cfg(feature = "wgpu")]
    with_wgpu_backend(|wgpu_client, wgpu_device| {
        let a_wgpu = Tensor::<numr::runtime::wgpu::WgpuRuntime>::try_from_slice(
            &data,
            &[2, 2],
            &wgpu_device,
        )
        .unwrap();
        let eig_wgpu = wgpu_client.eig_decompose_symmetric(&a_wgpu).unwrap();
        let wgpu_eigenvalues: Vec<f32> = eig_wgpu.eigenvalues.to_vec();
        for (i, (c, w)) in cpu_eigenvalues
            .iter()
            .zip(wgpu_eigenvalues.iter())
            .enumerate()
        {
            let diff = (c - w).abs();
            assert!(
                diff < 1e-5,
                "CPU/WGPU eigenvalue {} mismatch: {} vs {}",
                i,
                c,
                w
            );
        }
    });
}
