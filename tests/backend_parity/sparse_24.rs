//! Backend parity tests for 2:4 structured sparsity operations.
//!
//! Tests verify that CPU, CUDA, and WebGPU backends produce identical results
//! for prune, decompress, and sparse matmul operations.

use crate::backend_parity::helpers::assert_parity_f32;
use crate::common::create_cpu_client;
use numr::runtime::cpu::CpuRuntime;
use numr::sparse::Sparse24Ops;
use numr::tensor::Tensor;

// ============================================================================
// CPU-only correctness tests
// ============================================================================

#[test]
fn test_prune_to_24_correctness() {
    let (client, device) = create_cpu_client();

    // Matrix: 2x8, each row has 2 groups of 4
    let data: Vec<f32> = vec![
        1.0, -3.0, 2.0, 0.5, // group 0: top-2 = -3.0 (1), 2.0 (2)
        0.1, 0.2, 0.3, 0.4, // group 1: top-2 = 0.3 (2), 0.4 (3)
        4.0, 1.0, -5.0, 3.0, // group 2: top-2 = 4.0 (0), -5.0 (2)
        0.0, 0.0, 0.0, 0.0, // group 3: all zero, keeps (0), (1)
    ];
    let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 8], &device).unwrap();
    let sparse = client.prune_to_24(&dense).unwrap();

    assert_eq!(sparse.shape(), [2, 8]);
    assert_eq!(sparse.nnz(), 2 * 4); // 2 rows * 4 non-zeros per row
    assert!(sparse.is_valid());

    // Verify compressed values
    let vals: Vec<f32> = sparse.compressed_values().to_vec();
    // Row 0, group 0: -3.0 (idx 1), 2.0 (idx 2) → sorted by index
    assert_eq!(vals[0], -3.0);
    assert_eq!(vals[1], 2.0);
    // Row 0, group 1: 0.3 (idx 2), 0.4 (idx 3) → sorted by index
    assert_eq!(vals[2], 0.3);
    assert_eq!(vals[3], 0.4);
}

#[test]
fn test_sparse_24_roundtrip() {
    let (client, device) = create_cpu_client();

    let data: Vec<f32> = vec![
        1.0, -3.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 4.0, 1.0, -5.0, 3.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[2, 8], &device).unwrap();
    let sparse = client.prune_to_24(&dense).unwrap();
    let reconstructed = client.sparse_24_to_dense(&sparse).unwrap();

    let recon_data: Vec<f32> = reconstructed.to_vec();

    // After pruning and reconstruction, only top-2 per group survive
    // Row 0, group 0: kept idx 1,2 → [0, -3, 2, 0]
    assert_eq!(recon_data[0], 0.0);
    assert_eq!(recon_data[1], -3.0);
    assert_eq!(recon_data[2], 2.0);
    assert_eq!(recon_data[3], 0.0);

    // Row 0, group 1: kept idx 2,3 → [0, 0, 0.3, 0.4]
    assert_eq!(recon_data[4], 0.0);
    assert_eq!(recon_data[5], 0.0);
    assert_eq!(recon_data[6], 0.3);
    assert_eq!(recon_data[7], 0.4);
}

#[test]
fn test_sparse_24_matmul_matches_dense() {
    use numr::prelude::MatmulOps;

    let (client, device) = create_cpu_client();

    // Weight: [4, 8], Input: [2, 8]
    let weight_data: Vec<f32> = vec![
        1.0, -3.0, 2.0, 0.5, 0.1, 0.2, 0.3, 0.4, 4.0, 1.0, -5.0, 3.0, 0.5, 0.5, 0.5, 0.5, 2.0, 0.0,
        1.0, 0.0, 0.0, 3.0, 0.0, 1.0, 0.5, 1.5, 0.5, 1.5, 2.0, 0.0, 2.0, 0.0,
    ];
    let weight = Tensor::<CpuRuntime>::try_from_slice(&weight_data, &[4, 8], &device).unwrap();

    let input_data: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0,
    ];
    let input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &[2, 8], &device).unwrap();

    // Prune weight
    let sparse_weight = client.prune_to_24(&weight).unwrap();

    // Sparse matmul
    let sparse_result = client.sparse_24_matmul(&input, &sparse_weight).unwrap();

    // Dense matmul with pruned weight
    let dense_pruned = client.sparse_24_to_dense(&sparse_weight).unwrap();
    let dense_pruned_t = dense_pruned.t().unwrap();
    let dense_result = client.matmul(&input, &dense_pruned_t).unwrap();

    let sparse_out: Vec<f32> = sparse_result.to_vec();
    let dense_out: Vec<f32> = dense_result.to_vec();

    assert_parity_f32(&sparse_out, &dense_out, "sparse_24_matmul vs dense");
}

#[test]
fn test_sparse_24_matmul_larger() {
    use numr::prelude::MatmulOps;

    let (client, device) = create_cpu_client();

    // Larger: weight [16, 32], input [8, 32]
    let weight_data: Vec<f32> = (0..16 * 32).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
    let weight = Tensor::<CpuRuntime>::try_from_slice(&weight_data, &[16, 32], &device).unwrap();

    let input_data: Vec<f32> = (0..8 * 32).map(|i| (i as f32 * 0.07).cos() * 2.0).collect();
    let input = Tensor::<CpuRuntime>::try_from_slice(&input_data, &[8, 32], &device).unwrap();

    let sparse_weight = client.prune_to_24(&weight).unwrap();
    let sparse_result = client.sparse_24_matmul(&input, &sparse_weight).unwrap();

    let dense_pruned = client.sparse_24_to_dense(&sparse_weight).unwrap();
    let dense_pruned_t = dense_pruned.t().unwrap();
    let dense_result = client.matmul(&input, &dense_pruned_t).unwrap();

    let sparse_out: Vec<f32> = sparse_result.to_vec();
    let dense_out: Vec<f32> = dense_result.to_vec();

    assert_parity_f32(&sparse_out, &dense_out, "sparse_24_matmul_larger");
}

// ============================================================================
// CUDA backend parity tests
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_parity {
    use super::*;
    use crate::backend_parity::helpers::{assert_parity_f32, with_cuda_backend};
    use numr::runtime::cuda::CudaRuntime;
    use numr::sparse::Sparse24Ops;

    #[test]
    fn test_prune_to_24_parity_cuda() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let cpu_dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_dense).unwrap();
        let cpu_vals: Vec<f32> = cpu_sparse.compressed_values().to_vec();
        let cpu_meta: Vec<u32> = cpu_sparse.metadata().to_vec();

        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_dense =
                Tensor::<CudaRuntime>::try_from_slice(&data, &[4, 16], &cuda_device).unwrap();
            let cuda_sparse = cuda_client.prune_to_24(&cuda_dense).unwrap();
            let cuda_vals: Vec<f32> = cuda_sparse.compressed_values().to_vec();
            let cuda_meta: Vec<u32> = cuda_sparse.metadata().to_vec();

            assert_parity_f32(&cuda_vals, &cpu_vals, "prune_to_24 values CUDA vs CPU");
            assert_eq!(cuda_meta, cpu_meta, "prune_to_24 metadata CUDA vs CPU");
        });
    }

    #[test]
    fn test_sparse_24_roundtrip_parity_cuda() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let cpu_dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_dense).unwrap();
        let cpu_recon: Vec<f32> = cpu_client.sparse_24_to_dense(&cpu_sparse).unwrap().to_vec();

        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_dense =
                Tensor::<CudaRuntime>::try_from_slice(&data, &[4, 16], &cuda_device).unwrap();
            let cuda_sparse = cuda_client.prune_to_24(&cuda_dense).unwrap();
            let cuda_recon: Vec<f32> = cuda_client
                .sparse_24_to_dense(&cuda_sparse)
                .unwrap()
                .to_vec();

            assert_parity_f32(&cuda_recon, &cpu_recon, "roundtrip CUDA vs CPU");
        });
    }

    #[test]
    fn test_sparse_24_matmul_parity_cuda() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let weight_data: Vec<f32> = (0..8 * 16).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let input_data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.07).cos() * 2.0).collect();

        let cpu_weight =
            Tensor::<CpuRuntime>::try_from_slice(&weight_data, &[8, 16], &cpu_device).unwrap();
        let cpu_input =
            Tensor::<CpuRuntime>::try_from_slice(&input_data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_weight).unwrap();
        let cpu_result: Vec<f32> = cpu_client
            .sparse_24_matmul(&cpu_input, &cpu_sparse)
            .unwrap()
            .to_vec();

        with_cuda_backend(|cuda_client, cuda_device| {
            let cuda_weight =
                Tensor::<CudaRuntime>::try_from_slice(&weight_data, &[8, 16], &cuda_device)
                    .unwrap();
            let cuda_input =
                Tensor::<CudaRuntime>::try_from_slice(&input_data, &[4, 16], &cuda_device).unwrap();
            let cuda_sparse = cuda_client.prune_to_24(&cuda_weight).unwrap();
            let cuda_result: Vec<f32> = cuda_client
                .sparse_24_matmul(&cuda_input, &cuda_sparse)
                .unwrap()
                .to_vec();

            assert_parity_f32(&cuda_result, &cpu_result, "sparse_24_matmul CUDA vs CPU");
        });
    }
}

// ============================================================================
// WebGPU backend parity tests
// ============================================================================

#[cfg(feature = "wgpu")]
mod wgpu_parity {
    use super::*;
    use crate::backend_parity::helpers::{assert_parity_f32, with_wgpu_backend};
    use numr::runtime::wgpu::WgpuRuntime;
    use numr::sparse::Sparse24Ops;

    #[test]
    fn test_prune_to_24_parity_wgpu() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let cpu_dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_dense).unwrap();
        let cpu_vals: Vec<f32> = cpu_sparse.compressed_values().to_vec();
        let cpu_meta: Vec<u32> = cpu_sparse.metadata().to_vec();

        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_dense =
                Tensor::<WgpuRuntime>::try_from_slice(&data, &[4, 16], &wgpu_device).unwrap();
            let wgpu_sparse = wgpu_client.prune_to_24(&wgpu_dense).unwrap();
            let wgpu_vals: Vec<f32> = wgpu_sparse.compressed_values().to_vec();
            let wgpu_meta: Vec<u32> = wgpu_sparse.metadata().to_vec();

            assert_parity_f32(&wgpu_vals, &cpu_vals, "prune_to_24 values WGPU vs CPU");
            assert_eq!(wgpu_meta, cpu_meta, "prune_to_24 metadata WGPU vs CPU");
        });
    }

    #[test]
    fn test_sparse_24_roundtrip_parity_wgpu() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.13).sin() * 5.0).collect();
        let cpu_dense = Tensor::<CpuRuntime>::try_from_slice(&data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_dense).unwrap();
        let cpu_recon: Vec<f32> = cpu_client.sparse_24_to_dense(&cpu_sparse).unwrap().to_vec();

        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_dense =
                Tensor::<WgpuRuntime>::try_from_slice(&data, &[4, 16], &wgpu_device).unwrap();
            let wgpu_sparse = wgpu_client.prune_to_24(&wgpu_dense).unwrap();
            let wgpu_recon: Vec<f32> = wgpu_client
                .sparse_24_to_dense(&wgpu_sparse)
                .unwrap()
                .to_vec();

            assert_parity_f32(&wgpu_recon, &cpu_recon, "roundtrip WGPU vs CPU");
        });
    }

    #[test]
    fn test_sparse_24_matmul_parity_wgpu() {
        let (cpu_client, cpu_device) = create_cpu_client();

        let weight_data: Vec<f32> = (0..8 * 16).map(|i| (i as f32 * 0.1).sin() * 3.0).collect();
        let input_data: Vec<f32> = (0..4 * 16).map(|i| (i as f32 * 0.07).cos() * 2.0).collect();

        let cpu_weight =
            Tensor::<CpuRuntime>::try_from_slice(&weight_data, &[8, 16], &cpu_device).unwrap();
        let cpu_input =
            Tensor::<CpuRuntime>::try_from_slice(&input_data, &[4, 16], &cpu_device).unwrap();
        let cpu_sparse = cpu_client.prune_to_24(&cpu_weight).unwrap();
        let cpu_result: Vec<f32> = cpu_client
            .sparse_24_matmul(&cpu_input, &cpu_sparse)
            .unwrap()
            .to_vec();

        with_wgpu_backend(|wgpu_client, wgpu_device| {
            let wgpu_weight =
                Tensor::<WgpuRuntime>::try_from_slice(&weight_data, &[8, 16], &wgpu_device)
                    .unwrap();
            let wgpu_input =
                Tensor::<WgpuRuntime>::try_from_slice(&input_data, &[4, 16], &wgpu_device).unwrap();
            let wgpu_sparse = wgpu_client.prune_to_24(&wgpu_weight).unwrap();
            let wgpu_result: Vec<f32> = wgpu_client
                .sparse_24_matmul(&wgpu_input, &wgpu_sparse)
                .unwrap()
                .to_vec();

            assert_parity_f32(&wgpu_result, &cpu_result, "sparse_24_matmul WGPU vs CPU");
        });
    }
}
