//! CPU implementation of tensor decomposition algorithms.
//!
//! This module provides thin wrappers that delegate to the generic implementations
//! in `algorithm/linalg/tensor_decompose_core.rs` to ensure numerical parity.

use super::super::{CpuClient, CpuRuntime};
use crate::algorithm::linalg::tensor_decompose_core::{
    TensorDecomposeDTypeSupport, cp_decompose_impl, cp_reconstruct_impl, fold_impl, hosvd_impl,
    mode_n_product_impl, tensor_train_impl, tt_reconstruct_impl, tucker_impl,
    tucker_reconstruct_impl, unfold_impl,
};
use crate::algorithm::linalg::{
    CpDecomposition, CpOptions, TensorDecomposeAlgorithms, TensorTrainDecomposition,
    TuckerDecomposition, TuckerOptions,
};
use crate::error::Result;
use crate::tensor::Tensor;

impl TensorDecomposeAlgorithms<CpuRuntime> for CpuClient {
    fn unfold(&self, tensor: &Tensor<CpuRuntime>, mode: usize) -> Result<Tensor<CpuRuntime>> {
        unfold_impl(tensor, mode, TensorDecomposeDTypeSupport::FULL)
    }

    fn fold(
        &self,
        matrix: &Tensor<CpuRuntime>,
        mode: usize,
        shape: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        fold_impl(matrix, mode, shape)
    }

    fn mode_n_product(
        &self,
        tensor: &Tensor<CpuRuntime>,
        matrix: &Tensor<CpuRuntime>,
        mode: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        mode_n_product_impl(
            self,
            tensor,
            matrix,
            mode,
            TensorDecomposeDTypeSupport::FULL,
        )
    }

    fn hosvd(
        &self,
        tensor: &Tensor<CpuRuntime>,
        ranks: &[usize],
    ) -> Result<TuckerDecomposition<CpuRuntime>> {
        hosvd_impl(self, tensor, ranks, TensorDecomposeDTypeSupport::FULL)
    }

    fn tucker(
        &self,
        tensor: &Tensor<CpuRuntime>,
        ranks: &[usize],
        options: TuckerOptions,
    ) -> Result<TuckerDecomposition<CpuRuntime>> {
        tucker_impl(
            self,
            tensor,
            ranks,
            options,
            TensorDecomposeDTypeSupport::FULL,
        )
    }

    fn cp_decompose(
        &self,
        tensor: &Tensor<CpuRuntime>,
        rank: usize,
        options: CpOptions,
    ) -> Result<CpDecomposition<CpuRuntime>> {
        cp_decompose_impl(
            self,
            tensor,
            rank,
            options,
            TensorDecomposeDTypeSupport::FULL,
        )
    }

    fn tensor_train(
        &self,
        tensor: &Tensor<CpuRuntime>,
        max_rank: usize,
        tolerance: f64,
    ) -> Result<TensorTrainDecomposition<CpuRuntime>> {
        tensor_train_impl(
            self,
            tensor,
            max_rank,
            tolerance,
            TensorDecomposeDTypeSupport::FULL,
        )
    }

    fn tucker_reconstruct(
        &self,
        decomp: &TuckerDecomposition<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        tucker_reconstruct_impl(self, decomp, TensorDecomposeDTypeSupport::FULL)
    }

    fn cp_reconstruct(
        &self,
        decomp: &CpDecomposition<CpuRuntime>,
        shape: &[usize],
    ) -> Result<Tensor<CpuRuntime>> {
        cp_reconstruct_impl(self, decomp, shape, TensorDecomposeDTypeSupport::FULL)
    }

    fn tt_reconstruct(
        &self,
        decomp: &TensorTrainDecomposition<CpuRuntime>,
    ) -> Result<Tensor<CpuRuntime>> {
        tt_reconstruct_impl(self, decomp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    fn get_client() -> CpuClient {
        let device = CpuRuntime::default_device();
        CpuRuntime::default_client(&device)
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device).unwrap();

        // Test all modes
        for mode in 0..3 {
            let unfolded = client.unfold(&tensor, mode).expect("unfold should succeed");
            let refolded = client
                .fold(&unfolded, mode, &[2, 3, 4])
                .expect("fold should succeed");

            let orig: Vec<f32> = tensor.to_vec();
            let result: Vec<f32> = refolded.to_vec();
            assert_eq!(orig.len(), result.len());
            for (a, b) in orig.iter().zip(result.iter()) {
                assert!((a - b).abs() < 1e-6, "Mode {} roundtrip failed", mode);
            }
        }
    }

    #[test]
    fn test_mode_n_product() {
        let client = get_client();
        // Tensor [2, 3]
        let tensor = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &client.device,
        )
        .unwrap();
        // Matrix [4, 2] for mode-0 product
        let matrix = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            &[4, 2],
            &client.device,
        )
        .unwrap();

        let result = client
            .mode_n_product(&tensor, &matrix, 0)
            .expect("mode_n_product should succeed");
        assert_eq!(result.shape(), &[4, 3]);
    }

    #[test]
    fn test_hosvd_basic() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device).unwrap();

        let decomp = client
            .hosvd(&tensor, &[2, 3, 4])
            .expect("hosvd should succeed");

        // Full rank should give exact reconstruction
        assert_eq!(decomp.core.shape(), &[2, 3, 4]);
        assert_eq!(decomp.factors.len(), 3);
        assert_eq!(decomp.factors[0].shape(), &[2, 2]);
        assert_eq!(decomp.factors[1].shape(), &[3, 3]);
        assert_eq!(decomp.factors[2].shape(), &[4, 4]);

        // Reconstruction should be close to original
        let reconstructed = client
            .tucker_reconstruct(&decomp)
            .expect("reconstruction should succeed");
        let orig: Vec<f32> = tensor.to_vec();
        let recon: Vec<f32> = reconstructed.to_vec();
        for (a, b) in orig.iter().zip(recon.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "HOSVD reconstruction error: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_hosvd_truncated() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device).unwrap();

        // Truncated decomposition
        let decomp = client
            .hosvd(&tensor, &[1, 2, 2])
            .expect("hosvd should succeed");

        assert_eq!(decomp.core.shape(), &[1, 2, 2]);
        assert_eq!(decomp.factors[0].shape(), &[2, 1]);
        assert_eq!(decomp.factors[1].shape(), &[3, 2]);
        assert_eq!(decomp.factors[2].shape(), &[4, 2]);
    }

    #[test]
    fn test_cp_decompose_basic() {
        let client = get_client();
        // Simple rank-1 tensor: outer product of [1, 2] and [1, 1, 1]
        let tensor = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0],
            &[2, 3],
            &client.device,
        )
        .unwrap();

        let decomp = client
            .cp_decompose(&tensor, 1, CpOptions::default())
            .expect("cp_decompose should succeed");

        assert_eq!(decomp.factors.len(), 2);
        assert_eq!(decomp.factors[0].shape(), &[2, 1]);
        assert_eq!(decomp.factors[1].shape(), &[3, 1]);
        assert_eq!(decomp.weights.shape(), &[1]);

        // Reconstruction should be close to original
        let reconstructed = client
            .cp_reconstruct(&decomp, &[2, 3])
            .expect("reconstruction should succeed");
        let orig: Vec<f32> = tensor.to_vec();
        let recon: Vec<f32> = reconstructed.to_vec();
        for (a, b) in orig.iter().zip(recon.iter()) {
            assert!(
                (a - b).abs() < 1e-2,
                "CP reconstruction error: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_tensor_train_basic() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device).unwrap();

        // Full-rank TT decomposition
        let decomp = client
            .tensor_train(&tensor, 0, 0.0)
            .expect("tensor_train should succeed");

        assert_eq!(decomp.cores.len(), 3);
        // First core: [1, I_1, R_1]
        assert_eq!(decomp.cores[0].shape()[0], 1);
        assert_eq!(decomp.cores[0].shape()[1], 2);
        // Last core: [R_{n-1}, I_n, 1]
        assert_eq!(decomp.cores[2].shape()[1], 4);
        assert_eq!(decomp.cores[2].shape()[2], 1);

        // Reconstruction should be close to original
        let reconstructed = client
            .tt_reconstruct(&decomp)
            .expect("reconstruction should succeed");
        let orig: Vec<f32> = tensor.to_vec();
        let recon: Vec<f32> = reconstructed.to_vec();
        for (a, b) in orig.iter().zip(recon.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "TT reconstruction error: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_tensor_train_truncated() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device).unwrap();

        // Truncated TT decomposition with max_rank = 2
        let decomp = client
            .tensor_train(&tensor, 2, 0.0)
            .expect("tensor_train should succeed");

        // Ranks should be at most 2
        for &r in &decomp.ranks {
            assert!(r <= 2, "Rank {} exceeds max_rank 2", r);
        }
    }
}
