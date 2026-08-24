//! WebGPU implementation of tensor decomposition algorithms.
//!
//! This module provides thin wrappers that delegate to the generic implementations
//! in `algorithm/linalg/tensor_decompose_core.rs` to ensure numerical parity.
//!
//! # Note
//!
//! WebGPU only supports F32 (WGSL doesn't support F64).

use super::super::{WgpuClient, WgpuRuntime};
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

impl TensorDecomposeAlgorithms<WgpuRuntime> for WgpuClient {
    fn unfold(&self, tensor: &Tensor<WgpuRuntime>, mode: usize) -> Result<Tensor<WgpuRuntime>> {
        unfold_impl(tensor, mode, TensorDecomposeDTypeSupport::F32_ONLY)
    }

    fn fold(
        &self,
        matrix: &Tensor<WgpuRuntime>,
        mode: usize,
        shape: &[usize],
    ) -> Result<Tensor<WgpuRuntime>> {
        fold_impl(matrix, mode, shape)
    }

    fn mode_n_product(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        matrix: &Tensor<WgpuRuntime>,
        mode: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        mode_n_product_impl(
            self,
            tensor,
            matrix,
            mode,
            TensorDecomposeDTypeSupport::F32_ONLY,
        )
    }

    fn hosvd(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        ranks: &[usize],
    ) -> Result<TuckerDecomposition<WgpuRuntime>> {
        hosvd_impl(self, tensor, ranks, TensorDecomposeDTypeSupport::F32_ONLY)
    }

    fn tucker(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        ranks: &[usize],
        options: TuckerOptions,
    ) -> Result<TuckerDecomposition<WgpuRuntime>> {
        tucker_impl(
            self,
            tensor,
            ranks,
            options,
            TensorDecomposeDTypeSupport::F32_ONLY,
        )
    }

    fn cp_decompose(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        rank: usize,
        options: CpOptions,
    ) -> Result<CpDecomposition<WgpuRuntime>> {
        cp_decompose_impl(
            self,
            tensor,
            rank,
            options,
            TensorDecomposeDTypeSupport::F32_ONLY,
        )
    }

    fn tensor_train(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        max_rank: usize,
        tolerance: f64,
    ) -> Result<TensorTrainDecomposition<WgpuRuntime>> {
        tensor_train_impl(
            self,
            tensor,
            max_rank,
            tolerance,
            TensorDecomposeDTypeSupport::F32_ONLY,
        )
    }

    fn tucker_reconstruct(
        &self,
        decomp: &TuckerDecomposition<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        tucker_reconstruct_impl(self, decomp, TensorDecomposeDTypeSupport::F32_ONLY)
    }

    fn cp_reconstruct(
        &self,
        decomp: &CpDecomposition<WgpuRuntime>,
        shape: &[usize],
    ) -> Result<Tensor<WgpuRuntime>> {
        cp_reconstruct_impl(self, decomp, shape, TensorDecomposeDTypeSupport::F32_ONLY)
    }

    fn tt_reconstruct(
        &self,
        decomp: &TensorTrainDecomposition<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        tt_reconstruct_impl(self, decomp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;

    fn get_client() -> WgpuClient {
        let device = WgpuRuntime::default_device();
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        let client = get_client();
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let tensor =
            Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device_id).unwrap();

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
        let tensor = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            &[2, 3],
            &client.device_id,
        )
        .unwrap();
        // Matrix [4, 2] for mode-0 product
        let matrix = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, 1.0],
            &[4, 2],
            &client.device_id,
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
        let tensor =
            Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device_id).unwrap();

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
    fn test_cp_decompose_basic() {
        let client = get_client();
        // Simple rank-1 tensor: outer product of [1, 2] and [1, 1, 1]
        let tensor = Tensor::<WgpuRuntime>::from_slice(
            &[1.0f32, 1.0, 1.0, 2.0, 2.0, 2.0],
            &[2, 3],
            &client.device_id,
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
        let tensor =
            Tensor::<WgpuRuntime>::from_slice(&data, &[2, 3, 4], &client.device_id).unwrap();

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
}
