//! CUDA runtime operation helpers and TensorOps implementation.
//!
//! Trait implementations (CompareOps, ScalarOps, LogicalOps, etc.) live in `ops/cuda/`.
//! This module provides shared helpers, kernel launchers, and TensorOps.

pub(crate) mod helpers;
pub(crate) mod matmul_broadcast;
mod statistics;
mod tensor;

#[cfg(test)]
mod tests {
    use crate::ops::{
        ActivationOps, BinaryOps, IndexingOps, MatmulOps, NormalizationOps, ReduceOps,
    };
    use crate::runtime::Runtime;
    use crate::runtime::cuda::{CudaDevice, CudaRuntime, is_cuda_available};
    use crate::tensor::Tensor;

    #[test]
    fn test_cuda_tensor_add() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        let c = client.add(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_2x2() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CudaRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        let c = client.matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(result, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_cuda_tensor_matmul_3x2_2x4() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], &device)
                .unwrap();
        let b = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
            &device,
        )
        .unwrap();

        let c = client.matmul(&a, &b).unwrap();

        assert_eq!(c.shape(), &[3, 4]);
        let result: Vec<f32> = c.to_vec();
        assert_eq!(
            result,
            [
                11.0, 14.0, 17.0, 20.0, 23.0, 30.0, 37.0, 44.0, 35.0, 46.0, 57.0, 68.0
            ]
        );
    }

    #[test]
    fn test_cuda_tensor_relu() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[-1.0f32, 0.0, 1.0, -2.0], &[4], &device).unwrap();
        let b = client.relu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_cuda_tensor_sum() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
                .unwrap();

        let b = client.sum(&a, &[1], false).unwrap();

        assert_eq!(b.shape(), &[2]);
        let result: Vec<f32> = b.to_vec();
        assert_eq!(result, [6.0, 15.0]);
    }

    #[test]
    fn test_cuda_tensor_silu() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device)
            .unwrap();
        let b = client.silu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // SiLU(x) = x / (1 + exp(-x))
        // SiLU(0) = 0
        // SiLU(1) ≈ 0.731
        // SiLU(-1) ≈ -0.269
        assert!((result[2] - 0.0).abs() < 1e-5); // SiLU(0) = 0
        assert!((result[3] - 0.7310586).abs() < 1e-4); // SiLU(1) ≈ 0.731
        assert!((result[1] - (-0.2689414)).abs() < 1e-4); // SiLU(-1) ≈ -0.269
    }

    #[test]
    fn test_cuda_tensor_gelu() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        let a = Tensor::<CudaRuntime>::from_slice(&[-2.0f32, -1.0, 0.0, 1.0, 2.0], &[5], &device)
            .unwrap();
        let b = client.gelu(&a).unwrap();

        let result: Vec<f32> = b.to_vec();
        // GELU(0) = 0
        // GELU is approximately x for large positive x
        // GELU is approximately 0 for large negative x
        assert!((result[2] - 0.0).abs() < 1e-5); // GELU(0) = 0
        assert!((result[3] - 0.8413).abs() < 0.01); // GELU(1) ≈ 0.841
        assert!((result[4] - 1.9545).abs() < 0.01); // GELU(2) ≈ 1.955
    }

    #[test]
    fn test_cuda_tensor_rms_norm() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        )
        .unwrap();
        let weight =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();

        let out = client.rms_norm(&input, &weight, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], RMS = sqrt((1+4+9+16)/4) = sqrt(30/4) = sqrt(7.5) ≈ 2.739
        let rms1 = (30.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[0] - 1.0 / rms1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - 2.0 / rms1).abs() < 1e-3);
        assert!((result[2] - 3.0 / rms1).abs() < 1e-3);
        assert!((result[3] - 4.0 / rms1).abs() < 1e-3);

        // Row 2: [2, 4, 6, 8]
        let rms2 = (120.0f32 / 4.0 + 1e-5).sqrt();
        assert!((result[4] - 2.0 / rms2).abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_layer_norm() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // Input: 2 rows, 4 features each
        let input = Tensor::<CudaRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0],
            &[2, 4],
            &device,
        )
        .unwrap();
        let weight =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[4], &device).unwrap();
        let bias =
            Tensor::<CudaRuntime>::from_slice(&[0.0f32, 0.0, 0.0, 0.0], &[4], &device).unwrap();

        let out = client.layer_norm(&input, &weight, &bias, 1e-5).unwrap();
        let result: Vec<f32> = out.to_vec();

        // Row 1: [1, 2, 3, 4], mean = 2.5, var = 1.25, std = 1.118
        let mean1 = 2.5f32;
        let var1 = ((1.0 - mean1).powi(2)
            + (2.0 - mean1).powi(2)
            + (3.0 - mean1).powi(2)
            + (4.0 - mean1).powi(2))
            / 4.0;
        let std1 = (var1 + 1e-5).sqrt();
        assert!((result[0] - (1.0 - mean1) / std1).abs() < 1e-3); // Wider tolerance for GPU
        assert!((result[1] - (2.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[2] - (3.0 - mean1) / std1).abs() < 1e-3);
        assert!((result[3] - (4.0 - mean1) / std1).abs() < 1e-3);

        // Verify normalized outputs sum to approximately 0 (zero-centered)
        let row1_sum: f32 = result[0..4].iter().sum();
        assert!(row1_sum.abs() < 1e-3);
    }

    #[test]
    fn test_cuda_tensor_argmax() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
                .unwrap();

        // argmax along dim=1 (find max index in each row)
        let out = client.argmax(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [1, 2]); // Row 0: max at index 1 (5.0), Row 1: max at index 2 (6.0)

        // argmax along dim=0 (find max index in each column)
        let out = client.argmax(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [1, 0, 1]); // Col 0: max at 1 (4.0), Col 1: max at 0 (5.0), Col 2: max at 1 (6.0)

        // Test keepdim=true
        let out = client.argmax(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [1, 2]);
    }

    #[test]
    fn test_cuda_tensor_argmin() {
        if !is_cuda_available() {
            return;
        }
        let device = CudaDevice::new(0);
        let client = CudaRuntime::default_client(&device);

        // 2D tensor: [[1, 5, 3], [4, 2, 6]]
        let a =
            Tensor::<CudaRuntime>::from_slice(&[1.0f32, 5.0, 3.0, 4.0, 2.0, 6.0], &[2, 3], &device)
                .unwrap();

        // argmin along dim=1 (find min index in each row)
        let out = client.argmin(&a, 1, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(result, [0, 1]); // Row 0: min at index 0 (1.0), Row 1: min at index 1 (2.0)

        // argmin along dim=0 (find min index in each column)
        let out = client.argmin(&a, 0, false).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[3]);
        assert_eq!(result, [0, 1, 0]); // Col 0: min at 0 (1.0), Col 1: min at 1 (2.0), Col 2: min at 0 (3.0)

        // Test keepdim=true
        let out = client.argmin(&a, 1, true).unwrap();
        let result: Vec<i64> = out.to_vec();
        assert_eq!(out.shape(), &[2, 1]);
        assert_eq!(result, [0, 1]);
    }
}
