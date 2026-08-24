//! WebGPU runtime operation helpers and TensorOps implementation.
//!
//! Trait implementations (CompareOps, ScalarOps, LogicalOps, etc.) live in `ops/wgpu/`.
//! This module provides shared helpers, native GPU launchers, and TensorOps.

pub(crate) mod helpers;
pub(crate) mod native;
mod tensor;

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use crate::ops::{ActivationOps, BinaryOps, CompareOps, MatmulOps, ReduceOps, ScalarOps};
    use crate::runtime::Runtime;
    use crate::runtime::wgpu::{WgpuDevice, WgpuRuntime, is_wgpu_available};
    use crate::tensor::Tensor;

    fn create_test_tensor(data: &[f32], shape: &[usize]) -> Tensor<WgpuRuntime> {
        let device = WgpuDevice::new(0);
        Tensor::<WgpuRuntime>::try_from_slice(data, shape, &device).unwrap()
    }

    #[test]
    fn test_wgpu_add() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = create_test_tensor(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let result = client.add(&a, &b).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_wgpu_matmul() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        // 2x3 @ 3x2 = 2x2
        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = create_test_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);

        let result = client.matmul(&a, &b).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_wgpu_relu() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[-1.0, 0.0, 1.0, 2.0], &[4]);
        let result = client.relu(&a).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_wgpu_sum() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = client.sum(&a, &[0], false).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![4.0, 6.0]);
    }

    #[test]
    fn test_wgpu_mul_scalar() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = client.mul_scalar(&a, 2.0).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_wgpu_eq() {
        if !is_wgpu_available() {
            println!("No GPU available, skipping test");
            return;
        }

        let device = WgpuDevice::new(0);
        let client = WgpuRuntime::default_client(&device);

        let a = create_test_tensor(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let b = create_test_tensor(&[1.0, 0.0, 3.0, 0.0], &[4]);

        let result = client.eq(&a, &b).unwrap();
        let data: Vec<f32> = result.to_vec();

        assert_eq!(data, vec![1.0, 0.0, 1.0, 0.0]);
    }
}
