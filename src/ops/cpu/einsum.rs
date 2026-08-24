//! CPU implementation of einsum operations.

use crate::error::Result;
use crate::ops::EinsumOps;
use crate::ops::impl_generic::einsum::einsum_impl;
use crate::runtime::cpu::{CpuClient, CpuRuntime};
use crate::tensor::Tensor;

impl EinsumOps<CpuRuntime> for CpuClient {
    fn einsum(&self, notation: &str, inputs: &[&Tensor<CpuRuntime>]) -> Result<Tensor<CpuRuntime>> {
        einsum_impl(self, notation, inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{BinaryOps, MatmulOps};
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn create_client() -> (CpuClient, CpuDevice) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (client, device)
    }

    #[test]
    fn test_einsum_matmul() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ij,jk->ik", &[&a, &b]).unwrap();
        let expected = client.matmul(&a, &b).unwrap();

        let r: Vec<f32> = result.to_vec();
        let e: Vec<f32> = expected.to_vec();
        assert_eq!(r, e);
    }

    #[test]
    fn test_einsum_transpose() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &device)
                .unwrap();

        let result = client.einsum("ij->ji", &[&a]).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_einsum_trace() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ii->", &[&a]).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![5.0]); // 1 + 4
    }

    #[test]
    fn test_einsum_diagonal() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ii->i", &[&a]).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![1.0, 4.0]);
    }

    #[test]
    fn test_einsum_sum_all() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ij->", &[&a]).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![10.0]);
    }

    #[test]
    fn test_einsum_sum_rows() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ij->i", &[&a]).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![3.0, 7.0]);
    }

    #[test]
    fn test_einsum_outer_product() {
        let (client, device) = create_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0], &[2], &device).unwrap();

        let result = client.einsum("i,j->ij", &[&a, &b]).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]);
    }

    #[test]
    fn test_einsum_hadamard() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        let result = client.einsum("ij,ij->ij", &[&a, &b]).unwrap();
        let expected = client.mul(&a, &b).unwrap();

        let r: Vec<f32> = result.to_vec();
        let e: Vec<f32> = expected.to_vec();
        assert_eq!(r, e);
    }

    #[test]
    fn test_einsum_dot_product() {
        let (client, device) = create_client();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let b = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).unwrap();

        let result = client.einsum("i,i->", &[&a, &b]).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![32.0]); // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_einsum_implicit_notation() {
        let (client, device) = create_client();
        let a =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2], &device).unwrap();
        let b =
            Tensor::<CpuRuntime>::from_slice(&[5.0f32, 6.0, 7.0, 8.0], &[2, 2], &device).unwrap();

        // Implicit "ij,jk" = "ij,jk->ik"
        let result = client.einsum("ij,jk", &[&a, &b]).unwrap();
        let expected = client.matmul(&a, &b).unwrap();

        let r: Vec<f32> = result.to_vec();
        let e: Vec<f32> = expected.to_vec();
        assert_eq!(r, e);
    }
}
