//! CPU implementation of scalar operations.

use crate::error::Result;
use crate::ops::{BinaryOp, ScalarOps};
use crate::runtime::cpu::{
    CpuClient, CpuRuntime, helpers::fused_mul_add_scalar_impl,
    helpers::scalar::rsub_scalar_op_impl, helpers::scalar_op_impl,
};
use crate::tensor::Tensor;

impl ScalarOps<CpuRuntime> for CpuClient {
    fn add_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Add, a, scalar, "add_scalar")
    }

    fn sub_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Sub, a, scalar, "sub_scalar")
    }

    fn mul_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Mul, a, scalar, "mul_scalar")
    }

    fn div_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Div, a, scalar, "div_scalar")
    }

    fn pow_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        scalar_op_impl(self, BinaryOp::Pow, a, scalar, "pow_scalar")
    }

    fn rsub_scalar(&self, a: &Tensor<CpuRuntime>, scalar: f64) -> Result<Tensor<CpuRuntime>> {
        rsub_scalar_op_impl(self, a, scalar)
    }

    fn fused_mul_add_scalar(
        &self,
        a: &Tensor<CpuRuntime>,
        scale: f64,
        bias: f64,
    ) -> Result<Tensor<CpuRuntime>> {
        fused_mul_add_scalar_impl(self, a, scale, bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    #[test]
    fn test_rsub_scalar_f32() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4], &device).unwrap();
        let result = client.rsub_scalar(&a, 5.0).unwrap();
        let data: Vec<f32> = result.to_vec();
        assert_eq!(data, vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_rsub_scalar_f64() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a =
            Tensor::<CpuRuntime>::try_from_slice(&[0.3f64, 0.7, 1.0, 0.0], &[4], &device).unwrap();
        let result = client.rsub_scalar(&a, 1.0).unwrap();
        let data: Vec<f64> = result.to_vec();
        assert_eq!(data, vec![0.7, 0.30000000000000004, 0.0, 1.0]);
    }

    #[test]
    fn test_rsub_scalar_complement() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Common pattern: 1 - probability
        let p = Tensor::<CpuRuntime>::try_from_slice(&[0.1f32, 0.5, 0.9], &[3], &device).unwrap();
        let complement = client.rsub_scalar(&p, 1.0).unwrap();
        let data: Vec<f32> = complement.to_vec();
        for (a, b) in data.iter().zip([0.9f32, 0.5, 0.1].iter()) {
            assert!((a - b).abs() < 1e-6, "{a} != {b}");
        }
    }

    #[test]
    fn test_rsub_scalar_i32() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let a = Tensor::<CpuRuntime>::try_from_slice(&[1i32, 2, 3, 4], &[4], &device).unwrap();
        let result = client.rsub_scalar(&a, 10.0).unwrap();
        let data: Vec<i32> = result.to_vec();
        assert_eq!(data, vec![9, 8, 7, 6]);
    }
}
