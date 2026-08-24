//! Activation operations on dual tensors

use crate::autograd::DualTensor;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, BinaryOps, CompareOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Dual ReLU: relu(a, ȧ) = (relu(a), ȧ * (a > 0))
pub fn dual_relu<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R> + CompareOps<R> + BinaryOps<R> + TensorOps<R>,
{
    let primal = client.relu(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let zero = Tensor::zeros(a.primal().shape(), a.primal().dtype(), a.primal().device())?;
            let mask = client.gt(a.primal(), &zero)?;
            let mask_typed = client.cast(&mask, at.dtype())?;
            Some(client.mul(&mask_typed, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual sigmoid: sigmoid(a, ȧ) = (σ(a), σ(a) * (1 - σ(a)) * ȧ)
pub fn dual_sigmoid<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + ActivationOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let primal = client.sigmoid(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let one = Tensor::ones(primal.shape(), primal.dtype(), primal.device())?;
            let one_minus_sig = client.sub(&one, &primal)?;
            let grad = client.mul(&primal, &one_minus_sig)?;
            Some(client.mul(&grad, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_dual_relu_positive() {
        let (device, client) = setup();

        // At x=2 (positive): relu(2)=2, relu'(2)=1
        // tangent = 1 * v = 1 * 1.0 = 1.0
        let x_primal = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_relu(&x, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [2.0]);
        assert_eq!(y.tangent().unwrap().to_vec::<f32>(), [1.0]);
    }

    #[test]
    fn test_dual_relu_negative() {
        let (device, client) = setup();

        // At x=-1 (negative): relu(-1)=0, relu'(-1)=0
        // tangent = 0 * v = 0
        let x_primal = Tensor::<CpuRuntime>::from_slice(&[-1.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_relu(&x, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [0.0]);
        assert_eq!(y.tangent().unwrap().to_vec::<f32>(), [0.0]);
    }

    #[test]
    fn test_dual_relu_mixed() {
        let (device, client) = setup();

        // Mixed: [-1, 0, 2] -> relu = [0, 0, 2], gradient mask = [0, 0, 1]
        let x_primal =
            Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 2.0], &[3], &device).unwrap();
        let x_tangent =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[3], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_relu(&x, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [0.0, 0.0, 2.0]);
        assert_eq!(y.tangent().unwrap().to_vec::<f32>(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_dual_sigmoid_at_zero() {
        let (device, client) = setup();

        // At x=0: sigmoid(0)=0.5, sigmoid'(0)=0.5*(1-0.5)=0.25
        // tangent = 0.25 * v = 0.25 * 1.0 = 0.25
        let x_primal = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_sigmoid(&x, &client).unwrap();

        assert!((y.primal().to_vec::<f32>()[0] - 0.5).abs() < 1e-6);
        assert!((y.tangent().unwrap().to_vec::<f32>()[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_dual_sigmoid_large_positive() {
        let (device, client) = setup();

        // At large x: sigmoid(x) -> 1, sigmoid'(x) -> 0
        let x_primal = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_sigmoid(&x, &client).unwrap();

        // sigmoid(10) ≈ 0.9999546
        assert!(y.primal().to_vec::<f32>()[0] > 0.999);
        // sigmoid'(10) ≈ 0.0000454 (very small)
        assert!(y.tangent().unwrap().to_vec::<f32>()[0] < 0.001);
    }

    #[test]
    fn test_dual_sigmoid_large_negative() {
        let (device, client) = setup();

        // At large negative x: sigmoid(x) -> 0, sigmoid'(x) -> 0
        let x_primal = Tensor::<CpuRuntime>::from_slice(&[-10.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_sigmoid(&x, &client).unwrap();

        // sigmoid(-10) ≈ 0.0000454
        assert!(y.primal().to_vec::<f32>()[0] < 0.001);
        // sigmoid'(-10) ≈ 0.0000454 (very small)
        assert!(y.tangent().unwrap().to_vec::<f32>()[0] < 0.001);
    }
}
