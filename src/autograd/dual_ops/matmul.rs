//! Matrix multiplication operations on dual tensors

use crate::autograd::DualTensor;
use crate::error::Result;
use crate::ops::{BinaryOps, MatmulOps};
use crate::runtime::{Runtime, RuntimeClient};

/// Dual matrix multiplication: (A, Ȧ) @ (B, Ḃ) = (A@B, Ȧ@B + A@Ḃ)
///
/// Product rule for matrices: d(A@B) = dA@B + A@dB
pub fn dual_matmul<R, C>(a: &DualTensor<R>, b: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + MatmulOps<R> + BinaryOps<R>,
{
    let primal = client.matmul(a.primal(), b.primal())?;

    let tangent = match (a.tangent(), b.tangent()) {
        (Some(at), Some(bt)) => {
            let term1 = client.matmul(at, b.primal())?;
            let term2 = client.matmul(a.primal(), bt)?;
            Some(client.add(&term1, &term2)?)
        }
        (Some(at), None) => Some(client.matmul(at, b.primal())?),
        (None, Some(bt)) => Some(client.matmul(a.primal(), bt)?),
        (None, None) => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_dual_matmul() {
        let (device, client) = setup();

        let a_primal = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 2], &device).unwrap();
        let a_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[1, 2], &device).unwrap();
        let a = DualTensor::with_tangent(a_primal, a_tangent);

        let b_primal = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2, 1], &device).unwrap();
        let b = DualTensor::constant(b_primal);

        let c = dual_matmul(&a, &b, &client).unwrap();

        assert_eq!(c.primal().to_vec::<f32>(), [11.0]);
        assert_eq!(c.tangent().unwrap().to_vec::<f32>(), [3.0]);
    }
}
