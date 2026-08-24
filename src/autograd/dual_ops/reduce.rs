//! Reduction operations on dual tensors

use crate::autograd::DualTensor;
use crate::error::Result;
use crate::ops::ReduceOps;
use crate::runtime::{Runtime, RuntimeClient};

/// Dual sum: sum(a, ȧ) = (sum(a), sum(ȧ))
pub fn dual_sum<R, C>(
    a: &DualTensor<R>,
    dims: &[usize],
    keepdim: bool,
    client: &C,
) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ReduceOps<R>,
{
    let primal = client.sum(a.primal(), dims, keepdim)?;

    let tangent = match a.tangent() {
        Some(at) => Some(client.sum(at, dims, keepdim)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual mean: mean(a, ȧ) = (mean(a), mean(ȧ))
pub fn dual_mean<R, C>(
    a: &DualTensor<R>,
    dims: &[usize],
    keepdim: bool,
    client: &C,
) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ReduceOps<R>,
{
    let primal = client.mean(a.primal(), dims, keepdim)?;

    let tangent = match a.tangent() {
        Some(at) => Some(client.mean(at, dims, keepdim)?),
        None => None,
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
    fn test_dual_sum() {
        let (device, client) = setup();

        let x_primal =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let x_tangent =
            Tensor::<CpuRuntime>::try_from_slice(&[0.1f32, 0.2, 0.3], &[3], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_sum(&x, &[0], false, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [6.0]);
        assert!((y.tangent().unwrap().to_vec::<f32>()[0] - 0.6).abs() < 1e-6);
    }
}
