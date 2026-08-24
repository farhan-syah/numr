//! Scalar operations on dual tensors

use crate::autograd::DualTensor;
use crate::error::Result;
use crate::ops::{BinaryOps, ScalarOps};
use crate::runtime::{Runtime, RuntimeClient};

/// Dual add scalar: (a, ȧ) + c = (a + c, ȧ)
pub fn dual_add_scalar<R, C>(a: &DualTensor<R>, scalar: f64, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R>,
{
    let primal = client.add_scalar(a.primal(), scalar)?;
    Ok(DualTensor::new(primal, a.tangent().cloned()))
}

/// Dual subtract scalar: (a, ȧ) - c = (a - c, ȧ)
pub fn dual_sub_scalar<R, C>(a: &DualTensor<R>, scalar: f64, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R>,
{
    let primal = client.sub_scalar(a.primal(), scalar)?;
    Ok(DualTensor::new(primal, a.tangent().cloned()))
}

/// Dual multiply scalar: (a, ȧ) * c = (a * c, ȧ * c)
pub fn dual_mul_scalar<R, C>(a: &DualTensor<R>, scalar: f64, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R>,
{
    let primal = client.mul_scalar(a.primal(), scalar)?;
    let tangent = match a.tangent() {
        Some(t) => Some(client.mul_scalar(t, scalar)?),
        None => None,
    };
    Ok(DualTensor::new(primal, tangent))
}

/// Dual divide scalar: (a, ȧ) / c = (a / c, ȧ / c)
pub fn dual_div_scalar<R, C>(a: &DualTensor<R>, scalar: f64, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R>,
{
    let primal = client.div_scalar(a.primal(), scalar)?;
    let tangent = match a.tangent() {
        Some(t) => Some(client.div_scalar(t, scalar)?),
        None => None,
    };
    Ok(DualTensor::new(primal, tangent))
}

/// Dual power with scalar exponent: (a, ȧ)^n = (a^n, n * a^(n-1) * ȧ)
///
/// d(a^n)/da = n * a^(n-1)
pub fn dual_pow_scalar<R, C>(a: &DualTensor<R>, n: f64, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + ScalarOps<R> + BinaryOps<R>,
{
    let primal = client.pow_scalar(a.primal(), n)?;

    let tangent = match a.tangent() {
        Some(at) => {
            let a_pow_nm1 = client.pow_scalar(a.primal(), n - 1.0)?;
            let coeff = client.mul_scalar(&a_pow_nm1, n)?;
            Some(client.mul(&coeff, at)?)
        }
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
    fn test_dual_pow_scalar() {
        let (device, client) = setup();

        // f(x) = x³ at x=2, tangent v=1 → df = 3*2²*1 = 12
        let x_primal = Tensor::<CpuRuntime>::try_from_slice(&[2.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_pow_scalar(&x, 3.0, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [8.0]);
        assert!((y.tangent().unwrap().to_vec::<f32>()[0] - 12.0).abs() < 1e-5);
    }
}
