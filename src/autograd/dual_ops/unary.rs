//! Unary operations on dual tensors

use crate::autograd::DualTensor;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, ScalarOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Dual exponential: exp(a, ȧ) = (exp(a), exp(a) * ȧ)
pub fn dual_exp<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.exp(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => Some(client.mul(&primal, at)?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual natural logarithm: log(a, ȧ) = (log(a), ȧ / a)
pub fn dual_log<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.log(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => Some(client.div(at, a.primal())?),
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual square root: sqrt(a, ȧ) = (sqrt(a), ȧ / (2 * sqrt(a)))
pub fn dual_sqrt<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let primal = client.sqrt(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let two_sqrt = client.mul_scalar(&primal, 2.0)?;
            Some(client.div(at, &two_sqrt)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual square: square(a, ȧ) = (a², 2 * a * ȧ)
pub fn dual_square<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let primal = client.square(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let two_a = client.mul_scalar(a.primal(), 2.0)?;
            Some(client.mul(&two_a, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual sine: sin(a, ȧ) = (sin(a), cos(a) * ȧ)
pub fn dual_sin<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.sin(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let cos_a = client.cos(a.primal())?;
            Some(client.mul(&cos_a, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual cosine: cos(a, ȧ) = (cos(a), -sin(a) * ȧ)
pub fn dual_cos<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.cos(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let sin_a = client.sin(a.primal())?;
            let neg_sin_a = client.neg(&sin_a)?;
            Some(client.mul(&neg_sin_a, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual tangent: tan(a, ȧ) = (tan(a), (1 + tan²(a)) * ȧ)
pub fn dual_tan<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let primal = client.tan(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let tan_sq = client.square(&primal)?;
            let sec_sq = client.add_scalar(&tan_sq, 1.0)?;
            Some(client.mul(&sec_sq, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual hyperbolic tangent: tanh(a, ȧ) = (tanh(a), (1 - tanh²(a)) * ȧ)
pub fn dual_tanh<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R> + ScalarOps<R>,
{
    let primal = client.tanh(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let tanh_sq = client.square(&primal)?;
            let one = Tensor::ones(tanh_sq.shape(), tanh_sq.dtype(), tanh_sq.device())?;
            let sech_sq = client.sub(&one, &tanh_sq)?;
            Some(client.mul(&sech_sq, at)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual reciprocal: recip(a, ȧ) = (1/a, -ȧ / a²)
pub fn dual_recip<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.recip(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let a_sq = client.mul(a.primal(), a.primal())?;
            let neg_at = client.neg(at)?;
            Some(client.div(&neg_at, &a_sq)?)
        }
        None => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual absolute value: abs(a, ȧ) = (|a|, sign(a) * ȧ)
pub fn dual_abs<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R> + BinaryOps<R>,
{
    let primal = client.abs(a.primal())?;

    let tangent = match a.tangent() {
        Some(at) => {
            let sign = client.sign(a.primal())?;
            Some(client.mul(&sign, at)?)
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
    fn test_dual_exp() {
        let (device, client) = setup();

        let x_primal = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_exp(&x, &client).unwrap();

        assert!((y.primal().to_vec::<f32>()[0] - 1.0).abs() < 1e-6);
        assert!((y.tangent().unwrap().to_vec::<f32>()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dual_sin_cos() {
        let (device, client) = setup();

        let x_primal = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let sin_x = dual_sin(&x, &client).unwrap();
        assert!((sin_x.primal().to_vec::<f32>()[0] - 0.0).abs() < 1e-6);
        assert!((sin_x.tangent().unwrap().to_vec::<f32>()[0] - 1.0).abs() < 1e-6);

        let cos_x = dual_cos(&x, &client).unwrap();
        assert!((cos_x.primal().to_vec::<f32>()[0] - 1.0).abs() < 1e-6);
        assert!((cos_x.tangent().unwrap().to_vec::<f32>()[0] - 0.0).abs() < 1e-6);
    }
}
