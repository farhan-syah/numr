//! Arithmetic operations on dual tensors

use crate::autograd::DualTensor;
use crate::error::Result;
use crate::ops::{BinaryOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::Tensor;

/// Add two optional tangents: Some + Some = Some, Some + None = Some, None + None = None
pub(crate) fn add_tangents<R, C>(
    t1: Option<&Tensor<R>>,
    t2: Option<&Tensor<R>>,
    client: &C,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R>,
{
    match (t1, t2) {
        (Some(a), Some(b)) => Ok(Some(client.add(a, b)?)),
        (Some(a), None) => Ok(Some(a.clone())),
        (None, Some(b)) => Ok(Some(b.clone())),
        (None, None) => Ok(None),
    }
}

/// Subtract two optional tangents
pub(crate) fn sub_tangents<R, C>(
    t1: Option<&Tensor<R>>,
    t2: Option<&Tensor<R>>,
    client: &C,
) -> Result<Option<Tensor<R>>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R>,
{
    match (t1, t2) {
        (Some(a), Some(b)) => Ok(Some(client.sub(a, b)?)),
        (Some(a), None) => Ok(Some(a.clone())),
        (None, Some(b)) => Ok(Some(client.neg(b)?)),
        (None, None) => Ok(None),
    }
}

/// Dual addition: (a, ȧ) + (b, ḃ) = (a + b, ȧ + ḃ)
pub fn dual_add<R, C>(a: &DualTensor<R>, b: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R>,
{
    let primal = client.add(a.primal(), b.primal())?;
    let tangent = add_tangents(a.tangent(), b.tangent(), client)?;
    Ok(DualTensor::new(primal, tangent))
}

/// Dual subtraction: (a, ȧ) - (b, ḃ) = (a - b, ȧ - ḃ)
pub fn dual_sub<R, C>(a: &DualTensor<R>, b: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R>,
{
    let primal = client.sub(a.primal(), b.primal())?;
    let tangent = sub_tangents(a.tangent(), b.tangent(), client)?;
    Ok(DualTensor::new(primal, tangent))
}

/// Dual multiplication: (a, ȧ) * (b, ḃ) = (a*b, ȧ*b + a*ḃ)
///
/// Product rule: d(a*b) = da*b + a*db
pub fn dual_mul<R, C>(a: &DualTensor<R>, b: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R>,
{
    let primal = client.mul(a.primal(), b.primal())?;

    let tangent = match (a.tangent(), b.tangent()) {
        (Some(at), Some(bt)) => {
            let term1 = client.mul(at, b.primal())?;
            let term2 = client.mul(a.primal(), bt)?;
            Some(client.add(&term1, &term2)?)
        }
        (Some(at), None) => Some(client.mul(at, b.primal())?),
        (None, Some(bt)) => Some(client.mul(a.primal(), bt)?),
        (None, None) => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual division: (a, ȧ) / (b, ḃ) = (a/b, (ȧ*b - a*ḃ) / b²)
///
/// Quotient rule: d(a/b) = (da*b - a*db) / b²
pub fn dual_div<R, C>(a: &DualTensor<R>, b: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + BinaryOps<R> + UnaryOps<R>,
{
    let primal = client.div(a.primal(), b.primal())?;

    let tangent = match (a.tangent(), b.tangent()) {
        (Some(at), Some(bt)) => {
            let term1 = client.mul(at, b.primal())?;
            let term2 = client.mul(a.primal(), bt)?;
            let numer = client.sub(&term1, &term2)?;
            let b_sq = client.mul(b.primal(), b.primal())?;
            Some(client.div(&numer, &b_sq)?)
        }
        (Some(at), None) => Some(client.div(at, b.primal())?),
        (None, Some(bt)) => {
            let term = client.mul(a.primal(), bt)?;
            let b_sq = client.mul(b.primal(), b.primal())?;
            let neg_term = client.neg(&term)?;
            Some(client.div(&neg_term, &b_sq)?)
        }
        (None, None) => None,
    };

    Ok(DualTensor::new(primal, tangent))
}

/// Dual negation: -(a, ȧ) = (-a, -ȧ)
pub fn dual_neg<R, C>(a: &DualTensor<R>, client: &C) -> Result<DualTensor<R>>
where
    R: Runtime,
    C: RuntimeClient<R> + UnaryOps<R>,
{
    let primal = client.neg(a.primal())?;
    let tangent = match a.tangent() {
        Some(t) => Some(client.neg(t)?),
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
    fn test_dual_add() {
        let (device, client) = setup();

        let a_primal =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let a_tangent =
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 0.0, 0.0], &[3], &device).unwrap();
        let a = DualTensor::with_tangent(a_primal, a_tangent);

        let b_primal =
            Tensor::<CpuRuntime>::try_from_slice(&[4.0f32, 5.0, 6.0], &[3], &device).unwrap();
        let b_tangent =
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32, 1.0, 0.0], &[3], &device).unwrap();
        let b = DualTensor::with_tangent(b_primal, b_tangent);

        let c = dual_add(&a, &b, &client).unwrap();

        assert_eq!(c.primal().to_vec::<f32>(), [5.0, 7.0, 9.0]);
        assert_eq!(c.tangent().unwrap().to_vec::<f32>(), [1.0, 1.0, 0.0]);
    }

    #[test]
    fn test_dual_mul() {
        let (device, client) = setup();

        // f(x) = x² at x=3, tangent v=1 → df = 2*3*1 = 6
        let x_primal = Tensor::<CpuRuntime>::try_from_slice(&[3.0f32], &[1], &device).unwrap();
        let x_tangent = Tensor::<CpuRuntime>::try_from_slice(&[1.0f32], &[1], &device).unwrap();
        let x = DualTensor::with_tangent(x_primal, x_tangent);

        let y = dual_mul(&x, &x, &client).unwrap();

        assert_eq!(y.primal().to_vec::<f32>(), [9.0]);
        assert_eq!(y.tangent().unwrap().to_vec::<f32>(), [6.0]);
    }

    #[test]
    fn test_dual_constant_no_tangent() {
        let (device, client) = setup();

        let a = DualTensor::constant(
            Tensor::<CpuRuntime>::try_from_slice(&[1.0f32, 2.0], &[2], &device).unwrap(),
        );
        let b = DualTensor::constant(
            Tensor::<CpuRuntime>::try_from_slice(&[3.0f32, 4.0], &[2], &device).unwrap(),
        );

        let c = dual_add(&a, &b, &client).unwrap();

        assert_eq!(c.primal().to_vec::<f32>(), [4.0, 6.0]);
        assert!(!c.has_tangent());
    }
}
