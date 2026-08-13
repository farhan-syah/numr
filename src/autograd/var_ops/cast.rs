//! Autograd-aware dtype casting

use crate::autograd::Var;
use crate::autograd::var_ops::ops::CastBackward;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::TypeConversionOps;
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Cast a variable to a different dtype, preserving gradient flow.
///
/// The backward pass casts the gradient back to the input's original dtype.
///
/// # Arguments
/// * `a` - Input variable
/// * `dtype` - Target dtype
/// * `client` - Runtime client
pub fn var_cast<R, C>(a: &Var<R>, dtype: DType, client: &C) -> Result<Var<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TypeConversionOps<R>,
    R::Client: TypeConversionOps<R>,
{
    let input_dtype = a.tensor().dtype();

    // No-op if already the target dtype
    if input_dtype == dtype {
        return Ok(Var::with_id(a.tensor().clone(), a.id(), a.requires_grad()));
    }

    let output = client.cast(a.tensor(), dtype)?;

    if a.requires_grad() {
        let grad_fn = CastBackward::<R>::new(a.id(), input_dtype, a.grad_fn().cloned());
        Ok(Var::from_op(output, Arc::new(grad_fn)))
    } else {
        Ok(Var::new(output, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_cast_noop_same_dtype() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let v = Var::new(t, true);
        let result = var_cast(&v, DType::F32, &client).unwrap();
        // Same dtype returns clone — data should match
        assert_eq!(result.tensor().dtype(), DType::F32);
        let data = result.tensor().to_vec::<f32>();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_var_cast_f32_to_f64_gradient() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device);
        let x = Var::new(t, true);

        // Cast F32 → F64
        let y = var_cast(&x, DType::F64, &client).unwrap();
        assert_eq!(y.tensor().dtype(), DType::F64);

        // Sum to scalar for backward
        let sum = crate::autograd::var_sum(&y, &[], false, &client).unwrap();
        let grads = backward(&sum, &client).unwrap();

        // Gradient should be F32 (cast back from F64)
        let grad = grads.get(x.id()).unwrap();
        assert_eq!(grad.dtype(), DType::F32);
        let data = grad.to_vec::<f32>();
        assert_eq!(data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_var_cast_no_grad() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let t = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let v = Var::new(t, false);
        let result = var_cast(&v, DType::F64, &client).unwrap();
        assert!(!result.requires_grad());
        assert_eq!(result.tensor().dtype(), DType::F64);
    }
}
