//! Backward implementations for scalar operations
//!
//! These implement gradient computation for tensor-scalar operations.

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_div_scalar, var_mul, var_mul_scalar};
use crate::error::Result;
use crate::ops::{BinaryOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};

// ============================================================================
// AddScalarBackward
// ============================================================================

/// Backward for tensor + scalar: z = a + scalar
///
/// Gradient: dL/da = dL/dz (pass through)
pub struct AddScalarBackward<R: Runtime> {
    input_id: TensorId,
    input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> AddScalarBackward<R> {
    /// Create a new AddScalarBackward
    pub fn new(input_id: TensorId, input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for AddScalarBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        // Gradient passes through unchanged
        Ok(vec![Some(grad_output.clone())])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Gradient passes through unchanged — clone the Var itself.
        // Rebuilding it from `grad_output.tensor()` with `Var::new` would drop
        // the grad_fn and stop second-order backward here.
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<std::sync::Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "AddScalarBackward"
    }
}

// ============================================================================
// SubScalarBackward
// ============================================================================

/// Backward for tensor - scalar: z = a - scalar
///
/// Gradient: dL/da = dL/dz (pass through)
pub struct SubScalarBackward<R: Runtime> {
    input_id: TensorId,
    input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SubScalarBackward<R> {
    /// Create a new SubScalarBackward
    pub fn new(input_id: TensorId, input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for SubScalarBackward<R> {
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        // Gradient passes through unchanged
        Ok(vec![Some(grad_output.clone())])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        // Gradient passes through unchanged — clone the Var itself.
        // Rebuilding it from `grad_output.tensor()` with `Var::new` would drop
        // the grad_fn and stop second-order backward here.
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<std::sync::Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "SubScalarBackward"
    }
}

// ============================================================================
// MulScalarBackward
// ============================================================================

/// Backward for tensor * scalar: z = a * scalar
///
/// Gradient: dL/da = dL/dz * scalar
pub struct MulScalarBackward<R: Runtime> {
    input_id: TensorId,
    scalar: f64,
    input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> MulScalarBackward<R> {
    /// Create a new MulScalarBackward
    pub fn new(
        input_id: TensorId,
        scalar: f64,
        input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            scalar,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for MulScalarBackward<R>
where
    R::Client: ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        // dL/da = dL/dz * scalar
        let grad = client.mul_scalar(grad_output, self.scalar)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        // dL/da = dL/dz * scalar
        // Use var_mul_scalar to track gradients through grad_output
        let grad = var_mul_scalar(grad_output, self.scalar, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<std::sync::Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "MulScalarBackward"
    }
}

// ============================================================================
// DivScalarBackward
// ============================================================================

/// Backward for tensor / scalar: z = a / scalar
///
/// Gradient: dL/da = dL/dz / scalar
pub struct DivScalarBackward<R: Runtime> {
    input_id: TensorId,
    scalar: f64,
    input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> DivScalarBackward<R> {
    /// Create a new DivScalarBackward
    pub fn new(
        input_id: TensorId,
        scalar: f64,
        input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            scalar,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for DivScalarBackward<R>
where
    R::Client: ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        // dL/da = dL/dz / scalar
        let grad = client.div_scalar(grad_output, self.scalar)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        // dL/da = dL/dz / scalar
        // Use var_div_scalar to track gradients through grad_output
        let grad = var_div_scalar(grad_output, self.scalar, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<std::sync::Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "DivScalarBackward"
    }
}

// ============================================================================
// PowScalarBackward
// ============================================================================

/// Backward for tensor ^ scalar: z = a ^ scalar
///
/// Gradient: dL/da = dL/dz * scalar * a^(scalar-1)
pub struct PowScalarBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    scalar: f64,
    input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> PowScalarBackward<R> {
    /// Create a new PowScalarBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        scalar: f64,
        input_grad_fn: Option<std::sync::Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            scalar,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for PowScalarBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        // dL/da = dL/dz * scalar * a^(scalar-1)
        // = grad_output * scalar * a^(scalar-1)
        let a_pow_n_minus_1 = client.pow_scalar(&self.saved_input, self.scalar - 1.0)?;
        let scaled = client.mul_scalar(&a_pow_n_minus_1, self.scalar)?;
        let grad = client.mul(grad_output, &scaled)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R>,
    {
        use crate::autograd::var_ops::var_pow_scalar;

        let client = R::default_client(grad_output.tensor().device());
        // dL/da = dL/dz * scalar * a^(scalar-1)
        // = grad_output * scalar * a^(scalar-1)

        // Wrap saved input as Var with original input ID to track second-order gradients
        let a_var = Var::with_id(self.saved_input.clone(), self.input_id, true);

        // Compute a^(scalar-1) using var_pow_scalar to track gradients through a
        let a_pow_n_minus_1 = var_pow_scalar(&a_var, self.scalar - 1.0, &client)?;

        // Multiply by scalar
        let scaled = var_mul_scalar(&a_pow_n_minus_1, self.scalar, &client)?;

        // Multiply grad_output by scaled
        let grad = var_mul(grad_output, &scaled, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<std::sync::Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "PowScalarBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_add_scalar_backward() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = AddScalarBackward::<CpuRuntime>::new(a.id(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mul_scalar_backward() {
        let device = CpuDevice::new();

        // z = a * 3, dz/da = 3
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = MulScalarBackward::<CpuRuntime>::new(a.id(), 3.0, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_div_scalar_backward() {
        let device = CpuDevice::new();

        // z = a / 2, dz/da = 1/2 = 0.5
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32, 6.0, 8.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = DivScalarBackward::<CpuRuntime>::new(a.id(), 2.0, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_pow_scalar_backward() {
        let device = CpuDevice::new();

        // z = a^2, dz/da = 2*a
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0, 4.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = PowScalarBackward::<CpuRuntime>::new(a.id(), a.clone(), 2.0, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // dz/da = 2 * a = [4, 6, 8]
        assert_eq!(grad_a, vec![4.0, 6.0, 8.0]);
    }
}
