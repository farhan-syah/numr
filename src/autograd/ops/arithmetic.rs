//! Backward implementations for arithmetic operations
//!
//! These implement the gradient computation for basic binary operations.

use super::broadcast::{reduce_grad_for_broadcast, reduce_var_for_broadcast};
use crate::autograd::{GradFn, Var, var_div, var_log, var_mul, var_neg};
use crate::error::Result;
use crate::ops::{BinaryOps, TensorOps, UnaryOps};
use crate::runtime::Runtime;
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// AddBackward
// ============================================================================

/// Backward for element-wise addition: z = a + b
///
/// Gradients:
/// - dL/da = dL/dz (pass through)
/// - dL/db = dL/dz (pass through)
///
/// If broadcasting occurred, we need to sum over the broadcast dimensions.
pub struct AddBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> AddBackward<R> {
    /// Create a new AddBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a_shape: &[usize],
        b_shape: &[usize],
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.to_vec(),
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for AddBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        // For add, gradients are just the output gradient, possibly reduced
        // if broadcasting occurred
        let grad_a = reduce_grad_for_broadcast::<R>(grad_output, &self.a_shape)?;
        let grad_b = reduce_grad_for_broadcast::<R>(grad_output, &self.b_shape)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());

        // For add, gradients are just the output gradient (pass through)
        // No multiplication needed, so we just reduce for broadcasting
        let grad_a = reduce_var_for_broadcast(grad_output, &self.a_shape, &client)?;
        let grad_b = reduce_var_for_broadcast(grad_output, &self.b_shape, &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

// ============================================================================
// SubBackward
// ============================================================================

/// Backward for element-wise subtraction: z = a - b
///
/// Gradients:
/// - dL/da = dL/dz
/// - dL/db = -dL/dz
pub struct SubBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> SubBackward<R> {
    /// Create a new SubBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a_shape: &[usize],
        b_shape: &[usize],
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            a_shape: a_shape.to_vec(),
            b_shape: b_shape.to_vec(),
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for SubBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());

        // grad_a = grad_output
        let grad_a = reduce_grad_for_broadcast::<R>(grad_output, &self.a_shape)?;

        // grad_b = -grad_output
        let neg_grad = client.neg(grad_output)?;
        let grad_b = reduce_grad_for_broadcast::<R>(&neg_grad, &self.b_shape)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());

        // grad_a = grad_output (pass through)
        let grad_a = reduce_var_for_broadcast(grad_output, &self.a_shape, &client)?;

        // grad_b = -grad_output (using var_neg builds computation graph)
        let neg_grad = var_neg(grad_output, &client)?;
        let grad_b = reduce_var_for_broadcast(&neg_grad, &self.b_shape, &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

// ============================================================================
// MulBackward
// ============================================================================

/// Backward for element-wise multiplication: z = a * b
///
/// Gradients:
/// - dL/da = dL/dz * b
/// - dL/db = dL/dz * a
pub struct MulBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [a, b]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> MulBackward<R> {
    /// Create a new MulBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, b],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for MulBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // grad_a = grad_output * b
        let grad_a_full = client.mul(grad_output, saved_b)?;
        let grad_a = reduce_grad_for_broadcast::<R>(&grad_a_full, saved_a.shape())?;

        // grad_b = grad_output * a
        let grad_b_full = client.mul(grad_output, saved_a)?;
        let grad_b = reduce_grad_for_broadcast::<R>(&grad_b_full, saved_b.shape())?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // Wrap saved tensors as Vars with ORIGINAL input IDs AND grad_fns.
        // This is crucial for second-order differentiation: when we differentiate
        // the gradient computation, we need:
        // 1. gradients to accumulate to the original variable IDs
        // 2. the grad_fn chain to continue backward to even earlier inputs
        //
        // For example, if we computed y = (x + b)^2, then:
        // - saved_a = saved_b = x + b
        // - input_grad_fns[0] = AddBackward pointing to x and b
        // - When computing second derivatives, we need to continue from x+b to x
        let a_var = Var::with_id_and_grad_fn(
            saved_a.clone(),
            self.input_ids[0],
            self.input_grad_fns[0].clone(),
        );
        let b_var = Var::with_id_and_grad_fn(
            saved_b.clone(),
            self.input_ids[1],
            self.input_grad_fns[1].clone(),
        );

        // grad_a = grad_output * b (using var_mul builds computation graph)
        let grad_a_full = var_mul(grad_output, &b_var, &client)?;
        let grad_a = reduce_var_for_broadcast(&grad_a_full, saved_a.shape(), &client)?;

        // grad_b = grad_output * a (using var_mul builds computation graph)
        let grad_b_full = var_mul(grad_output, &a_var, &client)?;
        let grad_b = reduce_var_for_broadcast(&grad_b_full, saved_b.shape(), &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

// ============================================================================
// DivBackward
// ============================================================================

/// Backward for element-wise division: z = a / b
///
/// Gradients:
/// - dL/da = dL/dz / b
/// - dL/db = -dL/dz * a / b²
pub struct DivBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [a, b]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> DivBackward<R> {
    /// Create a new DivBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, b],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for DivBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // grad_a = grad_output / b
        let grad_a_full = client.div(grad_output, saved_b)?;
        let grad_a = reduce_grad_for_broadcast::<R>(&grad_a_full, saved_a.shape())?;

        // grad_b = -grad_output * a / b²
        // = -grad_output * a / (b * b)
        let neg_grad = client.neg(grad_output)?;
        let neg_grad_a = client.mul(&neg_grad, saved_a)?;
        let b_squared = client.mul(saved_b, saved_b)?;
        let grad_b_full = client.div(&neg_grad_a, &b_squared)?;
        let grad_b = reduce_grad_for_broadcast::<R>(&grad_b_full, saved_b.shape())?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];

        // Wrap saved tensors as Vars with original input IDs AND grad_fns
        // for proper second-order gradient chain propagation
        let a_var = Var::with_id_and_grad_fn(
            saved_a.clone(),
            self.input_ids[0],
            self.input_grad_fns[0].clone(),
        );
        let b_var = Var::with_id_and_grad_fn(
            saved_b.clone(),
            self.input_ids[1],
            self.input_grad_fns[1].clone(),
        );

        // grad_a = grad_output / b
        let grad_a_full = var_div(grad_output, &b_var, &client)?;
        let grad_a = reduce_var_for_broadcast(&grad_a_full, saved_a.shape(), &client)?;

        // grad_b = -grad_output * a / b²
        let neg_grad = var_neg(grad_output, &client)?;
        let neg_grad_a = var_mul(&neg_grad, &a_var, &client)?;
        let b_squared = var_mul(&b_var, &b_var, &client)?;
        let grad_b_full = var_div(&neg_grad_a, &b_squared, &client)?;
        let grad_b = reduce_var_for_broadcast(&grad_b_full, saved_b.shape(), &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

// ============================================================================
// PowBackward
// ============================================================================

/// Backward for element-wise power: z = a^b
///
/// Gradients:
/// - dL/da = dL/dz * b * a^(b-1)
/// - dL/db = dL/dz * a^b * ln(a)
pub struct PowBackward<R: Runtime> {
    input_ids: [TensorId; 2],
    saved_tensors: Vec<Tensor<R>>, // [a, b, output]
    input_grad_fns: [Option<Arc<dyn GradFn<R>>>; 2],
}

impl<R: Runtime> PowBackward<R> {
    /// Create a new PowBackward
    pub fn new(
        a_id: TensorId,
        b_id: TensorId,
        a: Tensor<R>,
        b: Tensor<R>,
        output: Tensor<R>,
        a_grad_fn: Option<Arc<dyn GradFn<R>>>,
        b_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_ids: [a_id, b_id],
            saved_tensors: vec![a, b, output],
            input_grad_fns: [a_grad_fn, b_grad_fn],
        }
    }
}

impl<R: Runtime> GradFn<R> for PowBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(&self, grad_output: &Tensor<R>) -> Result<Vec<Option<Tensor<R>>>> {
        let client = R::default_client(grad_output.device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];
        let saved_output = &self.saved_tensors[2];

        // grad_a = grad_output * b * a^(b-1)
        // = grad_output * b * (a^b / a)
        // = grad_output * b * output / a
        let grad_a_temp = client.mul(grad_output, saved_b)?;
        let grad_a_temp2 = client.mul(&grad_a_temp, saved_output)?;
        let grad_a_full = client.div(&grad_a_temp2, saved_a)?;
        let grad_a = reduce_grad_for_broadcast::<R>(&grad_a_full, saved_a.shape())?;

        // grad_b = grad_output * a^b * ln(a)
        // = grad_output * output * ln(a)
        let ln_a = client.log(saved_a)?;
        let grad_b_temp = client.mul(grad_output, saved_output)?;
        let grad_b_full = client.mul(&grad_b_temp, &ln_a)?;
        let grad_b = reduce_grad_for_broadcast::<R>(&grad_b_full, saved_b.shape())?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let saved_a = &self.saved_tensors[0];
        let saved_b = &self.saved_tensors[1];
        let saved_output = &self.saved_tensors[2];

        // Wrap saved tensors as Vars with original input IDs AND grad_fns
        // for proper second-order gradient chain propagation
        let a_var = Var::with_id_and_grad_fn(
            saved_a.clone(),
            self.input_ids[0],
            self.input_grad_fns[0].clone(),
        );
        let b_var = Var::with_id_and_grad_fn(
            saved_b.clone(),
            self.input_ids[1],
            self.input_grad_fns[1].clone(),
        );
        // Output is not an input, so it gets a fresh Var (no special ID needed)
        let output_var = Var::new(saved_output.clone(), false);

        // grad_a = grad_output * b * output / a
        let grad_a_temp = var_mul(grad_output, &b_var, &client)?;
        let grad_a_temp2 = var_mul(&grad_a_temp, &output_var, &client)?;
        let grad_a_full = var_div(&grad_a_temp2, &a_var, &client)?;
        let grad_a = reduce_var_for_broadcast(&grad_a_full, saved_a.shape(), &client)?;

        // grad_b = grad_output * output * ln(a)
        let ln_a = var_log(&a_var, &client)?;
        let grad_b_temp = var_mul(grad_output, &output_var, &client)?;
        let grad_b_full = var_mul(&grad_b_temp, &ln_a, &client)?;
        let grad_b = reduce_var_for_broadcast(&grad_b_full, saved_b.shape(), &client)?;

        Ok(vec![Some(grad_a), Some(grad_b)])
    }

    fn inputs(&self) -> &[TensorId] {
        &self.input_ids
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        self.input_grad_fns.to_vec()
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        &self.saved_tensors
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_add_backward_same_shape() {
        let device = CpuDevice::new();

        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32, 4.0], &[2], &device);

        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device);

        let backward =
            AddBackward::<CpuRuntime>::new(a.id(), b.id(), a.shape(), b.shape(), None, None);
        let grads = backward.backward(&grad_out).unwrap();

        assert_eq!(grads.len(), 2);

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let grad_b: Vec<f32> = grads[1].as_ref().unwrap().to_vec();

        assert_eq!(grad_a, vec![1.0, 1.0]);
        assert_eq!(grad_b, vec![1.0, 1.0]);
    }

    #[test]
    fn test_mul_backward() {
        let device = CpuDevice::new();

        // z = a * b where a = 2, b = 3
        // dz/da = b = 3
        // dz/db = a = 2
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device);

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device);

        let backward =
            MulBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let grad_b: Vec<f32> = grads[1].as_ref().unwrap().to_vec();

        assert_eq!(grad_a, vec![3.0]); // dz/da = b = 3
        assert_eq!(grad_b, vec![2.0]); // dz/db = a = 2
    }

    #[test]
    fn test_div_backward() {
        let device = CpuDevice::new();

        // z = a / b where a = 6, b = 2
        // z = 3
        // dz/da = 1/b = 0.5
        // dz/db = -a/b² = -6/4 = -1.5
        let a = Tensor::<CpuRuntime>::from_slice(&[6.0f32], &[1], &device);
        let b = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device);

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device);

        let backward =
            DivBackward::<CpuRuntime>::new(a.id(), b.id(), a.clone(), b.clone(), None, None);
        let grads = backward.backward(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let grad_b: Vec<f32> = grads[1].as_ref().unwrap().to_vec();

        assert!((grad_a[0] - 0.5).abs() < 1e-6);
        assert!((grad_b[0] - (-1.5)).abs() < 1e-6);
    }
}
