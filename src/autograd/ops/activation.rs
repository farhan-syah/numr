//! Backward implementations for activation functions
//!
//! Implements gradient computation for relu, sigmoid, silu, softplus, softmax, and log_softmax.

use crate::autograd::GradFn;
use crate::autograd::var::Var;
use crate::autograd::var_ops::{var_mul, var_sub, var_sum};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{ActivationOps, BinaryOps, CompareOps, ReduceOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// ReluBackward
// ============================================================================

/// Backward for ReLU: z = max(0, a)
///
/// Gradient: dL/da = dL/dz * (a > 0)
/// The gradient passes through where input was positive, zero elsewhere.
pub struct ReluBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ReluBackward<R> {
    /// Create a new ReluBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for ReluBackward<R>
where
    R::Client: TensorOps<R> + CompareOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // ReLU derivative: relu'(x) = 1 if x > 0, 0 otherwise
        // mask = (input > 0) -> returns 1.0 where true, 0.0 where false
        let zero = Tensor::<R>::zeros(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.saved_input.device(),
        )?;
        let mask = client.gt(&self.saved_input, &zero)?;

        // grad = grad_output * mask
        let grad = client.mul(grad_output, &mask)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + CompareOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // ReLU derivative: relu'(x) = 1 if x > 0, 0 otherwise
        // The mask is non-differentiable (step function), so treat as constant
        let zero = Tensor::<R>::zeros(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.saved_input.device(),
        )?;
        let mask = client.gt(&self.saved_input, &zero)?;

        // Wrap mask as Var without gradient tracking
        let mask_var = Var::new(mask, false);

        // grad = grad_output * mask using var_mul to track gradients through grad_output
        let grad = var_mul(grad_output, &mask_var, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

// ============================================================================
// SigmoidBackward
// ============================================================================

/// Backward for sigmoid: z = 1 / (1 + exp(-a))
///
/// Gradient: dL/da = dL/dz * z * (1 - z)
pub struct SigmoidBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // sigmoid(a)
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SigmoidBackward<R> {
    /// Create a new SigmoidBackward
    pub fn new(
        input_id: TensorId,
        output: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_output: output,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for SigmoidBackward<R>
where
    R::Client: TensorOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // dL/da = dL/dz * z * (1 - z)
        let one = Tensor::<R>::ones(
            self.saved_output.shape(),
            self.saved_output.dtype(),
            self.saved_output.device(),
        )?;
        let one_minus_sigmoid = client.sub(&one, &self.saved_output)?;
        let sigmoid_deriv = client.mul(&self.saved_output, &one_minus_sigmoid)?;
        let grad = client.mul(grad_output, &sigmoid_deriv)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        // dL/da = dL/dz * z * (1 - z)
        // The derivative z * (1 - z) is computed from saved output, treated as constant
        let one = Tensor::<R>::ones(
            self.saved_output.shape(),
            self.saved_output.dtype(),
            self.saved_output.device(),
        )?;
        let one_minus_sigmoid = client.sub(&one, &self.saved_output)?;
        let sigmoid_deriv = client.mul(&self.saved_output, &one_minus_sigmoid)?;

        // Wrap derivative as Var without gradient tracking (constant w.r.t. grad_output)
        let deriv_var = Var::new(sigmoid_deriv, false);

        // grad = grad_output * sigmoid_deriv using var_mul
        let grad = var_mul(grad_output, &deriv_var, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_output)
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

// ============================================================================
// SiluBackward
// ============================================================================

/// Backward for SiLU (Swish): z = a * sigmoid(a)
///
/// Gradient: dL/da = dL/dz * (sigmoid(a) + a * sigmoid(a) * (1 - sigmoid(a)))
///         = dL/dz * (sigmoid(a) * (1 + a * (1 - sigmoid(a))))
///         = dL/dz * (z/a * (1 + a - z))  [numerically: use saved input + output]
pub struct SiluBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    saved_output: Tensor<R>, // silu(a)
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SiluBackward<R> {
    /// Create a new SiluBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        output: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            saved_output: output,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for SiluBackward<R>
where
    R::Client: TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // silu'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
        //          = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        //          = sigmoid(x) * (1 + x - x*sigmoid(x))
        //          = sigmoid(x) * (1 + x - silu(x))
        let sigmoid = client.sigmoid(&self.saved_input)?;
        let one_plus_x = client.add_scalar(&self.saved_input, 1.0)?;
        let one_plus_x_minus_silu = client.sub(&one_plus_x, &self.saved_output)?;
        let deriv = client.mul(&sigmoid, &one_plus_x_minus_silu)?;
        let grad = client.mul(grad_output, &deriv)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ActivationOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let sigmoid = client.sigmoid(&self.saved_input)?;
        let one_plus_x = client.add_scalar(&self.saved_input, 1.0)?;
        let one_plus_x_minus_silu = client.sub(&one_plus_x, &self.saved_output)?;
        let deriv = client.mul(&sigmoid, &one_plus_x_minus_silu)?;

        let deriv_var = Var::new(deriv, false);
        let grad = var_mul(grad_output, &deriv_var, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        // Both saved_input and saved_output are stored internally for gradient computation.
        // The trait returns a slice, so we expose only the input here; saved_output is
        // accessed directly during backward() and backward_var().
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "SiluBackward"
    }
}

// ============================================================================
// SoftmaxBackward
// ============================================================================

/// Backward for softmax along a dimension: z_i = exp(a_i) / sum(exp(a))
///
/// Gradient: dL/da_i = z_i * (dL/dz_i - sum_j(z_j * dL/dz_j))
/// Simplified: dL/da = z * (dL/dz - sum(z * dL/dz, dim=dim))
pub struct SoftmaxBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // softmax(a)
    dim: isize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SoftmaxBackward<R> {
    /// Create a new SoftmaxBackward
    pub fn new(
        input_id: TensorId,
        output: Tensor<R>,
        dim: isize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_output: output,
            dim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime> GradFn<R> for SoftmaxBackward<R>
where
    R::Client: TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // Normalize dim
        let ndim = self.saved_output.ndim();
        let dim_idx = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // softmax gradient: grad_input = softmax * (grad_output - sum(softmax * grad_output, dim))
        // = z * (dy - sum(z * dy, dim))
        let z_dy = client.mul(&self.saved_output, grad_output)?;
        let sum_z_dy = client.sum(&z_dy, &[dim_idx], true)?;
        let dy_minus_sum = client.sub(grad_output, &sum_z_dy)?;
        let grad = client.mul(&self.saved_output, &dy_minus_sum)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ReduceOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // Normalize dim
        let ndim = self.saved_output.ndim();
        let dim_idx = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // softmax gradient: grad_input = softmax * (grad_output - sum(softmax * grad_output, dim))
        // = z * (dy - sum(z * dy, dim))

        // Wrap softmax output as Var without gradient (constant w.r.t. grad_output)
        let z_var = Var::new(self.saved_output.clone(), false);

        // z * dy
        let z_dy = var_mul(&z_var, grad_output, &client)?;

        // sum(z * dy, dim)
        let sum_z_dy = var_sum(&z_dy, &[dim_idx], true, &client)?;

        // dy - sum(z * dy, dim)
        let dy_minus_sum = var_sub(grad_output, &sum_z_dy, &client)?;

        // z * (dy - sum(z * dy, dim))
        let grad = var_mul(&z_var, &dy_minus_sum, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_output)
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ============================================================================
// LogSoftmaxBackward
// ============================================================================

/// Backward for log_softmax: z = log(softmax(a, dim))
///
/// Gradient: dL/da = dL/dz - softmax(a) * sum(dL/dz, dim)
///         = dL/dz - exp(z) * sum(dL/dz, dim)
pub struct LogSoftmaxBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // log_softmax(a)
    dim: isize,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> LogSoftmaxBackward<R> {
    /// Create a new LogSoftmaxBackward
    pub fn new(
        input_id: TensorId,
        output: Tensor<R>,
        dim: isize,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_output: output,
            dim,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for LogSoftmaxBackward<R>
where
    R::Client: TensorOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        let ndim = self.saved_output.ndim();
        let dim_idx = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // log_softmax gradient: grad_input = grad_output - exp(output) * sum(grad_output, dim)
        let softmax_output = client.exp(&self.saved_output)?;
        let sum_grad = client.sum(grad_output, &[dim_idx], true)?;
        let softmax_sum = client.mul(&softmax_output, &sum_grad)?;
        let grad = client.sub(grad_output, &softmax_sum)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + UnaryOps<R> + ReduceOps<R> + ScalarOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let ndim = self.saved_output.ndim();
        let dim_idx = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // exp(log_softmax(x)) = softmax(x), treated as constant
        let softmax_output = client.exp(&self.saved_output)?;
        let softmax_var = Var::new(softmax_output, false);

        let sum_grad = var_sum(grad_output, &[dim_idx], true, &client)?;
        let softmax_sum = var_mul(&softmax_var, &sum_grad, &client)?;
        let grad = var_sub(grad_output, &softmax_sum, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_output)
    }

    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }
}

// ============================================================================
// SoftplusBackward
// ============================================================================

/// Backward for softplus: `z = log(1 + exp(a))`
///
/// Gradient: `dL/da = dL/dz * sigmoid(a)`
///
/// `d/da log(1 + exp(a)) = exp(a) / (1 + exp(a)) = sigmoid(a)`
///
/// The backward is numerically stable since `sigmoid` is bounded in `(0, 1)`.
/// The forward must be computed via the stable form `relu(a) + log(1 + exp(-|a|))`
/// (see `softplus_impl`) — never the naive `log(1 + exp(a))` which overflows for
/// large positive inputs.
pub struct SoftplusBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SoftplusBackward<R> {
    /// Create a new SoftplusBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for SoftplusBackward<R>
where
    R::Client: TensorOps<R> + ActivationOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // softplus'(x) = sigmoid(x)
        let sigmoid = client.sigmoid(&self.saved_input)?;
        let grad = client.mul(grad_output, &sigmoid)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ActivationOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        let sigmoid = client.sigmoid(&self.saved_input)?;
        let sigmoid_var = Var::new(sigmoid, false);
        let grad = var_mul(grad_output, &sigmoid_var, &client)?;

        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[Tensor<R>] {
        std::slice::from_ref(&self.saved_input)
    }

    fn name(&self) -> &'static str {
        "SoftplusBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_relu_backward_positive() {
        let device = CpuDevice::new();

        // Input with positive values
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = ReluBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // All positive inputs, so gradient passes through
        for val in grad_data {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_relu_backward_mixed() {
        let device = CpuDevice::new();

        // Input with mixed values
        let input = Tensor::<CpuRuntime>::from_slice(&[-1.0f32, 0.0, 2.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = ReluBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // grad[0] should be ~0 (negative input)
        // grad[1] should be ~0 (zero input)
        // grad[2] should be ~1 (positive input)
        assert!(grad_data[0].abs() < 1e-5);
        assert!(grad_data[1].abs() < 1e-5);
        assert!((grad_data[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // sigmoid(0) = 0.5
        // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let output = client.sigmoid(&input).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SigmoidBackward::<CpuRuntime>::new(input.id(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_data[0] - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_silu_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // silu(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        // silu'(0) = sigmoid(0) * (1 + 0 * (1 - sigmoid(0))) = 0.5 * 1 = 0.5
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let output = client.silu(&input).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SiluBackward::<CpuRuntime>::new(input.id(), input.clone(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_silu_backward_nonzero() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // silu(1) = 1 * sigmoid(1) ≈ 0.7311
        // silu'(1) = sigmoid(1) * (1 + 1 * (1 - sigmoid(1)))
        //          ≈ 0.7311 * (1 + 1 * 0.2689) ≈ 0.7311 * 1.2689 ≈ 0.9277
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32], &[1], &device).unwrap();
        let output = client.silu(&input).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SiluBackward::<CpuRuntime>::new(input.id(), input.clone(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let sigmoid_1 = 1.0f32 / (1.0 + (-1.0f32).exp());
        let expected = sigmoid_1 * (1.0 + 1.0 * (1.0 - sigmoid_1));
        assert!((grad_data[0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_silu_backward_2d() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Shape [2, 3] — verifies element-wise gradient correctness on batched tensors.
        // silu'(x) = sigmoid(x) * (1 + x - silu(x))
        let data = [-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5];
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();
        let output = client.silu(&input).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device).unwrap();

        let backward =
            SiluBackward::<CpuRuntime>::new(input.id(), input.clone(), output.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let out_data: Vec<f32> = output.to_vec();

        for (i, &x) in data.iter().enumerate() {
            let sigmoid_x = 1.0f32 / (1.0 + (-x).exp());
            let expected = sigmoid_x * (1.0 + x - out_data[i]);
            assert!(
                (grad_data[i] - expected).abs() < 1e-5,
                "mismatch at index {i}: got {}, expected {expected}",
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_silu_backward_negative_gradient() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Verify chain rule: grad_output scales the derivative correctly.
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0], &[2], &device).unwrap();
        let output = client.silu(&input).unwrap();

        // grad_output = [2.0, 3.0] — non-unit upstream gradient
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap();

        let backward =
            SiluBackward::<CpuRuntime>::new(input.id(), input.clone(), output.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let out_data: Vec<f32> = output.to_vec();
        let upstream = [2.0f32, 3.0];

        for (i, (&x, &up)) in [1.0f32, -1.0].iter().zip(upstream.iter()).enumerate() {
            let sigmoid_x = 1.0f32 / (1.0 + (-x).exp());
            let local_deriv = sigmoid_x * (1.0 + x - out_data[i]);
            let expected = up * local_deriv;
            assert!(
                (grad_data[i] - expected).abs() < 1e-5,
                "mismatch at index {i}: got {}, expected {expected}",
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_softplus_backward() {
        let device = CpuDevice::new();

        // softplus(0) = log(1 + exp(0)) = log(2) ≈ 0.6931
        // softplus'(0) = sigmoid(0) = 0.5
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_data[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_softplus_backward_nonzero() {
        let device = CpuDevice::new();

        // softplus'(x) = sigmoid(x)
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, -1.0, 2.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        for (i, &x) in [1.0f32, -1.0, 2.0].iter().enumerate() {
            let expected = 1.0 / (1.0 + (-x).exp());
            assert!(
                (grad_data[i] - expected).abs() < 1e-5,
                "mismatch at {i}: got {}, expected {expected}",
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_softplus_backward_large_positive() {
        let device = CpuDevice::new();

        // For large positive x, sigmoid(x) → 1.0; must not produce NaN.
        // This exercises the numerical stability of the stable softplus formula.
        let input = Tensor::<CpuRuntime>::from_slice(&[100.0f32], &[1], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!(
            !grad_data[0].is_nan(),
            "gradient must not be NaN for large positive input"
        );
        assert!(
            !grad_data[0].is_infinite(),
            "gradient must not be Inf for large positive input"
        );
        // sigmoid(100) ≈ 1.0
        assert!((grad_data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_softplus_backward_large_negative() {
        let device = CpuDevice::new();

        // For large negative x, sigmoid(x) → 0.0; must not produce NaN.
        let input = Tensor::<CpuRuntime>::from_slice(&[-100.0f32], &[1], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!(
            !grad_data[0].is_nan(),
            "gradient must not be NaN for large negative input"
        );
        assert!(
            !grad_data[0].is_infinite(),
            "gradient must not be Inf for large negative input"
        );
        // sigmoid(-100) ≈ 0.0
        assert!(grad_data[0].abs() < 1e-5);
    }

    #[test]
    fn test_softplus_backward_2d() {
        let device = CpuDevice::new();

        // Shape [2, 3] — verifies element-wise gradient on batched tensors.
        let data = [-2.0f32, -1.0, 0.0, 1.0, 2.0, 100.0];
        let input = Tensor::<CpuRuntime>::from_slice(&data, &[2, 3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[2, 3], DType::F32, &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        for (i, &x) in data.iter().enumerate() {
            let expected = 1.0f32 / (1.0 + (-x).exp());
            assert!(
                !grad_data[i].is_nan(),
                "gradient NaN at index {i} for x={x}"
            );
            assert!(
                (grad_data[i] - expected).abs() < 1e-4,
                "mismatch at index {i} for x={x}: got {}, expected {expected}",
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_softplus_backward_non_unit_gradient() {
        let device = CpuDevice::new();

        // Verify chain rule: upstream gradient scales local derivative.
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 1.0], &[2], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[2.0f32, 3.0], &[2], &device).unwrap();

        let backward = SoftplusBackward::<CpuRuntime>::new(input.id(), input, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        let upstream = [2.0f32, 3.0];
        for (i, (&x, &up)) in [0.0f32, 1.0].iter().zip(upstream.iter()).enumerate() {
            let sigmoid_x = 1.0f32 / (1.0 + (-x).exp());
            let expected = up * sigmoid_x;
            assert!(
                (grad_data[i] - expected).abs() < 1e-5,
                "mismatch at index {i}: got {}, expected {expected}",
                grad_data[i]
            );
        }
    }

    #[test]
    fn test_softmax_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple 2-element softmax
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let output = client.softmax(&input, -1).unwrap(); // [0.5, 0.5]

        // dL/dz = [1, 0] - only first element contributes to loss
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device).unwrap();

        let backward = SoftmaxBackward::<CpuRuntime>::new(input.id(), output, -1, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // For softmax, when z = [0.5, 0.5] and dy = [1, 0]:
        // grad = z * (dy - sum(z * dy))
        // sum(z * dy) = 0.5 * 1 + 0.5 * 0 = 0.5
        // grad[0] = 0.5 * (1 - 0.5) = 0.25
        // grad[1] = 0.5 * (0 - 0.5) = -0.25
        assert!((grad_data[0] - 0.25).abs() < 1e-6);
        assert!((grad_data[1] - (-0.25)).abs() < 1e-6);
    }

    #[test]
    fn test_log_softmax_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // Simple 2-element log_softmax
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32, 0.0], &[2], &device).unwrap();
        let output = client.log_softmax(&input, -1).unwrap(); // [ln(0.5), ln(0.5)]

        let output_data: Vec<f32> = output.to_vec();
        let expected_log = (0.5f32).ln();
        assert!((output_data[0] - expected_log).abs() < 1e-6);
        assert!((output_data[1] - expected_log).abs() < 1e-6);

        // dL/dz = [1, 0]
        let grad_out = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0], &[2], &device).unwrap();

        let backward = LogSoftmaxBackward::<CpuRuntime>::new(input.id(), output, -1, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_data: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        // log_softmax gradient: grad = dy - exp(z) * sum(dy, dim)
        // exp(z) = [0.5, 0.5], sum(dy) = 1.0
        // grad[0] = 1.0 - 0.5 * 1.0 = 0.5
        // grad[1] = 0.0 - 0.5 * 1.0 = -0.5
        assert!((grad_data[0] - 0.5).abs() < 1e-6);
        assert!((grad_data[1] - (-0.5)).abs() < 1e-6);
    }
}
