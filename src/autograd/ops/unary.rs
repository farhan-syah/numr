//! Backward implementations for unary operations
//!
//! These implement the gradient computation for unary operations.

use crate::autograd::{
    GradFn, Var, var_abs, var_cos, var_div, var_mul, var_mul_scalar, var_neg, var_sin, var_square,
    var_sub,
};
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, CompareOps, ScalarOps, TensorOps, UnaryOps};
use crate::runtime::{Runtime, RuntimeClient};
use crate::tensor::{Tensor, TensorId};
use std::sync::Arc;

// ============================================================================
// NegBackward
// ============================================================================

/// Backward for negation: z = -a
///
/// Gradient: dL/da = -dL/dz
pub struct NegBackward<R: Runtime> {
    input_id: TensorId,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    _marker: std::marker::PhantomData<R>,
}

impl<R: Runtime> NegBackward<R> {
    /// Create a new NegBackward
    pub fn new(input_id: TensorId, input_grad_fn: Option<Arc<dyn GradFn<R>>>) -> Self {
        Self {
            input_id,
            input_grad_fn,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<R: Runtime> GradFn<R> for NegBackward<R>
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
        let grad = client.neg(grad_output)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        let grad = var_neg(grad_output, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

// ============================================================================
// ExpBackward
// ============================================================================

/// Backward for exponential: z = exp(a)
///
/// Gradient: dL/da = dL/dz * exp(a) = dL/dz * z
pub struct ExpBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // exp(a)
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ExpBackward<R> {
    /// Create a new ExpBackward
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

impl<R: Runtime> GradFn<R> for ExpBackward<R>
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
        // dL/da = dL/dz * exp(a) = grad_output * saved_output
        let grad = client.mul(grad_output, &self.saved_output)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved output (exp(a)) - output doesn't need original ID tracking
        let output_var = Var::new(self.saved_output.clone(), false);
        // dL/da = grad_output * exp(a)
        let grad = var_mul(grad_output, &output_var, &client)?;
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
        "ExpBackward"
    }
}

// ============================================================================
// LogBackward
// ============================================================================

/// Backward for natural logarithm: z = log(a)
///
/// Gradient: dL/da = dL/dz / a
pub struct LogBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> LogBackward<R> {
    /// Create a new LogBackward
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

impl<R: Runtime> GradFn<R> for LogBackward<R>
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
        // dL/da = dL/dz / a
        let grad = client.div(grad_output, &self.saved_input)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID for second-order differentiation
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // dL/da = grad_output / a
        let grad = var_div(grad_output, &input_var, &client)?;
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
        "LogBackward"
    }
}

// ============================================================================
// SqrtBackward
// ============================================================================

/// Backward for square root: z = sqrt(a)
///
/// Gradient: dL/da = dL/dz / (2 * sqrt(a)) = dL/dz / (2 * z)
pub struct SqrtBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // sqrt(a)
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SqrtBackward<R> {
    /// Create a new SqrtBackward
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

impl<R: Runtime> GradFn<R> for SqrtBackward<R>
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
        // dL/da = dL/dz / (2 * sqrt(a))
        // = grad_output / (2 * saved_output)
        let two_sqrt = client.mul_scalar(&self.saved_output, 2.0)?;
        let grad = client.div(grad_output, &two_sqrt)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved output (sqrt(a))
        let output_var = Var::new(self.saved_output.clone(), false);
        // dL/da = grad_output / (2 * sqrt(a))
        let two_sqrt = var_mul_scalar(&output_var, 2.0, &client)?;
        let grad = var_div(grad_output, &two_sqrt, &client)?;
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
        "SqrtBackward"
    }
}

// ============================================================================
// SinBackward
// ============================================================================

/// Backward for sine: z = sin(a)
///
/// Gradient: dL/da = dL/dz * cos(a)
pub struct SinBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SinBackward<R> {
    /// Create a new SinBackward
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

impl<R: Runtime> GradFn<R> for SinBackward<R>
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
        let cos_a = client.cos(&self.saved_input)?;
        let grad = client.mul(grad_output, &cos_a)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // dL/da = grad_output * cos(a)
        let cos_a = var_cos(&input_var, &client)?;
        let grad = var_mul(grad_output, &cos_a, &client)?;
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
        "SinBackward"
    }
}

// ============================================================================
// CosBackward
// ============================================================================

/// Backward for cosine: z = cos(a)
///
/// Gradient: dL/da = -dL/dz * sin(a)
pub struct CosBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> CosBackward<R> {
    /// Create a new CosBackward
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

impl<R: Runtime> GradFn<R> for CosBackward<R>
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
        let sin_a = client.sin(&self.saved_input)?;
        let neg_sin = client.neg(&sin_a)?;
        let grad = client.mul(grad_output, &neg_sin)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // dL/da = grad_output * (-sin(a))
        let sin_a = var_sin(&input_var, &client)?;
        let neg_sin = var_neg(&sin_a, &client)?;
        let grad = var_mul(grad_output, &neg_sin, &client)?;
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
        "CosBackward"
    }
}

// ============================================================================
// TanhBackward
// ============================================================================

/// Backward for hyperbolic tangent: z = tanh(a)
///
/// Gradient: dL/da = dL/dz * (1 - tanh²(a)) = dL/dz * (1 - z²)
pub struct TanhBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // tanh(a)
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> TanhBackward<R> {
    /// Create a new TanhBackward
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

impl<R: Runtime<DType = DType>> GradFn<R> for TanhBackward<R>
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
        // dL/da = dL/dz * (1 - tanh²(a))
        let tanh_squared = client.square(&self.saved_output)?;
        let one = Tensor::<R>::ones(
            self.saved_output.shape(),
            self.saved_output.dtype(),
            self.saved_output.device(),
        )?;
        let one_minus_tanh2 = client.sub(&one, &tanh_squared)?;
        let grad = client.mul(grad_output, &one_minus_tanh2)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved output (tanh(a))
        let output_var = Var::new(self.saved_output.clone(), false);
        // dL/da = grad_output * (1 - tanh²(a))
        let tanh_squared = var_square(&output_var, &client)?;
        let one = Tensor::<R>::ones(
            self.saved_output.shape(),
            self.saved_output.dtype(),
            self.saved_output.device(),
        )?;
        let one_var = Var::new(one, false);
        let one_minus_tanh2 = var_sub(&one_var, &tanh_squared, &client)?;
        let grad = var_mul(grad_output, &one_minus_tanh2, &client)?;
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
        "TanhBackward"
    }
}

// ============================================================================
// SquareBackward
// ============================================================================

/// Backward for square: z = a²
///
/// Gradient: dL/da = dL/dz * 2 * a
pub struct SquareBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> SquareBackward<R> {
    /// Create a new SquareBackward
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

impl<R: Runtime> GradFn<R> for SquareBackward<R>
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
        // dL/da = dL/dz * 2 * a
        let two_a = client.mul_scalar(&self.saved_input, 2.0)?;
        let grad = client.mul(grad_output, &two_a)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // dL/da = grad_output * 2 * a
        let two_a = var_mul_scalar(&input_var, 2.0, &client)?;
        let grad = var_mul(grad_output, &two_a, &client)?;
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
        "SquareBackward"
    }
}

// ============================================================================
// RecipBackward
// ============================================================================

/// Backward for reciprocal: z = 1/a
///
/// Gradient: dL/da = -dL/dz / a² = -dL/dz * z²
pub struct RecipBackward<R: Runtime> {
    input_id: TensorId,
    saved_output: Tensor<R>, // 1/a
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> RecipBackward<R> {
    /// Create a new RecipBackward
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

impl<R: Runtime> GradFn<R> for RecipBackward<R>
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
        // dL/da = -dL/dz * z²
        let z_squared = client.square(&self.saved_output)?;
        let neg_grad = client.neg(grad_output)?;
        let grad = client.mul(&neg_grad, &z_squared)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved output (1/a)
        let output_var = Var::new(self.saved_output.clone(), false);
        // dL/da = -grad_output * z²
        let z_squared = var_square(&output_var, &client)?;
        let neg_grad = var_neg(grad_output, &client)?;
        let grad = var_mul(&neg_grad, &z_squared, &client)?;
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
        "RecipBackward"
    }
}

// ============================================================================
// TanBackward
// ============================================================================

/// Backward for tangent: z = tan(a)
///
/// Gradient: dL/da = dL/dz * sec²(a) = dL/dz / cos²(a)
pub struct TanBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> TanBackward<R> {
    /// Create a new TanBackward
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

impl<R: Runtime> GradFn<R> for TanBackward<R>
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
        // dL/da = dL/dz / cos²(a)
        let cos_a = client.cos(&self.saved_input)?;
        let cos_squared = client.square(&cos_a)?;
        let grad = client.div(grad_output, &cos_squared)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // dL/da = grad_output / cos²(a)
        let cos_a = var_cos(&input_var, &client)?;
        let cos_squared = var_square(&cos_a, &client)?;
        let grad = var_div(grad_output, &cos_squared, &client)?;
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
        "TanBackward"
    }
}

// ============================================================================
// AbsBackward
// ============================================================================

/// Backward for absolute value: z = |a|
///
/// Gradient: dL/da = dL/dz * sign(a)
/// where sign(a) = 1 if a > 0, -1 if a < 0, 0 if a = 0
pub struct AbsBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> AbsBackward<R> {
    /// Create a new AbsBackward
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

impl<R: Runtime> GradFn<R> for AbsBackward<R>
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

        // Compute sign(a) by dividing a by |a|
        // sign = a / abs(a) where abs(a) != 0
        // For a = 0, this results in NaN, but the gradient should be 0 (subgradient)
        let abs_a = client.abs(&self.saved_input)?;
        let grad_sign = client.div(&self.saved_input, &abs_a)?;
        let grad = client.mul(grad_output, &grad_sign)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>> {
        let client = R::default_client(grad_output.tensor().device());
        // Wrap saved input with original ID
        let input_var = Var::with_id(self.saved_input.clone(), self.input_id, true);
        // sign(a) = a / |a|
        let abs_a = var_abs(&input_var, &client)?;
        let grad_sign = var_div(&input_var, &abs_a, &client)?;
        let grad = var_mul(grad_output, &grad_sign, &client)?;
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
        "AbsBackward"
    }
}

// ============================================================================
// ClampBackward
// ============================================================================

/// Backward for clamp: z = clamp(a, min, max)
///
/// Gradient: dL/da = dL/dz where min < a < max, 0 otherwise
/// The gradient flows through only where the value is within the clamped range.
pub struct ClampBackward<R: Runtime> {
    input_id: TensorId,
    saved_input: Tensor<R>,
    min_val: f64,
    max_val: f64,
    input_grad_fn: Option<Arc<dyn GradFn<R>>>,
}

impl<R: Runtime> ClampBackward<R> {
    /// Create a new ClampBackward
    pub fn new(
        input_id: TensorId,
        input: Tensor<R>,
        min_val: f64,
        max_val: f64,
        input_grad_fn: Option<Arc<dyn GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_input: input,
            min_val,
            max_val,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> GradFn<R> for ClampBackward<R>
where
    R::Client: TensorOps<R> + ScalarOps<R> + CompareOps<R>,
{
    fn backward(
        &self,
        grad_output: &Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());

        // Create mask where min < a < max (gradient flows through)
        // gradient is zero at boundaries
        let min_tensor = Tensor::<R>::full_scalar(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.min_val,
            self.saved_input.device(),
        )?;
        let max_tensor = Tensor::<R>::full_scalar(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.max_val,
            self.saved_input.device(),
        )?;

        // Check if value is greater than min
        let gt_min = client.gt(&self.saved_input, &min_tensor)?;
        // Check if value is less than max
        let lt_max = client.lt(&self.saved_input, &max_tensor)?;

        // Combine: gradient flows where min < a < max
        let mask = client.mul(&gt_min, &lt_max)?;

        // Apply mask to gradient
        let grad = client.mul(grad_output, &mask)?;

        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R> + ScalarOps<R> + CompareOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());

        // Create mask where min < a < max (gradient flows through)
        // The mask is non-differentiable (step function), so treat as constant
        let min_tensor = Tensor::<R>::full_scalar(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.min_val,
            self.saved_input.device(),
        )?;
        let max_tensor = Tensor::<R>::full_scalar(
            self.saved_input.shape(),
            self.saved_input.dtype(),
            self.max_val,
            self.saved_input.device(),
        )?;

        // Check if value is greater than min
        let gt_min = client.gt(&self.saved_input, &min_tensor)?;
        // Check if value is less than max
        let lt_max = client.lt(&self.saved_input, &max_tensor)?;

        // Combine: gradient flows where min < a < max
        let mask = client.mul(&gt_min, &lt_max)?;

        // Wrap mask as Var without gradient tracking (non-differentiable)
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
        "ClampBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_neg_backward() {
        let device = CpuDevice::new();
        let a = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device).unwrap();
        let grad_out = Tensor::<CpuRuntime>::ones(&[3], DType::F32, &device).unwrap();

        let backward = NegBackward::<CpuRuntime>::new(a.id(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert_eq!(grad_a, vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_exp_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = exp(a), dz/da = exp(a)
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap(); // exp(0) = 1
        let output = client.exp(&a).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = ExpBackward::<CpuRuntime>::new(a.id(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 1.0).abs() < 1e-6); // exp(0) = 1
    }

    #[test]
    fn test_log_backward() {
        let device = CpuDevice::new();

        // z = log(a), dz/da = 1/a
        let a = Tensor::<CpuRuntime>::from_slice(&[2.0f32], &[1], &device).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = LogBackward::<CpuRuntime>::new(a.id(), a.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 0.5).abs() < 1e-6); // 1/2 = 0.5
    }

    #[test]
    fn test_sqrt_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = sqrt(a), dz/da = 1/(2*sqrt(a))
        let a = Tensor::<CpuRuntime>::from_slice(&[4.0f32], &[1], &device).unwrap();
        let output = client.sqrt(&a).unwrap(); // sqrt(4) = 2

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SqrtBackward::<CpuRuntime>::new(a.id(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 0.25).abs() < 1e-6); // 1/(2*2) = 0.25
    }

    #[test]
    fn test_tanh_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        // z = tanh(a), dz/da = 1 - tanh²(a)
        // At a = 0, tanh(0) = 0, so dz/da = 1 - 0 = 1
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();
        let output = client.tanh(&a).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = TanhBackward::<CpuRuntime>::new(a.id(), output, None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_square_backward() {
        let device = CpuDevice::new();

        // z = a², dz/da = 2a
        let a = Tensor::<CpuRuntime>::from_slice(&[3.0f32], &[1], &device).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = SquareBackward::<CpuRuntime>::new(a.id(), a.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 6.0).abs() < 1e-6); // 2 * 3 = 6
    }

    #[test]
    fn test_tan_backward() {
        let device = CpuDevice::new();

        // z = tan(a), dz/da = 1/cos²(a)
        // At a = 0, cos(0) = 1, so dz/da = 1
        let a = Tensor::<CpuRuntime>::from_slice(&[0.0f32], &[1], &device).unwrap();

        let grad_out = Tensor::<CpuRuntime>::ones(&[1], DType::F32, &device).unwrap();

        let backward = TanBackward::<CpuRuntime>::new(a.id(), a.clone(), None);
        let grads = backward.backward_all(&grad_out).unwrap();

        let grad_a: Vec<f32> = grads[0].as_ref().unwrap().to_vec();
        assert!((grad_a[0] - 1.0).abs() < 1e-6); // 1/cos²(0) = 1/1 = 1
    }
}
