//! Unary operations (neg, exp, log, sqrt, sin, cos, tanh, square, recip, abs, tan)

use super::macros::*;
use super::ops::*;
use crate::autograd::Var;
use crate::error::Result;
use crate::ops::{ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};

impl_var_unary_op_id!(
    /// Negation: z = -a
    var_neg, neg, NegBackward
);

impl_var_unary_op_output!(
    /// Exponential: z = exp(a)
    var_exp, exp, ExpBackward
);

impl_var_unary_op_input!(
    /// Natural logarithm: z = log(a)
    var_log, log, LogBackward
);

impl_var_unary_op_output_scalar!(
    /// Square root: z = sqrt(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_sqrt, sqrt, SqrtBackward
);

impl_var_unary_op_input!(
    /// Sine: z = sin(a)
    var_sin, sin, SinBackward
);

impl_var_unary_op_input!(
    /// Cosine: z = cos(a)
    var_cos, cos, CosBackward
);

impl_var_unary_op_output_scalar!(
    /// Hyperbolic tangent: z = tanh(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_tanh, tanh, TanhBackward
);

impl_var_unary_op_input_scalar!(
    /// Square: z = a²
    ///
    /// Requires ScalarOps bound for backward computation.
    var_square, square, SquareBackward
);

impl_var_unary_op_output_scalar!(
    /// Reciprocal: z = 1/a
    ///
    /// Requires ScalarOps bound for backward computation.
    var_recip, recip, RecipBackward
);

impl_var_unary_op_input!(
    /// Absolute value: z = |a|
    var_abs, abs, AbsBackward
);

impl_var_unary_op_input_scalar!(
    /// Tangent: z = tan(a)
    ///
    /// Requires ScalarOps bound for backward computation.
    var_tan, tan, TanBackward
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};
    use crate::tensor::Tensor;

    #[test]
    fn test_var_exp_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_exp(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_var_tan_backward() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let x = Var::new(
            Tensor::<CpuRuntime>::try_from_slice(&[0.0f32], &[1], &device).unwrap(),
            true,
        );

        let z = var_tan(&x, &client).unwrap();
        let grads = backward(&z, &client).unwrap();

        let grad_x: Vec<f32> = grads.get(x.id()).unwrap().to_vec();
        assert!((grad_x[0] - 1.0).abs() < 1e-6);
    }
}
