//! Dropout operation with gradient support
//!
//! Dropout randomly zeroes elements with probability `p` during training,
//! scaling remaining elements by `1/(1-p)` (inverted dropout).
//! During inference, it's a no-op (identity function).

use crate::autograd::Var;
use crate::autograd::var_ops::var_mul;
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{BinaryOps, RandomOps, ScalarOps, TensorOps};
use crate::runtime::{Runtime, RuntimeClient};
use std::sync::Arc;

/// Dropout with inverted scaling: zero elements with probability `p`,
/// scale survivors by `1/(1-p)`.
///
/// Returns `(output, mask)` where mask is the binary mask scaled by `1/(1-p)`.
/// The mask is needed by the caller to store for potential reuse (e.g., in
/// the `Dropout` module) and is also saved internally for the backward pass.
///
/// When `p == 0.0`, this is an identity operation (no dropout applied).
pub fn var_dropout<R, C>(
    a: &Var<R>,
    p: f64,
    client: &C,
) -> Result<(Var<R>, crate::tensor::Tensor<R>)>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R> + TensorOps<R> + RandomOps<R> + ScalarOps<R>,
    R::Client: TensorOps<R> + ScalarOps<R>,
{
    if p == 0.0 {
        // No dropout — return input unchanged with a ones mask
        let mask = crate::tensor::Tensor::<R>::ones(
            a.tensor().shape(),
            a.tensor().dtype(),
            a.tensor().device(),
        );
        // Identity must preserve the grad_fn. Rebuilding with `Var::new` would
        // create a leaf and silently sever the graph for every caller that
        // passes p == 0.0 to disable dropout.
        let identity = if a.requires_grad() {
            Var::with_id_and_grad_fn(a.tensor().clone(), a.id(), a.grad_fn().cloned())
        } else {
            Var::with_id(a.tensor().clone(), a.id(), false)
        };
        return Ok((identity, mask));
    }

    if p >= 1.0 {
        // Drop everything — return zeros
        let zeros = crate::tensor::Tensor::<R>::zeros(
            a.tensor().shape(),
            a.tensor().dtype(),
            a.tensor().device(),
        );
        return Ok((Var::new(zeros.clone(), a.requires_grad()), zeros));
    }

    // Generate bernoulli mask: 1 with probability (1-p), 0 with probability p
    let keep_prob = 1.0 - p;
    let mask = client.bernoulli(keep_prob, a.tensor().shape(), a.tensor().dtype())?;

    // Scale mask by 1/(1-p) for inverted dropout
    let scale = 1.0 / keep_prob;
    let scaled_mask = client.mul_scalar(&mask, scale)?;

    // output = input * scaled_mask
    let output = client.mul(a.tensor(), &scaled_mask)?;

    if a.requires_grad() {
        let grad_fn = DropoutBackward::<R>::new(a.id(), scaled_mask.clone(), a.grad_fn().cloned());
        Ok((Var::from_op(output, Arc::new(grad_fn)), scaled_mask))
    } else {
        Ok((Var::new(output, false), scaled_mask))
    }
}

/// Backward for dropout.
///
/// Gradient: `dL/da = dL/dz * scaled_mask`
///
/// The same mask used in forward is applied to the gradient — zeroed positions
/// remain zeroed, surviving positions are scaled by `1/(1-p)`.
pub struct DropoutBackward<R: Runtime> {
    input_id: crate::tensor::TensorId,
    saved_mask: crate::tensor::Tensor<R>,
    input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
}

impl<R: Runtime> DropoutBackward<R> {
    pub fn new(
        input_id: crate::tensor::TensorId,
        mask: crate::tensor::Tensor<R>,
        input_grad_fn: Option<Arc<dyn crate::autograd::GradFn<R>>>,
    ) -> Self {
        Self {
            input_id,
            saved_mask: mask,
            input_grad_fn,
        }
    }
}

impl<R: Runtime<DType = DType>> crate::autograd::GradFn<R> for DropoutBackward<R>
where
    R::Client: TensorOps<R> + BinaryOps<R>,
{
    fn backward(
        &self,
        grad_output: &crate::tensor::Tensor<R>,
        _needed: &[bool],
    ) -> Result<Vec<Option<crate::tensor::Tensor<R>>>> {
        // Single input — nothing to skip.
        let client = R::default_client(grad_output.device());
        let grad = client.mul(grad_output, &self.saved_mask)?;
        Ok(vec![Some(grad)])
    }

    fn backward_var(&self, grad_output: &Var<R>) -> Result<Vec<Option<Var<R>>>>
    where
        R::Client: RuntimeClient<R> + TensorOps<R>,
    {
        let client = R::default_client(grad_output.tensor().device());
        let mask_var = Var::new(self.saved_mask.clone(), false);
        let grad = var_mul(grad_output, &mask_var, &client)?;
        Ok(vec![Some(grad)])
    }

    fn inputs(&self) -> &[crate::tensor::TensorId] {
        std::slice::from_ref(&self.input_id)
    }

    fn input_grad_fns(&self) -> Vec<Option<Arc<dyn crate::autograd::GradFn<R>>>> {
        vec![self.input_grad_fn.clone()]
    }

    fn saved_tensors(&self) -> &[crate::tensor::Tensor<R>] {
        std::slice::from_ref(&self.saved_mask)
    }

    fn name(&self) -> &'static str {
        "DropoutBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;
    use crate::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn test_dropout_zero_rate() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            crate::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );
        let (output, _mask) = var_dropout(&input, 0.0, &client).unwrap();

        let data: Vec<f32> = output.tensor().to_vec();
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }

    /// p == 0.0 is an identity, and an identity must not break the graph.
    ///
    /// Regression: this path returned `Var::new(a.tensor().clone(), ...)`, a
    /// LEAF with no grad_fn. Disabling dropout by setting p = 0.0 therefore
    /// silently detached everything upstream of it — the layers before the
    /// dropout simply stopped training, with no error and no NaN.
    #[test]
    fn dropout_zero_rate_preserves_gradient_flow() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let w = Var::new(
            crate::tensor::Tensor::<CpuRuntime>::from_slice(&[1.5f32, -2.0, 3.0], &[3], &device),
            true,
        );
        // An upstream op so the dropout input carries a grad_fn.
        let upstream = crate::autograd::var_mul(&w, &w, &client).unwrap();

        let (output, _mask) = var_dropout(&upstream, 0.0, &client).unwrap();
        let loss = crate::autograd::var_sum(&output, &[0], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();

        let grad = grads
            .get(w.id())
            .expect("p=0.0 dropout must not detach upstream parameters");
        let values: Vec<f32> = grad.contiguous().unwrap().to_vec();
        // d(sum(w^2))/dw = 2w
        assert_eq!(values, vec![3.0, -4.0, 6.0]);
    }

    #[test]
    fn test_dropout_full_rate() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            crate::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[3], &device),
            false,
        );
        // p=1.0 means drop everything
        let (output, _mask) = var_dropout(&input, 1.0, &client).unwrap();

        let data: Vec<f32> = output.tensor().to_vec();
        for val in data {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_dropout_scaling() {
        // With p=0.5, surviving elements should be scaled by 2.0
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            crate::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32; 1000], &[1000], &device),
            false,
        );
        let (output, _mask) = var_dropout(&input, 0.5, &client).unwrap();

        let data: Vec<f32> = output.tensor().to_vec();
        for val in &data {
            // Each element is either 0.0 or 2.0 (1.0 * 1/(1-0.5))
            assert!(*val == 0.0 || (*val - 2.0).abs() < 1e-5, "got {val}");
        }

        // Statistically, roughly half should be non-zero
        let nonzero = data.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero > 300 && nonzero < 700, "nonzero count: {nonzero}");
    }

    #[test]
    fn test_dropout_backward_gradient() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);

        let input = Var::new(
            crate::tensor::Tensor::<CpuRuntime>::from_slice(
                &[1.0f32, 2.0, 3.0, 4.0],
                &[4],
                &device,
            ),
            true,
        );
        let (output, mask) = var_dropout(&input, 0.5, &client).unwrap();

        // Sum to get scalar loss
        let loss = crate::autograd::var_sum(&output, &[], false, &client).unwrap();
        let grads = backward(&loss, &client).unwrap();
        let grad = grads.get(input.id()).unwrap();

        let grad_data: Vec<f32> = grad.to_vec();
        let mask_data: Vec<f32> = mask.to_vec();

        // Gradient should equal the mask (since d(sum(x*mask))/dx = mask)
        for (g, m) in grad_data.iter().zip(mask_data.iter()) {
            assert!((g - m).abs() < 1e-5, "grad {g} != mask {m}");
        }
    }
}
