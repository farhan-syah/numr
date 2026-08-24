//! Activation operations for WebGPU runtime

use crate::error::Result;
use crate::ops::ActivationOps;
use crate::ops::impl_generic::activation::{
    dropout_impl, log_softmax_impl, softmax_with_bias_impl, softplus_impl,
};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{
    native_fused_activation_mul_bwd, native_fused_activation_mul_fwd, native_parametric_activation,
    native_softmax, native_softmax_bwd, native_unary_op,
};
use crate::tensor::Tensor;

impl ActivationOps<WgpuRuntime> for WgpuClient {
    fn relu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "relu", a)
    }

    fn sigmoid(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sigmoid", a)
    }

    fn softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        native_softmax(self, a, dim)
    }

    /// `softmax(a + bias, dim)` via the shared generic implementation, as CPU
    /// does. Without this override wgpu inherited the trait's default, which
    /// returns `NotImplemented` — a silent backend gap, since every primitive
    /// the generic implementation needs (broadcast `add`, `softmax`) already
    /// exists here. CUDA additionally has a fused kernel; wgpu does not need
    /// one to be correct.
    fn softmax_with_bias(
        &self,
        a: &Tensor<WgpuRuntime>,
        bias: &Tensor<WgpuRuntime>,
        dim: isize,
    ) -> Result<Tensor<WgpuRuntime>> {
        softmax_with_bias_impl(self, a, bias, dim)
    }

    fn softmax_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        output: &Tensor<WgpuRuntime>,
        dim: isize,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_softmax_bwd(self, grad, output, dim)
    }

    fn silu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "silu", a)
    }

    fn gelu(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "gelu", a)
    }

    fn leaky_relu(
        &self,
        a: &Tensor<WgpuRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "leaky_relu", a, negative_slope)
    }

    fn elu(&self, a: &Tensor<WgpuRuntime>, alpha: f64) -> Result<Tensor<WgpuRuntime>> {
        native_parametric_activation(self, "elu", a, alpha)
    }

    fn silu_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_activation_mul_fwd(self, "silu_mul", a, b)
    }

    fn gelu_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_activation_mul_fwd(self, "gelu_mul", a, b)
    }

    fn relu_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_activation_mul_fwd(self, "relu_mul", a, b)
    }

    fn sigmoid_mul(
        &self,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_activation_mul_fwd(self, "sigmoid_mul", a, b)
    }

    fn silu_mul_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_activation_mul_bwd(self, "silu_mul_bwd", grad, a, b)
    }

    fn gelu_mul_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_activation_mul_bwd(self, "gelu_mul_bwd", grad, a, b)
    }

    fn relu_mul_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_activation_mul_bwd(self, "relu_mul_bwd", grad, a, b)
    }

    fn sigmoid_mul_bwd(
        &self,
        grad: &Tensor<WgpuRuntime>,
        a: &Tensor<WgpuRuntime>,
        b: &Tensor<WgpuRuntime>,
    ) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
        native_fused_activation_mul_bwd(self, "sigmoid_mul_bwd", grad, a, b)
    }

    fn softplus(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        softplus_impl(self, a)
    }

    fn log_softmax(&self, a: &Tensor<WgpuRuntime>, dim: isize) -> Result<Tensor<WgpuRuntime>> {
        log_softmax_impl(self, a, dim)
    }

    fn dropout(
        &self,
        a: &Tensor<WgpuRuntime>,
        p: f64,
        training: bool,
    ) -> Result<Tensor<WgpuRuntime>> {
        dropout_impl(self, a, p, training)
    }
}
