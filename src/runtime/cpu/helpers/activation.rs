//! Activation operation helpers for CPU tensors

use super::super::kernels;
use super::super::{CpuClient, CpuRuntime};
use crate::dispatch_dtype;
use crate::error::Result;
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Activation operation kind for kernel dispatch
#[derive(Copy, Clone)]
pub enum ActivationOp {
    Relu,
    Sigmoid,
    Silu,
    Gelu,
}

/// Parametric activation operation kind (activations that take a scalar parameter)
#[derive(Copy, Clone)]
pub enum ParametricActivationOp {
    /// LeakyReLU: x if x > 0, else negative_slope * x
    LeakyRelu,
    /// ELU: x if x > 0, else alpha * (exp(x) - 1)
    Elu,
}

/// Helper for activation operations (relu, sigmoid, silu, gelu)
pub fn activation_op_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ActivationOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            match op {
                ActivationOp::Relu => kernels::relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Sigmoid => kernels::sigmoid_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Silu => kernels::silu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
                ActivationOp::Gelu => kernels::gelu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

/// Fused activation-mul operation kind
#[derive(Copy, Clone)]
#[allow(clippy::enum_variant_names)]
pub enum FusedActivationMulOp {
    SiluMul,
    GeluMul,
    ReluMul,
    SigmoidMul,
}

/// Helper for fused activation-mul operations: activation(a) * b
pub fn fused_activation_mul_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    b: &Tensor<CpuRuntime>,
    op: FusedActivationMulOp,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    if b.dtype() != dtype {
        return Err(crate::error::Error::DTypeMismatch {
            lhs: dtype,
            rhs: b.dtype(),
        });
    }
    if a.shape() != b.shape() {
        return Err(crate::error::Error::ShapeMismatch {
            expected: a.shape().to_vec(),
            got: b.shape().to_vec(),
        });
    }

    let a_contig = ensure_contiguous(a)?;
    let b_contig = ensure_contiguous(b)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let b_ptr = b_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            match op {
                FusedActivationMulOp::SiluMul => kernels::silu_mul_kernel::<T>(
                    a_ptr as *const T, b_ptr as *const T, out_ptr as *mut T, len,
                ),
                FusedActivationMulOp::GeluMul => kernels::gelu_mul_kernel::<T>(
                    a_ptr as *const T, b_ptr as *const T, out_ptr as *mut T, len,
                ),
                FusedActivationMulOp::ReluMul => kernels::relu_mul_kernel::<T>(
                    a_ptr as *const T, b_ptr as *const T, out_ptr as *mut T, len,
                ),
                FusedActivationMulOp::SigmoidMul => kernels::sigmoid_mul_kernel::<T>(
                    a_ptr as *const T, b_ptr as *const T, out_ptr as *mut T, len,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for parametric activation operations (leaky_relu, elu)
///
/// These activations take a single f64 parameter in addition to the input tensor.
pub fn parametric_activation_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    op: ParametricActivationOp,
    param: f64,
    op_name: &'static str,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = a.dtype();
    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CpuRuntime>::empty(a.shape(), dtype, &client.device)?;

    let len = a.numel();
    let a_ptr = a_contig.ptr();
    let out_ptr = out.ptr();

    dispatch_dtype!(dtype, T => {
        unsafe {
            match op {
                ParametricActivationOp::LeakyRelu => kernels::leaky_relu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                    param,
                ),
                ParametricActivationOp::Elu => kernels::elu_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut T,
                    len,
                    param,
                ),
            }
        }
    }, op_name);

    Ok(out)
}

/// Helper for leaky_relu activation
#[inline]
pub fn leaky_relu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    negative_slope: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(
        client,
        a,
        ParametricActivationOp::LeakyRelu,
        negative_slope,
        "leaky_relu",
    )
}

/// Helper for ELU activation
#[inline]
pub fn elu_impl(
    client: &CpuClient,
    a: &Tensor<CpuRuntime>,
    alpha: f64,
) -> Result<Tensor<CpuRuntime>> {
    parametric_activation_impl(client, a, ParametricActivationOp::Elu, alpha, "elu")
}
