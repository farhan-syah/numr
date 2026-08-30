//! Unary operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::UnaryOps;
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::native::{native_cast_op, native_unary_op};
use crate::tensor::Tensor;

impl UnaryOps<WgpuRuntime> for WgpuClient {
    /// `neg` on an unsigned dtype wraps: it is `0 - a` in modular arithmetic,
    /// and element-wise integer ops wrap in this crate (`kernels::wide_acc`).
    /// So `neg(1u32)` is `u32::MAX`, matching `sub` and NumPy.
    fn neg(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "neg", a)
    }

    fn abs(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "abs", a)
    }

    fn sqrt(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sqrt", a)
    }

    fn exp(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "exp", a)
    }

    fn log(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "log", a)
    }

    fn sin(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sin", a)
    }

    fn cos(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "cos", a)
    }

    fn tan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "tan", a)
    }

    fn atan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "atan", a)
    }

    fn tanh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "tanh", a)
    }

    fn recip(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "recip", a)
    }

    fn square(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "square", a)
    }

    fn floor(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "floor", a)
    }

    fn ceil(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "ceil", a)
    }

    fn round(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "round", a)
    }

    fn round_ties_even(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "round_ties_even", a)
    }

    fn sign(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sign", a)
    }

    fn rsqrt(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "rsqrt", a)
    }

    fn cbrt(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "cbrt", a)
    }

    fn exp2(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "exp2", a)
    }

    fn expm1(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "expm1", a)
    }

    fn log2(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "log2", a)
    }

    fn log10(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "log10", a)
    }

    fn log1p(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "log1p", a)
    }

    fn asin(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "asin", a)
    }

    fn acos(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "acos", a)
    }

    fn sinh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "sinh", a)
    }

    fn cosh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "cosh", a)
    }

    fn asinh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "asinh", a)
    }

    fn acosh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "acosh", a)
    }

    fn atanh(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "atanh", a)
    }

    fn trunc(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        native_unary_op(self, "trunc", a)
    }

    fn isnan(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU boolean tensors are represented as U32 (no native U8 support).
        let out_f32 = native_unary_op(self, "isnan", a)?;
        native_cast_op(self, &out_f32, DType::U32)
    }

    fn isinf(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU boolean tensors are represented as U32 (no native U8 support).
        let out_f32 = native_unary_op(self, "isinf", a)?;
        native_cast_op(self, &out_f32, DType::U32)
    }
}
