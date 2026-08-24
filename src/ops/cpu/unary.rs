//! CPU implementation of unary operations.

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::UnaryOps;
use crate::runtime::cpu::{
    CpuClient, CpuRuntime,
    helpers::{UnaryOp, dispatch_dtype, ensure_contiguous, unary_op_impl},
    kernels,
};
use crate::tensor::Tensor;

/// UnaryOps implementation for CPU runtime.
impl UnaryOps<CpuRuntime> for CpuClient {
    fn neg(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Neg, a, "neg")
    }

    fn abs(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Abs, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sqrt, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Exp, a, "exp")
    }

    fn log(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log, a, "log")
    }

    fn sin(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sin, a, "sin")
    }

    fn cos(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Cos, a, "cos")
    }

    fn tanh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tanh, a, "tanh")
    }

    fn tan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Tan, a, "tan")
    }

    fn asin(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Asin, a, "asin")
    }

    fn acos(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Acos, a, "acos")
    }

    fn atan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Atan, a, "atan")
    }

    fn sinh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sinh, a, "sinh")
    }

    fn cosh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Cosh, a, "cosh")
    }

    fn asinh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Asinh, a, "asinh")
    }

    fn acosh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Acosh, a, "acosh")
    }

    fn atanh(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Atanh, a, "atanh")
    }

    fn recip(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Recip, a, "recip")
    }

    fn rsqrt(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Rsqrt, a, "rsqrt")
    }

    fn square(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Square, a, "square")
    }

    fn cbrt(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Cbrt, a, "cbrt")
    }

    fn exp2(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Exp2, a, "exp2")
    }

    fn expm1(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Expm1, a, "expm1")
    }

    fn log2(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log2, a, "log2")
    }

    fn log10(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log10, a, "log10")
    }

    fn log1p(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Log1p, a, "log1p")
    }

    fn floor(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Floor, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Ceil, a, "ceil")
    }

    fn round(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Round, a, "round")
    }

    fn trunc(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Trunc, a, "trunc")
    }

    fn sign(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        unary_op_impl(self, UnaryOp::Sign, a, "sign")
    }

    fn isnan(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::try_empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::isnan_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut u8,
                    numel,
                );
            }
        }, "isnan");

        Ok(out)
    }

    fn isinf(&self, a: &Tensor<CpuRuntime>) -> Result<Tensor<CpuRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CpuRuntime>::try_empty(a.shape(), DType::U8, &self.device)?;

        let a_ptr = a_contig.ptr();
        let out_ptr = out.ptr();
        let numel = a.numel();

        dispatch_dtype!(dtype, T => {
            unsafe {
                kernels::isinf_kernel::<T>(
                    a_ptr as *const T,
                    out_ptr as *mut u8,
                    numel,
                );
            }
        }, "isinf");

        Ok(out)
    }
}
