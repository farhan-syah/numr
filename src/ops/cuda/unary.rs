//! Unary operations for CUDA runtime
use crate::dtype::DType;
use crate::error::Result;
use crate::ops::UnaryOps;
use crate::runtime::cuda::kernels::{launch_isinf_op, launch_isnan_op};
use crate::runtime::cuda::ops::helpers::native_unary_op;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl UnaryOps<CudaRuntime> for CudaClient {
    fn neg(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "neg")
    }

    fn abs(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "abs")
    }

    fn sqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sqrt")
    }

    fn exp(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "exp")
    }

    fn log(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log")
    }

    fn sin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sin")
    }

    fn cos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cos")
    }

    fn tan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tan")
    }

    fn atan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "atan")
    }

    fn tanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "tanh")
    }

    fn recip(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "recip")
    }

    fn square(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "square")
    }

    fn floor(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "floor")
    }

    fn ceil(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "ceil")
    }

    fn round(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "round")
    }

    fn round_ties_even(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "round_ties_even")
    }

    fn sign(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sign")
    }

    fn rsqrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "rsqrt")
    }

    fn cbrt(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cbrt")
    }

    fn exp2(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "exp2")
    }

    fn expm1(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "expm1")
    }

    fn log2(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log2")
    }

    fn log10(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log10")
    }

    fn log1p(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "log1p")
    }

    fn asin(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "asin")
    }

    fn acos(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "acos")
    }

    fn sinh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "sinh")
    }

    fn cosh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "cosh")
    }

    fn asinh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "asinh")
    }

    fn acosh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "acosh")
    }

    fn atanh(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "atanh")
    }

    fn trunc(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        native_unary_op(self, a, "trunc")
    }

    fn isnan(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_isnan_op(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn isinf(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        // Output is always U8 (boolean)
        let out = Tensor::<CudaRuntime>::empty(a.shape(), DType::U8, &self.device)?;

        unsafe {
            launch_isinf_op(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }
}
