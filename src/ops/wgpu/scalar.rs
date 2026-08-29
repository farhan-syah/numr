//! WebGPU implementation of scalar operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ScalarOps;
use crate::runtime::pow_scalar_output_dtype;
use crate::runtime::wgpu::ops::native::{native_fused_mul_add_scalar, native_scalar_op};
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

impl ScalarOps<WgpuRuntime> for WgpuClient {
    fn add_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "add_scalar", a, scalar)
    }

    fn sub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "sub_scalar", a, scalar)
    }

    fn mul_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "mul_scalar", a, scalar)
    }

    fn div_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "div_scalar", a, scalar)
    }

    fn pow_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        let out_dtype = pow_scalar_output_dtype(a.dtype(), scalar);
        if out_dtype != a.dtype() {
            // An integer raised to a negative or fractional power promotes to
            // F64, and WebGPU is 32-bit only. This is the backend's documented
            // dtype limit, not a rule of the operation.
            return Err(Error::UnsupportedDType {
                dtype: DType::F64,
                op: "pow_scalar",
            });
        }
        native_scalar_op(self, "pow_scalar", a, scalar)
    }

    fn rsub_scalar(&self, a: &Tensor<WgpuRuntime>, scalar: f64) -> Result<Tensor<WgpuRuntime>> {
        native_scalar_op(self, "rsub_scalar", a, scalar)
    }

    fn fused_mul_add_scalar(
        &self,
        a: &Tensor<WgpuRuntime>,
        scale: f64,
        bias: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_fused_mul_add_scalar(self, a, scale, bias)
    }
}
