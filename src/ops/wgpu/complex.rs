//! Complex number operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ComplexOps;
use crate::ops::common::{
    validate_complex_real_inputs_f32_only, validate_make_complex_inputs_f32_only,
};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    UnaryParams, alloc_output, create_params_buffer, get_tensor_buffer,
};
use crate::runtime::{RuntimeClient, ensure_contiguous};
use crate::tensor::Tensor;

impl ComplexOps<WgpuRuntime> for WgpuClient {
    fn conj(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, conjugate is identity
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "conj" });
        }

        let a_contig = ensure_contiguous(a)?;
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "conj",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn real(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return copy
        if !dtype.is_complex() {
            return Ok(a.clone());
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "real" });
        }

        let a_contig = ensure_contiguous(a)?;
        let numel = a.numel();
        let out_dtype = DType::F32; // Complex64 → F32
        let out = alloc_output(self, a.shape(), out_dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "real",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn imag(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types, return zeros with same dtype
        if !dtype.is_complex() {
            return Tensor::zeros(a.shape(), dtype, self.device());
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "imag" });
        }

        // For complex types, extract imaginary part
        let out_dtype = DType::F32; // Complex64 → F32
        let a_contig = ensure_contiguous(a)?;
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), out_dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "imag",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn angle(&self, a: &Tensor<WgpuRuntime>) -> Result<Tensor<WgpuRuntime>> {
        let dtype = a.dtype();

        // For real types: angle(x) = 0 if x >= 0, π if x < 0
        if !dtype.is_complex() {
            match dtype {
                DType::F32 => {
                    // Use angle_real shader for F32
                    let a_contig = ensure_contiguous(a)?;
                    let numel = a.numel();
                    let out = alloc_output(self, a.shape(), dtype)?;

                    let a_buf = get_tensor_buffer(&a_contig)?;
                    let out_buf = get_tensor_buffer(&out)?;

                    let params = UnaryParams {
                        numel: numel as u32,
                    };
                    let params_buf = create_params_buffer(self, &params);

                    crate::runtime::wgpu::shaders::launch_angle_real(
                        self.pipeline_cache(),
                        self.wgpu_queue(),
                        &a_buf,
                        &out_buf,
                        &params_buf,
                        numel,
                    )?;

                    return Ok(out);
                }
                _ => {
                    // For other real types (integers, F64 not supported on WebGPU), return zeros
                    return Tensor::zeros(a.shape(), dtype, self.device());
                }
            }
        }

        // WebGPU only supports Complex64
        if dtype != DType::Complex64 {
            return Err(Error::UnsupportedDType { dtype, op: "angle" });
        }

        // For complex types, compute phase angle
        let out_dtype = DType::F32; // Complex64 → F32
        let a_contig = ensure_contiguous(a)?;
        let numel = a.numel();
        let out = alloc_output(self, a.shape(), out_dtype)?;

        let a_buf = get_tensor_buffer(&a_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_op(
            self.pipeline_cache(),
            self.wgpu_queue(),
            "angle",
            &a_buf,
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn make_complex(
        &self,
        real: &Tensor<WgpuRuntime>,
        imag: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_make_complex_inputs_f32_only(real, imag)?;

        let shape = real.shape();
        let numel = real.numel();
        let out_dtype = DType::Complex64; // F32 → Complex64

        let real_contig = ensure_contiguous(real)?;
        let imag_contig = ensure_contiguous(imag)?;
        let out = alloc_output(self, shape, out_dtype)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let real_buf = get_tensor_buffer(&real_contig)?;
        let imag_buf = get_tensor_buffer(&imag_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_from_real_imag(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &real_buf,
            &imag_buf,
            &out_buf,
            &params_buf,
            numel,
        )?;

        Ok(out)
    }

    fn complex_mul_real(
        &self,
        complex: &Tensor<WgpuRuntime>,
        real: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_complex_real_inputs_f32_only(complex, real, "complex_mul_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = alloc_output(self, shape, dtype)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let complex_buf = get_tensor_buffer(&complex_contig)?;
        let real_buf = get_tensor_buffer(&real_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_mul_real(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &complex_buf,
            &real_buf,
            &out_buf,
            &params_buf,
            numel,
        )?;

        Ok(out)
    }

    fn complex_div_real(
        &self,
        complex: &Tensor<WgpuRuntime>,
        real: &Tensor<WgpuRuntime>,
    ) -> Result<Tensor<WgpuRuntime>> {
        validate_complex_real_inputs_f32_only(complex, real, "complex_div_real")?;

        let dtype = complex.dtype();
        let shape = complex.shape();
        let numel = complex.numel();

        let complex_contig = ensure_contiguous(complex)?;
        let real_contig = ensure_contiguous(real)?;
        let out = alloc_output(self, shape, dtype)?;

        // Handle empty tensors
        if numel == 0 {
            return Ok(out);
        }

        let complex_buf = get_tensor_buffer(&complex_contig)?;
        let real_buf = get_tensor_buffer(&real_contig)?;
        let out_buf = get_tensor_buffer(&out)?;

        let params = UnaryParams {
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        crate::runtime::wgpu::shaders::launch_complex_div_real(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &complex_buf,
            &real_buf,
            &out_buf,
            &params_buf,
            numel,
        )?;

        Ok(out)
    }
}
