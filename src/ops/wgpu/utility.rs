//! Utility operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{ScalarOps, UtilityOps};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    ArangeParams, EyeParams, LinspaceParams, alloc_output, create_params_buffer, get_tensor_buffer,
};
use crate::runtime::wgpu::ops::native::native_clamp;
use crate::runtime::wgpu::shaders::shape;
use crate::runtime::{RuntimeClient, validate_arange, validate_eye};
use crate::tensor::Tensor;

impl UtilityOps<WgpuRuntime> for WgpuClient {
    fn clamp(
        &self,
        a: &Tensor<WgpuRuntime>,
        min_val: f64,
        max_val: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        native_clamp(self, a, min_val, max_val)
    }

    fn fill(&self, shape: &[usize], value: f64, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        let zeros = Tensor::zeros(shape, dtype, self.device())?;
        self.add_scalar(&zeros, value)
    }

    fn arange(
        &self,
        start: f64,
        stop: f64,
        step: f64,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // Use shared validation
        let numel = validate_arange(start, stop, step)?;

        // Handle empty tensor case
        if numel == 0 {
            return Tensor::empty(&[0], dtype, self.device());
        }

        // WebGPU only supports F32, I32, U32 natively (no F64, F16, I64, etc.)
        // This is a hardware limitation of WGSL.
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "arange",
            });
        }

        // Allocate output
        let out = alloc_output(self, &[numel], dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params (f32 precision - WebGPU limitation)
        let params = ArangeParams {
            numel: numel as u32,
            start: start as f32,
            step: step as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_arange(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }

    fn linspace(
        &self,
        start: f64,
        stop: f64,
        steps: usize,
        dtype: DType,
    ) -> Result<Tensor<WgpuRuntime>> {
        // WebGPU linspace only supports F32 because:
        // 1. WGSL has no F64 support, so computation must be in F32
        // 2. Integer linspace with F32 intermediate would lose precision
        // Use CPU backend for integer linspace if needed.
        if !matches!(dtype, DType::F32) {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "linspace (WebGPU only supports F32; use CPU for integer linspace)",
            });
        }

        if steps == 0 {
            return Tensor::empty(&[0], dtype, self.device());
        }

        if steps == 1 {
            return Ok(Tensor::from_slice(&[start as f32], &[1], &self.device_id)?);
        }

        // Allocate output
        let out = alloc_output(self, &[steps], dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params
        let params = LinspaceParams {
            steps: steps as u32,
            start: start as f32,
            stop: stop as f32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_linspace(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            steps,
            dtype,
        )?;

        Ok(out)
    }

    fn one_hot(
        &self,
        indices: &Tensor<WgpuRuntime>,
        num_classes: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        crate::ops::impl_generic::one_hot_impl(self, indices, num_classes)
    }

    fn meshgrid(
        &self,
        tensors: &[&Tensor<WgpuRuntime>],
        indexing: crate::ops::MeshgridIndexing,
    ) -> Result<Vec<Tensor<WgpuRuntime>>> {
        crate::ops::impl_generic::meshgrid_impl(tensors, indexing)
    }

    fn eye(&self, n: usize, m: Option<usize>, dtype: DType) -> Result<Tensor<WgpuRuntime>> {
        // Use shared validation
        let (rows, cols) = validate_eye(n, m);

        if rows == 0 || cols == 0 {
            return Tensor::empty(&[rows, cols], dtype, self.device());
        }

        // WebGPU only supports F32, I32, U32 natively (no F64, F16, I64, etc.)
        // This is a hardware limitation of WGSL.
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "eye" });
        }

        let numel = rows * cols;

        // Allocate output
        let out = alloc_output(self, &[rows, cols], dtype)?;
        let out_buf = get_tensor_buffer(&out)?;

        // Create params
        let params = EyeParams {
            n: rows as u32,
            m: cols as u32,
            numel: numel as u32,
        };
        let params_buf = create_params_buffer(self, &params);

        // Launch kernel
        shape::launch_eye(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &out_buf,
            &params_buf,
            numel,
            dtype,
        )?;

        Ok(out)
    }
}
