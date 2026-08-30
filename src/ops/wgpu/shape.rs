//! Shape operations for WebGPU runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::ShapeOps;
use crate::ops::impl_generic::{repeat_interleave_impl, unfold_impl};
use crate::runtime::common::shape_ops;
use crate::runtime::common::shape_ops::{validate_cat, validate_stack};
use crate::runtime::wgpu::WgpuClient;
use crate::runtime::wgpu::WgpuRuntime;
use crate::runtime::wgpu::ops::helpers::{
    CatShaderParams, MAX_DIMS, PadParamsF32, PadParamsI32, PadParamsU32, RepeatParams, RollParams,
    alloc_output, create_params_buffer, get_tensor_buffer, pack_u32_array,
};
use crate::runtime::wgpu::shaders::shape;
use crate::tensor::Tensor;

impl ShapeOps<WgpuRuntime> for WgpuClient {
    fn cat(&self, tensors: &[&Tensor<WgpuRuntime>], dim: isize) -> Result<Tensor<WgpuRuntime>> {
        let cat_params = validate_cat(tensors, dim)?;

        // Check dtype is supported by WebGPU (F32, I32, U32 are natively supported)
        if !matches!(cat_params.dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: cat_params.dtype,
                op: "cat",
            });
        }

        // Allocate output
        let out = alloc_output(self, &cat_params.out_shape, cat_params.dtype)?;

        // A zero-element result has nothing to copy, and `get_tensor_buffer` has no
        // buffer to return for a zero-byte allocation.
        if out.numel() == 0 {
            return Ok(out);
        }

        let out_buf = get_tensor_buffer(&out)?;

        // Copy data from each tensor using WGSL kernel
        let mut cat_offset = 0usize;
        for &tensor in tensors {
            let tensor_contig = tensor.contiguous()?;
            let src_cat_size = tensor.shape()[cat_params.dim_idx];
            let total_elements = cat_params.outer_size * src_cat_size * cat_params.inner_size;

            // An empty operand contributes nothing, and has no buffer to bind.
            if total_elements == 0 {
                continue;
            }

            let src_buf = get_tensor_buffer(&tensor_contig)?;

            let shader_params = CatShaderParams {
                outer_size: cat_params.outer_size as u32,
                src_cat_size: src_cat_size as u32,
                dst_cat_size: cat_params.cat_dim_total as u32,
                cat_offset: cat_offset as u32,
                inner_size: cat_params.inner_size as u32,
                total_elements: total_elements as u32,
            };
            let params_buf = create_params_buffer(self, &shader_params);

            shape::launch_cat_copy(
                self.pipeline_cache(),
                self.wgpu_queue(),
                &src_buf,
                &out_buf,
                &params_buf,
                total_elements,
                cat_params.dtype,
            )?;

            cat_offset += src_cat_size;
        }

        Ok(out)
    }

    fn stack(&self, tensors: &[&Tensor<WgpuRuntime>], dim: isize) -> Result<Tensor<WgpuRuntime>> {
        // Validate tensors and get normalized dimension
        let _ = validate_stack(tensors, dim)?;

        // stack(tensors, dim) = cat([t.unsqueeze(dim) for t in tensors], dim)
        let unsqueezed: Vec<Tensor<WgpuRuntime>> = tensors
            .iter()
            .map(|t| t.unsqueeze(dim))
            .collect::<Result<_>>()?;

        let refs: Vec<&Tensor<WgpuRuntime>> = unsqueezed.iter().collect();
        self.cat(&refs, dim)
    }

    fn split(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        split_size: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<WgpuRuntime>>> {
        shape_ops::split_impl(tensor, split_size, dim)
    }

    fn chunk(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        chunks: usize,
        dim: isize,
    ) -> Result<Vec<Tensor<WgpuRuntime>>> {
        shape_ops::chunk_impl(tensor, chunks, dim)
    }

    fn repeat(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        repeats: &[usize],
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_repeat(tensor, repeats)?;

        // No-op if all repeats are 1
        if repeats.iter().all(|&r| r == 1) {
            return tensor.contiguous();
        }

        // Check dtype is supported by WebGPU
        if !matches!(tensor.dtype(), DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: tensor.dtype(),
                op: "repeat",
            });
        }

        // Check ndim doesn't exceed shader limit
        if params.out_shape.len() > MAX_DIMS {
            return Err(Error::backend_limitation(
                "WebGPU",
                "repeat",
                format!(
                    "max {} dimensions, got {}",
                    MAX_DIMS,
                    params.out_shape.len()
                ),
            ));
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()?
        };

        let total_elements: usize = params.out_shape.iter().product();

        // Allocate output
        let out = alloc_output(self, &params.out_shape, tensor.dtype())?;
        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        // Build flat shape arrays, then pack for WGSL uniform buffer alignment
        let ndim = params.out_shape.len();
        let mut src_shape_flat = [0u32; 8];
        let mut out_shape_flat = [0u32; 8];
        for i in 0..ndim {
            src_shape_flat[i] = tensor.shape()[i] as u32;
            out_shape_flat[i] = params.out_shape[i] as u32;
        }

        let shader_params = RepeatParams {
            ndim: ndim as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
            _pad1: 0,
            src_shape: pack_u32_array(&src_shape_flat),
            out_shape: pack_u32_array(&out_shape_flat),
        };
        let params_buf = create_params_buffer(self, &shader_params);

        shape::launch_repeat(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            tensor.dtype(),
        )?;

        Ok(out)
    }

    fn pad(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        padding: &[usize],
        value: f64,
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_pad(tensor, padding)?;

        // No-op if all padding is zero
        if padding.iter().all(|&p| p == 0) {
            return tensor.contiguous();
        }

        let dtype = tensor.dtype();

        // Check dtype is supported by WebGPU
        if !matches!(dtype, DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType { dtype, op: "pad" });
        }

        // Check ndim doesn't exceed shader limit
        if params.out_shape.len() > MAX_DIMS {
            return Err(Error::backend_limitation(
                "WebGPU",
                "pad",
                format!(
                    "max {} dimensions, got {}",
                    MAX_DIMS,
                    params.out_shape.len()
                ),
            ));
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()?
        };

        let total_elements: usize = params.out_shape.iter().product();

        // Allocate output
        let out = alloc_output(self, &params.out_shape, dtype)?;
        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        // Build flat shape arrays, then pack for WGSL uniform buffer alignment
        let ndim = params.out_shape.len();
        let mut src_shape_flat = [0u32; 8];
        let mut out_shape_flat = [0u32; 8];
        let mut pad_before_flat = [0u32; 8];
        for i in 0..ndim {
            src_shape_flat[i] = tensor.shape()[i] as u32;
            out_shape_flat[i] = params.out_shape[i] as u32;
            pad_before_flat[i] = params.pad_per_dim[i].0 as u32;
        }

        // Pack arrays for WGSL uniform buffer 16-byte alignment
        let src_shape = pack_u32_array(&src_shape_flat);
        let out_shape = pack_u32_array(&out_shape_flat);
        let pad_before = pack_u32_array(&pad_before_flat);

        // Create dtype-specific params buffer
        let params_buf = match dtype {
            DType::F32 => {
                let shader_params = PadParamsF32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as f32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            DType::I32 => {
                let shader_params = PadParamsI32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as i32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            DType::U32 => {
                let shader_params = PadParamsU32 {
                    ndim: ndim as u32,
                    total_elements: total_elements as u32,
                    fill_value: value as u32,
                    _pad0: 0,
                    src_shape,
                    out_shape,
                    pad_before,
                };
                create_params_buffer(self, &shader_params)
            }
            _ => unreachable!("dtype validated above"),
        };

        shape::launch_pad(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            dtype,
        )?;

        Ok(out)
    }

    fn roll(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        shift: isize,
        dim: isize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let params = shape_ops::validate_roll(tensor, shift, dim)?;

        // Zero shift is a no-op
        if params.shift == 0 {
            return tensor.contiguous();
        }

        // Check dtype is supported by WebGPU
        if !matches!(tensor.dtype(), DType::F32 | DType::I32 | DType::U32) {
            return Err(Error::UnsupportedDType {
                dtype: tensor.dtype(),
                op: "roll",
            });
        }

        // Ensure contiguous input
        let tensor_contig = if tensor.is_contiguous() {
            tensor.clone()
        } else {
            tensor.contiguous()?
        };

        let total_elements = tensor.numel();
        let shape = tensor.shape();

        // Compute outer_size (product of dims before roll dim) and inner_size (product of dims after)
        let outer_size: usize = shape[..params.dim_idx].iter().product();
        let inner_size: usize = shape[params.dim_idx + 1..].iter().product();

        // Allocate output (same shape as input)
        let out = alloc_output(self, shape, tensor.dtype())?;

        // A zero-element tensor has nothing to roll, and `get_tensor_buffer` has no
        // buffer to return for a zero-byte allocation.
        if total_elements == 0 {
            return Ok(out);
        }

        let out_buf = get_tensor_buffer(&out)?;
        let src_buf = get_tensor_buffer(&tensor_contig)?;

        // Unclamped: the guard above rules a zero extent out, and clamping would
        // tell the shader about a row the allocation does not contain.
        let shader_params = RollParams {
            outer_size: outer_size as u32,
            dim_size: params.dim_size as u32,
            inner_size: inner_size as u32,
            shift: params.shift as u32,
            total_elements: total_elements as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        shape::launch_roll(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &src_buf,
            &out_buf,
            &params_buf,
            total_elements,
            tensor.dtype(),
        )?;

        Ok(out)
    }

    fn unfold(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        dim: isize,
        size: usize,
        step: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        unfold_impl(self, tensor, dim, size, step)
    }

    fn repeat_interleave(
        &self,
        tensor: &Tensor<WgpuRuntime>,
        repeats: usize,
        dim: Option<isize>,
    ) -> Result<Tensor<WgpuRuntime>> {
        repeat_interleave_impl(self, tensor, repeats, dim)
    }
}
