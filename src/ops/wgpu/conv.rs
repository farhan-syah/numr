//! WebGPU implementation of convolution operations.

use crate::error::Result;
use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};
use crate::ops::conv_transpose_common::validate_conv_transpose1d;
use crate::ops::{ConvOps, PaddingMode};
use crate::runtime::RuntimeClient;
use crate::runtime::ensure_contiguous;
use crate::runtime::wgpu::ops::helpers::{alloc_output, create_params_buffer, get_tensor_buffer};
use crate::runtime::wgpu::shaders::conv as conv_launcher;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::tensor::Tensor;

// ============================================================================
// Params Structs (must match WGSL shader structs)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv1dParams {
    batch: u32,
    c_in: u32,
    length: u32,
    c_out: u32,
    kernel_size: u32,
    output_length: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2dParams {
    batch: u32,
    c_in: u32,
    height: u32,
    width: u32,
    c_out: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    groups: u32,
    has_bias: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthwiseConv2dParams {
    batch: u32,
    channels: u32,
    height: u32,
    width: u32,
    kernel_h: u32,
    kernel_w: u32,
    output_h: u32,
    output_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    has_bias: u32,
    _pad: u32,
}

impl ConvOps<WgpuRuntime> for WgpuClient {
    fn conv1d(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_conv1d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            groups,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_length == 0 || params.batch == 0 {
            return Ok(Tensor::<WgpuRuntime>::try_empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                self.device(),
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = alloc_output(
            self,
            &[params.batch, params.c_out, params.output_length],
            dtype,
        )?;

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let weight_buf = get_tensor_buffer(&weight)?;
        let output_buf = get_tensor_buffer(&output)?;

        // Create a dummy bias buffer if no bias is provided
        let bias_buf = if let Some(ref b) = bias {
            get_tensor_buffer(b)?
        } else {
            // Create a small dummy buffer (1 element)
            let dummy = Tensor::<WgpuRuntime>::try_empty(&[1], dtype, self.device())?;
            get_tensor_buffer(&dummy)?
        };

        // Create params buffer
        let shader_params = Conv1dParams {
            batch: params.batch as u32,
            c_in: params.c_in as u32,
            length: params.length as u32,
            c_out: params.c_out as u32,
            kernel_size: params.kernel_size as u32,
            output_length: params.output_length as u32,
            stride: params.stride as u32,
            padding: params.pad_left as u32,
            dilation: params.dilation as u32,
            groups: params.groups as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        let total_output = params.batch * params.c_out * params.output_length;

        conv_launcher::launch_conv1d(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &weight_buf,
            &bias_buf,
            &output_buf,
            &params_buf,
            total_output,
            dtype,
        )?;

        Ok(output)
    }

    fn conv_transpose1d(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        stride: usize,
        padding: PaddingMode,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();

        let params = validate_conv_transpose1d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        if params.output_length == 0 || params.batch == 0 {
            return Ok(Tensor::<WgpuRuntime>::try_empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                self.device(),
            )?);
        }

        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        let output = alloc_output(
            self,
            &[params.batch, params.c_out, params.output_length],
            dtype,
        )?;

        let input_buf = get_tensor_buffer(&input)?;
        let weight_buf = get_tensor_buffer(&weight)?;
        let output_buf = get_tensor_buffer(&output)?;
        let bias_buf = if let Some(ref b) = bias {
            get_tensor_buffer(b)?
        } else {
            let dummy = Tensor::<WgpuRuntime>::try_empty(&[1], dtype, self.device())?;
            get_tensor_buffer(&dummy)?
        };

        let shader_params = Conv1dParams {
            batch: params.batch as u32,
            c_in: params.c_in as u32,
            length: params.length as u32,
            c_out: params.c_out as u32,
            kernel_size: params.kernel_size as u32,
            output_length: params.output_length as u32,
            stride: params.stride as u32,
            padding: params.pad_left as u32,
            dilation: params.dilation as u32,
            groups: params.groups as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        let total_output = params.batch * params.c_out * params.output_length;

        conv_launcher::launch_conv_transpose1d(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &weight_buf,
            &bias_buf,
            &output_buf,
            &params_buf,
            total_output,
            dtype,
        )?;

        Ok(output)
    }

    fn conv2d(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_conv2d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            groups,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_h == 0 || params.output_w == 0 || params.batch == 0 {
            return Ok(Tensor::<WgpuRuntime>::try_empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                self.device(),
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = alloc_output(
            self,
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
        )?;

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let weight_buf = get_tensor_buffer(&weight)?;
        let output_buf = get_tensor_buffer(&output)?;

        // Create a dummy bias buffer if no bias is provided
        let bias_buf = if let Some(ref b) = bias {
            get_tensor_buffer(b)?
        } else {
            let dummy = Tensor::<WgpuRuntime>::try_empty(&[1], dtype, self.device())?;
            get_tensor_buffer(&dummy)?
        };

        // Create params buffer
        let shader_params = Conv2dParams {
            batch: params.batch as u32,
            c_in: params.c_in as u32,
            height: params.height as u32,
            width: params.width as u32,
            c_out: params.c_out as u32,
            kernel_h: params.kernel_h as u32,
            kernel_w: params.kernel_w as u32,
            output_h: params.output_h as u32,
            output_w: params.output_w as u32,
            stride_h: params.stride_h as u32,
            stride_w: params.stride_w as u32,
            pad_h: params.pad_top as u32,
            pad_w: params.pad_left as u32,
            dilation_h: params.dilation_h as u32,
            dilation_w: params.dilation_w as u32,
            groups: params.groups as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        let total_output = params.batch * params.c_out * params.output_h * params.output_w;

        conv_launcher::launch_conv2d(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &weight_buf,
            &bias_buf,
            &output_buf,
            &params_buf,
            total_output,
            dtype,
        )?;

        Ok(output)
    }

    fn depthwise_conv2d(
        &self,
        input: &Tensor<WgpuRuntime>,
        weight: &Tensor<WgpuRuntime>,
        bias: Option<&Tensor<WgpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
    ) -> Result<Tensor<WgpuRuntime>> {
        let dtype = input.dtype();

        // Validate all inputs and get computed parameters
        let params = validate_depthwise_conv2d(
            input.shape(),
            weight.shape(),
            bias.map(|b| b.shape()),
            stride,
            padding,
            dilation,
            dtype,
            weight.dtype(),
            bias.map(|b| b.dtype()),
        )?;

        // Handle empty output
        if params.output_h == 0 || params.output_w == 0 || params.batch == 0 {
            return Ok(Tensor::<WgpuRuntime>::try_empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                self.device(),
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = alloc_output(
            self,
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
        )?;

        // Get buffers
        let input_buf = get_tensor_buffer(&input)?;
        let weight_buf = get_tensor_buffer(&weight)?;
        let output_buf = get_tensor_buffer(&output)?;

        // Create a dummy bias buffer if no bias is provided
        let bias_buf = if let Some(ref b) = bias {
            get_tensor_buffer(b)?
        } else {
            let dummy = Tensor::<WgpuRuntime>::try_empty(&[1], dtype, self.device())?;
            get_tensor_buffer(&dummy)?
        };

        // Create params buffer
        let shader_params = DepthwiseConv2dParams {
            batch: params.batch as u32,
            channels: params.c_in as u32,
            height: params.height as u32,
            width: params.width as u32,
            kernel_h: params.kernel_h as u32,
            kernel_w: params.kernel_w as u32,
            output_h: params.output_h as u32,
            output_w: params.output_w as u32,
            stride_h: params.stride_h as u32,
            stride_w: params.stride_w as u32,
            pad_h: params.pad_top as u32,
            pad_w: params.pad_left as u32,
            dilation_h: params.dilation_h as u32,
            dilation_w: params.dilation_w as u32,
            has_bias: if bias.is_some() { 1 } else { 0 },
            _pad: 0,
        };
        let params_buf = create_params_buffer(self, &shader_params);

        let total_output = params.batch * params.c_in * params.output_h * params.output_w;

        conv_launcher::launch_depthwise_conv2d(
            self.pipeline_cache(),
            self.wgpu_queue(),
            &input_buf,
            &weight_buf,
            &bias_buf,
            &output_buf,
            &params_buf,
            total_output,
            dtype,
        )?;

        Ok(output)
    }
}
