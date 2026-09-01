//! CUDA implementation of convolution operations.

use super::conv1d_im2col::{conv1d_im2col, use_conv1d_im2col};
use crate::error::Result;
use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};
use crate::ops::conv_transpose_common::validate_conv_transpose1d;
use crate::ops::{ConvOps, PaddingMode};
use crate::runtime::cuda::kernels::{
    launch_conv_transpose1d, launch_conv1d, launch_conv2d, launch_depthwise_conv2d,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ConvOps<CudaRuntime> for CudaClient {
    fn conv1d(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<CudaRuntime>> {
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
            return Ok(Tensor::<CudaRuntime>::empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                &self.device,
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Well-shaped convolutions run as im2col + GEMM; the direct kernel
        // below stays the path for every other shape.
        if use_conv1d_im2col(&params, dtype) {
            return conv1d_im2col(self, &input, &weight, bias.as_ref(), &params);
        }

        // Allocate output
        let output = Tensor::<CudaRuntime>::empty(
            &[params.batch, params.c_out, params.output_length],
            dtype,
            &self.device,
        )?;

        // Get device pointers
        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        // Launch CUDA kernel
        unsafe {
            launch_conv1d(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                params.batch,
                params.c_in,
                params.length,
                params.c_out,
                params.kernel_size,
                params.output_length,
                params.stride,
                params.pad_left, // Use pad_left as padding (symmetric)
                params.dilation,
                params.groups,
            )?;
        }

        Ok(output)
    }

    fn conv_transpose1d(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        stride: usize,
        padding: PaddingMode,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<CudaRuntime>> {
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
            return Ok(Tensor::<CudaRuntime>::empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                &self.device,
            )?);
        }

        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        let output = Tensor::<CudaRuntime>::empty(
            &[params.batch, params.c_out, params.output_length],
            dtype,
            &self.device,
        )?;

        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        unsafe {
            launch_conv_transpose1d(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                params.batch,
                params.c_in,
                params.length,
                params.c_out,
                params.kernel_size,
                params.output_length,
                params.stride,
                params.pad_left,
                params.dilation,
                params.groups,
            )?;
        }

        Ok(output)
    }

    fn conv2d(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor<CudaRuntime>> {
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
            return Ok(Tensor::<CudaRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = Tensor::<CudaRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        )?;

        // Get device pointers
        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        // Launch CUDA kernel
        unsafe {
            launch_conv2d(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                params.batch,
                params.c_in,
                params.height,
                params.width,
                params.c_out,
                params.kernel_h,
                params.kernel_w,
                params.output_h,
                params.output_w,
                params.stride_h,
                params.stride_w,
                params.pad_top,  // Use pad_top as pad_h (symmetric)
                params.pad_left, // Use pad_left as pad_w (symmetric)
                params.dilation_h,
                params.dilation_w,
                params.groups,
            )?;
        }

        Ok(output)
    }

    fn depthwise_conv2d(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: Option<&Tensor<CudaRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
    ) -> Result<Tensor<CudaRuntime>> {
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
            return Ok(Tensor::<CudaRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            )?);
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = Tensor::<CudaRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        )?;

        // Get device pointers
        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        // Launch CUDA kernel
        unsafe {
            launch_depthwise_conv2d(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_ptr,
                weight_ptr,
                bias_ptr,
                output_ptr,
                params.batch,
                params.c_in, // channels = c_in for depthwise
                params.height,
                params.width,
                params.kernel_h,
                params.kernel_w,
                params.output_h,
                params.output_w,
                params.stride_h,
                params.stride_w,
                params.pad_top,
                params.pad_left,
                params.dilation_h,
                params.dilation_w,
            )?;
        }

        Ok(output)
    }
}
