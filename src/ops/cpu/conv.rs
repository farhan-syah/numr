//! CPU implementation of convolution operations.

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::conv_common::{validate_conv1d, validate_conv2d, validate_depthwise_conv2d};
use crate::ops::conv_transpose_common::validate_conv_transpose1d;
use crate::ops::{ConvOps, PaddingMode};
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::runtime::cpu::kernels::simd::conv as simd_conv;
use crate::runtime::cpu::{CpuClient, CpuRuntime, helpers::ensure_contiguous, kernels};
use crate::tensor::Tensor;

/// Dispatch to convolution kernel for float types only
macro_rules! dispatch_float_dtype {
    ($dtype:expr, $T:ident => $body:block, $op:expr) => {
        match $dtype {
            DType::F32 => {
                type $T = f32;
                $body
            }
            DType::F64 => {
                type $T = f64;
                $body
            }
            #[cfg(feature = "f16")]
            DType::F16 => {
                type $T = half::f16;
                $body
            }
            #[cfg(feature = "f16")]
            DType::BF16 => {
                type $T = half::bf16;
                $body
            }
            #[cfg(feature = "fp8")]
            DType::FP8E4M3 => {
                type $T = crate::dtype::FP8E4M3;
                $body
            }
            #[cfg(feature = "fp8")]
            DType::FP8E5M2 => {
                type $T = crate::dtype::FP8E5M2;
                $body
            }
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: $dtype,
                    op: $op,
                })
            }
        }
    };
}

/// Dispatch to SIMD kernels for F32/F64 on x86_64 and aarch64, scalar for other
/// types/platforms.
///
/// This macro eliminates duplication across the conv1d, conv2d, depthwise_conv2d
/// and conv_transpose1d dispatch blocks. The `simd_conv::*` entry points detect
/// the `SimdLevel` themselves and fall back to the same generic
/// `kernels::*_kernel` used below when no vector path applies.
macro_rules! dispatch_conv {
    ($dtype:expr, $conv_name:ident, $input_ptr:expr, $weight_ptr:expr, $bias_ptr:expr, $output_ptr:expr, $params:expr) => {
        paste::paste! {
            match $dtype {
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                DType::F32 => unsafe {
                    simd_conv::[<$conv_name _f32>](
                        $input_ptr as *const f32,
                        $weight_ptr as *const f32,
                        $bias_ptr.map(|p| p as *const f32),
                        $output_ptr as *mut f32,
                        $params,
                    );
                },
                #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
                DType::F64 => unsafe {
                    simd_conv::[<$conv_name _f64>](
                        $input_ptr as *const f64,
                        $weight_ptr as *const f64,
                        $bias_ptr.map(|p| p as *const f64),
                        $output_ptr as *mut f64,
                        $params,
                    );
                },
                _ => {
                    dispatch_float_dtype!($dtype, T => {
                        unsafe {
                            kernels::[<$conv_name _kernel>]::<T>(
                                $input_ptr as *const T,
                                $weight_ptr as *const T,
                                $bias_ptr.map(|p| p as *const T),
                                $output_ptr as *mut T,
                                $params,
                            );
                        }
                    }, stringify!($conv_name));
                }
            }
        }
    };
}

impl ConvOps<CpuRuntime> for CpuClient {
    fn conv1d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: usize,
        padding: PaddingMode,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<CpuRuntime>> {
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
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_length],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_length],
            dtype,
            &self.device,
        );

        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        dispatch_conv!(
            dtype, conv1d, input_ptr, weight_ptr, bias_ptr, output_ptr, params
        );

        Ok(output)
    }

    fn conv2d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
        groups: usize,
    ) -> Result<Tensor<CpuRuntime>> {
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
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        );

        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        dispatch_conv!(
            dtype, conv2d, input_ptr, weight_ptr, bias_ptr, output_ptr, params
        );

        Ok(output)
    }

    fn depthwise_conv2d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: (usize, usize),
        padding: PaddingMode,
        dilation: (usize, usize),
    ) -> Result<Tensor<CpuRuntime>> {
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
            return Ok(Tensor::<CpuRuntime>::empty(
                &[params.batch, params.c_out, params.output_h, params.output_w],
                dtype,
                &self.device,
            ));
        }

        // Ensure contiguous
        let input = ensure_contiguous(input)?;
        let weight = ensure_contiguous(weight)?;
        let bias = bias.map(ensure_contiguous).transpose()?;

        // Allocate output
        let output = Tensor::<CpuRuntime>::empty(
            &[params.batch, params.c_out, params.output_h, params.output_w],
            dtype,
            &self.device,
        );

        let input_ptr = input.ptr();
        let weight_ptr = weight.ptr();
        let bias_ptr = bias.as_ref().map(|b| b.ptr());
        let output_ptr = output.ptr();

        dispatch_conv!(
            dtype,
            depthwise_conv2d,
            input_ptr,
            weight_ptr,
            bias_ptr,
            output_ptr,
            params
        );

        Ok(output)
    }

    fn conv_transpose1d(
        &self,
        input: &Tensor<CpuRuntime>,
        weight: &Tensor<CpuRuntime>,
        bias: Option<&Tensor<CpuRuntime>>,
        stride: usize,
        padding: PaddingMode,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<Tensor<CpuRuntime>> {
        conv_transpose1d_cpu(
            &self.device,
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )
    }
}

fn conv_transpose1d_cpu(
    device: &<CpuRuntime as crate::runtime::Runtime>::Device,
    input: &Tensor<CpuRuntime>,
    weight: &Tensor<CpuRuntime>,
    bias: Option<&Tensor<CpuRuntime>>,
    stride: usize,
    padding: PaddingMode,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Result<Tensor<CpuRuntime>> {
    let dtype = input.dtype();

    // Shared with the CUDA/WebGPU backends so all three agree on shapes,
    // padding and the `Same` upsampling convention.
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

    let (b, l_in) = (params.batch, params.length);
    let (c_out, kernel) = (params.c_out, params.kernel_size);
    let (pad_left, pad_right) = (params.pad_left, params.pad_right);
    let l_out = params.output_length;

    // `saturating_sub`: a zero-length input or zero-size kernel would otherwise
    // underflow here before the empty-output early return below.
    let unpadded_len = l_in.saturating_sub(1) * params.stride
        + params.dilation * kernel.saturating_sub(1)
        + params.output_padding
        + 1;
    if pad_left + pad_right > unpadded_len {
        return Err(Error::InvalidArgument {
            arg: "padding",
            reason: format!(
                "total padding ({}) exceeds raw output length ({unpadded_len})",
                pad_left + pad_right
            ),
        });
    }

    if b == 0 || l_out == 0 {
        return Ok(Tensor::<CpuRuntime>::empty(
            &[b, c_out, l_out],
            dtype,
            device,
        ));
    }

    let input = ensure_contiguous(input)?;
    let weight = ensure_contiguous(weight)?;
    let bias = bias.map(ensure_contiguous).transpose()?;
    let output = Tensor::<CpuRuntime>::empty(&[b, c_out, l_out], dtype, device);

    let input_ptr = input.ptr();
    let weight_ptr = weight.ptr();
    let bias_ptr = bias.as_ref().map(|b| b.ptr());
    let output_ptr = output.ptr();

    dispatch_conv!(
        dtype,
        conv_transpose1d,
        input_ptr,
        weight_ptr,
        bias_ptr,
        output_ptr,
        params
    );

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::RandomOps;
    use crate::runtime::Runtime;
    use crate::runtime::cpu::CpuDevice;

    fn setup() -> (CpuDevice, CpuClient) {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        (device, client)
    }

    #[test]
    fn test_conv1d_basic() {
        let (device, client) = setup();

        // Input: (1, 1, 5) = [1, 2, 3, 4, 5]
        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);

        // Weight: (1, 1, 3) = [1, 1, 1] (moving average kernel)
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 3]);
        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 6.0).abs() < 1e-5); // 1+2+3
        assert!((data[1] - 9.0).abs() < 1e-5); // 2+3+4
        assert!((data[2] - 12.0).abs() < 1e-5); // 3+4+5
    }

    #[test]
    fn test_conv1d_same_padding() {
        let (device, client) = setup();

        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 1, PaddingMode::Same, 1, 1)
            .unwrap();

        // With same padding and stride 1, output length == input length
        assert_eq!(output.shape(), &[1, 1, 5]);
    }

    #[test]
    fn test_conv1d_with_bias() {
        let (device, client) = setup();

        let input =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32], &[1], &device);

        let output = client
            .conv1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 1, 1)
            .unwrap();

        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 16.0).abs() < 1e-5); // 6 + 10
        assert!((data[1] - 19.0).abs() < 1e-5); // 9 + 10
        assert!((data[2] - 22.0).abs() < 1e-5); // 12 + 10
    }

    #[test]
    fn test_conv1d_stride() {
        let (device, client) = setup();

        let input = Tensor::<CpuRuntime>::from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            &[1, 1, 7],
            &device,
        );
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);

        let output = client
            .conv1d(&input, &weight, None, 2, PaddingMode::Valid, 1, 1)
            .unwrap();

        // Output length = (7 - 3) / 2 + 1 = 3
        assert_eq!(output.shape(), &[1, 1, 3]);
    }

    #[test]
    fn test_conv2d_basic() {
        let (device, client) = setup();

        #[rustfmt::skip]
        let input_data = [
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device);

        // 2x2 kernel of all ones
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device);

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
        let data: Vec<f32> = output.to_vec();
        assert!((data[0] - 12.0).abs() < 1e-5); // 1+2+4+5
        assert!((data[1] - 16.0).abs() < 1e-5); // 2+3+5+6
        assert!((data[2] - 24.0).abs() < 1e-5); // 4+5+7+8
        assert!((data[3] - 28.0).abs() < 1e-5); // 5+6+8+9
    }

    #[test]
    fn test_conv2d_same_padding() {
        let (device, client) = setup();

        #[rustfmt::skip]
        let input_data = [
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 1, 3, 3], &device);
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0, 1.0], &[1, 1, 2, 2], &device);

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1), 1)
            .unwrap();

        // With same padding and stride 1, output size == input size
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_conv2d_multi_channel() {
        let (_device, client) = setup();

        // 2 input channels, 4x4 images
        let input = client.randn(&[1, 2, 4, 4], DType::F32).unwrap();

        // 3 output channels, reading from 2 input channels, 3x3 kernels
        let weight = client.randn(&[3, 2, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        // Output: (1, 3, 2, 2) -> (4-3)/1+1 = 2
        assert_eq!(output.shape(), &[1, 3, 2, 2]);
    }

    #[test]
    fn test_conv2d_grouped() {
        let (_device, client) = setup();

        // 4 input channels, split into 2 groups
        let input = client.randn(&[1, 4, 4, 4], DType::F32).unwrap();

        // 6 output channels (3 per group), 2 input channels per group
        let weight = client.randn(&[6, 2, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(
                &input,
                &weight,
                None,
                (1, 1),
                PaddingMode::Valid,
                (1, 1),
                2, // 2 groups
            )
            .unwrap();

        assert_eq!(output.shape(), &[1, 6, 2, 2]);
    }

    #[test]
    fn test_depthwise_conv2d() {
        let (device, client) = setup();

        // 2 channels, 3x3 image
        #[rustfmt::skip]
        let input_data = [
            // Channel 0
            1.0f32, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            // Channel 1
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        let input = Tensor::<CpuRuntime>::from_slice(&input_data, &[1, 2, 3, 3], &device);

        // Depthwise: 2 output channels, 1 input per channel, 2x2 kernel
        let weight = Tensor::<CpuRuntime>::from_slice(
            &[
                1.0f32, 1.0, 1.0, 1.0, // channel 0: all 1s
                2.0, 2.0, 2.0, 2.0, // channel 1: all 2s
            ],
            &[2, 1, 2, 2],
            &device,
        );

        let output = client
            .depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1))
            .unwrap();

        assert_eq!(output.shape(), &[1, 2, 2, 2]);
        let data: Vec<f32> = output.to_vec();

        // Channel 0: 1+2+4+5=12, 2+3+5+6=16, 4+5+7+8=24, 5+6+8+9=28
        assert!((data[0] - 12.0).abs() < 1e-5);
        assert!((data[1] - 16.0).abs() < 1e-5);
        assert!((data[2] - 24.0).abs() < 1e-5);
        assert!((data[3] - 28.0).abs() < 1e-5);

        // Channel 1: (9+8+6+5)*2=56, (8+7+5+4)*2=48, (6+5+3+2)*2=32, (5+4+2+1)*2=24
        assert!((data[4] - 56.0).abs() < 1e-5);
        assert!((data[5] - 48.0).abs() < 1e-5);
        assert!((data[6] - 32.0).abs() < 1e-5);
        assert!((data[7] - 24.0).abs() < 1e-5);
    }

    #[test]
    fn test_depthwise_conv2d_same_padding() {
        let (_device, client) = setup();

        let input = client.randn(&[2, 32, 28, 28], DType::F32).unwrap();
        let weight = client.randn(&[32, 1, 3, 3], DType::F32).unwrap();

        let output = client
            .depthwise_conv2d(&input, &weight, None, (1, 1), PaddingMode::Same, (1, 1))
            .unwrap();

        // Same padding preserves spatial dimensions
        assert_eq!(output.shape(), &[2, 32, 28, 28]);
    }

    #[test]
    fn test_conv2d_dilation() {
        let (_device, client) = setup();

        // 5x5 input
        let input = client.randn(&[1, 1, 5, 5], DType::F32).unwrap();

        // 3x3 kernel with dilation 2 -> effective kernel size is 5x5
        let weight = client.randn(&[1, 1, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(
                &input,
                &weight,
                None,
                (1, 1),
                PaddingMode::Valid,
                (2, 2), // dilation 2
                1,
            )
            .unwrap();

        // Output: (5 - 2*(3-1) - 1) / 1 + 1 = 1
        assert_eq!(output.shape(), &[1, 1, 1, 1]);
    }

    #[test]
    fn test_conv1d_invalid_dimensions() {
        let (_device, client) = setup();

        // 2D tensor instead of 3D
        let input = client.randn(&[3, 10], DType::F32).unwrap();
        let weight = client.randn(&[16, 3, 3], DType::F32).unwrap();

        let result = client.conv1d(&input, &weight, None, 1, PaddingMode::Valid, 1, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_invalid_groups() {
        let (_device, client) = setup();

        // 5 channels not divisible by 2 groups
        let input = client.randn(&[1, 5, 8, 8], DType::F32).unwrap();
        let weight = client.randn(&[10, 2, 3, 3], DType::F32).unwrap();

        let result = client.conv2d(
            &input,
            &weight,
            None,
            (1, 1),
            PaddingMode::Valid,
            (1, 1),
            2, // 5 not divisible by 2
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_conv2d_batch() {
        let (_device, client) = setup();

        // Batch of 4 images
        let input = client.randn(&[4, 3, 8, 8], DType::F32).unwrap();
        let weight = client.randn(&[16, 3, 3, 3], DType::F32).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[4, 16, 6, 6]);
    }

    #[test]
    fn test_conv2d_f64() {
        let (_device, client) = setup();

        let input = client.randn(&[1, 1, 4, 4], DType::F64).unwrap();
        let weight = client.randn(&[1, 1, 2, 2], DType::F64).unwrap();

        let output = client
            .conv2d(&input, &weight, None, (1, 1), PaddingMode::Valid, (1, 1), 1)
            .unwrap();

        assert_eq!(output.shape(), &[1, 1, 3, 3]);
        assert_eq!(output.dtype(), DType::F64);
    }

    // -------- conv_transpose1d --------

    #[test]
    fn conv_transpose1d_stride1_valid_roundtrip() {
        // Known PyTorch reference: input=[1,2,3], weight=[1,1,1] of shape [1,1,3]
        // → output length = 3+2 = 5, value = convolve([1,2,3], [1,1,1]).
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
        let out = client
            .conv_transpose1d(&input, &weight, None, 1, PaddingMode::Valid, 0, 1, 1)
            .unwrap();
        assert_eq!(out.shape(), &[1, 1, 5]);
        let d: Vec<f32> = out.to_vec();
        // Positions: [1, 1+2, 1+2+3, 2+3, 3] = [1, 3, 6, 5, 3]
        assert_eq!(d, vec![1.0, 3.0, 6.0, 5.0, 3.0]);
    }

    #[test]
    fn conv_transpose1d_stride2_upsamples() {
        // stride=2 doubles the effective spacing between input samples.
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[1, 1, 2], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
        let out = client
            .conv_transpose1d(&input, &weight, None, 2, PaddingMode::Valid, 0, 1, 1)
            .unwrap();
        // L_out = (2-1)*2 + 3 = 5
        assert_eq!(out.shape(), &[1, 1, 5]);
        let d: Vec<f32> = out.to_vec();
        // Scatter: input[0]=1 → positions 0,1,2; input[1]=2 → positions 2,3,4.
        assert_eq!(d, vec![1.0, 1.0, 3.0, 2.0, 2.0]);
    }

    #[test]
    fn conv_transpose1d_bias_is_added_per_channel() {
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 1, 2], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[1, 2, 1], &device);
        let bias = Tensor::<CpuRuntime>::from_slice(&[10.0f32, 100.0], &[2], &device);
        let out = client
            .conv_transpose1d(&input, &weight, Some(&bias), 1, PaddingMode::Valid, 0, 1, 1)
            .unwrap();
        // L_out = (2-1)*1 + 0 + 0 + 1 = 2
        assert_eq!(out.shape(), &[1, 2, 2]);
        let d: Vec<f32> = out.to_vec();
        // Channel 0: input * 1 + 10 = [11, 11]. Channel 1: input * 1 + 100 = [101, 101].
        assert_eq!(d, vec![11.0, 11.0, 101.0, 101.0]);
    }

    #[test]
    fn conv_transpose1d_custom_padding_crops_output() {
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0, 3.0], &[1, 1, 3], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 1.0], &[1, 1, 3], &device);
        // Pad=1,1 drops one frame on each side of the raw 5-length output.
        let out = client
            .conv_transpose1d(
                &input,
                &weight,
                None,
                1,
                PaddingMode::Custom(1, 1, 0, 0),
                0,
                1,
                1,
            )
            .unwrap();
        assert_eq!(out.shape(), &[1, 1, 3]);
        let d: Vec<f32> = out.to_vec();
        // Crop of [1, 3, 6, 5, 3] → [3, 6, 5].
        assert_eq!(d, vec![3.0, 6.0, 5.0]);
    }

    #[test]
    fn conv_transpose1d_multichannel_accumulates_c_in() {
        // 2 input channels, 1 output channel — output should sum contributions.
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 2, 2], &device);
        // weight[c_in=0, c_out=0] = [1, 1]; weight[c_in=1, c_out=0] = [2, 2]
        let weight =
            Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0, 2.0, 2.0], &[2, 1, 2], &device);
        let out = client
            .conv_transpose1d(&input, &weight, None, 1, PaddingMode::Valid, 0, 1, 1)
            .unwrap();
        // L_out = (2-1)*1 + 1 + 1 = 3
        assert_eq!(out.shape(), &[1, 1, 3]);
        let d: Vec<f32> = out.to_vec();
        // c_in=0 contributes [1, 1, 0]; c_in=1 contributes [0, 2, 2]. Sum: [1, 3, 2].
        assert_eq!(d, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn conv_transpose1d_f64_output_dtype() {
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 2.0], &[1, 1, 2], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[1.0f64, 1.0], &[1, 1, 2], &device);
        let out = client
            .conv_transpose1d(&input, &weight, None, 1, PaddingMode::Valid, 0, 1, 1)
            .unwrap();
        assert_eq!(out.dtype(), DType::F64);
        assert_eq!(out.shape(), &[1, 1, 3]);
        let d: Vec<f64> = out.to_vec();
        assert_eq!(d, vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn conv_transpose1d_rejects_bad_group_division() {
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 3], &[1, 3, 1], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 6], &[3, 2, 1], &device);
        let res = client.conv_transpose1d(&input, &weight, None, 1, PaddingMode::Valid, 0, 1, 2);
        assert!(res.is_err());
    }

    #[test]
    fn conv_transpose1d_rejects_output_padding_ge_stride() {
        let (device, client) = setup();
        let input = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[1, 1, 2], &device);
        let weight = Tensor::<CpuRuntime>::from_slice(&[0.0f32; 2], &[1, 1, 2], &device);
        let res = client.conv_transpose1d(&input, &weight, None, 2, PaddingMode::Valid, 2, 1, 1);
        assert!(res.is_err());
    }
}
