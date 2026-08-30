//! Normalization operations for CUDA runtime
#[cfg(feature = "fp8")]
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::NormalizationOps;
#[cfg(feature = "fp8")]
use crate::ops::TypeConversionOps;
use crate::runtime::cuda::kernels::{
    launch_fused_add_layer_norm, launch_fused_add_layer_norm_bwd, launch_fused_add_rms_norm,
    launch_fused_add_rms_norm_bwd, launch_group_norm, launch_layer_norm, launch_rms_norm,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl NormalizationOps<CudaRuntime> for CudaClient {
    fn rms_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: weight.dtype(),
            });
        }

        // Weight must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last.
        // Unclamped: the 1D case already products to 1 over the empty slice, so a
        // clamp would only fabricate a row for a genuinely zero batch dim.
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();

        let input_contig = ensure_contiguous(input)?;
        let weight_contig = ensure_contiguous(weight)?;
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device)?;

        // A zero-element input normalizes nothing. The launcher takes its grid from
        // `batch_size` and its block from `hidden_size` without flooring either, so a
        // zero in one of them is a launch error rather than a wrong answer.
        if out.numel() == 0 {
            return Ok(out);
        }

        unsafe {
            launch_rms_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                out.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn layer_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();

        // Validate dtypes match
        if weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Weight and bias must be 1D with size matching input's last dimension
        let input_shape = input.shape();
        let hidden_size = input_shape[input_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }
        if bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: bias.shape().to_vec(),
            });
        }

        // Compute batch_size as product of all dimensions except last.
        // Unclamped: the 1D case already products to 1 over the empty slice, so a
        // clamp would only fabricate a row for a genuinely zero batch dim.
        let batch_size: usize = input_shape[..input_shape.len() - 1].iter().product();

        let input_contig = ensure_contiguous(input)?;
        let weight_contig = ensure_contiguous(weight)?;
        let bias_contig = ensure_contiguous(bias)?;
        let out = Tensor::<CudaRuntime>::empty(input_shape, dtype, &self.device)?;

        // A zero-element input normalizes nothing. The launcher takes its grid from
        // `batch_size` and its block from `hidden_size` without flooring either, so a
        // zero in one of them is a launch error rather than a wrong answer.
        if out.numel() == 0 {
            return Ok(out);
        }

        unsafe {
            launch_layer_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                bias_contig.ptr(),
                out.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok(out)
    }

    fn group_norm(
        &self,
        input: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        num_groups: usize,
        eps: f32,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = input.dtype();

        if weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        let shape = input.shape();
        if shape.len() < 2 {
            return Err(Error::InvalidArgument {
                arg: "input",
                reason: "group_norm requires at least 2D input [batch, channels, ...]".into(),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        if !channels.is_multiple_of(num_groups) {
            return Err(Error::InvalidArgument {
                arg: "num_groups",
                reason: format!("channels {channels} not divisible by num_groups {num_groups}"),
            });
        }
        let channels_per_group = channels / num_groups;
        // Unclamped: a 2D input already products to 1 over the empty trailing slice,
        // and a zero spatial dim must stay 0 so the kernel touches nothing.
        let spatial: usize = shape[2..].iter().product();

        if weight.shape() != [channels] || bias.shape() != [channels] {
            return Err(Error::ShapeMismatch {
                expected: vec![channels],
                got: if weight.shape() != [channels] {
                    weight.shape().to_vec()
                } else {
                    bias.shape().to_vec()
                },
            });
        }

        let input_contig = ensure_contiguous(input)?;
        let weight_contig = ensure_contiguous(weight)?;
        let bias_contig = ensure_contiguous(bias)?;
        let out = Tensor::<CudaRuntime>::empty(shape, dtype, &self.device)?;

        // A zero-element input normalizes nothing. The launcher takes its grid from
        // `batch * num_groups` and its block from `channels_per_group * spatial`
        // without flooring either, so a zero in one of them is a launch error.
        if out.numel() == 0 {
            return Ok(out);
        }

        unsafe {
            launch_group_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                input_contig.ptr(),
                weight_contig.ptr(),
                bias_contig.ptr(),
                out.ptr(),
                batch,
                channels,
                spatial,
                num_groups,
                channels_per_group,
                eps,
            )?;
        }

        Ok(out)
    }

    fn fused_add_rms_norm(
        &self,
        x: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = x.dtype();

        // Validate dtypes match
        if residual.dtype() != dtype || weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if residual.dtype() != dtype {
                    residual.dtype()
                } else {
                    weight.dtype()
                },
            });
        }

        // Weight must be 1D with size matching input's last dimension
        let x_shape = x.shape();
        let hidden_size = x_shape[x_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Residual must match x shape
        if residual.shape() != x_shape {
            return Err(Error::ShapeMismatch {
                expected: x_shape.to_vec(),
                got: residual.shape().to_vec(),
            });
        }

        // Unclamped: rank-1 already products to 1, a zero batch dim must stay 0.
        let batch_size: usize = x_shape[..x_shape.len() - 1].iter().product();

        let x_contig = ensure_contiguous(x)?;
        let residual_contig = ensure_contiguous(residual)?;
        let weight_contig = ensure_contiguous(weight)?;
        let output = Tensor::<CudaRuntime>::empty(x_shape, dtype, &self.device)?;
        let pre_norm = Tensor::<CudaRuntime>::empty(x_shape, dtype, &self.device)?;

        // Nothing to normalize. The launcher derives its grid from `batch_size` and
        // its block from `hidden_size` unfloored, so a zero is a launch error.
        if output.numel() == 0 {
            return Ok((output, pre_norm));
        }

        unsafe {
            launch_fused_add_rms_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x_contig.ptr(),
                residual_contig.ptr(),
                weight_contig.ptr(),
                output.ptr(),
                pre_norm.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok((output, pre_norm))
    }

    fn fused_add_rms_norm_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        pre_norm: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = grad.dtype();

        // Validate dtypes match
        if pre_norm.dtype() != dtype || weight.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if pre_norm.dtype() != dtype {
                    pre_norm.dtype()
                } else {
                    weight.dtype()
                },
            });
        }

        // Shapes must match
        let grad_shape = grad.shape();
        if pre_norm.shape() != grad_shape {
            return Err(Error::ShapeMismatch {
                expected: grad_shape.to_vec(),
                got: pre_norm.shape().to_vec(),
            });
        }

        let hidden_size = grad_shape[grad_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }

        // Unclamped: rank-1 already products to 1, a zero batch dim must stay 0.
        let batch_size: usize = grad_shape[..grad_shape.len() - 1].iter().product();

        let grad_contig = ensure_contiguous(grad)?;
        let pre_norm_contig = ensure_contiguous(pre_norm)?;
        let weight_contig = ensure_contiguous(weight)?;
        let d_input_residual = Tensor::<CudaRuntime>::empty(grad_shape, dtype, &self.device)?;
        let d_weight = Tensor::<CudaRuntime>::zeros(&[hidden_size], dtype, &self.device)?;

        // No rows contribute, so `d_weight` keeps the zeros it was seeded with —
        // exactly what CPU leaves behind. The launcher would take a zero grid or
        // block extent from `batch_size`/`hidden_size`, which is a launch error.
        if d_input_residual.numel() == 0 {
            return Ok((d_input_residual, d_weight));
        }

        unsafe {
            launch_fused_add_rms_norm_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                pre_norm_contig.ptr(),
                weight_contig.ptr(),
                d_input_residual.ptr(),
                d_weight.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok((d_input_residual, d_weight))
    }

    fn fused_add_layer_norm(
        &self,
        x: &Tensor<CudaRuntime>,
        residual: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = x.dtype();

        // Validate dtypes match
        if residual.dtype() != dtype || weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if residual.dtype() != dtype {
                    residual.dtype()
                } else if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Weight and bias must be 1D with size matching input's last dimension
        let x_shape = x.shape();
        let hidden_size = x_shape[x_shape.len() - 1];
        if weight.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: weight.shape().to_vec(),
            });
        }
        if bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: bias.shape().to_vec(),
            });
        }

        // Residual must match x shape
        if residual.shape() != x_shape {
            return Err(Error::ShapeMismatch {
                expected: x_shape.to_vec(),
                got: residual.shape().to_vec(),
            });
        }

        // Unclamped: rank-1 already products to 1, a zero batch dim must stay 0.
        let batch_size: usize = x_shape[..x_shape.len() - 1].iter().product();

        let x_contig = ensure_contiguous(x)?;
        let residual_contig = ensure_contiguous(residual)?;
        let weight_contig = ensure_contiguous(weight)?;
        let bias_contig = ensure_contiguous(bias)?;
        let output = Tensor::<CudaRuntime>::empty(x_shape, dtype, &self.device)?;
        let pre_norm = Tensor::<CudaRuntime>::empty(x_shape, dtype, &self.device)?;

        // Nothing to normalize. The launcher derives its grid from `batch_size` and
        // its block from `hidden_size` unfloored, so a zero is a launch error.
        if output.numel() == 0 {
            return Ok((output, pre_norm));
        }

        unsafe {
            launch_fused_add_layer_norm(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                x_contig.ptr(),
                residual_contig.ptr(),
                weight_contig.ptr(),
                bias_contig.ptr(),
                output.ptr(),
                pre_norm.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok((output, pre_norm))
    }

    fn fused_add_layer_norm_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        pre_norm: &Tensor<CudaRuntime>,
        weight: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        eps: f32,
    ) -> Result<(
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
        Tensor<CudaRuntime>,
    )> {
        let dtype = grad.dtype();

        // Validate dtypes match
        if pre_norm.dtype() != dtype || weight.dtype() != dtype || bias.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: if pre_norm.dtype() != dtype {
                    pre_norm.dtype()
                } else if weight.dtype() != dtype {
                    weight.dtype()
                } else {
                    bias.dtype()
                },
            });
        }

        // Shapes must match
        let grad_shape = grad.shape();
        if pre_norm.shape() != grad_shape {
            return Err(Error::ShapeMismatch {
                expected: grad_shape.to_vec(),
                got: pre_norm.shape().to_vec(),
            });
        }

        let hidden_size = grad_shape[grad_shape.len() - 1];
        if weight.shape() != [hidden_size] || bias.shape() != [hidden_size] {
            return Err(Error::ShapeMismatch {
                expected: vec![hidden_size],
                got: if weight.shape() != [hidden_size] {
                    weight.shape().to_vec()
                } else {
                    bias.shape().to_vec()
                },
            });
        }

        // Unclamped: rank-1 already products to 1, a zero batch dim must stay 0.
        let batch_size: usize = grad_shape[..grad_shape.len() - 1].iter().product();

        // FP8: compute backward in F32, then cast results back (FP8 precision too low for
        // multi-pass backward with atomicAdd accumulation)
        #[cfg(feature = "fp8")]
        if dtype == DType::FP8E4M3 || dtype == DType::FP8E5M2 {
            let grad_f32 = self.cast(grad, DType::F32)?;
            let pre_norm_f32 = self.cast(pre_norm, DType::F32)?;
            let weight_f32 = self.cast(weight, DType::F32)?;
            let bias_f32 = self.cast(bias, DType::F32)?;
            let (d_ir, d_w, d_b) = self.fused_add_layer_norm_bwd(
                &grad_f32,
                &pre_norm_f32,
                &weight_f32,
                &bias_f32,
                eps,
            )?;
            return Ok((
                self.cast(&d_ir, dtype)?,
                self.cast(&d_w, dtype)?,
                self.cast(&d_b, dtype)?,
            ));
        }

        let grad_contig = ensure_contiguous(grad)?;
        let pre_norm_contig = ensure_contiguous(pre_norm)?;
        let weight_contig = ensure_contiguous(weight)?;
        let d_input_residual = Tensor::<CudaRuntime>::empty(grad_shape, dtype, &self.device)?;
        let d_weight = Tensor::<CudaRuntime>::zeros(&[hidden_size], dtype, &self.device)?;
        let d_bias = Tensor::<CudaRuntime>::zeros(&[hidden_size], dtype, &self.device)?;

        // No rows contribute, so `d_weight` and `d_bias` keep the zeros they were
        // seeded with — exactly what CPU leaves behind. The launcher would take a
        // zero grid or block extent from `batch_size`/`hidden_size` otherwise.
        if d_input_residual.numel() == 0 {
            return Ok((d_input_residual, d_weight, d_bias));
        }

        unsafe {
            launch_fused_add_layer_norm_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                pre_norm_contig.ptr(),
                weight_contig.ptr(),
                d_input_residual.ptr(),
                d_weight.ptr(),
                d_bias.ptr(),
                batch_size,
                hidden_size,
                eps,
            )?;
        }

        Ok((d_input_residual, d_weight, d_bias))
    }
}
