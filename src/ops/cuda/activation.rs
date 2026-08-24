//! Activation functions for CUDA runtime
use crate::error::{Error, Result};
use crate::ops::ActivationOps;
use crate::ops::activation::normalize_softmax_dim;
use crate::ops::impl_generic::activation::{
    dropout_impl, log_softmax_impl, softmax_with_bias_impl, softplus_impl,
};
use crate::runtime::cuda::kernels::{
    launch_elu, launch_gelu, launch_gelu_mul, launch_gelu_mul_bwd, launch_leaky_relu, launch_relu,
    launch_relu_mul, launch_relu_mul_bwd, launch_sigmoid, launch_sigmoid_mul,
    launch_sigmoid_mul_bwd, launch_silu, launch_silu_mul, launch_silu_mul_bwd, launch_softmax,
    launch_softmax_bwd, launch_softmax_bwd_dim, launch_softmax_dim, launch_softmax_with_bias,
};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

impl ActivationOps<CudaRuntime> for CudaClient {
    fn relu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_relu(
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

    fn sigmoid(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_sigmoid(
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

    fn silu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_silu(
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

    fn gelu(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_gelu(
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

    fn silu_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: b.dtype(),
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_silu_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn gelu_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: b.dtype(),
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_gelu_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn relu_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: b.dtype(),
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_relu_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn sigmoid_mul(
        &self,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        if b.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                lhs: dtype,
                rhs: b.dtype(),
            });
        }
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_sigmoid_mul(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                b_contig.ptr(),
                out.ptr(),
                out.numel(),
            )?;
        }

        Ok(out)
    }

    fn silu_mul_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let grad_contig = ensure_contiguous(grad)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let d_a = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;
        let d_b = Tensor::<CudaRuntime>::empty(b.shape(), dtype, &self.device)?;

        unsafe {
            launch_silu_mul_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                a_contig.ptr(),
                b_contig.ptr(),
                d_a.ptr(),
                d_b.ptr(),
                a.numel(),
            )?;
        }

        Ok((d_a, d_b))
    }

    fn gelu_mul_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let grad_contig = ensure_contiguous(grad)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let d_a = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;
        let d_b = Tensor::<CudaRuntime>::empty(b.shape(), dtype, &self.device)?;

        unsafe {
            launch_gelu_mul_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                a_contig.ptr(),
                b_contig.ptr(),
                d_a.ptr(),
                d_b.ptr(),
                a.numel(),
            )?;
        }

        Ok((d_a, d_b))
    }

    fn relu_mul_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let grad_contig = ensure_contiguous(grad)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let d_a = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;
        let d_b = Tensor::<CudaRuntime>::empty(b.shape(), dtype, &self.device)?;

        unsafe {
            launch_relu_mul_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                a_contig.ptr(),
                b_contig.ptr(),
                d_a.ptr(),
                d_b.ptr(),
                a.numel(),
            )?;
        }

        Ok((d_a, d_b))
    }

    fn sigmoid_mul_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        a: &Tensor<CudaRuntime>,
        b: &Tensor<CudaRuntime>,
    ) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
        let dtype = a.dtype();
        let grad_contig = ensure_contiguous(grad)?;
        let a_contig = ensure_contiguous(a)?;
        let b_contig = ensure_contiguous(b)?;
        let d_a = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;
        let d_b = Tensor::<CudaRuntime>::empty(b.shape(), dtype, &self.device)?;

        unsafe {
            launch_sigmoid_mul_bwd(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                grad_contig.ptr(),
                a_contig.ptr(),
                b_contig.ptr(),
                d_a.ptr(),
                d_b.ptr(),
                a.numel(),
            )?;
        }

        Ok((d_a, d_b))
    }

    fn leaky_relu(
        &self,
        a: &Tensor<CudaRuntime>,
        negative_slope: f64,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_leaky_relu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
                negative_slope as f32,
            )?;
        }

        Ok(out)
    }

    fn elu(&self, a: &Tensor<CudaRuntime>, alpha: f64) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        unsafe {
            launch_elu(
                &self.context,
                &self.stream,
                self.device.index,
                dtype,
                a_contig.ptr(),
                out.ptr(),
                out.numel(),
                alpha as f32,
            )?;
        }

        Ok(out)
    }

    fn softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();

        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let a_contig = ensure_contiguous(a)?;
        let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

        let shape = a.shape();
        let outer_size: usize = shape[..dim_idx].iter().product();
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product();

        let outer_size = outer_size.max(1);
        let inner_size = inner_size.max(1);

        unsafe {
            if dim_idx == ndim - 1 {
                // Softmax over last dimension (optimized)
                launch_softmax(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    out.ptr(),
                    outer_size,
                    dim_size,
                )?;
            } else {
                // Softmax over non-last dimension
                launch_softmax_dim(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    out.ptr(),
                    outer_size,
                    dim_size,
                    inner_size,
                )?;
            }
        }

        Ok(out)
    }

    fn softmax_bwd(
        &self,
        grad: &Tensor<CudaRuntime>,
        output: &Tensor<CudaRuntime>,
        dim: isize,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = grad.dtype();
        let ndim = grad.ndim();
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        let grad_contig = ensure_contiguous(grad)?;
        let output_contig = ensure_contiguous(output)?;
        let d_input = Tensor::<CudaRuntime>::empty(grad.shape(), dtype, &self.device)?;

        let shape = grad.shape();
        let outer_size: usize = shape[..dim_idx].iter().product::<usize>().max(1);
        let dim_size = shape[dim_idx];
        let inner_size: usize = shape[dim_idx + 1..].iter().product::<usize>().max(1);

        unsafe {
            if dim_idx == ndim - 1 {
                launch_softmax_bwd(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    output_contig.ptr(),
                    d_input.ptr(),
                    outer_size,
                    dim_size,
                )?;
            } else {
                launch_softmax_bwd_dim(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    grad_contig.ptr(),
                    output_contig.ptr(),
                    d_input.ptr(),
                    outer_size,
                    dim_size,
                    inner_size,
                )?;
            }
        }

        Ok(d_input)
    }

    fn softmax_with_bias(
        &self,
        a: &Tensor<CudaRuntime>,
        bias: &Tensor<CudaRuntime>,
        dim: isize,
    ) -> Result<Tensor<CudaRuntime>> {
        let dtype = a.dtype();
        let ndim = a.ndim();
        let dim_idx =
            normalize_softmax_dim(ndim, dim).ok_or(Error::InvalidDimension { dim, ndim })?;

        // Fused path: last-dim softmax + bias whose last dim == softmax dim.
        // The fused kernel indexes bias by column (bias[col]) so we only need
        // the bias to be contiguous and have its last dim == dim_size.
        let a_shape = a.shape();
        let dim_size = a_shape[dim_idx];
        let bias_shape = bias.shape();
        let bias_last_dim = bias_shape.last().copied().unwrap_or(0);

        // The fused kernel reads bias[col] where col ∈ [0, dim_size).
        // This is correct only when the bias has exactly `dim_size` elements total
        // (i.e. all outer bias dims are 1, so bias effectively has shape [dim_size]).
        let bias_numel: usize = bias_shape.iter().product();
        let can_fuse = dim_idx == ndim - 1 && bias_last_dim == dim_size && bias_numel == dim_size; // bias is a single row: all outer dims are 1

        if can_fuse {
            let outer_size: usize = a_shape[..dim_idx].iter().product::<usize>().max(1);
            let a_contig = ensure_contiguous(a)?;
            let bias_contig = ensure_contiguous(bias)?;
            let out = Tensor::<CudaRuntime>::empty(a.shape(), dtype, &self.device)?;

            unsafe {
                launch_softmax_with_bias(
                    &self.context,
                    &self.stream,
                    self.device.index,
                    dtype,
                    a_contig.ptr(),
                    bias_contig.ptr(),
                    out.ptr(),
                    outer_size,
                    dim_size,
                )?;
            }

            Ok(out)
        } else {
            // Fall back to add+softmax for non-last dim or mismatched bias shapes.
            softmax_with_bias_impl(self, a, bias, dim)
        }
    }

    fn softplus(&self, a: &Tensor<CudaRuntime>) -> Result<Tensor<CudaRuntime>> {
        softplus_impl(self, a)
    }

    fn log_softmax(&self, a: &Tensor<CudaRuntime>, dim: isize) -> Result<Tensor<CudaRuntime>> {
        log_softmax_impl(self, a, dim)
    }

    fn dropout(
        &self,
        a: &Tensor<CudaRuntime>,
        p: f64,
        training: bool,
    ) -> Result<Tensor<CudaRuntime>> {
        dropout_impl(self, a, p, training)
    }
}
