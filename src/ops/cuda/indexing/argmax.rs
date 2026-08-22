//! Argmax/argmin operations for CUDA runtime

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::cuda::kernels::{launch_argmax_dim, launch_argmin_dim};
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::ensure_contiguous;
use crate::tensor::Tensor;

/// Execute argmax along dimension.
pub fn argmax(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
    let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &client.device);

    // `compute_reduce_strides` floors outer/inner at 1, so a zero-size output
    // would still make the kernel write one element past the empty allocation.
    if out.numel() == 0 {
        return Ok(out);
    }

    unsafe {
        launch_argmax_dim(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            outer_size,
            reduce_size,
            inner_size,
        )?;
    }

    Ok(out)
}

/// Execute argmin along dimension.
pub fn argmin(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: usize,
    keepdim: bool,
) -> Result<Tensor<CudaRuntime>> {
    let dtype = a.dtype();
    let shape = a.shape();
    let ndim = shape.len();

    // Validate dimension
    if dim >= ndim {
        return Err(Error::InvalidDimension {
            dim: dim as isize,
            ndim,
        });
    }

    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim);
    let out_shape = reduce_dim_output_shape(shape, dim, keepdim);

    let a_contig = ensure_contiguous(a)?;
    let out = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &client.device);

    // `compute_reduce_strides` floors outer/inner at 1, so a zero-size output
    // would still make the kernel write one element past the empty allocation.
    if out.numel() == 0 {
        return Ok(out);
    }

    unsafe {
        launch_argmin_dim(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            a_contig.ptr(),
            out.ptr(),
            outer_size,
            reduce_size,
            inner_size,
        )?;
    }

    Ok(out)
}
