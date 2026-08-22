//! Mode operation for CUDA runtime using native CUDA kernel

use crate::dtype::DType;
use crate::error::Result;
use crate::ops::{SortingOps, TypeConversionOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::cuda::kernels::launch_mode_dim;
use crate::runtime::cuda::{CudaClient, CudaRuntime};
use crate::runtime::{ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;

/// Compute mode (most frequent value) along a dimension using native CUDA kernel.
///
/// # Implementation Notes
///
/// Uses GPU-based sorting followed by native CUDA kernel for mode computation.
/// Entire operation runs on GPU with no CPU fallback - true hardware acceleration.
///
/// Supported dtypes: F32, F64, I32, I64, U32
/// Unsupported dtypes are cast to F32, computed, then cast back.
pub fn mode_impl(
    client: &CudaClient,
    a: &Tensor<CudaRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<(Tensor<CudaRuntime>, Tensor<CudaRuntime>)> {
    let dtype = a.dtype();

    // Validate dtype is supported by native kernel
    let native_supported = matches!(
        dtype,
        DType::F32 | DType::F64 | DType::I32 | DType::I64 | DType::U32
    );

    if !native_supported {
        // For unsupported dtypes (F16, BF16, etc.), cast to F32, compute, cast back
        let a_f32 = client.cast(a, DType::F32)?;
        let (values_f32, counts) = mode_impl(client, &a_f32, dim, keepdim)?;
        let values = client.cast(&values_f32, dtype)?;
        return Ok((values, counts));
    }

    // Handle None dim: flatten to 1D first
    if dim.is_none() {
        let numel = a.numel();
        if numel == 0 {
            let out_shape = if keepdim { vec![1; a.ndim()] } else { vec![] };
            let values = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
            let counts = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &client.device);
            return Ok((values, counts));
        }

        let flat = a.reshape(&[numel])?;
        return mode_impl(client, &flat, Some(0), keepdim);
    }

    let dim_val = dim.unwrap();
    let shape = a.shape();
    let ndim = shape.len();

    if ndim == 0 {
        // Scalar input: mode is itself with count 1
        let counts = Tensor::<CudaRuntime>::full_scalar(&[], DType::I64, 1.0, &client.device);
        return Ok((a.clone(), counts));
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        let values = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
        let counts = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &client.device);
        return Ok((values, counts));
    }

    // Sort along dimension using CUDA sort (entirely on GPU)
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);

    // Ensure sorted is contiguous for kernel access
    let sorted_contig = ensure_contiguous(&sorted)?;

    // Allocate output tensors on GPU
    let mode_values = Tensor::<CudaRuntime>::empty(&out_shape, dtype, &client.device);
    let mode_counts = Tensor::<CudaRuntime>::empty(&out_shape, DType::I64, &client.device);

    // `compute_reduce_strides` floors outer/inner at 1, so a zero-size output
    // would still make the kernel write one element past the empty allocations.
    if mode_values.numel() == 0 {
        return Ok((mode_values, mode_counts));
    }

    // Launch native CUDA kernel - no CPU fallback
    unsafe {
        launch_mode_dim(
            &client.context,
            &client.stream,
            client.device.index,
            dtype,
            sorted_contig.ptr(),
            mode_values.ptr(),
            mode_counts.ptr(),
            outer_size,
            reduce_size,
            inner_size,
        )?;
    }

    Ok((mode_values, mode_counts))
}
