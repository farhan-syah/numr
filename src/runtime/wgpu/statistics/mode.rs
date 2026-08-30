//! Mode operation for WebGPU runtime using native WGSL shader

use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{SortingOps, TypeConversionOps, compute_reduce_strides, reduce_dim_output_shape};
use crate::runtime::wgpu::client::get_buffer;
use crate::runtime::wgpu::shaders::launch_mode_dim;
use crate::runtime::wgpu::{WgpuClient, WgpuRuntime};
use crate::runtime::{RuntimeClient, ensure_contiguous, normalize_dim};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Compute mode (most frequent value) along a dimension using native WGSL shader.
///
/// # Implementation Notes
///
/// Uses GPU-based sorting followed by native WGSL shader for mode computation.
/// Entire operation runs on GPU with no CPU fallback - true hardware acceleration.
/// WebGPU natively supports F32, I32, U32. Other types are cast to F32.
pub fn mode_impl(
    client: &WgpuClient,
    a: &Tensor<WgpuRuntime>,
    dim: Option<isize>,
    keepdim: bool,
) -> Result<(Tensor<WgpuRuntime>, Tensor<WgpuRuntime>)> {
    let dtype = a.dtype();

    // Validate dtype is supported by native shader
    let native_supported = matches!(dtype, DType::F32 | DType::I32 | DType::U32);

    if !native_supported {
        // For unsupported dtypes (F64, F16, BF16, I64, etc.), cast to F32, compute, cast back
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
            let values = Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device())?;
            let counts = Tensor::<WgpuRuntime>::empty(&out_shape, DType::I32, client.device())?;
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
        let counts = Tensor::<WgpuRuntime>::full_scalar(&[], DType::I32, 1.0, client.device())?;
        return Ok((a.clone(), counts));
    }

    let dim_idx = normalize_dim(dim_val, ndim)?;
    let dim_size = shape[dim_idx];

    if dim_size == 0 {
        let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
        let values = Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device())?;
        let counts = Tensor::<WgpuRuntime>::empty(&out_shape, DType::I32, client.device())?;
        return Ok((values, counts));
    }

    // Sort along dimension using WebGPU sort (entirely on GPU)
    let sorted = client.sort(a, dim_val, false)?;

    // Compute output shape and strides
    let out_shape = reduce_dim_output_shape(shape, dim_idx, keepdim);
    let (outer_size, reduce_size, inner_size) = compute_reduce_strides(shape, dim_idx);
    let num_outputs = outer_size * inner_size;

    // Ensure sorted is contiguous for kernel access
    let sorted_contig = ensure_contiguous(&sorted)?;

    // Allocate output tensors on GPU
    // Note: WGSL shader uses i32 for counts (WebGPU doesn't support i64 in storage buffers)
    let mode_values = Tensor::<WgpuRuntime>::empty(&out_shape, dtype, client.device())?;
    let mode_counts = Tensor::<WgpuRuntime>::empty(&out_shape, DType::I32, client.device())?;

    // A zero-element output means a zero outer or inner extent — `dim_size == 0` is
    // handled above. `get_buffer` has no buffer to return for a zero-byte
    // allocation, so the empty results are handed back before the dispatch.
    if mode_values.numel() == 0 {
        return Ok((mode_values, mode_counts));
    }

    // Get wgpu buffers
    let sorted_buf = get_buffer(sorted_contig.ptr())
        .ok_or_else(|| Error::Internal("Failed to get sorted buffer".to_string()))?;
    let values_buf = get_buffer(mode_values.ptr())
        .ok_or_else(|| Error::Internal("Failed to get mode_values buffer".to_string()))?;
    let counts_buf = get_buffer(mode_counts.ptr())
        .ok_or_else(|| Error::Internal("Failed to get mode_counts buffer".to_string()))?;

    // Create params buffer: [outer_size, reduce_size, inner_size, pad]
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct ModeParams {
        outer_size: u32,
        reduce_size: u32,
        inner_size: u32,
        _pad: u32,
    }

    let params = ModeParams {
        outer_size: outer_size as u32,
        reduce_size: reduce_size as u32,
        inner_size: inner_size as u32,
        _pad: 0,
    };

    let params_buf = client
        .wgpu_device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mode_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // Launch native WGSL shader - no CPU fallback
    launch_mode_dim(
        client.pipeline_cache(),
        client.wgpu_queue(),
        &*sorted_buf,
        &*values_buf,
        &*counts_buf,
        &params_buf,
        num_outputs,
        dtype,
    )?;

    let mode_counts_i64 = client.cast(&mode_counts, DType::I64)?;
    Ok((mode_values, mode_counts_i64))
}
