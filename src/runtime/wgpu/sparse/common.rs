//! Common utilities for WebGPU sparse linear algebra.

use std::time::Duration;

use wgpu::{
    BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    BufferBindingType, ShaderStages,
};

use super::super::{WgpuClient, WgpuRuntime};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::ops::{CumulativeOps, ShapeOps};
use crate::runtime::RuntimeClient;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Workgroup size for sparse linear algebra kernels.
pub const WORKGROUP_SIZE: u32 = 256;

/// Validate dtype for WebGPU sparse linear algebra (F32 only).
pub fn validate_wgpu_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

/// Create bind group layout for ILU/IC kernels.
pub fn create_ilu_ic_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("ilu_ic_layout"),
        entries: &[
            // level_rows (read-only input)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // row_ptrs (read-only input)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // col_indices (read-only input)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // values (read_write for in-place factorization)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // diag_indices (read-only input, written by find_diag_indices separately)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

/// Create bind group layout for triangular solve kernels.
pub fn create_trsv_layout(device: &wgpu::Device) -> BindGroupLayout {
    device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("trsv_layout"),
        entries: &[
            // level_rows (read-only input)
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // row_ptrs (read-only input)
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // col_indices (read-only input)
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // values (read-only input)
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // b (read-only input)
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // x (read_write output)
            BindGroupLayoutEntry {
                binding: 5,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // params (uniform)
            BindGroupLayoutEntry {
                binding: 6,
                visibility: ShaderStages::COMPUTE,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    })
}

impl WgpuClient {
    /// Wait for GPU operations to complete.
    pub(super) fn poll_wait(&self) {
        let _ = self.wgpu_device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(60)),
        });
    }
}

/// Split factored LU matrix into L and U components - GPU-native implementation.
///
/// Keeps values entirely on GPU. Only row_ptrs and col_indices are on CPU
/// (they come from level scheduling which currently requires CPU).
pub fn split_lu_wgpu(
    client: &WgpuClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<WgpuRuntime>,
) -> Result<crate::algorithm::sparse_linalg::IluDecomposition<WgpuRuntime>> {
    use super::super::ops::helpers::get_tensor_buffer;

    let dtype = values_gpu.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "split_lu_wgpu (GPU version only supports F32)",
        });
    }

    // Upload row_ptrs and col_indices to GPU (convert I64 → I32 for WebGPU)
    let row_ptrs_i32: Vec<i32> = row_ptrs.iter().map(|&x| x as i32).collect();
    let col_indices_i32: Vec<i32> = col_indices.iter().map(|&x| x as i32).collect();

    let row_ptrs_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&row_ptrs_i32, &[n + 1], &client.device_id)?;
    let col_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &col_indices_i32,
        &[col_indices.len()],
        &client.device_id,
    )?;

    // Step 1: Count L and U non-zeros per row on GPU
    let l_counts_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, device)?;
    let u_counts_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, device)?;

    let row_ptrs_buf = get_tensor_buffer(&row_ptrs_gpu)?;
    let col_indices_buf = get_tensor_buffer(&col_indices_gpu)?;
    let l_counts_buf = get_tensor_buffer(&l_counts_gpu)?;
    let u_counts_buf = get_tensor_buffer(&u_counts_gpu)?;

    let count_params: [u32; 4] = [n as u32, 0, 0, 0];
    let count_params_buf = client.create_uniform_buffer("split_lu_count_params", 16);
    client.write_buffer(&count_params_buf, &count_params);

    // Launch count kernel using proper launcher
    super::super::shaders::launch_split_lu_count(
        client.pipeline_cache(),
        &client.queue,
        &*row_ptrs_buf,
        &*col_indices_buf,
        &*l_counts_buf,
        &*u_counts_buf,
        &count_params_buf,
        n,
    )?;

    // Step 2: Compute prefix sum to get row_ptrs
    // Prepend zero on GPU using concat
    let zero_i32 = Tensor::<WgpuRuntime>::try_zeros(&[1], DType::I32, device)?;
    let l_counts_with_zero = client.cat(&[&zero_i32, &l_counts_gpu], 0)?;
    let u_counts_with_zero = client.cat(&[&zero_i32, &u_counts_gpu], 0)?;

    let l_row_ptrs_i32 = client.cumsum(&l_counts_with_zero, 0)?;
    let u_row_ptrs_i32 = client.cumsum(&u_counts_with_zero, 0)?;

    // Get total sizes from last element of row_ptrs
    // This is the ONE necessary scalar read for allocation
    let l_nnz = {
        let last = l_row_ptrs_i32.narrow(0, n, 1)?.contiguous()?;
        last.to_vec::<i32>()[0] as usize
    };
    let u_nnz = {
        let last = u_row_ptrs_i32.narrow(0, n, 1)?.contiguous()?;
        last.to_vec::<i32>()[0] as usize
    };

    // Step 3: Allocate output buffers
    let l_col_indices_gpu = Tensor::<WgpuRuntime>::try_zeros(&[l_nnz], DType::I32, device)?;
    let l_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[l_nnz], dtype, device)?;
    let u_col_indices_gpu = Tensor::<WgpuRuntime>::try_zeros(&[u_nnz], DType::I32, device)?;
    let u_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[u_nnz], dtype, device)?;

    // Step 4: Scatter values into L and U on GPU
    let values_buf = get_tensor_buffer(values_gpu)?;
    let l_row_ptrs_buf = get_tensor_buffer(&l_row_ptrs_i32)?;
    let l_col_indices_buf = get_tensor_buffer(&l_col_indices_gpu)?;
    let l_values_buf = get_tensor_buffer(&l_values_gpu)?;
    let u_row_ptrs_buf = get_tensor_buffer(&u_row_ptrs_i32)?;
    let u_col_indices_buf = get_tensor_buffer(&u_col_indices_gpu)?;
    let u_values_buf = get_tensor_buffer(&u_values_gpu)?;

    let scatter_params: [u32; 4] = [n as u32, 0, 0, 0];
    let scatter_params_buf = client.create_uniform_buffer("split_lu_scatter_params", 16);
    client.write_buffer(&scatter_params_buf, &scatter_params);

    // Launch scatter kernels using proper launchers (split to stay within 8-buffer limit)
    super::super::shaders::launch_split_lu_scatter_l(
        client.pipeline_cache(),
        &client.queue,
        &*row_ptrs_buf,
        &*col_indices_buf,
        &*values_buf,
        &*l_row_ptrs_buf,
        &*l_col_indices_buf,
        &*l_values_buf,
        &scatter_params_buf,
        n,
        dtype,
    )?;

    super::super::shaders::launch_split_lu_scatter_u(
        client.pipeline_cache(),
        &client.queue,
        &*row_ptrs_buf,
        &*col_indices_buf,
        &*values_buf,
        &*u_row_ptrs_buf,
        &*u_col_indices_buf,
        &*u_values_buf,
        &scatter_params_buf,
        n,
        dtype,
    )?;

    client.poll_wait();

    // Convert I32 to I64 on CPU (WGSL limitation - no I64 support)
    // This transfers metadata only (row_ptrs, col_indices), not VALUES
    let l_row_ptrs_i32_vec: Vec<i32> = l_row_ptrs_i32.to_vec();
    let u_row_ptrs_i32_vec: Vec<i32> = u_row_ptrs_i32.to_vec();
    let l_col_indices_i32_vec: Vec<i32> = l_col_indices_gpu.to_vec();
    let u_col_indices_i32_vec: Vec<i32> = u_col_indices_gpu.to_vec();

    let l_row_ptrs_i64: Vec<i64> = l_row_ptrs_i32_vec.iter().map(|&x| x as i64).collect();
    let u_row_ptrs_i64: Vec<i64> = u_row_ptrs_i32_vec.iter().map(|&x| x as i64).collect();
    let l_col_indices_i64: Vec<i64> = l_col_indices_i32_vec.iter().map(|&x| x as i64).collect();
    let u_col_indices_i64: Vec<i64> = u_col_indices_i32_vec.iter().map(|&x| x as i64).collect();

    let l_row_ptrs_t = Tensor::<WgpuRuntime>::try_from_slice(&l_row_ptrs_i64, &[n + 1], device)?;
    let l_col_indices_t =
        Tensor::<WgpuRuntime>::try_from_slice(&l_col_indices_i64, &[l_nnz], device)?;
    let u_row_ptrs_t = Tensor::<WgpuRuntime>::try_from_slice(&u_row_ptrs_i64, &[n + 1], device)?;
    let u_col_indices_t =
        Tensor::<WgpuRuntime>::try_from_slice(&u_col_indices_i64, &[u_nnz], device)?;

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_gpu, [n, n])?;
    let u = CsrData::new(u_row_ptrs_t, u_col_indices_t, u_values_gpu, [n, n])?;

    Ok(crate::algorithm::sparse_linalg::IluDecomposition { l, u })
}

/// Extract lower triangular matrix after IC factorization - GPU-native implementation.
///
/// Keeps values entirely on GPU. Only row_ptrs and col_indices are on CPU
/// (they come from level scheduling which currently requires CPU).
pub fn extract_lower_wgpu(
    client: &WgpuClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<WgpuRuntime>,
) -> Result<crate::algorithm::sparse_linalg::IcDecomposition<WgpuRuntime>> {
    use super::super::ops::helpers::get_tensor_buffer;

    let dtype = values_gpu.dtype();
    let device = client.device();

    if dtype != DType::F32 {
        return Err(Error::UnsupportedDType {
            dtype,
            op: "extract_lower_wgpu (GPU version only supports F32)",
        });
    }

    // Upload row_ptrs and col_indices to GPU (convert I64 → I32 for WebGPU)
    let row_ptrs_i32: Vec<i32> = row_ptrs.iter().map(|&x| x as i32).collect();
    let col_indices_i32: Vec<i32> = col_indices.iter().map(|&x| x as i32).collect();

    let row_ptrs_gpu =
        Tensor::<WgpuRuntime>::try_from_slice(&row_ptrs_i32, &[n + 1], &client.device_id)?;
    let col_indices_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &col_indices_i32,
        &[col_indices.len()],
        &client.device_id,
    )?;

    // Step 1: Count lower triangle non-zeros per row on GPU
    let l_counts_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, device)?;

    let row_ptrs_buf = get_tensor_buffer(&row_ptrs_gpu)?;
    let col_indices_buf = get_tensor_buffer(&col_indices_gpu)?;
    let l_counts_buf = get_tensor_buffer(&l_counts_gpu)?;

    let count_params: [u32; 4] = [n as u32, 0, 0, 0];
    let count_params_buf = client.create_uniform_buffer("extract_lower_count_params", 16);
    client.write_buffer(&count_params_buf, &count_params);

    // Launch count kernel using proper launcher
    super::super::shaders::launch_extract_lower_count(
        client.pipeline_cache(),
        &client.queue,
        &*row_ptrs_buf,
        &*col_indices_buf,
        &*l_counts_buf,
        &count_params_buf,
        n,
    )?;

    // Step 2: Compute prefix sum to get row_ptrs
    // Prepend zero on GPU using concat
    let zero_i32 = Tensor::<WgpuRuntime>::try_zeros(&[1], DType::I32, device)?;
    let l_counts_with_zero = client.cat(&[&zero_i32, &l_counts_gpu], 0)?;

    let l_row_ptrs_i32 = client.cumsum(&l_counts_with_zero, 0)?;

    // Get total size from last element of row_ptrs
    // This is the ONE necessary scalar read for allocation
    let l_nnz = {
        let last = l_row_ptrs_i32.narrow(0, n, 1)?.contiguous()?;
        last.to_vec::<i32>()[0] as usize
    };

    // Step 3: Allocate output buffers
    let l_col_indices_gpu = Tensor::<WgpuRuntime>::try_zeros(&[l_nnz], DType::I32, device)?;
    let l_values_gpu = Tensor::<WgpuRuntime>::try_zeros(&[l_nnz], dtype, device)?;

    // Step 4: Scatter values into L on GPU
    let values_buf = get_tensor_buffer(values_gpu)?;
    let l_row_ptrs_buf = get_tensor_buffer(&l_row_ptrs_i32)?;
    let l_col_indices_buf = get_tensor_buffer(&l_col_indices_gpu)?;
    let l_values_buf = get_tensor_buffer(&l_values_gpu)?;

    let scatter_params: [u32; 4] = [n as u32, 0, 0, 0];
    let scatter_params_buf = client.create_uniform_buffer("extract_lower_scatter_params", 16);
    client.write_buffer(&scatter_params_buf, &scatter_params);

    // Launch scatter kernel using proper launcher
    super::super::shaders::launch_extract_lower_scatter(
        client.pipeline_cache(),
        &client.queue,
        &*row_ptrs_buf,
        &*col_indices_buf,
        &*values_buf,
        &*l_row_ptrs_buf,
        &*l_col_indices_buf,
        &*l_values_buf,
        &scatter_params_buf,
        n,
        dtype,
    )?;

    client.poll_wait();

    // Convert I32 to I64 on CPU (WGSL limitation - no I64 support)
    // This transfers metadata only (row_ptrs, col_indices), not VALUES
    let l_row_ptrs_i32_vec: Vec<i32> = l_row_ptrs_i32.to_vec();
    let l_col_indices_i32_vec: Vec<i32> = l_col_indices_gpu.to_vec();

    let l_row_ptrs_i64: Vec<i64> = l_row_ptrs_i32_vec.iter().map(|&x| x as i64).collect();
    let l_col_indices_i64: Vec<i64> = l_col_indices_i32_vec.iter().map(|&x| x as i64).collect();

    let l_row_ptrs_t = Tensor::<WgpuRuntime>::try_from_slice(&l_row_ptrs_i64, &[n + 1], device)?;
    let l_col_indices_t =
        Tensor::<WgpuRuntime>::try_from_slice(&l_col_indices_i64, &[l_nnz], device)?;

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_gpu, [n, n])?;

    Ok(crate::algorithm::sparse_linalg::IcDecomposition { l })
}

// ============================================================================
// GPU-native i64→i32 casting (avoids manual CPU conversion)
// ============================================================================

/// Cast i64 CSR indices to i32 on GPU to eliminate manual CPU conversion.
///
/// WebGPU doesn't support i64 compute types, but we can cast on GPU by reading
/// raw bytes (i64 stored as pair of u32, we extract low 32 bits).
///
/// **Why this matters**: Without this, code transfers i64 tensors to CPU, converts
/// to i32, then uploads back. This GPU-native cast stays entirely on the device.
pub fn cast_i64_to_i32_gpu(
    client: &WgpuClient,
    tensor_i64: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    use super::super::ops::helpers::get_tensor_buffer;

    let n = tensor_i64.numel();
    let device = client.device();

    // Allocate output i32 tensor
    let tensor_i32 = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, device)?;

    // Launch GPU-native cast kernel
    let input_buf = get_tensor_buffer(tensor_i64)?;
    let output_buf = get_tensor_buffer(&tensor_i32)?;

    super::super::shaders::sparse_level_compute::launch_cast_i64_to_i32(
        client.pipeline_cache(),
        &client.queue,
        &*input_buf,
        &*output_buf,
        n,
    )?;

    client.poll_wait();

    Ok(tensor_i32)
}
