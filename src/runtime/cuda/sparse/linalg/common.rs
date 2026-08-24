//! Common utilities for CUDA sparse linear algebra.

use super::super::{CudaClient, CudaRuntime};
use crate::algorithm::sparse_linalg::{IcDecomposition, IluDecomposition};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::cuda::kernels;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

/// Validate dtype for CUDA sparse linear algebra (F32 and F64 only).
pub fn validate_cuda_dtype(dtype: DType, op: &'static str) -> Result<()> {
    if dtype != DType::F32 && dtype != DType::F64 {
        return Err(Error::UnsupportedDType { dtype, op });
    }
    Ok(())
}

// ============================================================================
// GPU-Native Level Computation (eliminates GPU↔CPU transfers)
// ============================================================================

/// Cast i64 GPU tensor to i32 GPU tensor (no CPU transfer)
pub fn cast_i64_to_i32_gpu(
    client: &CudaClient,
    tensor: &Tensor<CudaRuntime>,
) -> Result<Tensor<CudaRuntime>> {
    if tensor.dtype() != DType::I64 {
        return Err(Error::DTypeMismatch {
            lhs: DType::I64,
            rhs: tensor.dtype(),
        });
    }

    let n = tensor.numel() as i32;
    let output = Tensor::<CudaRuntime>::try_zeros(&[tensor.numel()], DType::I32, &client.device)?;

    unsafe {
        kernels::launch_cast_i64_to_i32(
            &client.context,
            &client.stream,
            client.device.index,
            tensor.ptr(),
            output.ptr(),
            n,
        )?;
    }

    Ok(output)
}

/// Compute level schedule on GPU for lower triangular (returns level_ptrs on CPU, level_rows on GPU, num_levels)
///
/// The level_ptrs is a small O(num_levels+1) array - acceptable control-flow data.
/// level_rows is a full-size O(n) array that stays on GPU.
pub fn compute_levels_lower_gpu(
    client: &CudaClient,
    row_ptrs_i32: &Tensor<CudaRuntime>,
    col_indices_i32: &Tensor<CudaRuntime>,
    n: usize,
) -> Result<(Vec<i32>, Tensor<CudaRuntime>, usize)> {
    // Allocate levels[n] initialized to 0
    let levels_gpu = Tensor::<CudaRuntime>::try_zeros(&[n], DType::I32, &client.device)?;

    // Allocate changed flag (atomic integer)
    let changed_gpu = Tensor::<CudaRuntime>::try_zeros(&[1], DType::I32, &client.device)?;

    // Iterative level computation until convergence
    let max_iterations = n as i32 + 10; // Safety limit
    for _ in 0..max_iterations {
        // The changed flag is reset by GPU kernel (single integer atomic - acceptable control flow)

        // Launch level iteration kernel
        unsafe {
            kernels::launch_compute_levels_lower_iter(
                &client.context,
                &client.stream,
                client.device.index,
                row_ptrs_i32.ptr(),
                col_indices_i32.ptr(),
                levels_gpu.ptr(),
                changed_gpu.ptr(),
                n as i32,
            )?;
        }

        // Read changed flag (1 scalar - acceptable for control flow)
        client.stream.synchronize()?;
        let changed_vec: Vec<i32> = changed_gpu.to_vec();

        if changed_vec[0] == 0 {
            break;
        }
    }

    // Find max level (num_levels = max + 1)
    let max_level_gpu = Tensor::<CudaRuntime>::try_zeros(&[1], DType::I32, &client.device)?;
    unsafe {
        kernels::launch_reduce_max_i32(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            max_level_gpu.ptr(),
            n as i32,
        )?;
    }

    client.stream.synchronize()?;
    let max_level_vec: Vec<i32> = max_level_gpu.to_vec();
    let num_levels = (max_level_vec[0] + 1) as usize;

    // Compute histogram of levels
    let histogram_gpu =
        Tensor::<CudaRuntime>::try_zeros(&[num_levels], DType::I32, &client.device)?;
    unsafe {
        kernels::launch_histogram_levels(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            histogram_gpu.ptr(),
            n as i32,
        )?;
    }

    // Read histogram (O(num_levels) - acceptable)
    client.stream.synchronize()?;
    let histogram: Vec<i32> = histogram_gpu.to_vec();

    // Compute prefix sum -> level_ptrs on CPU
    let mut level_ptrs = vec![0i32; num_levels + 1];
    for i in 0..num_levels {
        level_ptrs[i + 1] = level_ptrs[i] + histogram[i];
    }

    // Upload level_ptrs to GPU for scatter
    let level_ptrs_gpu =
        Tensor::<CudaRuntime>::try_from_slice(&level_ptrs, &[num_levels + 1], &client.device)?;

    // Allocate level_rows[n] and level_counters[num_levels]
    let level_rows_gpu = Tensor::<CudaRuntime>::try_zeros(&[n], DType::I32, &client.device)?;
    let level_counters_gpu =
        Tensor::<CudaRuntime>::try_zeros(&[num_levels], DType::I32, &client.device)?;

    // Scatter rows by level
    unsafe {
        kernels::launch_scatter_by_level(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            level_ptrs_gpu.ptr(),
            level_rows_gpu.ptr(),
            level_counters_gpu.ptr(),
            n as i32,
        )?;
    }

    client.stream.synchronize()?;

    Ok((level_ptrs, level_rows_gpu, num_levels))
}

/// Compute level schedule on GPU for upper triangular (returns level_ptrs on CPU, level_rows on GPU, num_levels)
pub fn compute_levels_upper_gpu(
    client: &CudaClient,
    row_ptrs_i32: &Tensor<CudaRuntime>,
    col_indices_i32: &Tensor<CudaRuntime>,
    n: usize,
) -> Result<(Vec<i32>, Tensor<CudaRuntime>, usize)> {
    // Allocate levels[n] initialized to 0
    let levels_gpu = Tensor::<CudaRuntime>::try_zeros(&[n], DType::I32, &client.device)?;

    // Allocate changed flag
    let changed_gpu = Tensor::<CudaRuntime>::try_zeros(&[1], DType::I32, &client.device)?;

    // Iterative level computation until convergence
    let max_iterations = n as i32 + 10;
    for _ in 0..max_iterations {
        // Reset changed flag on GPU
        let _zero_tensor = Tensor::<CudaRuntime>::try_from_slice(&[0i32], &[1], &client.device)?;

        // Launch level iteration kernel
        unsafe {
            kernels::launch_compute_levels_upper_iter(
                &client.context,
                &client.stream,
                client.device.index,
                row_ptrs_i32.ptr(),
                col_indices_i32.ptr(),
                levels_gpu.ptr(),
                changed_gpu.ptr(),
                n as i32,
            )?;
        }

        // Read changed flag
        client.stream.synchronize()?;
        let changed_vec: Vec<i32> = changed_gpu.to_vec();

        if changed_vec[0] == 0 {
            break;
        }
    }

    // Find max level
    let max_level_gpu = Tensor::<CudaRuntime>::try_zeros(&[1], DType::I32, &client.device)?;
    unsafe {
        kernels::launch_reduce_max_i32(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            max_level_gpu.ptr(),
            n as i32,
        )?;
    }

    client.stream.synchronize()?;
    let max_level_vec: Vec<i32> = max_level_gpu.to_vec();
    let num_levels = (max_level_vec[0] + 1) as usize;

    // Compute histogram
    let histogram_gpu =
        Tensor::<CudaRuntime>::try_zeros(&[num_levels], DType::I32, &client.device)?;
    unsafe {
        kernels::launch_histogram_levels(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            histogram_gpu.ptr(),
            n as i32,
        )?;
    }

    client.stream.synchronize()?;
    let histogram: Vec<i32> = histogram_gpu.to_vec();

    // Compute prefix sum
    let mut level_ptrs = vec![0i32; num_levels + 1];
    for i in 0..num_levels {
        level_ptrs[i + 1] = level_ptrs[i] + histogram[i];
    }

    let level_ptrs_gpu =
        Tensor::<CudaRuntime>::try_from_slice(&level_ptrs, &[num_levels + 1], &client.device)?;

    // Allocate and scatter
    let level_rows_gpu = Tensor::<CudaRuntime>::try_zeros(&[n], DType::I32, &client.device)?;
    let level_counters_gpu =
        Tensor::<CudaRuntime>::try_zeros(&[num_levels], DType::I32, &client.device)?;

    unsafe {
        kernels::launch_scatter_by_level(
            &client.context,
            &client.stream,
            client.device.index,
            levels_gpu.ptr(),
            level_ptrs_gpu.ptr(),
            level_rows_gpu.ptr(),
            level_counters_gpu.ptr(),
            n as i32,
        )?;
    }

    client.stream.synchronize()?;

    Ok((level_ptrs, level_rows_gpu, num_levels))
}

/// Split factored LU matrix into L and U components.
///
/// Uses GPU kernel to scatter values - no GPU→CPU transfer of values.
pub fn split_lu_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IluDecomposition<CudaRuntime>> {
    let nnz = values_gpu.numel();

    // Build L and U structure on CPU (row_ptrs and col_indices are already on CPU)
    // Also build index mapping arrays for the GPU scatter kernel
    let mut l_row_ptrs = vec![0i64; n + 1];
    let mut l_col_indices = Vec::new();
    let mut u_row_ptrs = vec![0i64; n + 1];
    let mut u_col_indices = Vec::new();

    // l_map[i] = destination index in L values, or -1 if not in L
    // u_map[i] = destination index in U values, or -1 if not in U
    let mut l_map = vec![-1i32; nnz];
    let mut u_map = vec![-1i32; nnz];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;

            if j < i {
                // Goes to L (strictly lower triangular)
                l_map[idx] = l_col_indices.len() as i32;
                l_col_indices.push(j as i64);
            } else {
                // Goes to U (upper triangular including diagonal)
                u_map[idx] = u_col_indices.len() as i32;
                u_col_indices.push(j as i64);
            }
        }

        l_row_ptrs[i + 1] = l_col_indices.len() as i64;
        u_row_ptrs[i + 1] = u_col_indices.len() as i64;
    }

    let l_nnz = l_col_indices.len();
    let u_nnz = u_col_indices.len();

    // Create structure tensors on GPU
    let l_row_ptrs_t =
        Tensor::<CudaRuntime>::try_from_slice(&l_row_ptrs, &[n + 1], &client.device)?;
    let l_col_indices_t =
        Tensor::<CudaRuntime>::try_from_slice(&l_col_indices, &[l_nnz], &client.device)?;
    let u_row_ptrs_t =
        Tensor::<CudaRuntime>::try_from_slice(&u_row_ptrs, &[n + 1], &client.device)?;
    let u_col_indices_t =
        Tensor::<CudaRuntime>::try_from_slice(&u_col_indices, &[u_nnz], &client.device)?;

    // Upload mapping arrays to GPU
    let l_map_gpu = Tensor::<CudaRuntime>::try_from_slice(&l_map, &[nnz], &client.device)?;
    let u_map_gpu = Tensor::<CudaRuntime>::try_from_slice(&u_map, &[nnz], &client.device)?;

    // Allocate output value tensors on GPU
    let l_values_t = Tensor::<CudaRuntime>::try_empty(&[l_nnz], dtype, &client.device)?;
    let u_values_t = Tensor::<CudaRuntime>::try_empty(&[u_nnz], dtype, &client.device)?;

    // Launch GPU kernel to scatter values
    unsafe {
        match dtype {
            DType::F32 => {
                kernels::launch_split_lu_scatter_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.ptr(),
                    l_values_t.ptr(),
                    u_values_t.ptr(),
                    l_map_gpu.ptr(),
                    u_map_gpu.ptr(),
                    nnz as i32,
                )?;
            }
            DType::F64 => {
                kernels::launch_split_lu_scatter_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.ptr(),
                    l_values_t.ptr(),
                    u_values_t.ptr(),
                    l_map_gpu.ptr(),
                    u_map_gpu.ptr(),
                    nnz as i32,
                )?;
            }
            _ => unreachable!(),
        }
    }

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;
    let u = CsrData::new(u_row_ptrs_t, u_col_indices_t, u_values_t, [n, n])?;

    Ok(IluDecomposition { l, u })
}

/// Extract lower triangular matrix after IC factorization.
///
/// Uses GPU kernel to scatter values - no GPU→CPU transfer of values.
pub fn extract_lower_cuda(
    client: &CudaClient,
    n: usize,
    row_ptrs: &[i64],
    col_indices: &[i64],
    values_gpu: &Tensor<CudaRuntime>,
    dtype: DType,
) -> Result<IcDecomposition<CudaRuntime>> {
    let nnz = values_gpu.numel();

    // Build lower triangular structure on CPU (row_ptrs and col_indices are already on CPU)
    // Also build index mapping array for the GPU scatter kernel
    let mut new_row_ptrs = vec![0i64; n + 1];
    let mut new_col_indices = Vec::new();

    // lower_map[i] = destination index in output values, or -1 if not in lower
    let mut lower_map = vec![-1i32; nnz];

    for i in 0..n {
        let start = row_ptrs[i] as usize;
        let end = row_ptrs[i + 1] as usize;

        for idx in start..end {
            let j = col_indices[idx] as usize;
            if j <= i {
                // Include lower triangle (j <= i includes diagonal)
                lower_map[idx] = new_col_indices.len() as i32;
                new_col_indices.push(j as i64);
            }
        }

        new_row_ptrs[i + 1] = new_col_indices.len() as i64;
    }

    let lower_nnz = new_col_indices.len();

    // Create structure tensors on GPU
    let l_row_ptrs_t =
        Tensor::<CudaRuntime>::try_from_slice(&new_row_ptrs, &[n + 1], &client.device)?;
    let l_col_indices_t =
        Tensor::<CudaRuntime>::try_from_slice(&new_col_indices, &[lower_nnz], &client.device)?;

    // Upload mapping array to GPU
    let lower_map_gpu = Tensor::<CudaRuntime>::try_from_slice(&lower_map, &[nnz], &client.device)?;

    // Allocate output value tensor on GPU
    let l_values_t = Tensor::<CudaRuntime>::try_empty(&[lower_nnz], dtype, &client.device)?;

    // Launch GPU kernel to scatter values
    unsafe {
        match dtype {
            DType::F32 => {
                kernels::launch_extract_lower_scatter_f32(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.ptr(),
                    l_values_t.ptr(),
                    lower_map_gpu.ptr(),
                    nnz as i32,
                )?;
            }
            DType::F64 => {
                kernels::launch_extract_lower_scatter_f64(
                    &client.context,
                    &client.stream,
                    client.device.index,
                    values_gpu.ptr(),
                    l_values_t.ptr(),
                    lower_map_gpu.ptr(),
                    nnz as i32,
                )?;
            }
            _ => unreachable!(),
        }
    }

    let l = CsrData::new(l_row_ptrs_t, l_col_indices_t, l_values_t, [n, n])?;

    Ok(IcDecomposition { l })
}
