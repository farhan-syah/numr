//! WebGPU ILU(0) factorization implementation.

use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages};

use super::super::ops::helpers::get_tensor_buffer;
use super::super::{WgpuClient, WgpuRuntime};
use super::common::{
    WORKGROUP_SIZE, cast_i64_to_i32_gpu, create_ilu_ic_layout, split_lu_wgpu, validate_wgpu_dtype,
};
use crate::algorithm::sparse_linalg::{
    IluDecomposition, IluOptions, SymbolicIlu0, validate_square_sparse,
};
use crate::algorithm::sparse_linalg::{compute_levels_ilu, flatten_levels};
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::sparse::CsrData;
use crate::tensor::Tensor;

const FIND_DIAG_INDICES: &str = include_str!("../shaders/sparse_find_diag_indices.wgsl");
const ILU0_LEVEL_F32: &str = include_str!("../shaders/sparse_ilu0_level_f32.wgsl");

/// ILU(0) factorization for WebGPU.
pub fn ilu0_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    options: IluOptions,
) -> Result<IluDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "ilu0")?;

    // Extract CSR structure for level analysis.
    // Note: Symbolic level computation is O(n+nnz) on CPU, which is acceptable as one-time preprocessing.
    // Data transfers here are only structure metadata (row_ptrs, col_indices), not matrix values.
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule on CPU (symbolic preprocessing is fine here)
    let schedule = compute_levels_ilu(n, &row_ptrs, &col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Convert all indices to i32 on GPU (eliminates manual CPU conversion)
    let level_rows_i32: Vec<i32> = level_rows.iter().map(|&x| x as i32).collect();
    let row_ptrs_i64_gpu =
        Tensor::<WgpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], &client.device_id)?;
    let col_indices_i64_gpu =
        Tensor::<WgpuRuntime>::from_slice(&col_indices, &[col_indices.len()], &client.device_id)?;

    // Cast i64→i32 on GPU (native WGSL shader, avoids manual conversion)
    let row_ptrs_gpu = cast_i64_to_i32_gpu(client, &row_ptrs_i64_gpu)?;
    let col_indices_gpu = cast_i64_to_i32_gpu(client, &col_indices_i64_gpu)?;

    // Create GPU buffer for level rows
    let level_rows_gpu = Tensor::<WgpuRuntime>::from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    )?;

    // Clone values for in-place factorization
    let values_gpu = a.values().clone();

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &client.device_id)?;

    // Find diagonal indices on GPU
    launch_find_diag_indices(
        client,
        &row_ptrs_gpu,
        &col_indices_gpu,
        &diag_indices_gpu,
        n,
    )?;

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        launch_ilu0_level(
            client,
            &level_rows_gpu,
            level_start,
            level_size,
            &row_ptrs_gpu,
            &col_indices_gpu,
            &values_gpu,
            &diag_indices_gpu,
            n,
            options.diagonal_shift as f32,
        )?;
    }

    // Wait for GPU to complete
    client.poll_wait();

    // Split into L and U
    split_lu_wgpu(client, n, &row_ptrs, &col_indices, &values_gpu)
}

/// ILU(0) symbolic factorization for WebGPU (runs on CPU, returns reusable symbolic data).
///
/// The symbolic phase analyzes the sparsity pattern and precomputes the update schedule.
/// This can be reused for multiple numeric factorizations with different values.
pub fn ilu0_symbolic_wgpu(
    _client: &WgpuClient,
    pattern: &CsrData<WgpuRuntime>,
) -> Result<SymbolicIlu0> {
    let n = validate_square_sparse(pattern.shape)?;

    // Extract CSR structure for CPU-based symbolic analysis
    // This transfer is acceptable as symbolic analysis happens once per matrix structure
    let row_ptrs: Vec<i64> = pattern.row_ptrs().to_vec();
    let col_indices: Vec<i64> = pattern.col_indices().to_vec();

    // Delegate to shared implementation (pure CPU graph analysis)
    crate::algorithm::sparse_linalg::ilu0_symbolic_impl(n, &row_ptrs, &col_indices)
}

/// ILU(0) numeric factorization for WebGPU using precomputed symbolic data.
///
/// Uses the level schedule derived from the symbolic data for parallel execution.
pub fn ilu0_numeric_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    symbolic: &SymbolicIlu0,
    options: IluOptions,
) -> Result<IluDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "ilu0")?;

    if n != symbolic.n {
        return Err(Error::ShapeMismatch {
            expected: vec![symbolic.n, symbolic.n],
            got: vec![n, n],
        });
    }

    // Extract CSR structure - must match symbolic pattern.
    // Note: Symbolic level computation is O(n+nnz) on CPU, which is acceptable as one-time preprocessing.
    // Data transfers here are only structure metadata (row_ptrs, col_indices), not matrix values.
    let row_ptrs: Vec<i64> = a.row_ptrs().to_vec();
    let col_indices: Vec<i64> = a.col_indices().to_vec();

    // Compute level schedule from the pattern on CPU
    let schedule = compute_levels_ilu(n, &row_ptrs, &col_indices)?;
    let (level_ptrs, level_rows) = flatten_levels(&schedule);

    // Convert all indices to i32 on GPU (eliminates manual CPU conversion)
    let level_rows_i32: Vec<i32> = level_rows.iter().map(|&x| x as i32).collect();
    let row_ptrs_i64_gpu =
        Tensor::<WgpuRuntime>::from_slice(&row_ptrs, &[row_ptrs.len()], &client.device_id)?;
    let col_indices_i64_gpu =
        Tensor::<WgpuRuntime>::from_slice(&col_indices, &[col_indices.len()], &client.device_id)?;

    // Cast i64→i32 on GPU (native WGSL shader, avoids manual conversion)
    let row_ptrs_gpu = cast_i64_to_i32_gpu(client, &row_ptrs_i64_gpu)?;
    let col_indices_gpu = cast_i64_to_i32_gpu(client, &col_indices_i64_gpu)?;

    // Create GPU buffer for level rows
    let level_rows_gpu = Tensor::<WgpuRuntime>::from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    )?;

    // Clone values for in-place factorization
    let values_gpu = a.values().clone();

    // Allocate diagonal indices buffer
    let diag_indices_gpu = Tensor::<WgpuRuntime>::zeros(&[n], DType::I32, &client.device_id)?;

    // Find diagonal indices on GPU
    launch_find_diag_indices(
        client,
        &row_ptrs_gpu,
        &col_indices_gpu,
        &diag_indices_gpu,
        n,
    )?;

    // Process each level
    for level in 0..schedule.num_levels {
        let level_start = level_ptrs[level] as usize;
        let level_end = level_ptrs[level + 1] as usize;
        let level_size = level_end - level_start;

        if level_size == 0 {
            continue;
        }

        launch_ilu0_level(
            client,
            &level_rows_gpu,
            level_start,
            level_size,
            &row_ptrs_gpu,
            &col_indices_gpu,
            &values_gpu,
            &diag_indices_gpu,
            n,
            options.diagonal_shift as f32,
        )?;
    }

    // Wait for GPU to complete
    client.poll_wait();

    // Split into L and U
    split_lu_wgpu(client, n, &row_ptrs, &col_indices, &values_gpu)
}

/// Launch find diagonal indices kernel.
pub(super) fn launch_find_diag_indices(
    client: &WgpuClient,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    diag_indices: &Tensor<WgpuRuntime>,
    n: usize,
) -> Result<()> {
    let module = client
        .pipeline_cache
        .get_or_create_module("find_diag_indices", FIND_DIAG_INDICES);

    // Create bind group layout
    let layout = client
        .wgpu_device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("find_diag_indices_layout"),
            entries: &[
                // row_ptrs (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // col_indices (read-only input)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // diag_indices (output)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let pipeline = client.pipeline_cache.get_or_create_pipeline(
        "find_diag_indices",
        "find_diag_indices",
        &module,
        &layout,
    );

    // Create params buffer
    let params = [n as u32, 0u32, 0u32, 0u32];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("find_diag_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    // Get buffers
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let diag_indices_buf = get_tensor_buffer(diag_indices)?;

    // Create bind group
    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("find_diag_indices_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: row_ptrs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: col_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: diag_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    // Dispatch
    let workgroups = (n as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("find_diag_indices"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client.queue.submit(Some(encoder.finish()));

    Ok(())
}

/// Launch ILU0 level kernel.
#[allow(clippy::too_many_arguments)]
pub(super) fn launch_ilu0_level(
    client: &WgpuClient,
    level_rows: &Tensor<WgpuRuntime>,
    level_start: usize,
    level_size: usize,
    row_ptrs: &Tensor<WgpuRuntime>,
    col_indices: &Tensor<WgpuRuntime>,
    values: &Tensor<WgpuRuntime>,
    diag_indices: &Tensor<WgpuRuntime>,
    n: usize,
    diagonal_shift: f32,
) -> Result<()> {
    let module = client
        .pipeline_cache
        .get_or_create_module("ilu0_level_f32", ILU0_LEVEL_F32);

    let layout = create_ilu_ic_layout(&client.wgpu_device);

    let pipeline = client.pipeline_cache.get_or_create_pipeline(
        "ilu0_level_f32",
        "ilu0_level_f32",
        &module,
        &layout,
    );

    // Create params buffer: level_size, n, diagonal_shift, level_start
    let params: [u32; 4] = [
        level_size as u32,
        n as u32,
        diagonal_shift.to_bits(),
        level_start as u32,
    ];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ilu0_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    // Get buffers
    let level_rows_buf = get_tensor_buffer(level_rows)?;
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let values_buf = get_tensor_buffer(values)?;
    let diag_indices_buf = get_tensor_buffer(diag_indices)?;

    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("ilu0_level_bind_group"),
        layout: &layout,
        entries: &[
            BindGroupEntry {
                binding: 0,
                resource: level_rows_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 1,
                resource: row_ptrs_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 2,
                resource: col_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 3,
                resource: values_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 4,
                resource: diag_indices_buf.as_entire_binding(),
            },
            BindGroupEntry {
                binding: 5,
                resource: params_buffer.as_entire_binding(),
            },
        ],
    });

    let workgroups = (level_size as u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ilu0_level"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
    }
    client.queue.submit(Some(encoder.finish()));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithm::sparse_linalg::SparseLinAlgAlgorithms;
    use crate::runtime::Runtime;

    fn get_client() -> WgpuClient {
        let device = WgpuRuntime::default_device();
        WgpuRuntime::default_client(&device)
    }

    #[test]
    fn test_ilu0_basic() {
        let client = get_client();
        let device = &client.device_id;

        // Create a simple 3x3 sparse matrix in CSR format
        let row_ptrs = Tensor::<WgpuRuntime>::from_slice(&[0i64, 2, 5, 7], &[4], device).unwrap();
        let col_indices =
            Tensor::<WgpuRuntime>::from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device).unwrap();
        let values = Tensor::<WgpuRuntime>::from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ilu0(&a, IluOptions::default())
            .expect("ILU0 should succeed");

        assert_eq!(decomp.l.shape, [3, 3]);
        assert_eq!(decomp.u.shape, [3, 3]);
    }
}
