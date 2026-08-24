//! WebGPU IC(0) factorization implementation.

use wgpu::{BindGroupDescriptor, BindGroupEntry, BufferUsages};

use super::super::ops::helpers::get_tensor_buffer;
use super::super::{WgpuClient, WgpuRuntime};
use super::common::{
    WORKGROUP_SIZE, cast_i64_to_i32_gpu, create_ilu_ic_layout, extract_lower_wgpu,
    validate_wgpu_dtype,
};
use super::ilu0::launch_find_diag_indices;
use crate::algorithm::sparse_linalg::{IcDecomposition, IcOptions, validate_square_sparse};
use crate::algorithm::sparse_linalg::{compute_levels_ilu, flatten_levels};
use crate::dtype::DType;
use crate::error::Result;
use crate::sparse::CsrData;
use crate::tensor::Tensor;

const IC0_LEVEL_F32: &str = include_str!("../shaders/sparse_ic0_level_f32.wgsl");

/// IC(0) factorization for WebGPU.
pub fn ic0_wgpu(
    client: &WgpuClient,
    a: &CsrData<WgpuRuntime>,
    options: IcOptions,
) -> Result<IcDecomposition<WgpuRuntime>> {
    let n = validate_square_sparse(a.shape)?;
    let dtype = a.values().dtype();
    validate_wgpu_dtype(dtype, "ic0")?;

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
        Tensor::<WgpuRuntime>::try_from_slice(&row_ptrs, &[row_ptrs.len()], &client.device_id)?;
    let col_indices_i64_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &col_indices,
        &[col_indices.len()],
        &client.device_id,
    )?;

    // Cast i64→i32 on GPU (native WGSL shader, avoids manual conversion)
    let row_ptrs_gpu = cast_i64_to_i32_gpu(client, &row_ptrs_i64_gpu)?;
    let col_indices_gpu = cast_i64_to_i32_gpu(client, &col_indices_i64_gpu)?;

    // Create GPU buffer for level rows
    let level_rows_gpu = Tensor::<WgpuRuntime>::try_from_slice(
        &level_rows_i32,
        &[level_rows_i32.len()],
        &client.device_id,
    )?;

    let values_gpu = a.values().clone();
    let diag_indices_gpu = Tensor::<WgpuRuntime>::try_zeros(&[n], DType::I32, &client.device_id)?;

    // Find diagonal indices
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

        launch_ic0_level(
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

    client.poll_wait();

    // Extract lower triangular L
    extract_lower_wgpu(client, n, &row_ptrs, &col_indices, &values_gpu)
}

/// Launch IC0 level kernel.
#[allow(clippy::too_many_arguments)]
fn launch_ic0_level(
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
        .get_or_create_module("ic0_level_f32", IC0_LEVEL_F32);

    let layout = create_ilu_ic_layout(&client.wgpu_device);

    let pipeline = client.pipeline_cache.get_or_create_pipeline(
        "ic0_level_f32",
        "ic0_level_f32",
        &module,
        &layout,
    );

    let params: [u32; 4] = [
        level_size as u32,
        n as u32,
        diagonal_shift.to_bits(),
        level_start as u32,
    ];
    let params_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ic0_params"),
        size: 16,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&params_buffer, 0, bytemuck::cast_slice(&params));

    let level_rows_buf = get_tensor_buffer(level_rows)?;
    let row_ptrs_buf = get_tensor_buffer(row_ptrs)?;
    let col_indices_buf = get_tensor_buffer(col_indices)?;
    let values_buf = get_tensor_buffer(values)?;
    let diag_indices_buf = get_tensor_buffer(diag_indices)?;

    let bind_group = client.wgpu_device.create_bind_group(&BindGroupDescriptor {
        label: Some("ic0_level_bind_group"),
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
            label: Some("ic0_level"),
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
    fn test_ic0_basic() {
        let client = get_client();
        let device = &client.device_id;

        let row_ptrs =
            Tensor::<WgpuRuntime>::try_from_slice(&[0i64, 2, 5, 7], &[4], device).unwrap();
        let col_indices =
            Tensor::<WgpuRuntime>::try_from_slice(&[0i64, 1, 0, 1, 2, 1, 2], &[7], device).unwrap();
        let values = Tensor::<WgpuRuntime>::try_from_slice(
            &[4.0f32, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0],
            &[7],
            device,
        )
        .unwrap();

        let a = CsrData::new(row_ptrs, col_indices, values, [3, 3])
            .expect("CSR creation should succeed");

        let decomp = client
            .ic0(&a, IcOptions::default())
            .expect("IC0 should succeed");

        assert_eq!(decomp.l.shape, [3, 3]);
    }
}
