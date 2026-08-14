//! Global-memory stable bitonic sort for dimensions above the shared-memory limit.

use wgpu::{Buffer, Queue};

use super::pipeline::PipelineCache;
use crate::dtype::DType;
use crate::error::{Error, Result};

const GLOBAL_SORT_SHADER: &str = include_str!("sort_global.wgsl");

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalSortParams {
    outer_size: u32,
    sort_size: u32,
    inner_size: u32,
    padded_size: u32,
    segment_count: u32,
    dtype_tag: u32,
    descending: u32,
    k: u32,
    j: u32,
    total_padded: u32,
    padding_0: u32,
    padding_1: u32,
}

/// Launch the global-memory stable bitonic path used for sort dimensions above 512.
#[allow(clippy::too_many_arguments)]
pub fn launch_global_sort(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    values_output: Option<&Buffer>,
    indices_output: Option<&Buffer>,
    outer_size: usize,
    sort_size: usize,
    inner_size: usize,
    descending: bool,
    dtype: DType,
) -> Result<()> {
    let dtype_tag = match dtype {
        DType::U32 => 0,
        DType::I32 => 1,
        DType::F32 => 2,
        _ => {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "global_sort",
            });
        }
    };
    let padded_size = sort_size.checked_next_power_of_two().ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "sort dimension is too large")
    })?;
    let segment_count = outer_size.checked_mul(inner_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "segment count overflows usize")
    })?;
    let total_padded = segment_count.checked_mul(padded_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "global workspace size overflows usize")
    })?;
    let logical_total = segment_count.checked_mul(sort_size).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "output size overflows usize")
    })?;

    let outer_size_u32 = u32::try_from(outer_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "outer dimension exceeds u32"))?;
    let sort_size_u32 = u32::try_from(sort_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "sort dimension exceeds u32"))?;
    let inner_size_u32 = u32::try_from(inner_size)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "inner dimension exceeds u32"))?;
    let padded_size_u32 = u32::try_from(padded_size).map_err(|_| {
        Error::backend_limitation("WebGPU", "sort", "padded sort dimension exceeds u32")
    })?;
    let segment_count_u32 = u32::try_from(segment_count)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "segment count exceeds u32"))?;
    let total_padded_u32 = u32::try_from(total_padded).map_err(|_| {
        Error::backend_limitation("WebGPU", "sort", "global workspace exceeds u32 elements")
    })?;
    let _logical_total_u32 = u32::try_from(logical_total)
        .map_err(|_| Error::backend_limitation("WebGPU", "sort", "output exceeds u32 elements"))?;

    let scratch_bytes = (total_padded as u64).checked_mul(4).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "global workspace byte size overflows")
    })?;
    let limits = cache.device().limits();
    let binding_limit = limits.max_storage_buffer_binding_size;
    let allocation_limit = limits.max_buffer_size;
    let effective_limit = binding_limit.min(allocation_limit);
    if scratch_bytes > effective_limit {
        return Err(Error::backend_limitation(
            "WebGPU",
            "sort",
            format!(
                "global workspace binding requires {scratch_bytes} bytes, device limit is {effective_limit}"
            ),
        ));
    }

    let make_storage = |label: &'static str, size: u64| {
        cache.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let keys = make_storage("global_sort_keys", scratch_bytes);
    let values = make_storage("global_sort_values", scratch_bytes);
    let indices = make_storage("global_sort_indices", scratch_bytes);
    let step_dummy = make_storage("global_sort_step_dummy", 4);
    let logical_bytes = (logical_total as u64) * 4;

    let temporary_values;
    let values_output = match values_output {
        Some(output) => output,
        None => {
            temporary_values = make_storage("global_sort_temporary_values_output", logical_bytes);
            &temporary_values
        }
    };
    let temporary_indices;
    let indices_output = match indices_output {
        Some(output) => output,
        None => {
            temporary_indices = make_storage("global_sort_temporary_indices_output", logical_bytes);
            &temporary_indices
        }
    };

    let base_params = GlobalSortParams {
        outer_size: outer_size_u32,
        sort_size: sort_size_u32,
        inner_size: inner_size_u32,
        padded_size: padded_size_u32,
        segment_count: segment_count_u32,
        dtype_tag,
        descending: u32::from(descending),
        k: 0,
        j: 0,
        total_padded: total_padded_u32,
        padding_0: 0,
        padding_1: 0,
    };
    let mut stage_params = vec![base_params];
    // k <= 512 is fused into one shared-memory tile dispatch.
    let mut k = 1024u32;
    while k <= padded_size_u32 {
        let mut j = k >> 1;
        while j > 0 {
            stage_params.push(GlobalSortParams {
                k,
                j,
                ..base_params
            });
            j >>= 1;
        }
        k = k.checked_shl(1).unwrap_or(0);
        if k == 0 {
            break;
        }
    }

    let param_size = std::mem::size_of::<GlobalSortParams>();
    let alignment = limits.min_uniform_buffer_offset_alignment as usize;
    let stride = param_size.div_ceil(alignment) * alignment;
    let params_bytes_len = stride.checked_mul(stage_params.len()).ok_or_else(|| {
        Error::backend_limitation("WebGPU", "sort", "parameter buffer size overflows")
    })?;
    let mut params_bytes = vec![0u8; params_bytes_len];
    for (stage_index, params) in stage_params.iter().enumerate() {
        let bytes = bytemuck::bytes_of(params);
        let offset = stage_index * stride;
        params_bytes[offset..offset + bytes.len()].copy_from_slice(bytes);
    }
    let params_buffer = cache.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("global_sort_params"),
        size: params_bytes_len as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&params_buffer, 0, &params_bytes);

    let module = cache.get_or_create_module("global_sort", GLOBAL_SORT_SHADER);
    let layout = cache.get_or_create_dynamic_uniform_layout(4, 0);
    let pack_pipeline =
        cache.get_or_create_pipeline("global_sort_pack", "pack_global_sort", &module, &layout);
    let tile_pipeline =
        cache.get_or_create_pipeline("global_sort_tiles", "sort_global_tiles", &module, &layout);
    let step_pipeline =
        cache.get_or_create_pipeline("global_sort_step", "global_bitonic_step", &module, &layout);
    let scatter_pipeline = cache.get_or_create_pipeline(
        "global_sort_scatter",
        "scatter_global_sort",
        &module,
        &layout,
    );
    let uniform_binding_size = param_size as u64;
    let pack_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[input, &keys, &values, &indices],
        &params_buffer,
        uniform_binding_size,
    );
    let step_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[&keys, &values, &indices, &step_dummy],
        &params_buffer,
        uniform_binding_size,
    );
    let scatter_bind_group = cache.create_bind_group_with_dynamic_uniform(
        &layout,
        &[&values, &indices, values_output, indices_output],
        &params_buffer,
        uniform_binding_size,
    );

    let dispatch_grid = |items: usize| -> Result<(u32, u32)> {
        let groups = items.div_ceil(256);
        let x = groups.clamp(1, 65_535);
        let y = groups.div_ceil(x);
        if y > 65_535 {
            return Err(Error::backend_limitation(
                "WebGPU",
                "sort",
                "global sort dispatch exceeds WebGPU's 2-D dispatch grid",
            ));
        }
        Ok((x as u32, y as u32))
    };
    let (padded_x, padded_y) = dispatch_grid(total_padded)?;
    let (logical_x, logical_y) = dispatch_grid(logical_total)?;
    let tile_groups = total_padded / 512;
    let tile_x = tile_groups.clamp(1, 65_535);
    let tile_y = tile_groups.div_ceil(tile_x);
    if tile_y > 65_535 {
        return Err(Error::backend_limitation(
            "WebGPU",
            "sort",
            "global sort tile dispatch exceeds WebGPU's 2-D dispatch grid",
        ));
    }
    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("global_sort"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_pack"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pack_pipeline);
        pass.set_bind_group(0, Some(&pack_bind_group), &[0]);
        pass.dispatch_workgroups(padded_x, padded_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_tiles"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&tile_pipeline);
        pass.set_bind_group(0, Some(&step_bind_group), &[0]);
        pass.dispatch_workgroups(tile_x as u32, tile_y as u32, 1);
    }
    for stage_index in 1..stage_params.len() {
        let dynamic_offset = u32::try_from(stage_index * stride).map_err(|_| {
            Error::backend_limitation("WebGPU", "sort", "dynamic uniform offset exceeds u32")
        })?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_step"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&step_pipeline);
        pass.set_bind_group(0, Some(&step_bind_group), &[dynamic_offset]);
        pass.dispatch_workgroups(padded_x, padded_y, 1);
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("global_sort_scatter"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&scatter_pipeline);
        pass.set_bind_group(0, Some(&scatter_bind_group), &[0]);
        pass.dispatch_workgroups(logical_x, logical_y, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
