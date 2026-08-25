//! FFT kernel launchers for WebGPU
//!
//! Provides dispatch functions for FFT compute shaders (F32 only on WebGPU).

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::error::Result;
use wgpu::{Buffer, Queue};

/// Maximum FFT size for shared memory (workgroup) implementation.
/// Matches the shared memory array size in stockham_fft.wgsl.
pub const MAX_WORKGROUP_FFT_SIZE: usize = 256;

const STOCKHAM_FFT_SHADER: &str = include_str!("stockham_fft.wgsl");
// entry points: "stockham_fft_small", "stockham_fft_stage", "scale_complex"

const FFTSHIFT_SHADER: &str = include_str!("fftshift.wgsl");
// entry points: "fftshift", "ifftshift"

const RFFT_PACK_SHADER: &str = include_str!("rfft_pack.wgsl");
// entry point: "rfft_pack"

const IRFFT_UNPACK_SHADER: &str = include_str!("irfft_unpack.wgsl");
// entry point: "irfft_unpack"

const HERMITIAN_EXTEND_SHADER: &str = include_str!("hermitian_extend.wgsl");
// entry point: "hermitian_extend"

const RFFT_TRUNCATE_SHADER: &str = include_str!("rfft_truncate.wgsl");
// entry point: "rfft_truncate"

const COPY_COMPLEX_SHADER: &str = include_str!("copy_complex.wgsl");
// entry point: "copy_complex"

/// Launch batched Stockham FFT for small transforms (N <= MAX_WORKGROUP_FFT_SIZE)
///
/// Each workgroup processes one FFT using shared memory.
pub fn launch_stockham_fft_batched(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    if n > MAX_WORKGROUP_FFT_SIZE {
        return Err(crate::error::Error::Internal(format!(
            "FFT size {} exceeds max workgroup FFT size {}",
            n, MAX_WORKGROUP_FFT_SIZE
        )));
    }

    let module = pipeline_cache.get_or_create_module("stockham_fft", STOCKHAM_FFT_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = pipeline_cache.get_or_create_pipeline(
        "stockham_fft",
        "stockham_fft_small",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("stockham_fft_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per batch element
        pass.dispatch_workgroups(batch_size as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch single stage of Stockham FFT for large transforms
pub fn launch_stockham_fft_stage(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("stockham_fft", STOCKHAM_FFT_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = pipeline_cache.get_or_create_pipeline(
        "stockham_fft",
        "stockham_fft_stage",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft_stage_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft_stage_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One thread per butterfly operation
        let butterflies = n / 2;
        pass.dispatch_workgroups(workgroup_count(butterflies), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch scale complex shader
pub fn launch_scale_complex(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("stockham_fft", STOCKHAM_FFT_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("stockham_fft", "scale_complex", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("scale_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("scale_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch fftshift shader
pub fn launch_fftshift(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("fftshift", FFTSHIFT_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = pipeline_cache.get_or_create_pipeline("fftshift", "fftshift", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fftshift_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fftshift_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch ifftshift shader
pub fn launch_ifftshift(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("fftshift", FFTSHIFT_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = pipeline_cache.get_or_create_pipeline("fftshift", "ifftshift", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ifftshift_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ifftshift_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch rfft_pack shader (real to complex)
pub fn launch_rfft_pack(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("rfft_pack", RFFT_PACK_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("rfft_pack", "rfft_pack", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rfft_pack_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rfft_pack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch irfft_unpack shader (complex to real)
pub fn launch_irfft_unpack(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("irfft_unpack", IRFFT_UNPACK_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("irfft_unpack", "irfft_unpack", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("irfft_unpack_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("irfft_unpack_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch hermitian_extend shader
pub fn launch_hermitian_extend(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("hermitian_extend", HERMITIAN_EXTEND_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline = pipeline_cache.get_or_create_pipeline(
        "hermitian_extend",
        "hermitian_extend",
        &module,
        &layout,
    );

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("hermitian_extend_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("hermitian_extend_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch rfft_truncate shader
pub fn launch_rfft_truncate(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    half_n: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("rfft_truncate", RFFT_TRUNCATE_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("rfft_truncate", "rfft_truncate", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rfft_truncate_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rfft_truncate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(half_n), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch copy_complex shader
pub fn launch_copy_complex(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    output: &Buffer,
    params: &Buffer,
    n: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("copy_complex", COPY_COMPLEX_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("copy_complex", "copy_complex", &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("copy_complex_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy_complex_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

const FFT_BLUESTEIN_SHADER: &str = include_str!("fft_bluestein.wgsl");
// entry points: "bluestein_premultiply", "bluestein_pointwise_mul",
//               "bluestein_postmultiply"

/// Launch one Bluestein stage.
///
/// All three stages share a binding set and a dispatch shape — `input`,
/// `table`, `output`, `params` — so they share a launcher. `entry_point` picks
/// the stage; `rows` is the per-batch element count that stage writes (`m` for
/// premultiply and pointwise, `out_n` for postmultiply).
pub fn launch_bluestein_stage(
    pipeline_cache: &PipelineCache,
    queue: &Queue,
    entry_point: &'static str,
    input: &Buffer,
    table: &Buffer,
    output: &Buffer,
    params: &Buffer,
    rows: usize,
    batch_size: usize,
) -> Result<()> {
    let module = pipeline_cache.get_or_create_module("fft_bluestein", FFT_BLUESTEIN_SHADER);

    let layout = pipeline_cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });

    let pipeline =
        pipeline_cache.get_or_create_pipeline("fft_bluestein", entry_point, &module, &layout);

    let bind_group = pipeline_cache.create_bind_group(&layout, &[input, table, output, params]);

    let mut encoder =
        pipeline_cache
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bluestein_encoder"),
            });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bluestein_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(rows), batch_size as u32, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
