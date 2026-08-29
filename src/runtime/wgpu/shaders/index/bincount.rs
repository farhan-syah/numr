//! Bincount kernel launcher.
//!
//! The unweighted kernel counts integer values with integer atomics; the
//! weighted kernel accumulates F32 weights and is F32-only because it needs
//! float atomics.

use wgpu::{Buffer, Queue};

use super::super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

const BINCOUNT_UNWEIGHTED_SHADER: &str = include_str!("../bincount_i32.wgsl");
const BINCOUNT_WEIGHTED_SHADER_F32: &str = include_str!("../bincount_weighted_f32.wgsl");

/// Launch a bincount operation kernel.
///
/// Counts occurrences of each value in an integer tensor.
/// Input: integer tensor with values in `[0, minlength)`
/// Output: count tensor of shape `[minlength]`
pub fn launch_bincount(
    cache: &PipelineCache,
    queue: &Queue,
    input: &Buffer,
    weights: Option<&Buffer>,
    output: &Buffer,
    params_buffer: &Buffer,
    n: usize,
    weights_dtype: Option<DType>,
) -> Result<()> {
    let (name, shader) = if let Some(dtype) = weights_dtype {
        // bincount_weighted is F32 only (uses float atomics)
        if dtype != DType::F32 {
            return Err(Error::UnsupportedDType {
                dtype,
                op: "bincount_weighted",
            });
        }
        ("bincount_weighted_f32", BINCOUNT_WEIGHTED_SHADER_F32)
    } else {
        ("bincount_i32", BINCOUNT_UNWEIGHTED_SHADER)
    };

    let module = cache.get_or_create_module(name, shader);

    let (layout, bind_group) = if let Some(weights_buf) = weights {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 3,
            num_uniform_buffers: 1,
            num_readonly_storage: 2, // input and weights are read-only
        });
        let bind_group =
            cache.create_bind_group(&layout, &[input, weights_buf, output, params_buffer]);
        (layout, bind_group)
    } else {
        let layout = cache.get_or_create_layout(LayoutKey {
            num_storage_buffers: 2,
            num_uniform_buffers: 1,
            num_readonly_storage: 1, // input is read-only
        });
        let bind_group = cache.create_bind_group(&layout, &[input, output, params_buffer]);
        (layout, bind_group)
    };

    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("bincount"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bincount"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(n), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
