//! Shape operation WGSL kernel launchers
//!
//! Provides launchers for shape operations including:
//! - cat: Concatenate tensors along a dimension
//! - stack: Stack tensors along a new dimension (uses cat + unsqueeze)
//! - repeat: Tile tensor along all dimensions
//! - pad: Add padding around tensor
//! - roll: Circular shift along a dimension
//! - split/chunk: Zero-copy views using narrow (no kernel needed)
//!
//! All copy operations run entirely on GPU with no CPU fallback.
//!
//! dtype policy (Option C):
//! - cat, repeat, pad, roll → DATA-MOVEMENT → support F32, I32, U32
//! - arange, eye → can produce F32 / I32 / U32
//! - linspace → F32, I32, U32 (integer output computed in exact 64-bit math)
//! - rand, randn → F32 only (math)
//! - randint → I32 / U32 only
//! - multinomial → F32 only (math)

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders — cat (data-movement: F32 / I32 / U32)
// ============================================================================

const CAT_COPY_SHADER_F32: &str = include_str!("cat_copy_f32.wgsl");
const CAT_COPY_SHADER_I32: &str = include_str!("cat_copy_i32.wgsl");
const CAT_COPY_SHADER_U32: &str = include_str!("cat_copy_u32.wgsl");

// ============================================================================
// Static shaders — repeat (data-movement: F32 / I32 / U32)
// ============================================================================

const REPEAT_SHADER_F32: &str = include_str!("repeat_f32.wgsl");
const REPEAT_SHADER_I32: &str = include_str!("repeat_i32.wgsl");
const REPEAT_SHADER_U32: &str = include_str!("repeat_u32.wgsl");

// ============================================================================
// Static shaders — pad (data-movement: F32 / I32 / U32)
// ============================================================================

const PAD_SHADER_F32: &str = include_str!("pad_f32.wgsl");
const PAD_SHADER_I32: &str = include_str!("pad_i32.wgsl");
const PAD_SHADER_U32: &str = include_str!("pad_u32.wgsl");

// ============================================================================
// Static shaders — roll (data-movement: F32 / I32 / U32)
// ============================================================================

const ROLL_SHADER_F32: &str = include_str!("roll_f32.wgsl");
const ROLL_SHADER_I32: &str = include_str!("roll_i32.wgsl");
const ROLL_SHADER_U32: &str = include_str!("roll_u32.wgsl");

// ============================================================================
// Static shaders — arange (F32 / I32 / U32)
// ============================================================================

const ARANGE_SHADER_F32: &str = include_str!("arange_f32.wgsl");

// The integer variants evaluate in f32 and clamp at the store, so they need the
// shared conversion guards. WGSL has neither an include nor forward
// declarations, so the order below is load-bearing.
const ARANGE_SHADER_I32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_from_float.wgsl"),
    include_str!("arange_i32.wgsl"),
);
const ARANGE_SHADER_U32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_from_float.wgsl"),
    include_str!("arange_u32.wgsl"),
);

// ============================================================================
// Static shaders — linspace (F32 / I32 / U32)
// ============================================================================

const LINSPACE_SHADER_F32: &str = include_str!("linspace_f32.wgsl");

// The integer variants need the shared 64-bit helpers, and WGSL has neither an
// include nor forward declarations, so the order below is load-bearing.
const LINSPACE_SHADER_I32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("int_wide_div.wgsl"),
    include_str!("int_from_float.wgsl"),
    include_str!("linspace_i32.wgsl"),
);
const LINSPACE_SHADER_U32: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("int_matmul_acc.wgsl"),
    include_str!("int_wide_div.wgsl"),
    include_str!("int_from_float.wgsl"),
    include_str!("linspace_u32.wgsl"),
);

// ============================================================================
// Static shaders — eye (F32 / I32 / U32)
// ============================================================================

const EYE_SHADER_F32: &str = include_str!("eye_f32.wgsl");
const EYE_SHADER_I32: &str = include_str!("eye_i32.wgsl");
const EYE_SHADER_U32: &str = include_str!("eye_u32.wgsl");

// ============================================================================
// Static shaders — rand / randn (F32 only)
// ============================================================================

const RAND_SHADER_F32: &str = include_str!("rand_f32.wgsl");
const RANDN_SHADER_F32: &str = include_str!("randn_f32.wgsl");

// ============================================================================
// Static shaders — randint (I32 / U32 only)
// ============================================================================

const RANDINT_SHADER_I32: &str = include_str!("randint_i32.wgsl");
const RANDINT_SHADER_U32: &str = include_str!("randint_u32.wgsl");

// ============================================================================
// Static shaders — multinomial (F32 only)
// ============================================================================

const MULTINOMIAL_WITH_REPLACEMENT_SHADER_F32: &str =
    include_str!("multinomial_with_replacement_f32.wgsl");
const MULTINOMIAL_WITHOUT_REPLACEMENT_SHADER_F32: &str =
    include_str!("multinomial_without_replacement_f32.wgsl");

// ============================================================================
// Helper: shader_info returns (shader_source, module_key, entry_point)
// ============================================================================

fn shader_info(
    op: &'static str,
    dtype: DType,
) -> Result<(&'static str, &'static str, &'static str)> {
    match (op, dtype) {
        // cat_copy
        ("cat_copy", DType::F32) => Ok((CAT_COPY_SHADER_F32, "cat_copy_f32", "cat_copy_f32")),
        ("cat_copy", DType::I32) => Ok((CAT_COPY_SHADER_I32, "cat_copy_i32", "cat_copy_i32")),
        ("cat_copy", DType::U32) => Ok((CAT_COPY_SHADER_U32, "cat_copy_u32", "cat_copy_u32")),
        // repeat
        ("repeat", DType::F32) => Ok((REPEAT_SHADER_F32, "repeat_f32", "repeat_f32")),
        ("repeat", DType::I32) => Ok((REPEAT_SHADER_I32, "repeat_i32", "repeat_i32")),
        ("repeat", DType::U32) => Ok((REPEAT_SHADER_U32, "repeat_u32", "repeat_u32")),
        // pad
        ("pad", DType::F32) => Ok((PAD_SHADER_F32, "pad_f32", "pad_f32")),
        ("pad", DType::I32) => Ok((PAD_SHADER_I32, "pad_i32", "pad_i32")),
        ("pad", DType::U32) => Ok((PAD_SHADER_U32, "pad_u32", "pad_u32")),
        // roll
        ("roll", DType::F32) => Ok((ROLL_SHADER_F32, "roll_f32", "roll_f32")),
        ("roll", DType::I32) => Ok((ROLL_SHADER_I32, "roll_i32", "roll_i32")),
        ("roll", DType::U32) => Ok((ROLL_SHADER_U32, "roll_u32", "roll_u32")),
        // arange
        ("arange", DType::F32) => Ok((ARANGE_SHADER_F32, "arange_f32", "arange_f32")),
        ("arange", DType::I32) => Ok((ARANGE_SHADER_I32, "arange_i32", "arange_i32")),
        ("arange", DType::U32) => Ok((ARANGE_SHADER_U32, "arange_u32", "arange_u32")),
        // linspace
        ("linspace", DType::F32) => Ok((LINSPACE_SHADER_F32, "linspace_f32", "linspace_f32")),
        ("linspace", DType::I32) => Ok((LINSPACE_SHADER_I32, "linspace_i32", "linspace_i32")),
        ("linspace", DType::U32) => Ok((LINSPACE_SHADER_U32, "linspace_u32", "linspace_u32")),
        // eye
        ("eye", DType::F32) => Ok((EYE_SHADER_F32, "eye_f32", "eye_f32")),
        ("eye", DType::I32) => Ok((EYE_SHADER_I32, "eye_i32", "eye_i32")),
        ("eye", DType::U32) => Ok((EYE_SHADER_U32, "eye_u32", "eye_u32")),
        // rand
        ("rand", DType::F32) => Ok((RAND_SHADER_F32, "rand_f32", "rand_f32")),
        // randn
        ("randn", DType::F32) => Ok((RANDN_SHADER_F32, "randn_f32", "randn_f32")),
        // randint
        ("randint", DType::I32) => Ok((RANDINT_SHADER_I32, "randint_i32", "randint_i32")),
        ("randint", DType::U32) => Ok((RANDINT_SHADER_U32, "randint_u32", "randint_u32")),
        _ => Err(Error::UnsupportedDType { dtype, op }),
    }
}

// ============================================================================
// Cat Copy Operation
// ============================================================================

/// Launch a cat_copy operation kernel.
///
/// Copies data from a source tensor to the appropriate position in the
/// concatenated output tensor. This is called once per input tensor.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer (must be contiguous)
/// * `dst` - Destination tensor buffer (output)
/// * `params_buffer` - Uniform buffer containing CatParams
/// * `total_elements` - Total elements in source tensor
/// * `dtype` - Data type of tensors
#[allow(clippy::too_many_arguments)]
pub fn launch_cat_copy(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("cat_copy", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cat_copy"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cat_copy"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Arange Operation
// ============================================================================

/// Launch an arange operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing ArangeParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output
pub fn launch_arange(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("arange", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("arange"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("arange"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Linspace Operation
// ============================================================================

/// Launch a linspace operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing LinspaceParams
/// * `steps` - Number of steps to generate
/// * `dtype` - Data type of output (F32, I32 or U32)
pub fn launch_linspace(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    steps: usize,
    dtype: DType,
) -> Result<()> {
    if steps == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("linspace", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("linspace"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("linspace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(steps), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Eye Operation
// ============================================================================

/// Launch an eye (identity matrix) operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing EyeParams
/// * `numel` - Total elements (n * m)
/// * `dtype` - Data type of output
pub fn launch_eye(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("eye", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("eye") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("eye"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Random Operations
// ============================================================================

/// Launch a rand operation kernel (uniform [0, 1)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be F32)
pub fn launch_rand(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("rand", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rand"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rand"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a randn operation kernel (standard normal N(0, 1)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandnParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be F32)
pub fn launch_randn(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("randn", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("randn"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("randn"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a randint operation kernel (uniform integers in [low, high)).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `out` - Output buffer
/// * `params_buffer` - Uniform buffer containing RandintParams
/// * `numel` - Number of elements to generate
/// * `dtype` - Data type of output (must be I32 or U32)
pub fn launch_randint(
    cache: &PipelineCache,
    queue: &Queue,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    if numel == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("randint", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 1,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("randint"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("randint"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Repeat Operation
// ============================================================================

/// Launch a repeat operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing RepeatParams
/// * `total_elements` - Total elements in output tensor
/// * `dtype` - Data type of tensors
pub fn launch_repeat(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("repeat", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("repeat"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("repeat"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Pad Operation
// ============================================================================

/// Launch a pad operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing PadParams
/// * `total_elements` - Total elements in output tensor
/// * `dtype` - Data type of tensors
pub fn launch_pad(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("pad", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("pad") });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("pad"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Roll Operation
// ============================================================================

/// Launch a roll operation kernel.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `src` - Source tensor buffer
/// * `dst` - Destination tensor buffer
/// * `params_buffer` - Uniform buffer containing RollParams
/// * `total_elements` - Total elements in tensor
/// * `dtype` - Data type of tensors
pub fn launch_roll(
    cache: &PipelineCache,
    queue: &Queue,
    src: &Buffer,
    dst: &Buffer,
    params_buffer: &Buffer,
    total_elements: usize,
    dtype: DType,
) -> Result<()> {
    if total_elements == 0 {
        return Ok(());
    }

    let (shader, module_key, entry_point) = shader_info("roll", dtype)?;
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_key, entry_point, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[src, dst, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("roll"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("roll"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_elements), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

// ============================================================================
// Multinomial Operations
// ============================================================================

/// Launch multinomial sampling with replacement kernel.
///
/// Samples indices from categorical distributions defined by probability rows.
/// Each thread independently samples one output using the CDF inversion method.
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `probs` - Input probability buffer (num_distributions × num_categories)
/// * `out` - Output buffer for sampled indices (num_distributions × num_samples)
/// * `params_buffer` - Uniform buffer containing MultinomialWithReplacementParams
/// * `total_samples` - Total number of samples (num_distributions × num_samples)
/// * `input_dtype` - Data type of probability tensor (F32 only for WebGPU)
#[allow(clippy::too_many_arguments)]
pub fn launch_multinomial_with_replacement(
    cache: &PipelineCache,
    queue: &Queue,
    probs: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    total_samples: usize,
    input_dtype: DType,
) -> Result<()> {
    if total_samples == 0 {
        return Ok(());
    }

    // Only F32 input is supported for WebGPU multinomial
    if !matches!(input_dtype, DType::F32) {
        return Err(Error::UnsupportedDType {
            dtype: input_dtype,
            op: "multinomial",
        });
    }

    let name = "multinomial_with_replacement_f32";
    let module = cache.get_or_create_module(name, MULTINOMIAL_WITH_REPLACEMENT_SHADER_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[probs, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multinomial_with_replacement"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multinomial_with_replacement"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(total_samples), 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch multinomial sampling without replacement kernel.
///
/// Samples indices from categorical distributions without replacement using
/// shared memory to track modified probabilities. Thread 0 does sequential
/// sampling within each workgroup (one workgroup per distribution).
///
/// # Arguments
///
/// * `cache` - Pipeline cache for shader compilation
/// * `queue` - WGPU command queue
/// * `probs` - Input probability buffer (num_distributions × num_categories)
/// * `out` - Output buffer for sampled indices (num_distributions × num_samples)
/// * `params_buffer` - Uniform buffer containing MultinomialWithoutReplacementParams
/// * `num_distributions` - Number of distributions (one workgroup per distribution)
/// * `input_dtype` - Data type of probability tensor (F32 only for WebGPU)
#[allow(clippy::too_many_arguments)]
pub fn launch_multinomial_without_replacement(
    cache: &PipelineCache,
    queue: &Queue,
    probs: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    num_distributions: usize,
    input_dtype: DType,
) -> Result<()> {
    if num_distributions == 0 {
        return Ok(());
    }

    // Only F32 input is supported for WebGPU multinomial
    if !matches!(input_dtype, DType::F32) {
        return Err(Error::UnsupportedDType {
            dtype: input_dtype,
            op: "multinomial",
        });
    }

    let name = "multinomial_without_replacement_f32";
    let module = cache.get_or_create_module(name, MULTINOMIAL_WITHOUT_REPLACEMENT_SHADER_F32);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(name, name, &module, &layout);

    let bind_group = cache.create_bind_group(&layout, &[probs, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("multinomial_without_replacement"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multinomial_without_replacement"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        // One workgroup per distribution - thread 0 handles all samples
        pass.dispatch_workgroups(num_distributions as u32, 1, 1);
    }

    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
