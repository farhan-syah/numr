//! Element-wise WGSL kernel launchers
//!
//! Binary and broadcast-binary ops support F32, I32, U32.
//! Unary ops: most are F32 only; neg/abs/sign support I32, abs supports U32.
//! Scalar ops: F32, I32, U32.
//! Compare ops: F32, I32, U32.

use wgpu::{Buffer, Queue};

use super::pipeline::{LayoutKey, PipelineCache, workgroup_count};
use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static Shader Sources
// ============================================================================

const BINARY_F32_SHADER: &str = include_str!("binary.wgsl");
// The integer shaders share one saturating-power helper and one guarded
// integer division, prepended to each module that calls them. WGSL has no
// include, so this is where the single copy in the repository reaches the three
// I32 and three U32 modules. Order matters: WGSL has no forward declarations,
// so a helper must be concatenated ahead of its callers.
const BINARY_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_i32.wgsl"),
    include_str!("idiv_i32.wgsl"),
    include_str!("binary_i32.wgsl")
);
const BINARY_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_u32.wgsl"),
    include_str!("idiv_u32.wgsl"),
    include_str!("binary_u32.wgsl")
);
const BINARY_BROADCAST_F32_SHADER: &str = include_str!("binary_broadcast.wgsl");
const BINARY_BROADCAST_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_i32.wgsl"),
    include_str!("idiv_i32.wgsl"),
    include_str!("binary_broadcast_i32.wgsl")
);
const BINARY_BROADCAST_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_u32.wgsl"),
    include_str!("idiv_u32.wgsl"),
    include_str!("binary_broadcast_u32.wgsl")
);
const UNARY_SHADER: &str = include_str!("unary.wgsl");
const UNARY_I32_SHADER: &str = include_str!("unary_i32.wgsl");
const UNARY_U32_SHADER: &str = include_str!("unary_u32.wgsl");
const SCALAR_SHADER: &str = include_str!("scalar.wgsl");
const SCALAR_I32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_i32.wgsl"),
    include_str!("idiv_i32.wgsl"),
    include_str!("scalar_i32.wgsl")
);
const SCALAR_U32_SHADER: &str = concat!(
    include_str!("int_saturate.wgsl"),
    include_str!("ipow_u32.wgsl"),
    include_str!("idiv_u32.wgsl"),
    include_str!("scalar_u32.wgsl")
);
const COMPARE_SHADER: &str = include_str!("compare.wgsl");
const COMPARE_I32_SHADER: &str = include_str!("compare_i32.wgsl");
const COMPARE_U32_SHADER: &str = include_str!("compare_u32.wgsl");

const CAST_F32_TO_I32_SHADER: &str = include_str!("cast_f32_to_i32.wgsl");
const CAST_F32_TO_U32_SHADER: &str = include_str!("cast_f32_to_u32.wgsl");
const CAST_I32_TO_F32_SHADER: &str = include_str!("cast_i32_to_f32.wgsl");
const CAST_I32_TO_U32_SHADER: &str = include_str!("cast_i32_to_u32.wgsl");
const CAST_U32_TO_F32_SHADER: &str = include_str!("cast_u32_to_f32.wgsl");
const CAST_U32_TO_I32_SHADER: &str = include_str!("cast_u32_to_i32.wgsl");

// ============================================================================
// Binary Operations
// ============================================================================

/// Launch a binary element-wise operation: `out[i] = a[i] op b[i]`. F32, I32, U32.
pub fn launch_binary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let op_name = match op {
        "maximum" => "max",
        "minimum" => "min",
        _ => op,
    };

    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("binary_f32", BINARY_F32_SHADER, "f32"),
        DType::I32 => ("binary_i32", BINARY_I32_SHADER, "i32"),
        DType::U32 => ("binary_u32", BINARY_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    // atan2 has no integer meaning, so it stays float-only. Integer pow is
    // exact by squaring and saturating, matching CPU.
    if op_name == "atan2" && dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let entry_point: String = format!("{}_{}", op_name, suffix);
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

/// Launch a broadcast binary operation. F32, I32, U32.
/// Also serves broadcast compare: its mask dtype equals the input dtype, so the
/// buffer shape and bind-group layout are identical.
#[allow(clippy::too_many_arguments)]
pub fn launch_broadcast_binary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    a_strides: &Buffer,
    b_strides: &Buffer,
    out_strides: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let op_name = match op {
        "maximum" => "max",
        "minimum" => "min",
        _ => op,
    };

    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("binary_broadcast_f32", BINARY_BROADCAST_F32_SHADER, "f32"),
        DType::I32 => ("binary_broadcast_i32", BINARY_BROADCAST_I32_SHADER, "i32"),
        DType::U32 => ("binary_broadcast_u32", BINARY_BROADCAST_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = format!("broadcast_{}_{}", op_name, suffix);
    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 6,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(
        &layout,
        &[a, b, out, a_strides, b_strides, out_strides, params_buffer],
    );

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("broadcast_binary"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("broadcast_binary"),
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
// Unary Operations
// ============================================================================

/// Launch a unary operation: `out[i] = op(a[i])`.
/// Most ops are F32 only. neg/abs/sign support I32, neg/abs support U32.
pub fn launch_unary_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    // For I32, only neg, abs and sign are supported; for U32, neg and abs.
    match dtype {
        DType::F32 => {}
        DType::I32 => {
            if !matches!(op, "neg" | "abs" | "sign") {
                return Err(Error::UnsupportedDType { dtype, op });
            }
        }
        DType::U32 => {
            if !matches!(op, "neg" | "abs") {
                return Err(Error::UnsupportedDType { dtype, op });
            }
        }
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    }

    let (module_key, shader, entry_point): (&str, &str, String) = match dtype {
        DType::I32 => ("unary_i32", UNARY_I32_SHADER, format!("{}_i32", op)),
        DType::U32 => ("unary_u32", UNARY_U32_SHADER, format!("{}_u32", op)),
        DType::F32 => {
            let ep: &'static str = match op {
                "neg" => "neg_f32",
                "abs" => "abs_f32",
                "sqrt" => "sqrt_f32",
                "exp" => "exp_f32",
                "log" => "log_f32",
                "sin" => "sin_f32",
                "cos" => "cos_f32",
                "tan" => "tan_f32",
                "atan" => "atan_f32",
                "tanh" => "tanh_f32",
                "recip" => "recip_f32",
                "floor" => "floor_f32",
                "ceil" => "ceil_f32",
                "round" => "round_f32",
                "round_ties_even" => "round_ties_even_f32",
                "trunc" => "trunc_f32",
                "rsqrt" => "rsqrt_f32",
                "cbrt" => "cbrt_f32",
                "exp2" => "exp2_f32",
                "expm1" => "expm1_f32",
                "log2" => "log2_f32",
                "log10" => "log10_f32",
                "log1p" => "log1p_f32",
                "asin" => "asin_f32",
                "acos" => "acos_f32",
                "sinh" => "sinh_f32",
                "cosh" => "cosh_f32",
                "asinh" => "asinh_f32",
                "acosh" => "acosh_f32",
                "atanh" => "atanh_f32",
                "square" => "square_f32",
                "sign" => "sign_f32",
                "relu" => "relu_f32",
                "sigmoid" => "sigmoid_f32",
                "silu" => "silu_f32",
                "gelu" => "gelu_f32",
                "isnan" => "isnan_f32",
                "isinf" => "isinf_f32",
                _ => return Err(Error::Internal(format!("Unknown unary op: {}", op))),
            };
            ("unary_f32", UNARY_SHADER, ep.to_string())
        }
        _ => unreachable!(),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
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
// Scalar Operations
// ============================================================================

/// Launch a scalar operation: `out[i] = a[i] op scalar`. F32, I32, U32.
/// leaky_relu and elu are F32-only.
pub fn launch_scalar_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    // leaky_relu and elu have no integer meaning, so they stay float-only.
    if matches!(op, "leaky_relu" | "elu") && dtype != DType::F32 {
        return Err(Error::UnsupportedDType { dtype, op });
    }

    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("scalar_f32", SCALAR_SHADER, "f32"),
        DType::I32 => ("scalar_i32", SCALAR_I32_SHADER, "i32"),
        DType::U32 => ("scalar_u32", SCALAR_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = match dtype {
        DType::F32 => {
            // F32 uses static entry points
            let ep: &'static str = match op {
                "add_scalar" => "add_scalar_f32",
                "sub_scalar" => "sub_scalar_f32",
                "rsub_scalar" => "rsub_scalar_f32",
                "mul_scalar" => "mul_scalar_f32",
                "div_scalar" => "div_scalar_f32",
                "pow_scalar" => "pow_scalar_f32",
                "leaky_relu" => "leaky_relu_f32",
                "elu" => "elu_f32",
                _ => return Err(Error::Internal(format!("Unknown scalar op: {}", op))),
            };
            ep.to_string()
        }
        _ => {
            // I32/U32: format entry point
            match op {
                "add_scalar" | "sub_scalar" | "rsub_scalar" | "mul_scalar" | "div_scalar"
                | "pow_scalar" => format!("{}_{}", op, suffix),
                _ => return Err(Error::Internal(format!("Unknown scalar op: {}", op))),
            }
        }
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
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
// Comparison Operations
// ============================================================================

/// Launch a comparison operation: `out[i] = (a[i] op b[i]) ? 1 : 0`. F32, I32, U32.
/// The mask carries the input dtype, matching CPU and CUDA.
pub fn launch_compare_op(
    cache: &PipelineCache,
    queue: &Queue,
    op: &'static str,
    a: &Buffer,
    b: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    dtype: DType,
) -> Result<()> {
    let (module_key, shader, suffix) = match dtype {
        DType::F32 => ("compare_f32", COMPARE_SHADER, "f32"),
        DType::I32 => ("compare_i32", COMPARE_I32_SHADER, "i32"),
        DType::U32 => ("compare_u32", COMPARE_U32_SHADER, "u32"),
        _ => return Err(Error::UnsupportedDType { dtype, op }),
    };

    let entry_point: String = match op {
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => format!("{}_{}", op, suffix),
        _ => return Err(Error::Internal(format!("Unknown compare op: {}", op))),
    };

    let module = cache.get_or_create_module(module_key, shader);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 3,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_dynamic_pipeline(module_key, &entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, b, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(op) });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(op),
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
// Cast Operations
// ============================================================================

/// Launch a cast operation: `out[i] = DstType(a[i])`. Supports F32 ↔ I32 ↔ U32.
pub fn launch_cast_op(
    cache: &PipelineCache,
    queue: &Queue,
    a: &Buffer,
    out: &Buffer,
    params_buffer: &Buffer,
    numel: usize,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<()> {
    if src_dtype == dst_dtype {
        return Ok(());
    }

    let (module_name, entry_point, shader_source): (&'static str, &'static str, &'static str) =
        match (src_dtype, dst_dtype) {
            (DType::F32, DType::I32) => ("cast_f32_i32", "cast_f32_to_i32", CAST_F32_TO_I32_SHADER),
            (DType::F32, DType::U32) => ("cast_f32_u32", "cast_f32_to_u32", CAST_F32_TO_U32_SHADER),
            (DType::I32, DType::F32) => ("cast_i32_f32", "cast_i32_to_f32", CAST_I32_TO_F32_SHADER),
            (DType::I32, DType::U32) => ("cast_i32_u32", "cast_i32_to_u32", CAST_I32_TO_U32_SHADER),
            (DType::U32, DType::F32) => ("cast_u32_f32", "cast_u32_to_f32", CAST_U32_TO_F32_SHADER),
            (DType::U32, DType::I32) => ("cast_u32_i32", "cast_u32_to_i32", CAST_U32_TO_I32_SHADER),
            _ => {
                return Err(Error::UnsupportedDType {
                    dtype: src_dtype,
                    op: "cast",
                });
            }
        };

    let module = cache.get_or_create_module(module_name, shader_source);
    let layout = cache.get_or_create_layout(LayoutKey {
        num_storage_buffers: 2,
        num_uniform_buffers: 1,
        num_readonly_storage: 0,
    });
    let pipeline = cache.get_or_create_pipeline(module_name, entry_point, &module, &layout);
    let bind_group = cache.create_bind_group(&layout, &[a, out, params_buffer]);

    let mut encoder = cache
        .device()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("cast"),
        });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cast"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroup_count(numel), 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}
