//! WGSL source table for the cat, repeat, pad, roll, arange, linspace, eye and
//! random kernels.
//!
//! One `(op, dtype)` lookup returns the shader source, its module cache key and
//! the entry point to dispatch, so the dtype policy for this whole family is
//! stated once here instead of once per launcher.

use crate::dtype::DType;
use crate::error::{Error, Result};

// ============================================================================
// Static shaders — cat (data-movement: F32 / I32 / U32)
// ============================================================================

const CAT_COPY_SHADER_F32: &str = include_str!("../cat_copy_f32.wgsl");
const CAT_COPY_SHADER_I32: &str = include_str!("../cat_copy_i32.wgsl");
const CAT_COPY_SHADER_U32: &str = include_str!("../cat_copy_u32.wgsl");

// ============================================================================
// Static shaders — repeat (data-movement: F32 / I32 / U32)
// ============================================================================

const REPEAT_SHADER_F32: &str = include_str!("../repeat_f32.wgsl");
const REPEAT_SHADER_I32: &str = include_str!("../repeat_i32.wgsl");
const REPEAT_SHADER_U32: &str = include_str!("../repeat_u32.wgsl");

// ============================================================================
// Static shaders — pad (data-movement: F32 / I32 / U32)
// ============================================================================

const PAD_SHADER_F32: &str = include_str!("../pad_f32.wgsl");
const PAD_SHADER_I32: &str = include_str!("../pad_i32.wgsl");
const PAD_SHADER_U32: &str = include_str!("../pad_u32.wgsl");

// ============================================================================
// Static shaders — roll (data-movement: F32 / I32 / U32)
// ============================================================================

const ROLL_SHADER_F32: &str = include_str!("../roll_f32.wgsl");
const ROLL_SHADER_I32: &str = include_str!("../roll_i32.wgsl");
const ROLL_SHADER_U32: &str = include_str!("../roll_u32.wgsl");

// ============================================================================
// Static shaders — arange (F32 / I32 / U32)
// ============================================================================

const ARANGE_SHADER_F32: &str = include_str!("../arange_f32.wgsl");

// The integer variants evaluate in f32 and clamp at the store, so they need the
// shared conversion guards. WGSL has neither an include nor forward
// declarations, so the order below is load-bearing.
const ARANGE_SHADER_I32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_from_float.wgsl"),
    include_str!("../arange_i32.wgsl"),
);
const ARANGE_SHADER_U32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_from_float.wgsl"),
    include_str!("../arange_u32.wgsl"),
);

// ============================================================================
// Static shaders — linspace (F32 / I32 / U32)
// ============================================================================

const LINSPACE_SHADER_F32: &str = include_str!("../linspace_f32.wgsl");

// The integer variants need the shared 64-bit helpers, and WGSL has neither an
// include nor forward declarations, so the order below is load-bearing.
const LINSPACE_SHADER_I32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_matmul_acc.wgsl"),
    include_str!("../int_wide_div.wgsl"),
    include_str!("../int_from_float.wgsl"),
    include_str!("../linspace_i32.wgsl"),
);
const LINSPACE_SHADER_U32: &str = concat!(
    include_str!("../int_saturate.wgsl"),
    include_str!("../int_matmul_acc.wgsl"),
    include_str!("../int_wide_div.wgsl"),
    include_str!("../int_from_float.wgsl"),
    include_str!("../linspace_u32.wgsl"),
);

// ============================================================================
// Static shaders — eye (F32 / I32 / U32)
// ============================================================================

const EYE_SHADER_F32: &str = include_str!("../eye_f32.wgsl");
const EYE_SHADER_I32: &str = include_str!("../eye_i32.wgsl");
const EYE_SHADER_U32: &str = include_str!("../eye_u32.wgsl");

// ============================================================================
// Static shaders — rand / randn (F32 only)
// ============================================================================

const RAND_SHADER_F32: &str = include_str!("../rand_f32.wgsl");
const RANDN_SHADER_F32: &str = include_str!("../randn_f32.wgsl");

// ============================================================================
// Static shaders — randint (I32 / U32 only)
// ============================================================================

const RANDINT_SHADER_I32: &str = include_str!("../randint_i32.wgsl");
const RANDINT_SHADER_U32: &str = include_str!("../randint_u32.wgsl");

pub(super) fn shader_info(
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
