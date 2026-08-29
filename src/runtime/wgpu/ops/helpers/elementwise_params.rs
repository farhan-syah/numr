//! Params structs for the element-wise kernels: binary, unary, scalar, clamp,
//! where and cast.
//!
//! `ScalarParams` and `ClampParams` carry their scalars as raw bits already
//! encoded in the tensor dtype, which is why they have constructors instead of
//! plain field initialization.

use crate::dtype::DType;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BinaryParams {
    pub(crate) numel: u32,
}

/// Parameters for broadcast binary operations.
/// Matches the BroadcastBinaryParams struct in WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BroadcastBinaryParams {
    pub(crate) numel: u32,
    pub(crate) ndim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct UnaryParams {
    pub(crate) numel: u32,
}

/// Parameters for tensor-scalar operations.
///
/// `scalar_bits` is the scalar re-encoded per dtype, not a plain `f32`. The
/// scalar_f32/scalar_i32/scalar_u32 WGSL shaders all read this same 4-byte
/// field but declare it as their own type (`f32`, `i32`, `u32`), so the bit
/// pattern written here must already match the tensor's dtype — a raw `f32`
/// cast would be bit-reinterpreted as garbage by the integer shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScalarParams {
    pub(crate) numel: u32,
    pub(crate) scalar_bits: u32,
}

impl ScalarParams {
    /// Build params for `numel` elements of `dtype`, encoding `scalar` the
    /// same way the CPU backend does: convert to the element type first
    /// (`as` cast, saturating for integers), then take that value's bits.
    pub(crate) fn new(numel: u32, scalar: f64, dtype: DType) -> Self {
        let scalar_bits = match dtype {
            DType::I32 => (scalar as i32).to_ne_bytes(),
            DType::U32 => (scalar as u32).to_ne_bytes(),
            _ => (scalar as f32).to_ne_bytes(),
        };
        Self {
            numel,
            scalar_bits: u32::from_ne_bytes(scalar_bits),
        }
    }
}

/// Parameters for clamp operation.
/// Padding ensures 16-byte alignment for WebGPU uniform buffers.
///
/// `min_bits` and `max_bits` are the bounds re-encoded per dtype, not plain
/// `f32`s. clamp_f32/clamp_i32/clamp_u32 all read these same two 4-byte fields
/// but declare them as their own type, so the bit pattern written here must
/// already match the tensor's dtype - the same rule [`ScalarParams`] follows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ClampParams {
    pub(crate) numel: u32,
    pub(crate) min_bits: u32,
    pub(crate) max_bits: u32,
    /// Padding for 16-byte alignment (WebGPU uniform buffer requirement)
    pub(crate) _pad0: u32,
}

impl ClampParams {
    /// Build params for `numel` elements of `dtype`, encoding each bound the
    /// way the CPU backend converts it: an `as` cast to the element type, which
    /// truncates toward zero and saturates for integers.
    pub(crate) fn new(numel: u32, min_val: f64, max_val: f64, dtype: DType) -> Self {
        let encode = |v: f64| -> u32 {
            match dtype {
                DType::I32 => u32::from_ne_bytes((v as i32).to_ne_bytes()),
                DType::U32 => v as u32,
                _ => u32::from_ne_bytes((v as f32).to_ne_bytes()),
            }
        };
        Self {
            numel,
            min_bits: encode(min_val),
            max_bits: encode(max_val),
            _pad0: 0,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct WhereParams {
    pub(crate) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct WhereBroadcastParams {
    pub(crate) numel: u32,
    pub(crate) ndim: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CastParams {
    pub(crate) numel: u32,
}
