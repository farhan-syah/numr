//! Params structs for the shape kernels: cat, repeat, pad and roll.
//!
//! WGSL uniform buffers need 16-byte aligned array elements, so shape arrays
//! travel packed as `array<vec4<u32>, 2>` and are capped at `MAX_DIMS`.

/// Maximum number of dimensions supported by WebGPU shape operation shaders.
/// WGSL doesn't support dynamic arrays in uniform buffers, so we use fixed-size arrays.
pub const MAX_DIMS: usize = 8;

/// Pack a flat `` `[u32; 8]` `` array into `` `[[u32; 4]; 2]` `` for WGSL uniform buffer alignment.
///
/// WGSL uniform buffers require 16-byte alignment for array elements. Since `u32` is 4 bytes,
/// `` `array<u32, 8>` `` would have 4-byte stride which violates this requirement. By packing into
/// `` `array<vec4<u32>, 2>` ``, each element is 16 bytes and properly aligned.
#[inline]
pub(crate) fn pack_u32_array(values: &[u32; 8]) -> [[u32; 4]; 2] {
    [
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5], values[6], values[7]],
    ]
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CatShaderParams {
    pub(crate) outer_size: u32,
    pub(crate) src_cat_size: u32,
    pub(crate) dst_cat_size: u32,
    pub(crate) cat_offset: u32,
    pub(crate) inner_size: u32,
    pub(crate) total_elements: u32,
}

/// Params for repeat operation (tile tensor along all dimensions)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RepeatParams {
    pub(crate) ndim: u32,
    pub(crate) total_elements: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    /// Source tensor shape (8 values packed as `` `2 vec4<u32>` `` for alignment)
    pub(crate) src_shape: [[u32; 4]; 2],
    /// Output tensor shape (8 values packed as `` `2 vec4<u32>` `` for alignment)
    pub(crate) out_shape: [[u32; 4]; 2],
}

/// Params for pad operation with `F32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PadParamsF32 {
    pub(crate) ndim: u32,
    pub(crate) total_elements: u32,
    pub(crate) fill_value: f32,
    pub(crate) _pad0: u32,
    pub(crate) src_shape: [[u32; 4]; 2],
    pub(crate) out_shape: [[u32; 4]; 2],
    pub(crate) pad_before: [[u32; 4]; 2],
}

/// Params for pad operation with `I32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PadParamsI32 {
    pub(crate) ndim: u32,
    pub(crate) total_elements: u32,
    pub(crate) fill_value: i32,
    pub(crate) _pad0: u32,
    pub(crate) src_shape: [[u32; 4]; 2],
    pub(crate) out_shape: [[u32; 4]; 2],
    pub(crate) pad_before: [[u32; 4]; 2],
}

/// Params for pad operation with `U32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct PadParamsU32 {
    pub(crate) ndim: u32,
    pub(crate) total_elements: u32,
    pub(crate) fill_value: u32,
    pub(crate) _pad0: u32,
    pub(crate) src_shape: [[u32; 4]; 2],
    pub(crate) out_shape: [[u32; 4]; 2],
    pub(crate) pad_before: [[u32; 4]; 2],
}

/// Params for roll operation (circular shift along a dimension)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RollParams {
    pub(crate) outer_size: u32,
    pub(crate) dim_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) shift: u32,
    pub(crate) total_elements: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}
