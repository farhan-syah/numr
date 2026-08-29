//! Params structs for the normalization kernels: RMS norm, layer norm and
//! group norm.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct RmsNormParams {
    pub(crate) batch_size: u32,
    pub(crate) hidden_size: u32,
    pub(crate) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LayerNormParams {
    pub(crate) batch_size: u32,
    pub(crate) hidden_size: u32,
    pub(crate) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GroupNormParams {
    pub(crate) batch_size: u32,
    pub(crate) channels: u32,
    pub(crate) spatial: u32,
    pub(crate) num_groups: u32,
    pub(crate) channels_per_group: u32,
    pub(crate) eps: f32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}
