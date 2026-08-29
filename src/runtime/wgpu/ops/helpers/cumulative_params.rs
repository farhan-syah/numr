//! Params structs for the running-accumulation kernels: cumsum, cumprod and
//! logsumexp, each in a contiguous and a strided form.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CumsumParams {
    pub(crate) scan_size: u32,
    pub(crate) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CumsumStridedParams {
    pub(crate) scan_size: u32,
    pub(crate) outer_size: u32,
    pub(crate) inner_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CumprodParams {
    pub(crate) scan_size: u32,
    pub(crate) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CumprodStridedParams {
    pub(crate) scan_size: u32,
    pub(crate) outer_size: u32,
    pub(crate) inner_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LogsumexpParams {
    pub(crate) reduce_size: u32,
    pub(crate) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LogsumexpStridedParams {
    pub(crate) reduce_size: u32,
    pub(crate) outer_size: u32,
    pub(crate) inner_size: u32,
}
