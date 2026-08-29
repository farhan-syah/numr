//! Params struct for the matmul kernels.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MatmulParams {
    pub(crate) m: u32,
    pub(crate) k: u32,
    pub(crate) n: u32,
    pub(crate) batch_size: u32,
}
