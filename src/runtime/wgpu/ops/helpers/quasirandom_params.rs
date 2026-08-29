//! Params structs for the low-discrepancy sequence kernels: Sobol, Halton and
//! Latin hypercube.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SobolParams {
    pub(crate) n_points: u32,
    pub(crate) dimension: u32,
    pub(crate) skip: u32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct HaltonParams {
    pub(crate) n_points: u32,
    pub(crate) dimension: u32,
    pub(crate) skip: u32,
    pub(crate) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct LatinHypercubeParams {
    pub(crate) n_samples: u32,
    pub(crate) dimension: u32,
    pub(crate) seed: u32,
    pub(crate) _pad: u32,
}
