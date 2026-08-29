//! Params structs for the reduction kernels: dimension reduce, full reduce,
//! softmax and the index-returning reductions.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ReduceParams {
    pub(crate) reduce_size: u32,
    pub(crate) outer_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FullReduceParams {
    pub(crate) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SoftmaxParams {
    pub(crate) batch_size: u32,
    pub(crate) dim_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ArgReduceParams {
    pub(crate) reduce_size: u32,
    pub(crate) outer_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) numel_out: u32,
}
