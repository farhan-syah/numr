//! Params structs for the sort family: sort, top-k, searchsorted, nonzero
//! counting, flat-to-multi index expansion and unique-with-counts.

/// Params for sort operation (sort, argsort)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SortParams {
    pub(crate) outer_size: u32,
    pub(crate) sort_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) descending: u32,
}

/// Params for topk operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct TopkParams {
    pub(crate) outer_size: u32,
    pub(crate) sort_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) k: u32,
    pub(crate) largest: u32,
    pub(crate) sorted: u32,
}

/// Params for searchsorted operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SearchsortedParams {
    pub(crate) seq_len: u32,
    pub(crate) num_values: u32,
    pub(crate) right: u32,
    pub(crate) _pad: u32,
}

/// Params for count operations (nonzero, unique)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CountParams {
    pub(crate) numel: u32,
}

/// Params for flat_to_multi_index operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct FlatToMultiParams {
    pub(crate) nnz: u32,
    pub(crate) ndim: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) shape: [[u32; 4]; 2],
}

/// Params for unique_with_counts operations
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct UniqueCountsParams {
    pub(crate) numel: u32,
    pub(crate) num_unique: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}
