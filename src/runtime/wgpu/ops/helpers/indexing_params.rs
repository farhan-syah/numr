//! Params structs for the index, gather, scatter and mask kernels.

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct IndexSelectParams {
    pub(crate) outer_size: u32,
    pub(crate) dim_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) index_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GatherParams {
    pub(crate) ndim: u32,
    pub(crate) dim: u32,
    pub(crate) total_elements: u32,
    pub(crate) _padding: u32,
    pub(crate) input_shape: [u32; 4],
    pub(crate) input_strides: [u32; 4],
    pub(crate) output_shape: [u32; 4],
    pub(crate) output_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScatterParams {
    pub(crate) ndim: u32,
    pub(crate) dim: u32,
    pub(crate) src_total: u32,
    pub(crate) _padding: u32,
    pub(crate) output_shape: [u32; 4],
    pub(crate) output_strides: [u32; 4],
    pub(crate) src_shape: [u32; 4],
    pub(crate) src_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CopyParams {
    pub(crate) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaskedFillParams {
    pub(crate) numel: u32,
    pub(crate) fill_value: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaskedCountParams {
    pub(crate) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaskedSelectParams {
    pub(crate) numel: u32,
}

/// Params for embedding lookup operation
/// Looks up embeddings from a 2D embedding table `` `[vocab_size, embedding_dim]` ``
/// using indices `` `[num_indices]` ``. Output shape is `` `[num_indices, embedding_dim]` ``.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct EmbeddingLookupParams {
    pub(crate) num_indices: u32,
    pub(crate) vocab_size: u32,
    pub(crate) embedding_dim: u32,
    pub(crate) _pad0: u32,
}

/// Params for index bounds validation kernel
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ValidateIndicesParams {
    pub(crate) index_len: u32,
    pub(crate) dim_size: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

/// Params for gather_nd operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GatherNdParams {
    pub(crate) num_slices: u32,
    pub(crate) slice_size: u32,
    pub(crate) index_depth: u32,
    pub(crate) ndim: u32,
    pub(crate) input_shape: [u32; 8],
    pub(crate) input_strides: [u32; 8],
}

/// Params for bincount operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct BincountParams {
    pub(crate) n: u32,
    pub(crate) minlength: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
}

/// Params for scatter_reduce operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct ScatterReduceParams {
    pub(crate) dim: u32,
    pub(crate) outer_size: u32,
    pub(crate) dim_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) src_dim_size: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

/// Params for scatter_reduce mean division
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MeanDivParams {
    pub(crate) n: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

/// Params for gather_2d operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Gather2dParams {
    pub(crate) nrows: u32,
    pub(crate) ncols: u32,
    pub(crate) num_indices: u32,
    pub(crate) _pad: u32,
}

/// Params for slice_assign operations
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SliceAssignParams {
    pub(crate) outer_size: u32,
    pub(crate) dst_dim_size: u32,
    pub(crate) src_dim_size: u32,
    pub(crate) inner_size: u32,
    pub(crate) start: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}
