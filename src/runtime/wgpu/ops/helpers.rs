//! Helper functions and parameter structs for WebGPU operations.
//!
//! This module provides shared utilities for operation implementation including
//! buffer creation, tensor access, and parameter struct definitions.

use wgpu::BufferUsages;

/// Maximum number of dimensions supported by WebGPU shape operation shaders.
/// WGSL doesn't support dynamic arrays in uniform buffers, so we use fixed-size arrays.
pub const MAX_DIMS: usize = 8;

use super::super::WgpuRuntime;
use super::super::client::get_buffer;
use crate::dtype::DType;
use crate::error::{Error, Result};
use crate::runtime::RuntimeClient;
use crate::tensor::Tensor;

/// Create a uniform buffer with the given data.
pub(super) fn create_params_buffer<T: bytemuck::Pod>(
    client: &super::super::WgpuClient,
    data: &T,
) -> wgpu::Buffer {
    let buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: std::mem::size_of::<T>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    client
        .queue
        .write_buffer(&buffer, 0, bytemuck::bytes_of(data));
    buffer
}

/// Get the wgpu buffer from a tensor's storage pointer.
pub(crate) fn get_tensor_buffer(
    tensor: &Tensor<WgpuRuntime>,
) -> Result<std::sync::Arc<wgpu::Buffer>> {
    let ptr = tensor.ptr();
    get_buffer(ptr).ok_or_else(|| Error::Internal("Buffer not found in registry".to_string()))
}

/// Allocate output tensor with given shape and dtype.
pub(crate) fn alloc_output(
    client: &super::super::WgpuClient,
    shape: &[usize],
    dtype: DType,
) -> Result<Tensor<WgpuRuntime>> {
    Tensor::empty(shape, dtype, client.device())
}

// ============================================================================
// Params Structs (must match WGSL shader structs)
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BinaryParams {
    pub(super) numel: u32,
}

/// Parameters for broadcast binary operations.
/// Matches the BroadcastBinaryParams struct in WGSL shaders.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BroadcastBinaryParams {
    pub(super) numel: u32,
    pub(super) ndim: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct UnaryParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScalarParams {
    pub(super) numel: u32,
    pub(super) scalar: f32,
}

/// Parameters for clamp operation.
/// Padding ensures 16-byte alignment for WebGPU uniform buffers.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ClampParams {
    pub(super) numel: u32,
    pub(super) min_val: f32,
    pub(super) max_val: f32,
    /// Padding for 16-byte alignment (WebGPU uniform buffer requirement)
    pub(super) _pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WhereParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct WhereBroadcastParams {
    pub(super) numel: u32,
    pub(super) ndim: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CastParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ReduceParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
    pub(super) numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FullReduceParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SoftmaxParams {
    pub(super) batch_size: u32,
    pub(super) dim_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArgReduceParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
    pub(super) numel_out: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MatmulParams {
    pub(super) m: u32,
    pub(super) k: u32,
    pub(super) n: u32,
    pub(super) batch_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RmsNormParams {
    pub(super) batch_size: u32,
    pub(super) hidden_size: u32,
    pub(super) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LayerNormParams {
    pub(super) batch_size: u32,
    pub(super) hidden_size: u32,
    pub(super) eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GroupNormParams {
    pub(super) batch_size: u32,
    pub(super) channels: u32,
    pub(super) spatial: u32,
    pub(super) num_groups: u32,
    pub(super) channels_per_group: u32,
    pub(super) eps: f32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CatShaderParams {
    pub(super) outer_size: u32,
    pub(super) src_cat_size: u32,
    pub(super) dst_cat_size: u32,
    pub(super) cat_offset: u32,
    pub(super) inner_size: u32,
    pub(super) total_elements: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct IndexSelectParams {
    pub(super) outer_size: u32,
    pub(super) dim_size: u32,
    pub(super) inner_size: u32,
    pub(super) index_len: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GatherParams {
    pub(super) ndim: u32,
    pub(super) dim: u32,
    pub(super) total_elements: u32,
    pub(super) _padding: u32,
    pub(super) input_shape: [u32; 4],
    pub(super) input_strides: [u32; 4],
    pub(super) output_shape: [u32; 4],
    pub(super) output_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ScatterParams {
    pub(super) ndim: u32,
    pub(super) dim: u32,
    pub(super) src_total: u32,
    pub(super) _padding: u32,
    pub(super) output_shape: [u32; 4],
    pub(super) output_strides: [u32; 4],
    pub(super) src_shape: [u32; 4],
    pub(super) src_strides: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CopyParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedFillParams {
    pub(super) numel: u32,
    pub(super) fill_value: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedCountParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MaskedSelectParams {
    pub(super) numel: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ArangeParams {
    pub(super) numel: u32,
    pub(super) start: f32,
    pub(super) step: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LinspaceParams {
    pub(super) steps: u32,
    pub(super) start: f32,
    pub(super) stop: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EyeParams {
    pub(super) n: u32,
    pub(super) m: u32,
    pub(super) numel: u32,
}

// Cumulative operation params

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CumsumParams {
    pub(super) scan_size: u32,
    pub(super) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CumsumStridedParams {
    pub(super) scan_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CumprodParams {
    pub(super) scan_size: u32,
    pub(super) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CumprodStridedParams {
    pub(super) scan_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LogsumexpParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LogsumexpStridedParams {
    pub(super) reduce_size: u32,
    pub(super) outer_size: u32,
    pub(super) inner_size: u32,
}

// Random operation params

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RandParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RandnParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

/// Randint params for signed integer types (`I32`)
/// The `low` field is i32 to properly handle negative bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RandintParamsI32 {
    pub(super) numel: u32,
    pub(super) low: i32, // Signed low bound
    pub(super) range: u32,
    pub(super) seed: u32,
}

/// Randint params for unsigned integer types (`U32`)
/// The `low` field is u32 for unsigned bounds.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RandintParamsU32 {
    pub(super) numel: u32,
    pub(super) low: u32, // Unsigned low bound
    pub(super) range: u32,
    pub(super) seed: u32,
}

// Shape operation params
// Note: Array sizes must match MAX_DIMS (8) for binary layout compatibility with WGSL shaders.
// WGSL doesn't support dynamic arrays in uniform buffers, so we use fixed-size arrays.
// Arrays in WGSL uniform buffers must have 16-byte aligned elements, so we pack 8 u32s
// into 2 vec4<u32> (represented as [[u32; 4]; 2] in Rust).

/// Pack a flat `` `[u32; 8]` `` array into `` `[[u32; 4]; 2]` `` for WGSL uniform buffer alignment.
///
/// WGSL uniform buffers require 16-byte alignment for array elements. Since `u32` is 4 bytes,
/// `` `array<u32, 8>` `` would have 4-byte stride which violates this requirement. By packing into
/// `` `array<vec4<u32>, 2>` ``, each element is 16 bytes and properly aligned.
#[inline]
pub(super) fn pack_u32_array(values: &[u32; 8]) -> [[u32; 4]; 2] {
    [
        [values[0], values[1], values[2], values[3]],
        [values[4], values[5], values[6], values[7]],
    ]
}

/// Params for repeat operation (tile tensor along all dimensions)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RepeatParams {
    pub(super) ndim: u32,
    pub(super) total_elements: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    /// Source tensor shape (8 values packed as `` `2 vec4<u32>` `` for alignment)
    pub(super) src_shape: [[u32; 4]; 2],
    /// Output tensor shape (8 values packed as `` `2 vec4<u32>` `` for alignment)
    pub(super) out_shape: [[u32; 4]; 2],
}

/// Params for pad operation with `F32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PadParamsF32 {
    pub(super) ndim: u32,
    pub(super) total_elements: u32,
    pub(super) fill_value: f32,
    pub(super) _pad0: u32,
    pub(super) src_shape: [[u32; 4]; 2],
    pub(super) out_shape: [[u32; 4]; 2],
    pub(super) pad_before: [[u32; 4]; 2],
}

/// Params for pad operation with `I32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PadParamsI32 {
    pub(super) ndim: u32,
    pub(super) total_elements: u32,
    pub(super) fill_value: i32,
    pub(super) _pad0: u32,
    pub(super) src_shape: [[u32; 4]; 2],
    pub(super) out_shape: [[u32; 4]; 2],
    pub(super) pad_before: [[u32; 4]; 2],
}

/// Params for pad operation with `U32` fill value
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PadParamsU32 {
    pub(super) ndim: u32,
    pub(super) total_elements: u32,
    pub(super) fill_value: u32,
    pub(super) _pad0: u32,
    pub(super) src_shape: [[u32; 4]; 2],
    pub(super) out_shape: [[u32; 4]; 2],
    pub(super) pad_before: [[u32; 4]; 2],
}

/// Params for roll operation (circular shift along a dimension)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct RollParams {
    pub(super) outer_size: u32,
    pub(super) dim_size: u32,
    pub(super) inner_size: u32,
    pub(super) shift: u32,
    pub(super) total_elements: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

/// Params for embedding lookup operation
/// Looks up embeddings from a 2D embedding table `` `[vocab_size, embedding_dim]` ``
/// using indices `` `[num_indices]` ``. Output shape is `` `[num_indices, embedding_dim]` ``.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct EmbeddingLookupParams {
    pub(super) num_indices: u32,
    pub(super) vocab_size: u32,
    pub(super) embedding_dim: u32,
    pub(super) _pad0: u32,
}

/// Params for multinomial sampling operation (with replacement)
/// Samples indices from categorical distributions defined by probability rows.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MultinomialWithReplacementParams {
    pub(super) num_distributions: u32,
    pub(super) num_categories: u32,
    pub(super) num_samples: u32,
    pub(super) seed: u32,
}

/// Params for multinomial sampling operation (without replacement)
/// Samples indices from categorical distributions without replacement.
/// Uses workgroup shared memory for modified probabilities.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct MultinomialWithoutReplacementParams {
    pub(super) num_distributions: u32,
    pub(super) num_categories: u32,
    pub(super) num_samples: u32,
    pub(super) seed: u32,
}

// ============================================================================
// Distribution Sampling Params
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BernoulliParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) p: f32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BetaDistParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) alpha: f32,
    pub(super) beta: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct GammaDistParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) shape: f32,
    pub(super) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ExponentialParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) rate: f32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct PoissonParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) lambda: f32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct BinomialParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) n_trials: u32,
    pub(super) p: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LaplaceParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) loc: f32,
    pub(super) scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ChiSquaredParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) df: f32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct StudentTParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) df: f32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FDistributionParams {
    pub(super) numel: u32,
    pub(super) seed: u32,
    pub(super) df1: f32,
    pub(super) df2: f32,
}

// ============================================================================
// Quasi-Random Sequence Params
// ============================================================================

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SobolParams {
    pub(super) n_points: u32,
    pub(super) dimension: u32,
    pub(super) skip: u32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct HaltonParams {
    pub(super) n_points: u32,
    pub(super) dimension: u32,
    pub(super) skip: u32,
    pub(super) _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct LatinHypercubeParams {
    pub(super) n_samples: u32,
    pub(super) dimension: u32,
    pub(super) seed: u32,
    pub(super) _pad: u32,
}

// ============================================================================
// Sort Operation Params
// ============================================================================

/// Params for sort operation (sort, argsort)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SortParams {
    pub(super) outer_size: u32,
    pub(super) sort_size: u32,
    pub(super) inner_size: u32,
    pub(super) descending: u32,
}

/// Params for topk operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct TopkParams {
    pub(super) outer_size: u32,
    pub(super) sort_size: u32,
    pub(super) inner_size: u32,
    pub(super) k: u32,
    pub(super) largest: u32,
    pub(super) sorted: u32,
}

/// Params for searchsorted operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct SearchsortedParams {
    pub(super) seq_len: u32,
    pub(super) num_values: u32,
    pub(super) right: u32,
    pub(super) _pad: u32,
}

/// Params for count operations (nonzero, unique)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct CountParams {
    pub(super) numel: u32,
}

/// Params for flat_to_multi_index operation
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct FlatToMultiParams {
    pub(super) nnz: u32,
    pub(super) ndim: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) shape: [[u32; 4]; 2],
}

/// Params for index bounds validation kernel
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ValidateIndicesParams {
    pub(super) index_len: u32,
    pub(super) dim_size: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
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

/// Params for unique_with_counts operations
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct UniqueCountsParams {
    pub(super) numel: u32,
    pub(super) num_unique: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

// ============================================================================
// Broadcast Helpers
// ============================================================================

/// Compute broadcast strides for an input tensor relative to an output shape.
///
/// For each dimension in the output shape:
/// - If the input dimension matches, use the original stride
/// - If the input dimension is 1 (broadcast), use stride 0
/// - If the input doesn't have this dimension (prepended), use stride 0
pub fn compute_broadcast_strides(input_shape: &[usize], output_shape: &[usize]) -> Vec<u32> {
    let mut strides = vec![0u32; output_shape.len()];
    let input_ndim = input_shape.len();
    let output_ndim = output_shape.len();

    // Compute input strides (row-major)
    let mut input_strides = vec![1usize; input_ndim];
    for i in (0..input_ndim.saturating_sub(1)).rev() {
        input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
    }

    // Map input dimensions to output dimensions (right-aligned)
    let offset = output_ndim - input_ndim;
    for i in 0..output_ndim {
        if i < offset {
            // Dimension doesn't exist in input, broadcast with stride 0
            strides[i] = 0;
        } else {
            let input_idx = i - offset;
            if input_shape[input_idx] == 1 {
                // Broadcasting dimension, stride 0
                strides[i] = 0;
            } else {
                // Normal dimension, use input stride
                strides[i] = input_strides[input_idx] as u32;
            }
        }
    }

    strides
}

/// Create a storage buffer with the given data.
pub(super) fn create_storage_buffer<T: bytemuck::Pod>(
    client: &super::super::WgpuClient,
    data: &[T],
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    client
        .wgpu_device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("storage_buffer"),
            contents: bytemuck::cast_slice(data),
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        })
}

/// Generate a random seed for WebGPU RNG operations.
/// Combines system time with an atomic counter to ensure uniqueness across calls.
pub(super) fn generate_wgpu_seed() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static SEED_COUNTER: AtomicU32 = AtomicU32::new(0);

    let counter = SEED_COUNTER.fetch_add(1, Ordering::Relaxed);
    let time_seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u32)
        .unwrap_or(12345u32);
    time_seed.wrapping_add(counter)
}

/// Read a single u32 value from a GPU buffer (synchronous)
pub(crate) fn read_u32_from_buffer(
    client: &super::super::WgpuClient,
    buffer: &wgpu::Buffer,
) -> Result<u32> {
    let staging_buffer = client.wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("staging_read"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = client
        .wgpu_device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read_u32"),
        });
    encoder.copy_buffer_to_buffer(buffer, 0, &staging_buffer, 0, 4);
    client.queue.submit(std::iter::once(encoder.finish()));

    // Block until GPU work is done
    let (tx, rx) = std::sync::mpsc::channel();
    staging_buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    let _ = client.wgpu_device.poll(wgpu::PollType::Wait {
        submission_index: None,
        timeout: Some(std::time::Duration::from_secs(60)),
    });
    rx.recv()
        .map_err(|_| Error::Internal("Failed to read from GPU buffer".to_string()))?
        .map_err(|e| Error::Internal(format!("Buffer map failed: {:?}", e)))?;

    let data = staging_buffer.slice(..).get_mapped_range();
    let value = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    staging_buffer.unmap();

    Ok(value)
}

/// Cast indices tensor to I32 for WebGPU shaders.
/// WebGPU natively supports I32 indices; I64 indices are cast on GPU.
/// Returns an error for unsupported index dtypes.
pub(crate) fn ensure_i32_indices(
    client: &super::super::WgpuClient,
    indices: &Tensor<WgpuRuntime>,
) -> Result<Tensor<WgpuRuntime>> {
    use crate::ops::TypeConversionOps;
    match indices.dtype() {
        DType::I32 => Ok(indices.clone()),
        DType::I64 => client.cast(indices, DType::I32),
        other => Err(Error::DTypeMismatch {
            lhs: DType::I32,
            rhs: other,
        }),
    }
}
