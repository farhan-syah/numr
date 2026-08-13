// Indexing CUDA kernels - gather, scatter, index_select, masked_select, masked_fill
// Supports: f32, f64, f16, bf16, i32, i64
//
// Each operation is defined per-dtype (no templates for extern "C" compatibility)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Helper Macros for Multi-Dtype Kernel Generation
// ============================================================================

// Maximum number of dimensions supported (must match MAX_DIMS in index/gather.rs)
#ifndef INDEX_MAX_DIMS
#define INDEX_MAX_DIMS 8
#endif

// Macro for gather kernel.
// Shape and stride arrays are passed as 4*MAX_DIMS=32 individual scalar u32 args
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused slots are zero-padded by the Rust launcher.
#define DEFINE_GATHER_KERNEL(suffix, dtype) \
__global__ void gather_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int ndim, \
    unsigned int dim, \
    unsigned int input_shape0, unsigned int input_shape1, unsigned int input_shape2, unsigned int input_shape3, \
    unsigned int input_shape4, unsigned int input_shape5, unsigned int input_shape6, unsigned int input_shape7, \
    unsigned int input_strides0, unsigned int input_strides1, unsigned int input_strides2, unsigned int input_strides3, \
    unsigned int input_strides4, unsigned int input_strides5, unsigned int input_strides6, unsigned int input_strides7, \
    unsigned int output_shape0, unsigned int output_shape1, unsigned int output_shape2, unsigned int output_shape3, \
    unsigned int output_shape4, unsigned int output_shape5, unsigned int output_shape6, unsigned int output_shape7, \
    unsigned int output_strides0, unsigned int output_strides1, unsigned int output_strides2, unsigned int output_strides3, \
    unsigned int output_strides4, unsigned int output_strides5, unsigned int output_strides6, unsigned int output_strides7, \
    unsigned int total_elements \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= total_elements) return; \
    \
    /* Load scalar args into shared arrays for indexed access */ \
    __shared__ unsigned int s_input_shape[INDEX_MAX_DIMS]; \
    __shared__ unsigned int s_input_strides[INDEX_MAX_DIMS]; \
    __shared__ unsigned int s_output_strides[INDEX_MAX_DIMS]; \
    if (threadIdx.x == 0) { \
        s_input_shape[0]   = input_shape0;   s_input_shape[1]   = input_shape1; \
        s_input_shape[2]   = input_shape2;   s_input_shape[3]   = input_shape3; \
        s_input_shape[4]   = input_shape4;   s_input_shape[5]   = input_shape5; \
        s_input_shape[6]   = input_shape6;   s_input_shape[7]   = input_shape7; \
        s_input_strides[0] = input_strides0; s_input_strides[1] = input_strides1; \
        s_input_strides[2] = input_strides2; s_input_strides[3] = input_strides3; \
        s_input_strides[4] = input_strides4; s_input_strides[5] = input_strides5; \
        s_input_strides[6] = input_strides6; s_input_strides[7] = input_strides7; \
        s_output_strides[0] = output_strides0; s_output_strides[1] = output_strides1; \
        s_output_strides[2] = output_strides2; s_output_strides[3] = output_strides3; \
        s_output_strides[4] = output_strides4; s_output_strides[5] = output_strides5; \
        s_output_strides[6] = output_strides6; s_output_strides[7] = output_strides7; \
    } \
    __syncthreads(); \
    \
    unsigned int remaining = idx; \
    unsigned int src_offset = 0; \
    \
    for (unsigned int d = 0; d < ndim; d++) { \
        unsigned int coord = remaining / s_output_strides[d]; \
        remaining %= s_output_strides[d]; \
        \
        if (d == dim) { \
            long long index_val = indices[idx]; \
            if (index_val < 0 || (unsigned int)index_val >= s_input_shape[d]) { \
                output[idx] = (dtype)0; \
                return; \
            } \
            src_offset += (unsigned int)index_val * s_input_strides[d]; \
        } else { \
            src_offset += coord * s_input_strides[d]; \
        } \
    } \
    \
    output[idx] = input[src_offset]; \
}

// Macro for scatter kernel.
// Shape and stride arrays are passed as 4*INDEX_MAX_DIMS=32 individual scalar u32 args
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused slots are zero-padded by the Rust launcher.
#define DEFINE_SCATTER_KERNEL(suffix, dtype) \
__global__ void scatter_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    const dtype* __restrict__ src, \
    dtype* __restrict__ output, \
    unsigned int ndim, \
    unsigned int dim, \
    unsigned int output_shape0, unsigned int output_shape1, unsigned int output_shape2, unsigned int output_shape3, \
    unsigned int output_shape4, unsigned int output_shape5, unsigned int output_shape6, unsigned int output_shape7, \
    unsigned int output_strides0, unsigned int output_strides1, unsigned int output_strides2, unsigned int output_strides3, \
    unsigned int output_strides4, unsigned int output_strides5, unsigned int output_strides6, unsigned int output_strides7, \
    unsigned int src_shape0, unsigned int src_shape1, unsigned int src_shape2, unsigned int src_shape3, \
    unsigned int src_shape4, unsigned int src_shape5, unsigned int src_shape6, unsigned int src_shape7, \
    unsigned int src_strides0, unsigned int src_strides1, unsigned int src_strides2, unsigned int src_strides3, \
    unsigned int src_strides4, unsigned int src_strides5, unsigned int src_strides6, unsigned int src_strides7, \
    unsigned int src_total \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= src_total) return; \
    \
    /* Load scalar args into shared arrays for indexed access */ \
    __shared__ unsigned int s_output_shape[INDEX_MAX_DIMS]; \
    __shared__ unsigned int s_output_strides[INDEX_MAX_DIMS]; \
    __shared__ unsigned int s_src_strides[INDEX_MAX_DIMS]; \
    if (threadIdx.x == 0) { \
        s_output_shape[0]   = output_shape0;   s_output_shape[1]   = output_shape1; \
        s_output_shape[2]   = output_shape2;   s_output_shape[3]   = output_shape3; \
        s_output_shape[4]   = output_shape4;   s_output_shape[5]   = output_shape5; \
        s_output_shape[6]   = output_shape6;   s_output_shape[7]   = output_shape7; \
        s_output_strides[0] = output_strides0; s_output_strides[1] = output_strides1; \
        s_output_strides[2] = output_strides2; s_output_strides[3] = output_strides3; \
        s_output_strides[4] = output_strides4; s_output_strides[5] = output_strides5; \
        s_output_strides[6] = output_strides6; s_output_strides[7] = output_strides7; \
        s_src_strides[0]    = src_strides0;    s_src_strides[1]    = src_strides1; \
        s_src_strides[2]    = src_strides2;    s_src_strides[3]    = src_strides3; \
        s_src_strides[4]    = src_strides4;    s_src_strides[5]    = src_strides5; \
        s_src_strides[6]    = src_strides6;    s_src_strides[7]    = src_strides7; \
    } \
    __syncthreads(); \
    \
    unsigned int remaining = idx; \
    unsigned int dst_offset = 0; \
    \
    for (unsigned int d = 0; d < ndim; d++) { \
        unsigned int coord = remaining / s_src_strides[d]; \
        remaining %= s_src_strides[d]; \
        \
        if (d == dim) { \
            long long index_val = indices[idx]; \
            if (index_val < 0 || (unsigned int)index_val >= s_output_shape[d]) { \
                return; \
            } \
            dst_offset += (unsigned int)index_val * s_output_strides[d]; \
        } else { \
            dst_offset += coord * s_output_strides[d]; \
        } \
    } \
    \
    output[dst_offset] = src[idx]; \
}

// Macro for copy kernel (for scatter initialization)
#define DEFINE_COPY_KERNEL(suffix, dtype) \
__global__ void copy_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ dst, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx < n) { \
        dst[idx] = src[idx]; \
    } \
}

// Macro for index_select kernel
#define DEFINE_INDEX_SELECT_KERNEL(suffix, dtype) \
__global__ void index_select_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int index_len \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * index_len * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int sel_idx = (idx / inner_size) % index_len; \
    unsigned int outer = idx / (index_len * inner_size); \
    \
    long long index_val = indices[sel_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    unsigned int src_offset = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    output[idx] = input[src_offset]; \
}

// Macro for masked_select kernel
#define DEFINE_MASKED_SELECT_KERNEL(suffix, dtype) \
__global__ void masked_select_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    const unsigned int* __restrict__ prefix_sum, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    if (mask[idx] != 0) { \
        unsigned int out_idx = prefix_sum[idx]; \
        output[out_idx] = input[idx]; \
    } \
}

// Macro for masked_fill kernel (float types)
#define DEFINE_MASKED_FILL_KERNEL(suffix, dtype) \
__global__ void masked_fill_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    dtype fill_value, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    output[idx] = (mask[idx] != 0) ? fill_value : input[idx]; \
}

// Macro for index_put kernel
// Inverse of index_select: puts values from src at positions specified by indices.
// Output is pre-initialized with a copy of input tensor.
// For each position (outer, sel_idx, inner):
//   out[outer * dim_size * inner_size + indices[sel_idx] * inner_size + inner] = src[outer * index_len * inner_size + sel_idx * inner_size + inner]
#define DEFINE_INDEX_PUT_KERNEL(suffix, dtype) \
__global__ void index_put_##suffix( \
    const long long* __restrict__ indices, \
    const dtype* __restrict__ src, \
    dtype* __restrict__ output, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int index_len \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * index_len * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int sel_idx = (idx / inner_size) % index_len; \
    unsigned int outer = idx / (index_len * inner_size); \
    \
    long long index_val = indices[sel_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        return; /* Out of bounds - skip */ \
    } \
    \
    unsigned int dst_offset = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    output[dst_offset] = src[idx]; \
}

// Macro for embedding_lookup kernel
// Industry-standard embedding table lookup operation for neural networks.
// Input: embeddings [vocab_size, embedding_dim], indices [num_indices]
// Output: output [num_indices, embedding_dim]
// Each thread handles one index, copying the entire embedding row to output.
#define DEFINE_EMBEDDING_LOOKUP_KERNEL(suffix, dtype) \
__global__ void embedding_lookup_##suffix( \
    const dtype* __restrict__ embeddings, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int num_indices, \
    unsigned int vocab_size, \
    unsigned int embedding_dim \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= num_indices) return; \
    \
    long long index_val = indices[idx]; \
    dtype* out_row = output + (idx * embedding_dim); \
    \
    /* Check bounds */ \
    if (index_val < 0 || (unsigned int)index_val >= vocab_size) { \
        /* Out of bounds - fill with zeros */ \
        for (unsigned int i = 0; i < embedding_dim; i++) { \
            out_row[i] = (dtype)0; \
        } \
        return; \
    } \
    \
    const dtype* emb_row = embeddings + ((unsigned int)index_val * embedding_dim); \
    \
    /* Copy the entire embedding row to output (coalesced writes) */ \
    for (unsigned int i = 0; i < embedding_dim; i++) { \
        out_row[i] = emb_row[i]; \
    } \
}

extern "C" {

// ============================================================================
// Masked Count Kernel (dtype-independent)
// ============================================================================

__global__ void masked_count_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ count,
    unsigned int n
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        if (mask[i] != 0) {
            local_count++;
        }
    }

    atomicAdd(&shared_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, shared_count);
    }
}

// ============================================================================
// Masked Prefix Sum Kernel (dtype-independent)
// ============================================================================

__global__ void masked_prefix_sum_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    unsigned int n
) {
    // Simple sequential prefix sum - for small tensors or as fallback
    // For large tensors, a parallel scan algorithm would be used
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < n; i++) {
            prefix_sum[i] = sum;
            if (mask[i] != 0) {
                sum++;
            }
        }
    }
}

// ============================================================================
// Broadcast Masked Operations
// These kernels support broadcasting the mask to the input/output shape.
// Uses stride-based indexing where stride=0 means broadcast dimension.
// ============================================================================

// Device function to compute mask index with broadcasting
__device__ __forceinline__ unsigned int compute_broadcast_index(
    unsigned int linear_idx,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim
) {
    unsigned int mask_offset = 0;
    unsigned int remaining = linear_idx;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        mask_offset += coord * mask_strides[d];
    }
    return mask_offset;
}

// Broadcast masked count - counts true elements in mask broadcast to output shape
__global__ void masked_count_broadcast_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ count,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim,
    unsigned int n
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        unsigned int mask_idx = compute_broadcast_index(i, mask_strides, out_shape, ndim);
        if (mask[mask_idx] != 0) {
            local_count++;
        }
    }

    atomicAdd(&shared_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, shared_count);
    }
}

// Broadcast masked prefix sum - computes prefix sum with broadcast mask
__global__ void masked_prefix_sum_broadcast_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim,
    unsigned int n
) {
    // Sequential prefix sum (for small tensors or as fallback)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int sum = 0;
        for (unsigned int i = 0; i < n; i++) {
            prefix_sum[i] = sum;
            unsigned int mask_idx = compute_broadcast_index(i, mask_strides, out_shape, ndim);
            if (mask[mask_idx] != 0) {
                sum++;
            }
        }
    }
}

// Macro for broadcast masked_select kernel
#define DEFINE_MASKED_SELECT_BROADCAST_KERNEL(suffix, dtype) \
__global__ void masked_select_broadcast_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    const unsigned int* __restrict__ prefix_sum, \
    const unsigned int* __restrict__ mask_strides, \
    const unsigned int* __restrict__ out_shape, \
    unsigned int ndim, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim); \
    if (mask[mask_idx] != 0) { \
        unsigned int out_idx = prefix_sum[idx]; \
        output[out_idx] = input[idx]; \
    } \
}

// Macro for broadcast masked_fill kernel
#define DEFINE_MASKED_FILL_BROADCAST_KERNEL(suffix, dtype) \
__global__ void masked_fill_broadcast_##suffix( \
    const dtype* __restrict__ input, \
    const unsigned char* __restrict__ mask, \
    dtype* __restrict__ output, \
    dtype fill_value, \
    const unsigned int* __restrict__ mask_strides, \
    const unsigned int* __restrict__ out_shape, \
    unsigned int ndim, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim); \
    output[idx] = (mask[mask_idx] != 0) ? fill_value : input[idx]; \
}

// Instantiate broadcast masked operations for all types
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f32, float)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f64, double)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(f16, __half)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(i32, int)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(i64, long long)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_MASKED_SELECT_BROADCAST_KERNEL(fp8_e5m2, numr_fp8_e5m2)

DEFINE_MASKED_FILL_BROADCAST_KERNEL(f32, float)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(f64, double)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(f16, __half)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(i32, int)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(i64, long long)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_MASKED_FILL_BROADCAST_KERNEL(fp8_e5m2, numr_fp8_e5m2)

// ============================================================================
// Index Bounds Validation Kernel (dtype-independent)
// ============================================================================

// Validates that all indices are within bounds [0, dim_size).
// Atomically counts the number of out-of-bounds indices.
// Returns count in error_count[0]. If count > 0, some indices are invalid.
__global__ void validate_indices_kernel(
    const long long* __restrict__ indices,
    unsigned int* __restrict__ error_count,
    unsigned int index_len,
    unsigned int dim_size
) {
    __shared__ unsigned int shared_count;
    if (threadIdx.x == 0) {
        shared_count = 0;
    }
    __syncthreads();

    unsigned int local_count = 0;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < index_len; i += blockDim.x * gridDim.x) {
        long long idx = indices[i];
        if (idx < 0 || idx >= (long long)dim_size) {
            local_count++;
        }
    }

    if (local_count > 0) {
        atomicAdd(&shared_count, local_count);
    }
    __syncthreads();

    if (threadIdx.x == 0 && shared_count > 0) {
        atomicAdd(error_count, shared_count);
    }
}

// ============================================================================
// F32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f32, float)
DEFINE_SCATTER_KERNEL(f32, float)
DEFINE_COPY_KERNEL(f32, float)
DEFINE_INDEX_SELECT_KERNEL(f32, float)
DEFINE_INDEX_PUT_KERNEL(f32, float)
DEFINE_MASKED_SELECT_KERNEL(f32, float)
DEFINE_MASKED_FILL_KERNEL(f32, float)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f32, float)

// ============================================================================
// F64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f64, double)
DEFINE_SCATTER_KERNEL(f64, double)
DEFINE_COPY_KERNEL(f64, double)
DEFINE_INDEX_SELECT_KERNEL(f64, double)
DEFINE_INDEX_PUT_KERNEL(f64, double)
DEFINE_MASKED_SELECT_KERNEL(f64, double)
DEFINE_MASKED_FILL_KERNEL(f64, double)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f64, double)

// ============================================================================
// F16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(f16, __half)
DEFINE_SCATTER_KERNEL(f16, __half)
DEFINE_COPY_KERNEL(f16, __half)
DEFINE_INDEX_SELECT_KERNEL(f16, __half)
DEFINE_INDEX_PUT_KERNEL(f16, __half)
DEFINE_MASKED_SELECT_KERNEL(f16, __half)
DEFINE_MASKED_FILL_KERNEL(f16, __half)
DEFINE_EMBEDDING_LOOKUP_KERNEL(f16, __half)

// ============================================================================
// BF16 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(bf16, __nv_bfloat16)
DEFINE_SCATTER_KERNEL(bf16, __nv_bfloat16)
DEFINE_COPY_KERNEL(bf16, __nv_bfloat16)
DEFINE_INDEX_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_INDEX_PUT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_SELECT_KERNEL(bf16, __nv_bfloat16)
DEFINE_MASKED_FILL_KERNEL(bf16, __nv_bfloat16)
DEFINE_EMBEDDING_LOOKUP_KERNEL(bf16, __nv_bfloat16)

// ============================================================================
// I32 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i32, int)
DEFINE_SCATTER_KERNEL(i32, int)
DEFINE_COPY_KERNEL(i32, int)
DEFINE_INDEX_SELECT_KERNEL(i32, int)
DEFINE_INDEX_PUT_KERNEL(i32, int)
DEFINE_MASKED_SELECT_KERNEL(i32, int)
DEFINE_MASKED_FILL_KERNEL(i32, int)
DEFINE_EMBEDDING_LOOKUP_KERNEL(i32, int)

// ============================================================================
// I64 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(i64, long long)
DEFINE_SCATTER_KERNEL(i64, long long)
DEFINE_COPY_KERNEL(i64, long long)
DEFINE_INDEX_SELECT_KERNEL(i64, long long)
DEFINE_INDEX_PUT_KERNEL(i64, long long)
DEFINE_MASKED_SELECT_KERNEL(i64, long long)
DEFINE_MASKED_FILL_KERNEL(i64, long long)
DEFINE_EMBEDDING_LOOKUP_KERNEL(i64, long long)

// ============================================================================
// FP8 E4M3 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_SCATTER_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_COPY_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_INDEX_SELECT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_INDEX_PUT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_MASKED_SELECT_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_MASKED_FILL_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_EMBEDDING_LOOKUP_KERNEL(fp8_e4m3, numr_fp8_e4m3)

// ============================================================================
// FP8 E5M2 Kernels
// ============================================================================

DEFINE_GATHER_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_SCATTER_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_COPY_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_INDEX_SELECT_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_INDEX_PUT_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_MASKED_SELECT_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_MASKED_FILL_KERNEL(fp8_e5m2, numr_fp8_e5m2)
DEFINE_EMBEDDING_LOOKUP_KERNEL(fp8_e5m2, numr_fp8_e5m2)

// ============================================================================
// Gather ND - N-dimensional gather operation
// Gathers slices from input at positions specified by indices tensor.
// indices: (num_slices, index_depth) where index_depth <= ndim
// output: (num_slices, remaining_dims...)
//
// Shape and stride arrays are passed as 2*MAX_DIMS=16 individual scalar u32 args
// rather than device pointers, making this safe for CUDA graph capture/replay.
// Unused slots are zero-padded by the Rust launcher.
// ============================================================================

#define DEFINE_GATHER_ND_KERNEL(suffix, dtype) \
__global__ void gather_nd_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ output, \
    unsigned int input_shape0, unsigned int input_shape1, unsigned int input_shape2, unsigned int input_shape3, \
    unsigned int input_shape4, unsigned int input_shape5, unsigned int input_shape6, unsigned int input_shape7, \
    unsigned int input_strides0, unsigned int input_strides1, unsigned int input_strides2, unsigned int input_strides3, \
    unsigned int input_strides4, unsigned int input_strides5, unsigned int input_strides6, unsigned int input_strides7, \
    unsigned int num_slices, \
    unsigned int slice_size, \
    unsigned int index_depth, \
    unsigned int ndim \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = num_slices * slice_size; \
    if (idx >= total) return; \
    \
    /* Load scalar args into shared arrays for indexed access */ \
    __shared__ unsigned int s_input_shape[INDEX_MAX_DIMS]; \
    __shared__ unsigned int s_input_strides[INDEX_MAX_DIMS]; \
    if (threadIdx.x == 0) { \
        s_input_shape[0]   = input_shape0;   s_input_shape[1]   = input_shape1; \
        s_input_shape[2]   = input_shape2;   s_input_shape[3]   = input_shape3; \
        s_input_shape[4]   = input_shape4;   s_input_shape[5]   = input_shape5; \
        s_input_shape[6]   = input_shape6;   s_input_shape[7]   = input_shape7; \
        s_input_strides[0] = input_strides0; s_input_strides[1] = input_strides1; \
        s_input_strides[2] = input_strides2; s_input_strides[3] = input_strides3; \
        s_input_strides[4] = input_strides4; s_input_strides[5] = input_strides5; \
        s_input_strides[6] = input_strides6; s_input_strides[7] = input_strides7; \
    } \
    __syncthreads(); \
    \
    unsigned int slice_idx = idx / slice_size; \
    unsigned int within_slice = idx % slice_size; \
    \
    /* Compute offset into input from indices */ \
    unsigned int src_offset = 0; \
    bool out_of_bounds = false; \
    for (unsigned int d = 0; d < index_depth; d++) { \
        long long index_val = indices[slice_idx * index_depth + d]; \
        if (index_val < 0 || (unsigned int)index_val >= s_input_shape[d]) { \
            out_of_bounds = true; \
            break; \
        } \
        src_offset += (unsigned int)index_val * s_input_strides[d]; \
    } \
    \
    if (out_of_bounds) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    /* Add offset for remaining dimensions */ \
    src_offset += within_slice; \
    output[idx] = input[src_offset]; \
}

// Instantiate gather_nd for all dtypes
DEFINE_GATHER_ND_KERNEL(f32, float)
DEFINE_GATHER_ND_KERNEL(f64, double)
DEFINE_GATHER_ND_KERNEL(f16, __half)
DEFINE_GATHER_ND_KERNEL(bf16, __nv_bfloat16)
DEFINE_GATHER_ND_KERNEL(i32, int)
DEFINE_GATHER_ND_KERNEL(i64, long long)
DEFINE_GATHER_ND_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_GATHER_ND_KERNEL(fp8_e5m2, numr_fp8_e5m2)

// ============================================================================
// Bincount - Count occurrences of each value in an integer tensor
// input: 1D tensor of non-negative integers
// weights: Optional 1D tensor of weights (same length as input)
// output: 1D tensor of counts/sums with length = minlength
// Uses atomicAdd for thread-safe accumulation
// ============================================================================

// Bincount without weights (counting) - output is I64
__global__ void bincount_i32(
    const int* __restrict__ input,
    long long* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd((unsigned long long*)&output[val], 1ULL);
    }
}

__global__ void bincount_i64(
    const long long* __restrict__ input,
    long long* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long val = input[idx];
    if (val >= 0 && (unsigned long long)val < minlength) {
        atomicAdd((unsigned long long*)&output[val], 1ULL);
    }
}

// Bincount with f32 weights
__global__ void bincount_weighted_f32(
    const int* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// Bincount with f64 weights
__global__ void bincount_weighted_f64(
    const int* __restrict__ input,
    const double* __restrict__ weights,
    double* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    int val = input[idx];
    if (val >= 0 && (unsigned int)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// Bincount with i64 input and f32 weights
__global__ void bincount_i64_weighted_f32(
    const long long* __restrict__ input,
    const float* __restrict__ weights,
    float* __restrict__ output,
    unsigned int n,
    unsigned int minlength
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    long long val = input[idx];
    if (val >= 0 && (unsigned long long)val < minlength) {
        atomicAdd(&output[val], weights[idx]);
    }
}

// ============================================================================
// Scatter Reduce - Scatter with reduction operations
// Scatters values from src to dst at positions specified by index with a
// reduction operation (sum, prod, max, min, mean).
// Uses atomic operations for thread-safe reduction.
// Reduce ops: 0=sum, 1=prod, 2=max, 3=min, 4=mean
// ============================================================================

// Atomic max for float using CAS
__device__ __forceinline__ float atomicMaxFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fmaxf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic min for float using CAS
__device__ __forceinline__ float atomicMinFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = fminf(old_val, val);
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

// Atomic max for double using CAS
__device__ __forceinline__ double atomicMaxDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmax(old_val, val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Atomic min for double using CAS
__device__ __forceinline__ double atomicMinDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = fmin(old_val, val);
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

// Scatter reduce for F32 - sum operation
__global__ void scatter_reduce_sum_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F32 - max operation
__global__ void scatter_reduce_max_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMaxFloat(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F32 - min operation
__global__ void scatter_reduce_min_f32(
    const float* __restrict__ src,
    const long long* __restrict__ indices,
    float* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMinFloat(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - sum operation
__global__ void scatter_reduce_sum_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - max operation
__global__ void scatter_reduce_max_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMaxDouble(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for F64 - min operation
__global__ void scatter_reduce_min_f64(
    const double* __restrict__ src,
    const long long* __restrict__ indices,
    double* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMinDouble(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - sum operation
__global__ void scatter_reduce_sum_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicAdd(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - max operation
__global__ void scatter_reduce_max_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMax(&dst[dst_idx], src[src_idx]);
}

// Scatter reduce for I32 - min operation
__global__ void scatter_reduce_min_i32(
    const int* __restrict__ src,
    const long long* __restrict__ indices,
    int* __restrict__ dst,
    unsigned int dim,
    unsigned int outer_size,
    unsigned int dim_size,
    unsigned int inner_size,
    unsigned int src_dim_size
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int src_d = (idx / inner_size) % src_dim_size;
    unsigned int outer = idx / (src_dim_size * inner_size);

    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner;

    // The index tensor is element-wise (same shape as src), so the index is read
    // at the source element's own position, not at its coordinate along dim.
    long long index_val = indices[src_idx];
    if (index_val < 0 || (unsigned int)index_val >= dim_size) {
        return;
    }

    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;

    atomicMin(&dst[dst_idx], src[src_idx]);
}

// ============================================================================
// Gather 2D - Gathers from 2D matrix at (row, col) positions
// input: 2D tensor [nrows, ncols]
// rows: 1D tensor of row indices [num_indices]
// cols: 1D tensor of column indices [num_indices]
// output: 1D tensor [num_indices]
// ============================================================================

#define DEFINE_GATHER_2D_KERNEL(suffix, dtype) \
__global__ void gather_2d_##suffix( \
    const dtype* __restrict__ input, \
    const long long* __restrict__ rows, \
    const long long* __restrict__ cols, \
    dtype* __restrict__ output, \
    unsigned int nrows, \
    unsigned int ncols, \
    unsigned int num_indices \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= num_indices) return; \
    \
    long long r = rows[idx]; \
    long long c = cols[idx]; \
    \
    /* Bounds checking */ \
    if (r < 0 || (unsigned int)r >= nrows || c < 0 || (unsigned int)c >= ncols) { \
        output[idx] = (dtype)0; \
        return; \
    } \
    \
    /* Row-major indexing: input[r, c] = input[r * ncols + c] */ \
    unsigned int input_idx = (unsigned int)r * ncols + (unsigned int)c; \
    output[idx] = input[input_idx]; \
}

// Instantiate gather_2d for all dtypes
DEFINE_GATHER_2D_KERNEL(f32, float)
DEFINE_GATHER_2D_KERNEL(f64, double)
DEFINE_GATHER_2D_KERNEL(f16, __half)
DEFINE_GATHER_2D_KERNEL(bf16, __nv_bfloat16)
DEFINE_GATHER_2D_KERNEL(i32, int)
DEFINE_GATHER_2D_KERNEL(i64, long long)
DEFINE_GATHER_2D_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_GATHER_2D_KERNEL(fp8_e5m2, numr_fp8_e5m2)

// ============================================================================
// Scatter Reduce - Prod (atomic multiply via CAS)
// ============================================================================

__device__ __forceinline__ float atomicMulFloat(float* address, float val) {
    int* address_as_int = (int*)address;
    int old = *address_as_int;
    int assumed;
    do {
        assumed = old;
        float old_val = __int_as_float(assumed);
        float new_val = old_val * val;
        old = atomicCAS(address_as_int, assumed, __float_as_int(new_val));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ __forceinline__ double atomicMulDouble(double* address, double val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull;
    unsigned long long assumed;
    do {
        assumed = old;
        double old_val = __longlong_as_double(assumed);
        double new_val = old_val * val;
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(new_val));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__device__ __forceinline__ int atomicMulInt(int* address, int val) {
    int old = *address;
    int assumed;
    do {
        assumed = old;
        int new_val = assumed * val;
        old = atomicCAS(address, assumed, new_val);
    } while (assumed != old);
    return old;
}

#define DEFINE_SCATTER_REDUCE_PROD_KERNEL(suffix, dtype, atomic_mul_fn) \
__global__ void scatter_reduce_prod_##suffix( \
    const dtype* __restrict__ src, \
    const long long* __restrict__ indices, \
    dtype* __restrict__ dst, \
    unsigned int dim, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int src_dim_size \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * src_dim_size * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int src_d = (idx / inner_size) % src_dim_size; \
    unsigned int outer = idx / (src_dim_size * inner_size); \
    \
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner; \
    \
    long long index_val = indices[src_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        return; \
    } \
    \
    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    \
    atomic_mul_fn(&dst[dst_idx], src[src_idx]); \
}

DEFINE_SCATTER_REDUCE_PROD_KERNEL(f32, float, atomicMulFloat)
DEFINE_SCATTER_REDUCE_PROD_KERNEL(f64, double, atomicMulDouble)
DEFINE_SCATTER_REDUCE_PROD_KERNEL(i32, int, atomicMulInt)

// ============================================================================
// Scatter Reduce - Count (for mean: atomicAdd 1 to count buffer per scatter)
// ============================================================================

#define DEFINE_SCATTER_REDUCE_COUNT_KERNEL(suffix, dtype) \
__global__ void scatter_reduce_count_##suffix( \
    const long long* __restrict__ indices, \
    dtype* __restrict__ count, \
    unsigned int dim, \
    unsigned int outer_size, \
    unsigned int dim_size, \
    unsigned int inner_size, \
    unsigned int src_dim_size \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * src_dim_size * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int src_d = (idx / inner_size) % src_dim_size; \
    unsigned int outer = idx / (src_dim_size * inner_size); \
    \
    unsigned int src_idx = outer * src_dim_size * inner_size + src_d * inner_size + inner; \
    \
    long long index_val = indices[src_idx]; \
    if (index_val < 0 || (unsigned int)index_val >= dim_size) { \
        return; \
    } \
    \
    unsigned int dst_idx = outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner; \
    \
    atomicAdd(&count[dst_idx], (dtype)1); \
}

DEFINE_SCATTER_REDUCE_COUNT_KERNEL(f32, float)
DEFINE_SCATTER_REDUCE_COUNT_KERNEL(f64, double)

// ============================================================================
// Scatter Reduce - Mean Divide (element-wise: out[i] = sum[i] / count[i])
// ============================================================================

#define DEFINE_SCATTER_REDUCE_MEAN_DIV_KERNEL(suffix, dtype) \
__global__ void scatter_reduce_mean_div_##suffix( \
    const dtype* __restrict__ sum_buf, \
    const dtype* __restrict__ count_buf, \
    dtype* __restrict__ output, \
    unsigned int n \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= n) return; \
    \
    dtype c = count_buf[idx]; \
    if (c > (dtype)0) { \
        output[idx] = sum_buf[idx] / c; \
    } else { \
        output[idx] = (dtype)0; \
    } \
}

DEFINE_SCATTER_REDUCE_MEAN_DIV_KERNEL(f32, float)
DEFINE_SCATTER_REDUCE_MEAN_DIV_KERNEL(f64, double)

// ============================================================================
// Slice Assign - Copy src into a slice of dst along a dimension
// dst: full destination tensor (outer_size * dst_dim_size * inner_size)
// src: source tensor (outer_size * src_dim_size * inner_size)
// output: pre-copied dst, then src overwrites the slice region
// ============================================================================

#define DEFINE_SLICE_ASSIGN_KERNEL(suffix, dtype) \
__global__ void slice_assign_##suffix( \
    const dtype* __restrict__ src, \
    dtype* __restrict__ output, \
    unsigned int outer_size, \
    unsigned int dst_dim_size, \
    unsigned int src_dim_size, \
    unsigned int inner_size, \
    unsigned int start \
) { \
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int total = outer_size * src_dim_size * inner_size; \
    if (idx >= total) return; \
    \
    unsigned int inner = idx % inner_size; \
    unsigned int s = (idx / inner_size) % src_dim_size; \
    unsigned int o = idx / (src_dim_size * inner_size); \
    \
    unsigned int dst_offset = o * dst_dim_size * inner_size + (start + s) * inner_size + inner; \
    output[dst_offset] = src[idx]; \
}

DEFINE_SLICE_ASSIGN_KERNEL(f32, float)
DEFINE_SLICE_ASSIGN_KERNEL(f64, double)
DEFINE_SLICE_ASSIGN_KERNEL(f16, __half)
DEFINE_SLICE_ASSIGN_KERNEL(bf16, __nv_bfloat16)
DEFINE_SLICE_ASSIGN_KERNEL(i32, int)
DEFINE_SLICE_ASSIGN_KERNEL(i64, long long)
DEFINE_SLICE_ASSIGN_KERNEL(fp8_e4m3, numr_fp8_e4m3)
DEFINE_SLICE_ASSIGN_KERNEL(fp8_e5m2, numr_fp8_e5m2)

} // extern "C"
