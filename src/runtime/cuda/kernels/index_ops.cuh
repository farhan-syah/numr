// Shared machinery for the per-element indexing kernels: one body template per
// operation plus the macros that expand one dtype into a full row of
// `extern "C"` wrappers.
//
// Instantiated by index.cu (gather, scatter, copy, index_select, index_put,
// masked_select, masked_fill, embedding_lookup) and index_nd.cu (gather_nd,
// gather_2d, slice_assign, the broadcast masked pair).
//
// Semantics, all of them matching the CPU reference in
// src/runtime/cpu/kernels/index.rs:
//
//  * An out-of-range index writes zero on a read op (gather, index_select,
//    gather_nd, gather_2d, embedding_lookup) and writes nothing on a write op
//    (scatter, index_put). CPU spells both out the same way.
//  * A mask element is true when its byte is non-zero, matching the CPU
//    backend's one-byte Bool layout.
//
// Shape and stride arrays arrive as individual scalar kernel arguments rather
// than device pointers, which is what makes these kernels safe to capture into
// a CUDA graph: a pointer argument would encode a host-side Vec address that
// dangles on replay. Unused trailing slots are zero-padded by the Rust
// launcher.

#ifndef NUMR_INDEX_OPS_CUH
#define NUMR_INDEX_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// Must match MAX_DIMS in src/runtime/cuda/kernels/index/gather.rs.
#ifndef INDEX_MAX_DIMS
#define INDEX_MAX_DIMS 8
#endif

// ============================================================================
// Scalar-argument packs
// ============================================================================
// Each pack is one INDEX_MAX_DIMS-wide u32 array flattened into the kernel
// parameter block. The NAME token keeps the parameter names distinct when a
// kernel takes several packs.

#define NUMR_DIM_ARGS(NAME)                                                     \
    unsigned int NAME##0, unsigned int NAME##1, unsigned int NAME##2,           \
    unsigned int NAME##3, unsigned int NAME##4, unsigned int NAME##5,           \
    unsigned int NAME##6, unsigned int NAME##7

#define NUMR_DIM_PACK(NAME)                                                     \
    { NAME##0, NAME##1, NAME##2, NAME##3, NAME##4, NAME##5, NAME##6, NAME##7 }

#define NUMR_DIM_CALL(NAME)                                                     \
    NAME##0, NAME##1, NAME##2, NAME##3, NAME##4, NAME##5, NAME##6, NAME##7

// A pack the kernel body does not read, declared without parameter names. The
// host still passes it, so the slot must stay in the signature; naming it would
// only draw an unused-parameter warning.
#define NUMR_DIM_ARGS_UNUSED                                                    \
    unsigned int, unsigned int, unsigned int, unsigned int,                     \
    unsigned int, unsigned int, unsigned int, unsigned int

// The packs are unpacked into per-thread local arrays, not shared memory. The
// original kernels staged them in shared memory behind a `threadIdx.x == 0`
// write and a `__syncthreads()` placed *after* the out-of-range early return,
// so any block with an inactive thread reached a partial barrier. A local array
// of eight u32 lives in registers and needs no barrier at all.

// ============================================================================
// Kernel body templates
// ============================================================================

// gather: output[idx] = input[... indices[idx] along dim ...]
template<typename T>
__device__ __forceinline__ void gather_impl(
    const T* __restrict__ input, const long long* __restrict__ indices,
    T* __restrict__ output, unsigned int ndim, unsigned int dim,
    NUMR_DIM_ARGS(input_shape), NUMR_DIM_ARGS(input_strides),
    NUMR_DIM_ARGS(output_strides), unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    const unsigned int in_shape[INDEX_MAX_DIMS] = NUMR_DIM_PACK(input_shape);
    const unsigned int in_strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(input_strides);
    const unsigned int out_strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(output_strides);

    unsigned int remaining = idx;
    unsigned int src_offset = 0;

    for (unsigned int d = 0; d < ndim; d++) {
        unsigned int coord = remaining / out_strides[d];
        remaining %= out_strides[d];

        if (d == dim) {
            long long index_val = indices[idx];
            if (index_val < 0 || (unsigned long long)index_val >= in_shape[d]) {
                output[idx] = (T)0;
                return;
            }
            src_offset += (unsigned int)index_val * in_strides[d];
        } else {
            src_offset += coord * in_strides[d];
        }
    }

    output[idx] = input[src_offset];
}

// scatter: output[... indices[idx] along dim ...] = src[idx]
template<typename T>
__device__ __forceinline__ void scatter_impl(
    const long long* __restrict__ indices, const T* __restrict__ src,
    T* __restrict__ output, unsigned int ndim, unsigned int dim,
    NUMR_DIM_ARGS(output_shape), NUMR_DIM_ARGS(output_strides),
    NUMR_DIM_ARGS(src_strides), unsigned int src_total
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= src_total) return;

    const unsigned int out_shape[INDEX_MAX_DIMS] = NUMR_DIM_PACK(output_shape);
    const unsigned int out_strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(output_strides);
    const unsigned int s_strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(src_strides);

    unsigned int remaining = idx;
    unsigned int dst_offset = 0;

    for (unsigned int d = 0; d < ndim; d++) {
        unsigned int coord = remaining / s_strides[d];
        remaining %= s_strides[d];

        if (d == dim) {
            long long index_val = indices[idx];
            if (index_val < 0 || (unsigned long long)index_val >= out_shape[d]) {
                return;
            }
            dst_offset += (unsigned int)index_val * out_strides[d];
        } else {
            dst_offset += coord * out_strides[d];
        }
    }

    output[dst_offset] = src[idx];
}

template<typename T>
__device__ __forceinline__ void copy_impl(
    const T* __restrict__ src, T* __restrict__ dst, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// index_select and index_put share the (outer, selected, inner) decomposition:
// index_select reads at the indexed position, index_put writes to it.
template<typename T>
__device__ __forceinline__ void index_select_impl(
    const T* __restrict__ input, const long long* __restrict__ indices,
    T* __restrict__ output, unsigned int outer_size, unsigned int dim_size,
    unsigned int inner_size, unsigned int index_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * index_len * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int sel_idx = (idx / inner_size) % index_len;
    unsigned int outer = idx / (index_len * inner_size);

    long long index_val = indices[sel_idx];
    if (index_val < 0 || (unsigned long long)index_val >= dim_size) {
        output[idx] = (T)0;
        return;
    }

    unsigned int src_offset =
        outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    output[idx] = input[src_offset];
}

template<typename T>
__device__ __forceinline__ void index_put_impl(
    const long long* __restrict__ indices, const T* __restrict__ src,
    T* __restrict__ output, unsigned int outer_size, unsigned int dim_size,
    unsigned int inner_size, unsigned int index_len
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * index_len * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int sel_idx = (idx / inner_size) % index_len;
    unsigned int outer = idx / (index_len * inner_size);

    long long index_val = indices[sel_idx];
    if (index_val < 0 || (unsigned long long)index_val >= dim_size) {
        return;
    }

    unsigned int dst_offset =
        outer * dim_size * inner_size + (unsigned int)index_val * inner_size + inner;
    output[dst_offset] = src[idx];
}

// masked_select writes the kept elements compactly, using the exclusive prefix
// sum of the mask the caller computed as the destination index.
template<typename T>
__device__ __forceinline__ void masked_select_impl(
    const T* __restrict__ input, const unsigned char* __restrict__ mask,
    T* __restrict__ output, const unsigned int* __restrict__ prefix_sum,
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (mask[idx] != 0) {
        output[prefix_sum[idx]] = input[idx];
    }
}

template<typename T>
__device__ __forceinline__ void masked_fill_impl(
    const T* __restrict__ input, const unsigned char* __restrict__ mask,
    T* __restrict__ output, T fill_value, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    output[idx] = (mask[idx] != 0) ? fill_value : input[idx];
}

// embedding_lookup: one thread per index, copying a whole embedding row.
template<typename T>
__device__ __forceinline__ void embedding_lookup_impl(
    const T* __restrict__ embeddings, const long long* __restrict__ indices,
    T* __restrict__ output, unsigned int num_indices, unsigned int vocab_size,
    unsigned int embedding_dim
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    long long index_val = indices[idx];
    T* out_row = output + (idx * embedding_dim);

    if (index_val < 0 || (unsigned long long)index_val >= vocab_size) {
        for (unsigned int i = 0; i < embedding_dim; i++) {
            out_row[i] = (T)0;
        }
        return;
    }

    const T* emb_row = embeddings + ((unsigned int)index_val * embedding_dim);
    for (unsigned int i = 0; i < embedding_dim; i++) {
        out_row[i] = emb_row[i];
    }
}

// ----------------------------------------------------------------------------
// Broadcast masked operations
// ----------------------------------------------------------------------------
// The mask is broadcast to the output shape by stride, where a stride of 0
// marks a broadcast dimension. The dtype-independent count and prefix-sum
// kernels in index.cu share this index computation, which is why it lives here.

__device__ __forceinline__ unsigned int compute_broadcast_index(
    unsigned int linear_idx,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim
) {
    unsigned int mask_offset = 0;
    unsigned int remaining = linear_idx;

    for (int d = (int)ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % out_shape[d];
        remaining /= out_shape[d];
        mask_offset += coord * mask_strides[d];
    }
    return mask_offset;
}

template<typename T>
__device__ __forceinline__ void masked_select_broadcast_impl(
    const T* __restrict__ input, const unsigned char* __restrict__ mask,
    T* __restrict__ output, const unsigned int* __restrict__ prefix_sum,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim);
    if (mask[mask_idx] != 0) {
        output[prefix_sum[idx]] = input[idx];
    }
}

template<typename T>
__device__ __forceinline__ void masked_fill_broadcast_impl(
    const T* __restrict__ input, const unsigned char* __restrict__ mask,
    T* __restrict__ output, T fill_value,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim, unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int mask_idx = compute_broadcast_index(idx, mask_strides, out_shape, ndim);
    output[idx] = (mask[mask_idx] != 0) ? fill_value : input[idx];
}

// ============================================================================
// Instantiation macro
// ============================================================================
// NUMR_INDEX_ROW emits the ten kernels of one dtype. The suffixes are the
// ones dtype_suffix() produces in src/runtime/cuda/kernels/loader.rs.

#define NUMR_INDEX_ROW(T, S)                                                    \
    __global__ void gather_##S(                                                 \
        const T* __restrict__ input, const long long* __restrict__ indices,     \
        T* __restrict__ output, unsigned int ndim, unsigned int dim,            \
        NUMR_DIM_ARGS(input_shape), NUMR_DIM_ARGS(input_strides),               \
        NUMR_DIM_ARGS_UNUSED, NUMR_DIM_ARGS(output_strides),                    \
        unsigned int total_elements) {                                          \
        gather_impl<T>(input, indices, output, ndim, dim,                       \
                       NUMR_DIM_CALL(input_shape), NUMR_DIM_CALL(input_strides),\
                       NUMR_DIM_CALL(output_strides), total_elements);          \
    }                                                                           \
    __global__ void scatter_##S(                                                \
        const T*, const long long* __restrict__ indices,                        \
        const T* __restrict__ src, T* __restrict__ output,                      \
        unsigned int ndim, unsigned int dim,                                    \
        NUMR_DIM_ARGS(output_shape), NUMR_DIM_ARGS(output_strides),             \
        NUMR_DIM_ARGS_UNUSED, NUMR_DIM_ARGS(src_strides),                       \
        unsigned int src_total) {                                               \
        scatter_impl<T>(indices, src, output, ndim, dim,                        \
                        NUMR_DIM_CALL(output_shape),                            \
                        NUMR_DIM_CALL(output_strides),                          \
                        NUMR_DIM_CALL(src_strides), src_total);                 \
    }                                                                           \
    __global__ void copy_##S(                                                   \
        const T* __restrict__ src, T* __restrict__ dst, unsigned int n) {       \
        copy_impl<T>(src, dst, n);                                              \
    }                                                                           \
    __global__ void index_select_##S(                                           \
        const T* __restrict__ input, const long long* __restrict__ indices,     \
        T* __restrict__ output, unsigned int outer_size, unsigned int dim_size, \
        unsigned int inner_size, unsigned int index_len) {                      \
        index_select_impl<T>(input, indices, output, outer_size, dim_size,      \
                             inner_size, index_len);                            \
    }                                                                           \
    __global__ void index_put_##S(                                              \
        const long long* __restrict__ indices, const T* __restrict__ src,       \
        T* __restrict__ output, unsigned int outer_size, unsigned int dim_size, \
        unsigned int inner_size, unsigned int index_len) {                      \
        index_put_impl<T>(indices, src, output, outer_size, dim_size,           \
                          inner_size, index_len);                               \
    }                                                                           \
    __global__ void masked_select_##S(                                          \
        const T* __restrict__ input, const unsigned char* __restrict__ mask,    \
        T* __restrict__ output, const unsigned int* __restrict__ prefix_sum,    \
        unsigned int n) {                                                       \
        masked_select_impl<T>(input, mask, output, prefix_sum, n);              \
    }                                                                           \
    __global__ void masked_fill_##S(                                            \
        const T* __restrict__ input, const unsigned char* __restrict__ mask,    \
        T* __restrict__ output, T fill_value, unsigned int n) {                 \
        masked_fill_impl<T>(input, mask, output, fill_value, n);                \
    }                                                                           \
    __global__ void embedding_lookup_##S(                                       \
        const T* __restrict__ embeddings,                                       \
        const long long* __restrict__ indices, T* __restrict__ output,          \
        unsigned int num_indices, unsigned int vocab_size,                      \
        unsigned int embedding_dim) {                                           \
        embedding_lookup_impl<T>(embeddings, indices, output, num_indices,      \
                                 vocab_size, embedding_dim);                    \
    }                                                                           \
    __global__ void masked_select_broadcast_##S(                                \
        const T* __restrict__ input, const unsigned char* __restrict__ mask,    \
        T* __restrict__ output, const unsigned int* __restrict__ prefix_sum,    \
        const unsigned int* __restrict__ mask_strides,                          \
        const unsigned int* __restrict__ out_shape,                             \
        unsigned int ndim, unsigned int n) {                                    \
        masked_select_broadcast_impl<T>(input, mask, output, prefix_sum,        \
                                        mask_strides, out_shape, ndim, n);      \
    }                                                                           \
    __global__ void masked_fill_broadcast_##S(                                  \
        const T* __restrict__ input, const unsigned char* __restrict__ mask,    \
        T* __restrict__ output, T fill_value,                                   \
        const unsigned int* __restrict__ mask_strides,                          \
        const unsigned int* __restrict__ out_shape,                             \
        unsigned int ndim, unsigned int n) {                                    \
        masked_fill_broadcast_impl<T>(input, mask, output, fill_value,          \
                                      mask_strides, out_shape, ndim, n);        \
    }

#endif // NUMR_INDEX_OPS_CUH
