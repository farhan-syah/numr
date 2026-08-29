// Shared machinery for the coordinate-addressed indexing kernels instantiated
// in index_nd.cu: gather_nd, gather_2d, and slice_assign.
//
// These are split out of index_ops.cuh because they take a different argument
// shape — gather_nd reads a whole coordinate vector per output slice — and
// because index.cu is at its size limit with the per-element family alone.
//
// Semantics match the CPU reference in src/runtime/cpu/kernels/index.rs
// (`gather_nd_kernel`, `gather_2d_kernel`, `slice_assign_kernel`): an
// out-of-range coordinate yields zero, and slice_assign writes only the slice
// region, its caller having pre-copied the rest.

#ifndef NUMR_INDEX_ND_OPS_CUH
#define NUMR_INDEX_ND_OPS_CUH

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "index_ops.cuh"

// ============================================================================
// Kernel body templates
// ============================================================================

// gather_nd: `indices` holds `num_slices` coordinate vectors of `index_depth`
// components each; every output element is one element of the addressed slice.
template<typename T>
__device__ __forceinline__ void gather_nd_impl(
    const T* __restrict__ input, const long long* __restrict__ indices,
    T* __restrict__ output,
    NUMR_DIM_ARGS(input_shape), NUMR_DIM_ARGS(input_strides),
    unsigned int num_slices, unsigned int slice_size, unsigned int index_depth
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_slices * slice_size;
    if (idx >= total) return;

    const unsigned int in_shape[INDEX_MAX_DIMS] = NUMR_DIM_PACK(input_shape);
    const unsigned int in_strides[INDEX_MAX_DIMS] = NUMR_DIM_PACK(input_strides);

    unsigned int slice_idx = idx / slice_size;
    unsigned int within_slice = idx % slice_size;

    unsigned int src_offset = 0;
    for (unsigned int d = 0; d < index_depth; d++) {
        long long index_val = indices[slice_idx * index_depth + d];
        if (index_val < 0 || (unsigned long long)index_val >= in_shape[d]) {
            output[idx] = (T)0;
            return;
        }
        src_offset += (unsigned int)index_val * in_strides[d];
    }

    // The trailing dimensions are contiguous, so the offset within the slice is
    // the flat position itself.
    output[idx] = input[src_offset + within_slice];
}

// gather_2d: output[i] = input[rows[i], cols[i]] over a row-major 2D input.
template<typename T>
__device__ __forceinline__ void gather_2d_impl(
    const T* __restrict__ input, const long long* __restrict__ rows,
    const long long* __restrict__ cols, T* __restrict__ output,
    unsigned int nrows, unsigned int ncols, unsigned int num_indices
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_indices) return;

    long long r = rows[idx];
    long long c = cols[idx];

    if (r < 0 || (unsigned long long)r >= nrows || c < 0 || (unsigned long long)c >= ncols) {
        output[idx] = (T)0;
        return;
    }

    output[idx] = input[(unsigned int)r * ncols + (unsigned int)c];
}

// slice_assign: copy src into the [start, start + src_dim_size) window of the
// destination dimension. `output` already holds a copy of the destination.
template<typename T>
__device__ __forceinline__ void slice_assign_impl(
    const T* __restrict__ src, T* __restrict__ output,
    unsigned int outer_size, unsigned int dst_dim_size,
    unsigned int src_dim_size, unsigned int inner_size, unsigned int start
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = outer_size * src_dim_size * inner_size;
    if (idx >= total) return;

    unsigned int inner = idx % inner_size;
    unsigned int s = (idx / inner_size) % src_dim_size;
    unsigned int o = idx / (src_dim_size * inner_size);

    output[o * dst_dim_size * inner_size + (start + s) * inner_size + inner] = src[idx];
}

// ============================================================================
// Instantiation macro
// ============================================================================
// NUMR_INDEX_ND_ROW emits the three kernels of one dtype. The suffixes are the
// ones dtype_suffix() produces in src/runtime/cuda/kernels/loader.rs.

#define NUMR_INDEX_ND_ROW(T, S)                                                 \
    __global__ void gather_nd_##S(                                              \
        const T* __restrict__ input, const long long* __restrict__ indices,     \
        T* __restrict__ output,                                                 \
        NUMR_DIM_ARGS(input_shape), NUMR_DIM_ARGS(input_strides),               \
        unsigned int num_slices, unsigned int slice_size,                       \
        unsigned int index_depth, unsigned int) {                               \
        gather_nd_impl<T>(input, indices, output,                               \
                          NUMR_DIM_CALL(input_shape),                           \
                          NUMR_DIM_CALL(input_strides),                         \
                          num_slices, slice_size, index_depth);                 \
    }                                                                           \
    __global__ void gather_2d_##S(                                              \
        const T* __restrict__ input, const long long* __restrict__ rows,        \
        const long long* __restrict__ cols, T* __restrict__ output,             \
        unsigned int nrows, unsigned int ncols, unsigned int num_indices) {     \
        gather_2d_impl<T>(input, rows, cols, output, nrows, ncols,              \
                          num_indices);                                         \
    }                                                                           \
    __global__ void slice_assign_##S(                                           \
        const T* __restrict__ src, T* __restrict__ output,                      \
        unsigned int outer_size, unsigned int dst_dim_size,                     \
        unsigned int src_dim_size, unsigned int inner_size,                     \
        unsigned int start) {                                                   \
        slice_assign_impl<T>(src, output, outer_size, dst_dim_size,             \
                             src_dim_size, inner_size, start);                  \
    }

#endif // NUMR_INDEX_ND_OPS_CUH
