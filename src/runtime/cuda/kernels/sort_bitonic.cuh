// Block-level bitonic sort, and the three kernel bodies built on it: sort,
// argsort, and topk. One block handles one (outer, inner) slice.
//
// Included by sort.cu, which instantiates one row per dtype.

#ifndef NUMR_SORT_BITONIC_CUH
#define NUMR_SORT_BITONIC_CUH

#include "sort_compare.cuh"

// Shared memory layout, sized by sort_shared_mem_size() in
// src/runtime/cuda/kernels/sort.rs: n values of T, then n i64 indices, with the
// index array aligned to 8 bytes. n is sort_size rounded up to a power of two,
// which is what the bitonic network needs.
template<typename T>
__device__ __forceinline__ void bitonic_shared_layout(
    unsigned int n, T** vals, long long** idx
) {
    extern __shared__ char shared_mem[];
    T* v = (T*)shared_mem;
    char* idx_start = (char*)(v + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    *vals = v;
    *idx = (long long*)idx_start;
}

__device__ __forceinline__ unsigned int bitonic_padded_size(unsigned int sort_size) {
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;
    return n;
}

// Load one slice into shared memory, pad it out to n, and run the network.
// Returns false when this block has no slice, in which case every thread of the
// block returns together and no barrier is skipped by only some of them.
template<typename T>
__device__ __forceinline__ bool bitonic_sort_slice(
    const T* input, unsigned int outer_size, unsigned int sort_size,
    unsigned int inner_size, bool descending,
    T** out_vals, long long** out_idx,
    unsigned int* out_outer, unsigned int* out_inner
) {
    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    if (outer_idx >= outer_size || inner_idx >= inner_size) return false;

    unsigned int n = bitonic_padded_size(sort_size);
    T* shared_vals;
    long long* shared_idx;
    bitonic_shared_layout<T>(n, &shared_vals, &shared_idx);

    unsigned int tid = threadIdx.x;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    // Pad entries carry the maximum rank and an index past the slice, so they
    // sort into the discarded tail and never win a tie against a real element.
    T pad_val = sort_pad_rank<T>(descending);
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    for (unsigned int k = 2; k <= n; k *= 2) {
        for (unsigned int j = k / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;
                bool ascending_local = ((ij / k) % 2 == 0);

                if (ij_pair < n) {
                    bitonic_cas_indexed(
                        shared_vals[ij], shared_idx[ij],
                        shared_vals[ij_pair], shared_idx[ij_pair],
                        ascending_local, descending
                    );
                }
            }
            __syncthreads();
        }
    }

    *out_vals = shared_vals;
    *out_idx = shared_idx;
    *out_outer = outer_idx;
    *out_inner = inner_idx;
    return true;
}

// Sort along a dimension, optionally also writing the permutation.
template<typename T>
__device__ void sort_dim_impl(
    const T* input, T* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending, bool output_indices
) {
    T* shared_vals;
    long long* shared_idx;
    unsigned int outer_idx, inner_idx;
    if (!bitonic_sort_slice<T>(input, outer_size, sort_size, inner_size, descending,
                               &shared_vals, &shared_idx, &outer_idx, &inner_idx)) {
        return;
    }

    for (unsigned int i = threadIdx.x; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        output[out_idx] = shared_vals[i];
        if (output_indices && indices != nullptr) {
            indices[out_idx] = shared_idx[i];
        }
    }
}

// Argsort: the same network, writing only the permutation.
template<typename T>
__device__ void argsort_dim_impl(
    const T* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    T* shared_vals;
    long long* shared_idx;
    unsigned int outer_idx, inner_idx;
    if (!bitonic_sort_slice<T>(input, outer_size, sort_size, inner_size, descending,
                               &shared_vals, &shared_idx, &outer_idx, &inner_idx)) {
        return;
    }

    for (unsigned int i = threadIdx.x; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// Top-K: fully sort the slice, then keep the leading k entries. `largest`
// selects the sort direction, so the wanted end is always the front.
template<typename T>
__device__ void topk_dim_impl(
    const T* input, T* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    T* shared_vals;
    long long* shared_idx;
    unsigned int outer_idx, inner_idx;
    if (!bitonic_sort_slice<T>(input, outer_size, sort_size, inner_size, largest,
                               &shared_vals, &shared_idx, &outer_idx, &inner_idx)) {
        return;
    }

    // `sorted == false` asks for the k results in input order instead of rank
    // order. k is small in every caller, so one thread reorders them in place.
    if (!sorted) {
        for (unsigned int i = 0; i < k && threadIdx.x == 0; i++) {
            for (unsigned int j = i + 1; j < k; j++) {
                if (shared_idx[i] > shared_idx[j]) {
                    T tmp_val = shared_vals[i];
                    shared_vals[i] = shared_vals[j];
                    shared_vals[j] = tmp_val;
                    long long tmp_idx = shared_idx[i];
                    shared_idx[i] = shared_idx[j];
                    shared_idx[j] = tmp_idx;
                }
            }
        }
        __syncthreads();
    }

    for (unsigned int i = threadIdx.x; i < k; i += blockDim.x) {
        unsigned int out_idx = outer_idx * k * inner_size + i * inner_size + inner_idx;
        out_values[out_idx] = shared_vals[i];
        out_indices[out_idx] = shared_idx[i];
    }
}

#endif // NUMR_SORT_BITONIC_CUH
