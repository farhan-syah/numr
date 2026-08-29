// Sorting and search CUDA kernels: sort, sort_values_only, argsort, topk,
// count_nonzero, gather_nonzero, searchsorted, count_unique, extract_unique —
// nine kernels per dtype from one row macro.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/sort.rs from dtype_suffix() in loader.rs:
// {op}_{suffix}, e.g. count_unique_u32. Those launchers have no dtype gate, so
// a dtype missing from the rows below fails at kernel lookup, not with an
// UnsupportedDType error.
//
// The bodies live in sort_compare.cuh (total order, padding, bitonic swap),
// sort_bitonic.cuh (sort/argsort/topk), and sort_scan.cuh (the linear-scan
// search and counting family).

#include "sort_bitonic.cuh"
#include "sort_scan.cuh"

// ============================================================================
// Instantiation macro
// ============================================================================

#define NUMR_SORT_ROW(T, S)                                                     \
    __global__ void sort_##S(                                                   \
        const T* input, T* output, long long* indices,                          \
        unsigned int outer_size, unsigned int sort_size,                        \
        unsigned int inner_size, bool descending) {                             \
        sort_dim_impl<T>(input, output, indices, outer_size, sort_size,         \
                         inner_size, descending, true);                         \
    }                                                                           \
    __global__ void sort_values_only_##S(                                       \
        const T* input, T* output, unsigned int outer_size,                     \
        unsigned int sort_size, unsigned int inner_size, bool descending) {     \
        sort_dim_impl<T>(input, output, nullptr, outer_size, sort_size,         \
                         inner_size, descending, false);                        \
    }                                                                           \
    __global__ void argsort_##S(                                                \
        const T* input, long long* indices, unsigned int outer_size,            \
        unsigned int sort_size, unsigned int inner_size, bool descending) {     \
        argsort_dim_impl<T>(input, indices, outer_size, sort_size, inner_size,  \
                            descending);                                        \
    }                                                                           \
    __global__ void topk_##S(                                                   \
        const T* input, T* out_values, long long* out_indices,                  \
        unsigned int outer_size, unsigned int sort_size,                        \
        unsigned int inner_size, unsigned int k, bool largest, bool sorted) {   \
        topk_dim_impl<T>(input, out_values, out_indices, outer_size, sort_size, \
                         inner_size, k, largest, sorted);                       \
    }                                                                           \
    __global__ void count_nonzero_##S(                                          \
        const T* input, unsigned int* count, unsigned int n) {                  \
        count_nonzero_impl<T>(input, count, n);                                 \
    }                                                                           \
    __global__ void gather_nonzero_##S(                                         \
        const T* input, long long* indices, unsigned int* counter,              \
        unsigned int n) {                                                       \
        gather_nonzero_impl<T>(input, indices, counter, n);                     \
    }                                                                           \
    __global__ void searchsorted_##S(                                           \
        const T* seq, const T* values, long long* output,                       \
        unsigned int seq_len, unsigned int num_values, bool right) {            \
        searchsorted_impl<T>(seq, values, output, seq_len, num_values, right);  \
    }                                                                           \
    __global__ void count_unique_##S(                                           \
        const T* input, unsigned int* count, unsigned int n) {                  \
        count_unique_impl<T>(input, count, n);                                  \
    }                                                                           \
    __global__ void extract_unique_##S(                                         \
        const T* input, T* output, unsigned int* counter, unsigned int n) {     \
        extract_unique_impl<T>(input, output, counter, n);                      \
    }

extern "C" {

// ============================================================================
// Dtype-independent kernels
// ============================================================================

// Expand flat positions into row-major coordinates, the shape `nonzero`
// returns.
__global__ void flat_to_multi_index(
    const long long* flat_indices, long long* multi_indices,
    unsigned int nnz, unsigned int ndim,
    const unsigned int* shape
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nnz) return;

    long long flat_idx = flat_indices[tid];

    for (int d = (int)ndim - 1; d >= 0; d--) {
        multi_indices[tid * ndim + d] = flat_idx % shape[d];
        flat_idx /= shape[d];
    }
}

// Counts how many times each bin index appears; `unique_with_counts` runs this
// over the inverse-index array to get per-value occurrence counts.
__global__ void bincount(const long long* indices, long long* counts,
                         unsigned int n, unsigned int num_bins) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        long long idx = indices[i];
        if (idx >= 0 && idx < (long long)num_bins) {
            atomicAdd((unsigned long long*)&counts[idx], 1ULL);
        }
    }
}

// ============================================================================
// Per-dtype rows
// ============================================================================

NUMR_SORT_ROW(float, f32)
NUMR_SORT_ROW(double, f64)
NUMR_SORT_ROW(__half, f16)
NUMR_SORT_ROW(__nv_bfloat16, bf16)
NUMR_SORT_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_SORT_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_SORT_ROW(long long, i64)
NUMR_SORT_ROW(int, i32)
NUMR_SORT_ROW(short, i16)
NUMR_SORT_ROW(signed char, i8)
NUMR_SORT_ROW(unsigned long long, u64)
NUMR_SORT_ROW(unsigned int, u32)
NUMR_SORT_ROW(unsigned short, u16)
NUMR_SORT_ROW(unsigned char, u8)

} // extern "C"
