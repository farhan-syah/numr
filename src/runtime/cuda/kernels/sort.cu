// Sorting and search CUDA kernels
// Supports: sort, argsort, topk, unique, nonzero, searchsorted
// Types: f32, f64, i32, i64, u32, u64
//
// Uses bitonic sort for GPU-friendly parallel sorting within shared memory blocks.
// For dimension-wise operations, uses the (outer, sort, inner) decomposition pattern.

#include <cuda_runtime.h>
#include "dtype_traits.cuh"

// ============================================================================
// FP8 comparison operators for templated sort/search kernels
// ============================================================================

__device__ __forceinline__ bool operator<(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    return fp8_e4m3_to_f32(a.data) < fp8_e4m3_to_f32(b.data);
}
__device__ __forceinline__ bool operator>(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    return fp8_e4m3_to_f32(a.data) > fp8_e4m3_to_f32(b.data);
}
__device__ __forceinline__ bool operator==(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    return fp8_e4m3_to_f32(a.data) == fp8_e4m3_to_f32(b.data);
}
__device__ __forceinline__ bool operator!=(numr_fp8_e4m3 a, numr_fp8_e4m3 b) {
    return fp8_e4m3_to_f32(a.data) != fp8_e4m3_to_f32(b.data);
}

__device__ __forceinline__ bool operator<(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    return fp8_e5m2_to_f32(a.data) < fp8_e5m2_to_f32(b.data);
}
__device__ __forceinline__ bool operator>(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    return fp8_e5m2_to_f32(a.data) > fp8_e5m2_to_f32(b.data);
}
__device__ __forceinline__ bool operator==(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    return fp8_e5m2_to_f32(a.data) == fp8_e5m2_to_f32(b.data);
}
__device__ __forceinline__ bool operator!=(numr_fp8_e5m2 a, numr_fp8_e5m2 b) {
    return fp8_e5m2_to_f32(a.data) != fp8_e5m2_to_f32(b.data);
}

// ============================================================================
// Sort padding value helpers (type-safe max/min for bitonic sort padding)
// ============================================================================

template<typename T> __device__ __forceinline__ T sort_pad_max();
template<typename T> __device__ __forceinline__ T sort_pad_min();

template<> __device__ __forceinline__ float sort_pad_max<float>() { return 1e38f; }
template<> __device__ __forceinline__ float sort_pad_min<float>() { return -1e38f; }
template<> __device__ __forceinline__ double sort_pad_max<double>() { return 1e308; }
template<> __device__ __forceinline__ double sort_pad_min<double>() { return -1e308; }
template<> __device__ __forceinline__ int sort_pad_max<int>() { return INT_MAX; }
template<> __device__ __forceinline__ int sort_pad_min<int>() { return INT_MIN; }
template<> __device__ __forceinline__ long long sort_pad_max<long long>() { return LLONG_MAX; }
template<> __device__ __forceinline__ long long sort_pad_min<long long>() { return LLONG_MIN; }
template<> __device__ __forceinline__ unsigned int sort_pad_max<unsigned int>() { return UINT_MAX; }
template<> __device__ __forceinline__ unsigned int sort_pad_min<unsigned int>() { return 0u; }
template<> __device__ __forceinline__ unsigned long long sort_pad_max<unsigned long long>() { return ULLONG_MAX; }
template<> __device__ __forceinline__ unsigned long long sort_pad_min<unsigned long long>() { return 0ull; }
template<> __device__ __forceinline__ __half sort_pad_max<__half>() { return __float2half(65504.0f); }
template<> __device__ __forceinline__ __half sort_pad_min<__half>() { return __float2half(-65504.0f); }
template<> __device__ __forceinline__ __nv_bfloat16 sort_pad_max<__nv_bfloat16>() { return __float2bfloat16(1e38f); }
template<> __device__ __forceinline__ __nv_bfloat16 sort_pad_min<__nv_bfloat16>() { return __float2bfloat16(-1e38f); }
template<> __device__ __forceinline__ numr_fp8_e4m3 sort_pad_max<numr_fp8_e4m3>() { return numr_fp8_e4m3(f32_to_fp8_e4m3(FP8_E4M3_MAX)); }
template<> __device__ __forceinline__ numr_fp8_e4m3 sort_pad_min<numr_fp8_e4m3>() { return numr_fp8_e4m3(f32_to_fp8_e4m3(FP8_E4M3_MIN)); }
template<> __device__ __forceinline__ numr_fp8_e5m2 sort_pad_max<numr_fp8_e5m2>() { return numr_fp8_e5m2(f32_to_fp8_e5m2(FP8_E5M2_MAX)); }
template<> __device__ __forceinline__ numr_fp8_e5m2 sort_pad_min<numr_fp8_e5m2>() { return numr_fp8_e5m2(f32_to_fp8_e5m2(FP8_E5M2_MIN)); }

// ============================================================================
// Comparison helpers for sorting
// ============================================================================

// NaN detection. Integer types have no NaN, so the default is always false.
// __half/__nv_bfloat16 need __hisnan because their operator!= is ordered and
// returns false when either operand is NaN.
template<typename T> __device__ __forceinline__ bool sort_is_nan(T) { return false; }
template<> __device__ __forceinline__ bool sort_is_nan<float>(float v) { return isnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<double>(double v) { return isnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<__half>(__half v) { return __hisnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<__nv_bfloat16>(__nv_bfloat16 v) { return __hisnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<numr_fp8_e4m3>(numr_fp8_e4m3 v) { return isnan(fp8_e4m3_to_f32(v.data)); }
template<> __device__ __forceinline__ bool sort_is_nan<numr_fp8_e5m2>(numr_fp8_e5m2 v) { return isnan(fp8_e5m2_to_f32(v.data)); }

// Total order shared by every backend: NaN compares greater than all non-NaN
// values, NaNs tie with each other, and -0.0 ties with +0.0. Mirrors
// `Element::sort_cmp` on the CPU side.
template<typename T>
__device__ __forceinline__ int sort_cmp(T a, T b) {
    bool a_nan = sort_is_nan(a);
    bool b_nan = sort_is_nan(b);
    if (a_nan || b_nan) {
        if (a_nan && b_nan) return 0;
        return a_nan ? 1 : -1;
    }
    if (a < b) return -1;
    if (b < a) return 1;
    return 0;
}

// Padding value of maximum rank, so pad entries sort into the discarded tail.
// Ascending that is NaN for float types (NaN is the greatest value); descending
// it is -inf. Both must be beyond any real value, otherwise real infinities get
// pushed past the padding and dropped.
template<typename T> __device__ __forceinline__ T sort_pad_rank(bool descending) {
    return descending ? sort_pad_min<T>() : sort_pad_max<T>();
}
template<> __device__ __forceinline__ float sort_pad_rank<float>(bool descending) {
    return descending ? -INFINITY : NAN;
}
template<> __device__ __forceinline__ double sort_pad_rank<double>(bool descending) {
    return descending ? -INFINITY : NAN;
}
template<> __device__ __forceinline__ __half sort_pad_rank<__half>(bool descending) {
    return __float2half(descending ? -INFINITY : NAN);
}
template<> __device__ __forceinline__ __nv_bfloat16 sort_pad_rank<__nv_bfloat16>(bool descending) {
    return __float2bfloat16(descending ? -INFINITY : NAN);
}
template<> __device__ __forceinline__ numr_fp8_e4m3 sort_pad_rank<numr_fp8_e4m3>(bool descending) {
    return numr_fp8_e4m3(f32_to_fp8_e4m3(descending ? -INFINITY : NAN));
}
template<> __device__ __forceinline__ numr_fp8_e5m2 sort_pad_rank<numr_fp8_e5m2>(bool descending) {
    return numr_fp8_e5m2(f32_to_fp8_e5m2(descending ? -INFINITY : NAN));
}

// ============================================================================
// Bitonic sort primitives (for small arrays in shared memory)
// ============================================================================

// Rank order: the requested output order, ties broken by original index. The
// network always sorts ascending in this rank space, so `descending` flips the
// value comparison only and stability holds in both directions.
template<typename T>
__device__ __forceinline__ int sort_rank_cmp(T a_val, long long a_idx,
                                             T b_val, long long b_idx,
                                             bool descending) {
    int c = sort_cmp(a_val, b_val);
    if (descending) c = -c;
    if (c != 0) return c;
    if (a_idx == b_idx) return 0;
    return a_idx < b_idx ? -1 : 1;
}

// Bitonic compare-and-swap with indices
template<typename T>
__device__ __forceinline__ void bitonic_cas_indexed(T& a_val, long long& a_idx,
                                                     T& b_val, long long& b_idx,
                                                     bool ascending_local,
                                                     bool descending) {
    int c = sort_rank_cmp(a_val, a_idx, b_val, b_idx, descending);
    bool swap = ascending_local ? (c > 0) : (c < 0);
    if (swap) {
        T tmp_val = a_val;
        a_val = b_val;
        b_val = tmp_val;
        long long tmp_idx = a_idx;
        a_idx = b_idx;
        b_idx = tmp_idx;
    }
}

// ============================================================================
// Templated sort implementations
// ============================================================================

// Sort along dimension - uses block-level sorting
// Each block handles one (outer, inner) slice
template<typename T>
__device__ void sort_dim_impl(
    const T* input, T* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending, bool output_indices
) {
    // Calculate padded size FIRST (needed for shared memory layout)
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    // Shared memory for sort buffer (max 1024 elements per block)
    // Layout: [n values of type T][n indices of type long long]
    extern __shared__ char shared_mem[];
    T* shared_vals = (T*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data into shared memory
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    // Pad with max/min values
    T pad_val = sort_pad_rank<T>(descending);
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size; // Invalid index
    }
    __syncthreads();

    // Bitonic sort network
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

    // Write results back
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        output[out_idx] = shared_vals[i];
        if (output_indices && indices != nullptr) {
            indices[out_idx] = shared_idx[i];
        }
    }
}

// Top-K along dimension
// Uses partial sort - only keep k smallest/largest
template<typename T>
__device__ void topk_dim_impl(
    const T* input, T* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    T* shared_vals = (T*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    // For small k, use selection approach
    // For larger k, do full sort then take top k

    // Full bitonic sort for simplicity (can optimize for partial sort later)

    T pad_val = sort_pad_rank<T>(largest);
    for (unsigned int i = tid + sort_size; i < n; i += blockDim.x) {
        shared_vals[i] = pad_val;
        shared_idx[i] = sort_size;
    }
    __syncthreads();

    bool descending = largest;

    for (unsigned int kk = 2; kk <= n; kk *= 2) {
        for (unsigned int j = kk / 2; j > 0; j /= 2) {
            for (unsigned int i = tid; i < n / 2; i += blockDim.x) {
                unsigned int ij = (i / j) * 2 * j + (i % j);
                unsigned int ij_pair = ij + j;

                bool ascending_local = ((ij / kk) % 2 == 0);

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

    // If not sorted, re-sort by original index (to preserve input order)
    if (!sorted) {
        // Simple bubble sort for top-k elements (k is usually small)
        for (unsigned int i = 0; i < k && tid == 0; i++) {
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

    // Write top-k results
    for (unsigned int i = tid; i < k; i += blockDim.x) {
        unsigned int out_idx = outer_idx * k * inner_size + i * inner_size + inner_idx;
        out_values[out_idx] = shared_vals[i];
        out_indices[out_idx] = shared_idx[i];
    }
}

// Count nonzero elements
template<typename T>
__device__ void count_nonzero_impl(
    const T* input, unsigned int* count, unsigned int n
) {
    __shared__ unsigned int block_count;

    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_count = 0;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] != (T)0) {
            local_count++;
        }
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Gather nonzero indices
template<typename T>
__device__ void gather_nonzero_impl(
    const T* input, long long* flat_indices, unsigned int* counter, unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (input[i] != (T)0) {
            unsigned int pos = atomicAdd(counter, 1);
            flat_indices[pos] = i;
        }
    }
}

// Binary search (searchsorted)
template<typename T>
__device__ void searchsorted_impl(
    const T* sorted_seq, const T* values,
    long long* output, unsigned int seq_len, unsigned int num_values, bool right
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < num_values; i += blockDim.x * gridDim.x) {
        T val = values[i];

        // Binary search
        unsigned int lo = 0;
        unsigned int hi = seq_len;

        while (lo < hi) {
            unsigned int mid = lo + (hi - lo) / 2;
            T mid_val = sorted_seq[mid];

            // Same total order the sequence was sorted by, so NaN keys land at
            // the trailing NaN run instead of collapsing to position 0.
            int c = sort_cmp(mid_val, val);
            bool go_left = right ? (c <= 0) : (c < 0);

            if (go_left) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        output[i] = lo;
    }
}

// Count unique elements in sorted array
template<typename T>
__device__ void count_unique_impl(
    const T* sorted_input, unsigned int* count, unsigned int n
) {
    __shared__ unsigned int block_count;

    if (threadIdx.x == 0) block_count = 0;
    __syncthreads();

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_count = 0;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (i == 0 || sorted_input[i] != sorted_input[i - 1]) {
            local_count++;
        }
    }

    atomicAdd(&block_count, local_count);
    __syncthreads();

    if (threadIdx.x == 0) {
        atomicAdd(count, block_count);
    }
}

// Extract unique elements from sorted array
template<typename T>
__device__ void extract_unique_impl(
    const T* sorted_input, T* unique_output,
    unsigned int* counter, unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        if (i == 0 || sorted_input[i] != sorted_input[i - 1]) {
            unsigned int pos = atomicAdd(counter, 1);
            unique_output[pos] = sorted_input[i];
        }
    }
}

// Bincount: count occurrences of each index in the input array
// Used for counting unique element occurrences in unique_with_counts
__device__ void bincount_impl(
    const long long* indices, long long* counts,
    unsigned int n, unsigned int num_bins
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < n; i += blockDim.x * gridDim.x) {
        long long idx = indices[i];
        if (idx >= 0 && idx < (long long)num_bins) {
            atomicAdd((unsigned long long*)&counts[idx], 1ULL);
        }
    }
}

// ============================================================================
// Templated argsort (indices only, no values output)
// ============================================================================

template<typename T>
__device__ void argsort_dim_impl(
    const T* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    T* shared_vals = (T*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ============================================================================
// extern "C" wrapper kernels for Rust FFI
// ============================================================================

extern "C" {

// Convert flat index to multi-dimensional index
__global__ void flat_to_multi_index(
    const long long* flat_indices, long long* multi_indices,
    unsigned int nnz, unsigned int ndim,
    const unsigned int* shape
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= nnz) return;

    long long flat_idx = flat_indices[tid];

    // Compute multi-index (row-major order)
    for (int d = ndim - 1; d >= 0; d--) {
        multi_indices[tid * ndim + d] = flat_idx % shape[d];
        flat_idx /= shape[d];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - F32
// ----------------------------------------------------------------------------
__global__ void sort_f32(
    const float* input, float* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<float>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<float>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_f32(
    const float* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    float* shared_vals = (float*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    // Load data
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    float pad_val = sort_pad_rank<float>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    // Write indices only
    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - F64
// ----------------------------------------------------------------------------
__global__ void sort_f64(
    const double* input, double* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<double>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<double>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_f64(
    const double* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    double* shared_vals = (double*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    double pad_val = sort_pad_rank<double>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - I32
// ----------------------------------------------------------------------------
__global__ void sort_i32(
    const int* input, int* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<int>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_i32(
    const int* input, int* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<int>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_i32(
    const int* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    int* shared_vals = (int*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    int pad_val = sort_pad_rank<int>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - I64
// ----------------------------------------------------------------------------
__global__ void sort_i64(
    const long long* input, long long* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<long long>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_i64(
    const long long* input, long long* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<long long>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_i64(
    const long long* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    long long* shared_vals = (long long*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    long long pad_val = sort_pad_rank<long long>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - U32
// ----------------------------------------------------------------------------
__global__ void sort_u32(
    const unsigned int* input, unsigned int* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned int>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_u32(
    const unsigned int* input, unsigned int* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned int>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_u32(
    const unsigned int* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    unsigned int* shared_vals = (unsigned int*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    unsigned int pad_val = sort_pad_rank<unsigned int>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Sort with indices - U64
// ----------------------------------------------------------------------------
__global__ void sort_u64(
    const unsigned long long* input, unsigned long long* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned long long>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_u64(
    const unsigned long long* input, unsigned long long* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<unsigned long long>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_u64(
    const unsigned long long* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    // Calculate padded size FIRST for correct shared memory layout
    unsigned int n = 1;
    while (n < sort_size) n <<= 1;

    extern __shared__ char shared_mem[];
    unsigned long long* shared_vals = (unsigned long long*)shared_mem;
    // Align to 8 bytes for long long access
    char* idx_start = (char*)(shared_vals + n);
    idx_start = (char*)(((unsigned long long)idx_start + 7) & ~7ULL);
    long long* shared_idx = (long long*)idx_start;

    unsigned int outer_idx = blockIdx.x;
    unsigned int inner_idx = blockIdx.y;
    unsigned int tid = threadIdx.x;

    if (outer_idx >= outer_size || inner_idx >= inner_size) return;

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        shared_vals[i] = input[idx];
        shared_idx[i] = i;
    }
    __syncthreads();

    unsigned long long pad_val = sort_pad_rank<unsigned long long>(descending);
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
                    bitonic_cas_indexed(shared_vals[ij], shared_idx[ij],
                                       shared_vals[ij_pair], shared_idx[ij_pair],
                                       ascending_local, descending);
                }
            }
            __syncthreads();
        }
    }

    for (unsigned int i = tid; i < sort_size; i += blockDim.x) {
        unsigned int out_idx = outer_idx * sort_size * inner_size + i * inner_size + inner_idx;
        indices[out_idx] = shared_idx[i];
    }
}

// ----------------------------------------------------------------------------
// Top-K kernels
// ----------------------------------------------------------------------------
__global__ void topk_f32(
    const float* input, float* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<float>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_f64(
    const double* input, double* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<double>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_i32(
    const int* input, int* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<int>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void topk_i64(
    const long long* input, long long* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<long long>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

// ----------------------------------------------------------------------------
// Nonzero count kernels
// ----------------------------------------------------------------------------
__global__ void count_nonzero_f32(const float* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<float>(input, count, n);
}

__global__ void count_nonzero_f64(const double* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<double>(input, count, n);
}

__global__ void count_nonzero_i32(const int* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<int>(input, count, n);
}

__global__ void count_nonzero_i64(const long long* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<long long>(input, count, n);
}

// ----------------------------------------------------------------------------
// Nonzero gather kernels
// ----------------------------------------------------------------------------
__global__ void gather_nonzero_f32(const float* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<float>(input, indices, counter, n);
}

__global__ void gather_nonzero_f64(const double* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<double>(input, indices, counter, n);
}

__global__ void gather_nonzero_i32(const int* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<int>(input, indices, counter, n);
}

__global__ void gather_nonzero_i64(const long long* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<long long>(input, indices, counter, n);
}

// ----------------------------------------------------------------------------
// Searchsorted kernels
// ----------------------------------------------------------------------------
__global__ void searchsorted_f32(const float* seq, const float* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<float>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_f64(const double* seq, const double* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<double>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_i32(const int* seq, const int* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<int>(seq, values, output, seq_len, num_values, right);
}

__global__ void searchsorted_i64(const long long* seq, const long long* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<long long>(seq, values, output, seq_len, num_values, right);
}

// ----------------------------------------------------------------------------
// Unique count kernels
// ----------------------------------------------------------------------------
__global__ void count_unique_f32(const float* input, unsigned int* count, unsigned int n) {
    count_unique_impl<float>(input, count, n);
}

__global__ void count_unique_f64(const double* input, unsigned int* count, unsigned int n) {
    count_unique_impl<double>(input, count, n);
}

__global__ void count_unique_i32(const int* input, unsigned int* count, unsigned int n) {
    count_unique_impl<int>(input, count, n);
}

__global__ void count_unique_i64(const long long* input, unsigned int* count, unsigned int n) {
    count_unique_impl<long long>(input, count, n);
}

// ----------------------------------------------------------------------------
// Unique extract kernels
// ----------------------------------------------------------------------------
__global__ void extract_unique_f32(const float* input, float* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<float>(input, output, counter, n);
}

__global__ void extract_unique_f64(const double* input, double* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<double>(input, output, counter, n);
}

__global__ void extract_unique_i32(const int* input, int* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<int>(input, output, counter, n);
}

__global__ void extract_unique_i64(const long long* input, long long* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<long long>(input, output, counter, n);
}

// ----------------------------------------------------------------------------
// Bincount kernel - counts occurrences of indices
// ----------------------------------------------------------------------------
__global__ void bincount(const long long* indices, long long* counts,
                         unsigned int n, unsigned int num_bins) {
    bincount_impl(indices, counts, n, num_bins);
}

// ============================================================================
// F16 (__half) sort/search kernels
// ============================================================================

__global__ void sort_f16(
    const __half* input, __half* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<__half>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<__half>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_f16(
    const __half* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    argsort_dim_impl<__half>(input, indices, outer_size, sort_size, inner_size, descending);
}

__global__ void topk_f16(
    const __half* input, __half* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<__half>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void count_nonzero_f16(const __half* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<__half>(input, count, n);
}

__global__ void gather_nonzero_f16(const __half* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<__half>(input, indices, counter, n);
}

__global__ void searchsorted_f16(const __half* seq, const __half* values, long long* output,
                                  unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<__half>(seq, values, output, seq_len, num_values, right);
}

__global__ void count_unique_f16(const __half* input, unsigned int* count, unsigned int n) {
    count_unique_impl<__half>(input, count, n);
}

__global__ void extract_unique_f16(const __half* input, __half* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<__half>(input, output, counter, n);
}

// ============================================================================
// BF16 (__nv_bfloat16) sort/search kernels
// ============================================================================

__global__ void sort_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<__nv_bfloat16>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<__nv_bfloat16>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_bf16(
    const __nv_bfloat16* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    argsort_dim_impl<__nv_bfloat16>(input, indices, outer_size, sort_size, inner_size, descending);
}

__global__ void topk_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<__nv_bfloat16>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void count_nonzero_bf16(const __nv_bfloat16* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<__nv_bfloat16>(input, count, n);
}

__global__ void gather_nonzero_bf16(const __nv_bfloat16* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<__nv_bfloat16>(input, indices, counter, n);
}

__global__ void searchsorted_bf16(const __nv_bfloat16* seq, const __nv_bfloat16* values, long long* output,
                                   unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<__nv_bfloat16>(seq, values, output, seq_len, num_values, right);
}

__global__ void count_unique_bf16(const __nv_bfloat16* input, unsigned int* count, unsigned int n) {
    count_unique_impl<__nv_bfloat16>(input, count, n);
}

__global__ void extract_unique_bf16(const __nv_bfloat16* input, __nv_bfloat16* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<__nv_bfloat16>(input, output, counter, n);
}

// ============================================================================
// FP8 E4M3 sort/search kernels
// ============================================================================

__global__ void sort_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<numr_fp8_e4m3>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<numr_fp8_e4m3>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_fp8_e4m3(
    const numr_fp8_e4m3* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    argsort_dim_impl<numr_fp8_e4m3>(input, indices, outer_size, sort_size, inner_size, descending);
}

__global__ void topk_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<numr_fp8_e4m3>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void count_nonzero_fp8_e4m3(const numr_fp8_e4m3* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<numr_fp8_e4m3>(input, count, n);
}

__global__ void gather_nonzero_fp8_e4m3(const numr_fp8_e4m3* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<numr_fp8_e4m3>(input, indices, counter, n);
}

__global__ void searchsorted_fp8_e4m3(const numr_fp8_e4m3* seq, const numr_fp8_e4m3* values, long long* output,
                                       unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<numr_fp8_e4m3>(seq, values, output, seq_len, num_values, right);
}

__global__ void count_unique_fp8_e4m3(const numr_fp8_e4m3* input, unsigned int* count, unsigned int n) {
    count_unique_impl<numr_fp8_e4m3>(input, count, n);
}

__global__ void extract_unique_fp8_e4m3(const numr_fp8_e4m3* input, numr_fp8_e4m3* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<numr_fp8_e4m3>(input, output, counter, n);
}

// ============================================================================
// FP8 E5M2 sort/search kernels
// ============================================================================

__global__ void sort_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<numr_fp8_e5m2>(input, output, indices, outer_size, sort_size, inner_size, descending, true);
}

__global__ void sort_values_only_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    sort_dim_impl<numr_fp8_e5m2>(input, output, nullptr, outer_size, sort_size, inner_size, descending, false);
}

__global__ void argsort_fp8_e5m2(
    const numr_fp8_e5m2* input, long long* indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    bool descending
) {
    argsort_dim_impl<numr_fp8_e5m2>(input, indices, outer_size, sort_size, inner_size, descending);
}

__global__ void topk_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* out_values, long long* out_indices,
    unsigned int outer_size, unsigned int sort_size, unsigned int inner_size,
    unsigned int k, bool largest, bool sorted
) {
    topk_dim_impl<numr_fp8_e5m2>(input, out_values, out_indices, outer_size, sort_size, inner_size, k, largest, sorted);
}

__global__ void count_nonzero_fp8_e5m2(const numr_fp8_e5m2* input, unsigned int* count, unsigned int n) {
    count_nonzero_impl<numr_fp8_e5m2>(input, count, n);
}

__global__ void gather_nonzero_fp8_e5m2(const numr_fp8_e5m2* input, long long* indices, unsigned int* counter, unsigned int n) {
    gather_nonzero_impl<numr_fp8_e5m2>(input, indices, counter, n);
}

__global__ void searchsorted_fp8_e5m2(const numr_fp8_e5m2* seq, const numr_fp8_e5m2* values, long long* output,
                                       unsigned int seq_len, unsigned int num_values, bool right) {
    searchsorted_impl<numr_fp8_e5m2>(seq, values, output, seq_len, num_values, right);
}

__global__ void count_unique_fp8_e5m2(const numr_fp8_e5m2* input, unsigned int* count, unsigned int n) {
    count_unique_impl<numr_fp8_e5m2>(input, count, n);
}

__global__ void extract_unique_fp8_e5m2(const numr_fp8_e5m2* input, numr_fp8_e5m2* output, unsigned int* counter, unsigned int n) {
    extract_unique_impl<numr_fp8_e5m2>(input, output, counter, n);
}

} // extern "C"
