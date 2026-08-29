// Linear-scan search and counting kernel bodies: count_nonzero,
// gather_nonzero, searchsorted, count_unique, extract_unique.
//
// Included by sort.cu, which instantiates one row per dtype.
//
// Semantics match the CPU reference in src/runtime/cpu/kernels/sort.rs:
//
//  * "nonzero" is `value != 0` in the element type, which is what CPU's
//    `count_nonzero_kernel` / `nonzero_flat_kernel` spell as `to_f64() != 0.0`.
//    The two agree for every integer dtype: no non-zero integer converts to
//    0.0, however wide it is.
//  * "unique" compares each element of an already sorted array against its
//    predecessor with `!=`, matching CPU's `extract_unique_kernel`.

#ifndef NUMR_SORT_SCAN_CUH
#define NUMR_SORT_SCAN_CUH

#include "sort_compare.cuh"

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

// Writes the flat position of each nonzero element. The slots are claimed with
// an atomic counter, so the output order is not the input order; the caller
// sorts the result when it needs one.
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

template<typename T>
__device__ void searchsorted_impl(
    const T* sorted_seq, const T* values,
    long long* output, unsigned int seq_len, unsigned int num_values, bool right
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int i = tid; i < num_values; i += blockDim.x * gridDim.x) {
        T val = values[i];

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

#endif // NUMR_SORT_SCAN_CUH
