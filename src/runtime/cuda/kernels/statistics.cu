// Statistics CUDA kernels - Mode computation on sorted data
// Types: f32, f64, i32, i64, u32
//
// Mode operation finds the most frequent value along a dimension.
// Input must be pre-sorted along the reduce dimension.
//
// Algorithm: Run-length encoding on sorted data
// - Each thread block processes one output element (one slice)
// - Sequential scan within slice to find longest run
// - Parallel across slices (outer_size * inner_size blocks)

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Mode Device Function Implementation
// ============================================================================

// Mode device function for dimension-wise reduction on sorted data
// Each block computes mode for one (outer, inner) position
template<typename T>
__device__ void mode_dim_impl(
    const T* __restrict__ sorted,      // [outer_size * reduce_size * inner_size]
    T* __restrict__ mode_values,       // [outer_size * inner_size]
    long long* __restrict__ mode_counts, // [outer_size * inner_size]
    unsigned int outer_size,
    unsigned int reduce_size,
    unsigned int inner_size
) {
    // Only thread 0 does the sequential scan (mode is inherently sequential).
    // There is no __syncthreads() below, so returning early here is safe.
    if (threadIdx.x != 0) return;

    // Handle empty slice: nothing to scan, and no defined mode to write.
    if (reduce_size == 0) {
        return;
    }

    // One block owns one (outer, inner) output element at a time. Both axes
    // grid-stride: the launcher clamps each grid axis to the architectural
    // maximum ([2147483647, 65535, 65535]) and these loops cover the remainder,
    // so inner_size > 65535 neither fails the launch nor leaves outputs unwritten.
    for (unsigned int outer = blockIdx.x; outer < outer_size; outer += gridDim.x) {
        for (unsigned int inner = blockIdx.y; inner < inner_size; inner += gridDim.y) {
            // Base offset for this slice
            unsigned int base = outer * reduce_size * inner_size + inner;

            // Initialize with first element
            T best_val = sorted[base];
            long long best_count = 1;

            T curr_val = best_val;
            long long curr_count = 1;

            // Scan through sorted slice
            for (unsigned int r = 1; r < reduce_size; r++) {
                unsigned int idx = base + r * inner_size;
                T val = sorted[idx];

                if (val == curr_val) {
                    curr_count++;
                } else {
                    // End of current run
                    if (curr_count > best_count) {
                        best_val = curr_val;
                        best_count = curr_count;
                    }
                    curr_val = val;
                    curr_count = 1;
                }
            }

            // Check final run
            if (curr_count > best_count) {
                best_val = curr_val;
                best_count = curr_count;
            }

            unsigned int out_idx = outer * inner_size + inner;
            mode_values[out_idx] = best_val;
            mode_counts[out_idx] = best_count;
        }
    }
}

// Full mode device function (reduce entire tensor to single mode value)
template<typename T>
__device__ void mode_full_impl(
    const T* __restrict__ sorted,      // [numel]
    T* __restrict__ mode_value,        // [1]
    long long* __restrict__ mode_count, // [1]
    unsigned int numel
) {
    // Single thread processes entire array
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    if (numel == 0) return;

    T best_val = sorted[0];
    long long best_count = 1;

    T curr_val = best_val;
    long long curr_count = 1;

    for (unsigned int i = 1; i < numel; i++) {
        T val = sorted[i];

        if (val == curr_val) {
            curr_count++;
        } else {
            if (curr_count > best_count) {
                best_val = curr_val;
                best_count = curr_count;
            }
            curr_val = val;
            curr_count = 1;
        }
    }

    if (curr_count > best_count) {
        best_val = curr_val;
        best_count = curr_count;
    }

    mode_value[0] = best_val;
    mode_count[0] = best_count;
}

// ============================================================================
// Extern C Wrappers
// ============================================================================

extern "C" {

// F32 mode kernels
__global__ void mode_dim_f32(
    const float* sorted, float* mode_values, long long* mode_counts,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    mode_dim_impl<float>(sorted, mode_values, mode_counts, outer_size, reduce_size, inner_size);
}

__global__ void mode_full_f32(
    const float* sorted, float* mode_value, long long* mode_count, unsigned int numel
) {
    mode_full_impl<float>(sorted, mode_value, mode_count, numel);
}

// F64 mode kernels
__global__ void mode_dim_f64(
    const double* sorted, double* mode_values, long long* mode_counts,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    mode_dim_impl<double>(sorted, mode_values, mode_counts, outer_size, reduce_size, inner_size);
}

__global__ void mode_full_f64(
    const double* sorted, double* mode_value, long long* mode_count, unsigned int numel
) {
    mode_full_impl<double>(sorted, mode_value, mode_count, numel);
}

// I32 mode kernels
__global__ void mode_dim_i32(
    const int* sorted, int* mode_values, long long* mode_counts,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    mode_dim_impl<int>(sorted, mode_values, mode_counts, outer_size, reduce_size, inner_size);
}

__global__ void mode_full_i32(
    const int* sorted, int* mode_value, long long* mode_count, unsigned int numel
) {
    mode_full_impl<int>(sorted, mode_value, mode_count, numel);
}

// I64 mode kernels
__global__ void mode_dim_i64(
    const long long* sorted, long long* mode_values, long long* mode_counts,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    mode_dim_impl<long long>(sorted, mode_values, mode_counts, outer_size, reduce_size, inner_size);
}

__global__ void mode_full_i64(
    const long long* sorted, long long* mode_value, long long* mode_count, unsigned int numel
) {
    mode_full_impl<long long>(sorted, mode_value, mode_count, numel);
}

// U32 mode kernels
__global__ void mode_dim_u32(
    const unsigned int* sorted, unsigned int* mode_values, long long* mode_counts,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    mode_dim_impl<unsigned int>(sorted, mode_values, mode_counts, outer_size, reduce_size, inner_size);
}

__global__ void mode_full_u32(
    const unsigned int* sorted, unsigned int* mode_value, long long* mode_count, unsigned int numel
) {
    mode_full_impl<unsigned int>(sorted, mode_value, mode_count, numel);
}

} // extern "C"
