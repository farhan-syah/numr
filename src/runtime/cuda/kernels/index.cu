// Per-element indexing CUDA kernels: gather, scatter, copy, index_select,
// index_put, masked_select, masked_fill, embedding_lookup, and the broadcast
// masked pair — ten kernels per dtype from one row macro.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8, bool
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/index/ from dtype_suffix() in loader.rs:
// {op}_{suffix}, e.g. gather_u32 or masked_fill_broadcast_i16.
//
// The operation bodies and the row macro live in index_ops.cuh, which also
// documents the out-of-range and mask conventions. Coordinate-addressed
// indexing (gather_nd, gather_2d, slice_assign) is in index_nd.cu and
// scatter-with-reduction is in scatter_reduce.cu; both are separate PTX
// modules, named by the *_MODULE constants in loader.rs.

#include "index_ops.cuh"

extern "C" {

// ============================================================================
// Mask counting and prefix sum (dtype-independent: the mask is always u8)
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

// Exclusive prefix sum of the mask, run on a single thread. masked_select needs
// the destination slots in input order, and a parallel scan would have to
// preserve that same order to be worth it.
__global__ void masked_prefix_sum_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    unsigned int n
) {
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

__global__ void masked_prefix_sum_broadcast_kernel(
    const unsigned char* __restrict__ mask,
    unsigned int* __restrict__ prefix_sum,
    const unsigned int* __restrict__ mask_strides,
    const unsigned int* __restrict__ out_shape,
    unsigned int ndim,
    unsigned int n
) {
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

// ============================================================================
// Index bounds validation (dtype-independent: indices are always i64)
// ============================================================================

// Counts indices outside [0, dim_size) into error_count[0]. A non-zero count
// tells the host to raise the error rather than launch the indexing kernel.
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
// Bincount
// ============================================================================
// Counts occurrences of each non-negative value below minlength. The input
// dtype and the weight dtype vary independently, so these do not fit the row
// macro; the four combinations below are the ones launch_bincount_weighted
// dispatches.

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
// Per-dtype rows
// ============================================================================
// The CPU backend dispatches indexing over EVERY dtype (`dispatch_dtype!`), so
// a dtype missing here is one that indexes on CPU and fails on CUDA with
// "named symbol not found" at launch — a gap that shows up only on the device,
// never at compile time. `bool` is one byte per element, matching the CPU
// backend's Bool layout, so it is a byte row like u8's.

NUMR_INDEX_ROW(float, f32)
NUMR_INDEX_ROW(double, f64)
NUMR_INDEX_ROW(__half, f16)
NUMR_INDEX_ROW(__nv_bfloat16, bf16)
NUMR_INDEX_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_INDEX_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_INDEX_ROW(int64_t, i64)
NUMR_INDEX_ROW(int32_t, i32)
NUMR_INDEX_ROW(int16_t, i16)
NUMR_INDEX_ROW(int8_t, i8)
NUMR_INDEX_ROW(uint64_t, u64)
NUMR_INDEX_ROW(uint32_t, u32)
NUMR_INDEX_ROW(uint16_t, u16)
NUMR_INDEX_ROW(uint8_t, u8)
NUMR_INDEX_ROW(unsigned char, bool)

} // extern "C"
