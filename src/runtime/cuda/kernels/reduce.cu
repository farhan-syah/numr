// Reduction CUDA kernels - Templated implementation using AccumTraits
// Supports: sum, max, min, mean
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2
//
// Uses AccumTraits<T, Acc> for DRY reduction code across all type/accumulation combinations.
// Similar to CPU's Accumulator trait pattern.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// ============================================================================
// Templated Reduction Device Functions
// These are called by the extern "C" __global__ wrapper kernels
// ============================================================================

// Global sum reduction - produces partial sums per block
template<typename T, typename Acc>
__device__ void reduce_sum_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32]; // One element per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc sum = Traits::zero();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        sum = Traits::add(sum, Traits::load(input, i));
    }

    sum = Traits::warp_sum(sum);

    if (lane == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        sum = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::zero();
        sum = Traits::warp_sum(sum);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, sum);
        }
    }
}

// Global max reduction
template<typename T, typename Acc>
__device__ void reduce_max_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc max_val = Traits::neg_inf();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        max_val = Traits::max(max_val, Traits::load(input, i));
    }

    max_val = Traits::warp_max(max_val);

    if (lane == 0) {
        shared[warp_id] = max_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        max_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::neg_inf();
        max_val = Traits::warp_max(max_val);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, max_val);
        }
    }
}

// Global min reduction
template<typename T, typename Acc>
__device__ void reduce_min_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32];

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc min_val = Traits::pos_inf();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        min_val = Traits::min(min_val, Traits::load(input, i));
    }

    min_val = Traits::warp_min(min_val);

    if (lane == 0) {
        shared[warp_id] = min_val;
    }
    __syncthreads();

    if (warp_id == 0) {
        min_val = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::pos_inf();
        min_val = Traits::warp_min(min_val);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, min_val);
        }
    }
}

// Global prod reduction - produces partial products per block
template<typename T, typename Acc>
__device__ void reduce_prod_impl(const T* input, T* output, unsigned int n) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[32]; // One element per warp

    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int lane = tid % 32;
    unsigned int warp_id = tid / 32;

    Acc prod = Traits::one();
    for (unsigned int i = idx; i < n; i += blockDim.x * gridDim.x) {
        prod = Traits::mul(prod, Traits::load(input, i));
    }

    prod = Traits::warp_prod(prod);

    if (lane == 0) {
        shared[warp_id] = prod;
    }
    __syncthreads();

    if (warp_id == 0) {
        prod = (tid < (blockDim.x + 31) / 32) ? shared[lane] : Traits::one();
        prod = Traits::warp_prod(prod);
        if (tid == 0) {
            Traits::store(output, blockIdx.x, prod);
        }
    }
}

// ============================================================================
// Dimension-wise reductions: grid-stride over BOTH output axes
//
// The tensor is viewed as [outer_size, reduce_size, inner_size] and reduced
// along the middle axis. One thread block owns one (outer, inner) output
// element at a time.
//
// CUDA caps grid dimensions at [2147483647, 65535, 65535], so mapping outer to
// blockIdx.x and inner to blockIdx.y one-to-one makes any launch with
// inner_size > 65535 fail with CUDA_ERROR_INVALID_VALUE (hit by e.g. summing
// [1, 32, 4096, 64] over dim 1, where inner_size = 262144). Both axes therefore
// grid-stride: the launcher clamps each grid axis to its architectural maximum
// and these loops cover the remainder.
//
// The stride loops change only WHICH block computes an output element, never
// how that element is accumulated: each output is still reduced by a single
// block, sequentially over `i += blockDim.x` then through the same shared-memory
// tree. Results are bit-identical to the pre-fix kernels.
//
// The trip counts depend only on blockIdx/gridDim/sizes, so they are uniform
// across every thread of a block and the __syncthreads() calls stay legal.
// ============================================================================

// Dimension-wise sum reduction
template<typename T, typename Acc>
__device__ void reduce_sum_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            Acc sum = Traits::zero();
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                sum = Traits::add(sum, Traits::load(input, idx));
            }

            shared[tid] = sum;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = Traits::add(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
            }
            __syncthreads();
        }
    }
}

// Dimension-wise max reduction
template<typename T, typename Acc>
__device__ void reduce_max_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            Acc max_val = Traits::neg_inf();
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                max_val = Traits::max(max_val, Traits::load(input, idx));
            }

            shared[tid] = max_val;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = Traits::max(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
            }
            __syncthreads();
        }
    }
}

// Dimension-wise min reduction
template<typename T, typename Acc>
__device__ void reduce_min_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            Acc min_val = Traits::pos_inf();
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                min_val = Traits::min(min_val, Traits::load(input, idx));
            }

            shared[tid] = min_val;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = Traits::min(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
            }
            __syncthreads();
        }
    }
}

// Argmax along dimension - uses float for comparison (or double for f64)
template<typename T, typename CompareType>
__device__ void argmax_dim_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ CompareType shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            CompareType max_val = Traits::neg_inf();
            long long max_idx = 0;

            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                CompareType val = Traits::load(input, idx);
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            shared_val[tid] = max_val;
            shared_idx[tid] = max_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (shared_val[tid + s] > shared_val[tid]) {
                        shared_val[tid] = shared_val[tid + s];
                        shared_idx[tid] = shared_idx[tid + s];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[outer_idx * inner_size + inner_idx] = shared_idx[0];
            }
            __syncthreads();
        }
    }
}

// Argmin along dimension
template<typename T, typename CompareType>
__device__ void argmin_dim_impl(
    const T* input, long long* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ CompareType shared_val[256];
    __shared__ long long shared_idx[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            CompareType min_val = Traits::pos_inf();
            long long min_idx = 0;

            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                CompareType val = Traits::load(input, idx);
                if (val < min_val) {
                    min_val = val;
                    min_idx = i;
                }
            }

            shared_val[tid] = min_val;
            shared_idx[tid] = min_idx;
            __syncthreads();

            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    if (shared_val[tid + s] < shared_val[tid]) {
                        shared_val[tid] = shared_val[tid + s];
                        shared_idx[tid] = shared_idx[tid + s];
                    }
                }
                __syncthreads();
            }

            if (tid == 0) {
                output[outer_idx * inner_size + inner_idx] = shared_idx[0];
            }
            __syncthreads();
        }
    }
}

// Dimension-wise prod reduction
template<typename T, typename Acc>
__device__ void reduce_prod_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, Acc>;
    __shared__ Acc shared[256];
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            Acc prod = Traits::one();
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                prod = Traits::mul(prod, Traits::load(input, idx));
            }

            shared[tid] = prod;
            __syncthreads();

            // Tree reduction with multiplication
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = Traits::mul(shared[tid], shared[tid + s]);
                }
                __syncthreads();
            }

            if (tid == 0) {
                Traits::store(output, outer_idx * inner_size + inner_idx, shared[0]);
            }
            __syncthreads();
        }
    }
}

// Dimension-wise any reduction (logical OR - true if any element is non-zero)
// Works on any dtype by converting to bool (non-zero = true)
template<typename T, typename CompareType>
__device__ void reduce_any_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            int any_true = 0;  // false
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                CompareType val = Traits::load(input, idx);
                // Non-zero means true
                if (val != Traits::zero()) {
                    any_true = 1;
                }
            }

            shared[tid] = any_true;
            __syncthreads();

            // Tree reduction with OR
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] | shared[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                // Store 1 if any true, 0 otherwise (in output dtype)
                Traits::store(output, outer_idx * inner_size + inner_idx,
                              shared[0] ? Traits::one() : Traits::zero());
            }
            __syncthreads();
        }
    }
}

// Dimension-wise all reduction (logical AND - true if all elements are non-zero)
template<typename T, typename CompareType>
__device__ void reduce_all_dim_impl(
    const T* input, T* output,
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size
) {
    using Traits = AccumTraits<T, CompareType>;
    __shared__ int shared[256];  // Use int for boolean reduction
    unsigned int tid = threadIdx.x;

    for (unsigned int outer_idx = blockIdx.x; outer_idx < outer_size; outer_idx += gridDim.x) {
        for (unsigned int inner_idx = blockIdx.y; inner_idx < inner_size; inner_idx += gridDim.y) {
            int all_true = 1;  // true
            for (unsigned int i = tid; i < reduce_size; i += blockDim.x) {
                unsigned int idx = outer_idx * reduce_size * inner_size + i * inner_size + inner_idx;
                CompareType val = Traits::load(input, idx);
                // Zero means false
                if (val == Traits::zero()) {
                    all_true = 0;
                }
            }

            shared[tid] = all_true;
            __syncthreads();

            // Tree reduction with AND
            for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s) {
                    shared[tid] = shared[tid] & shared[tid + s];
                }
                __syncthreads();
            }

            if (tid == 0) {
                // Store 1 if all true, 0 otherwise (in output dtype)
                Traits::store(output, outer_idx * inner_size + inner_idx,
                              shared[0] ? Traits::one() : Traits::zero());
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Instantiation macros - eliminates boilerplate in extern "C" wrappers
// ============================================================================

// Global reductions: reduce_sum_SUFFIX, reduce_max_SUFFIX, etc.
#define INSTANTIATE_GLOBAL_REDUCE(T, Acc, SUFFIX) \
__global__ void reduce_sum_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_sum_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_max_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_max_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_min_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_min_impl<T, Acc>(input, output, n); \
} \
__global__ void reduce_prod_##SUFFIX(const T* input, T* output, unsigned int n) { \
    reduce_prod_impl<T, Acc>(input, output, n); \
}

// Dim reductions: reduce_sum_dim_SUFFIX, reduce_max_dim_SUFFIX, reduce_min_dim_SUFFIX
#define INSTANTIATE_DIM_REDUCE_CORE(T, Acc, SUFFIX) \
__global__ void reduce_sum_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_sum_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_max_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_max_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_min_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_min_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
}

// Argmax/argmin dim: argmax_dim_SUFFIX, argmin_dim_SUFFIX
#define INSTANTIATE_ARG_REDUCE(T, CompareType, SUFFIX) \
__global__ void argmax_dim_##SUFFIX( \
    const T* input, long long* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    argmax_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void argmin_dim_##SUFFIX( \
    const T* input, long long* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    argmin_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
}

// Prod dim: reduce_prod_dim_SUFFIX
#define INSTANTIATE_PROD_DIM(T, Acc, SUFFIX) \
__global__ void reduce_prod_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_prod_dim_impl<T, Acc>(input, output, outer_size, reduce_size, inner_size); \
}

// Any/all dim: reduce_any_dim_SUFFIX, reduce_all_dim_SUFFIX
#define INSTANTIATE_LOGICAL_DIM(T, CompareType, SUFFIX) \
__global__ void reduce_any_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_any_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
} \
__global__ void reduce_all_dim_##SUFFIX( \
    const T* input, T* output, \
    unsigned int outer_size, unsigned int reduce_size, unsigned int inner_size \
) { \
    reduce_all_dim_impl<T, CompareType>(input, output, outer_size, reduce_size, inner_size); \
}

// ============================================================================
// extern "C" wrapper kernels for Rust FFI
//
// Accumulation Strategy:
//   Each dtype has multiple accumulation variants selected by suffix:
//   - Native (e.g. f32): Accumulates in the same type. Fastest, but may
//     lose precision for large reductions on low-precision types.
//   - fp32acc (e.g. f16_fp32acc): Accumulates in f32. Default for f16/bf16
//     reductions — good precision/speed tradeoff.
//   - fp64acc (e.g. f16_fp64acc): Accumulates in f64. Maximum precision for
//     scientific computing workloads. Slowest due to f64 throughput on GPU.
//   - bf16acc (e.g. fp8_e4m3_bf16acc): FP8 types accumulate in bf16 as a
//     fast alternative to fp32acc.
//
//   The Rust dispatch layer selects the appropriate suffix at runtime based
//   on the requested precision level.
// ============================================================================

extern "C" {

// --- F32 (native accumulation) ---
INSTANTIATE_GLOBAL_REDUCE(float, float, f32)
INSTANTIATE_DIM_REDUCE_CORE(float, float, f32)
INSTANTIATE_DIM_REDUCE_CORE(float, double, f32_fp64acc)
INSTANTIATE_ARG_REDUCE(float, float, f32)
INSTANTIATE_PROD_DIM(float, float, f32)
INSTANTIATE_PROD_DIM(float, double, f32_fp64acc)
INSTANTIATE_LOGICAL_DIM(float, float, f32)

// --- F64 (native accumulation only) ---
INSTANTIATE_GLOBAL_REDUCE(double, double, f64)
INSTANTIATE_DIM_REDUCE_CORE(double, double, f64)
INSTANTIATE_ARG_REDUCE(double, double, f64)
INSTANTIATE_PROD_DIM(double, double, f64)
INSTANTIATE_LOGICAL_DIM(double, double, f64)

// --- F16 (native, fp32acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(__half, __half, f16)
INSTANTIATE_GLOBAL_REDUCE(__half, float, f16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__half, __half, f16)
INSTANTIATE_DIM_REDUCE_CORE(__half, float, f16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__half, double, f16_fp64acc)
INSTANTIATE_ARG_REDUCE(__half, float, f16)
INSTANTIATE_PROD_DIM(__half, __half, f16)
INSTANTIATE_PROD_DIM(__half, float, f16_fp32acc)
INSTANTIATE_PROD_DIM(__half, double, f16_fp64acc)
INSTANTIATE_LOGICAL_DIM(__half, float, f16)

// --- BF16 (native, fp32acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_GLOBAL_REDUCE(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_DIM_REDUCE_CORE(__nv_bfloat16, double, bf16_fp64acc)
INSTANTIATE_ARG_REDUCE(__nv_bfloat16, float, bf16)
INSTANTIATE_PROD_DIM(__nv_bfloat16, __nv_bfloat16, bf16)
INSTANTIATE_PROD_DIM(__nv_bfloat16, float, bf16_fp32acc)
INSTANTIATE_PROD_DIM(__nv_bfloat16, double, bf16_fp64acc)
INSTANTIATE_LOGICAL_DIM(__nv_bfloat16, float, bf16)

// --- FP8 E4M3 (fp32 default, bf16acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e4m3, double, fp8_e4m3_fp64acc)
INSTANTIATE_ARG_REDUCE(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, float, fp8_e4m3)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, __nv_bfloat16, fp8_e4m3_bf16acc)
INSTANTIATE_PROD_DIM(numr_fp8_e4m3, double, fp8_e4m3_fp64acc)
INSTANTIATE_LOGICAL_DIM(numr_fp8_e4m3, float, fp8_e4m3)

// --- FP8 E5M2 (fp32 default, bf16acc, fp64acc) ---
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_GLOBAL_REDUCE(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_DIM_REDUCE_CORE(numr_fp8_e5m2, double, fp8_e5m2_fp64acc)
INSTANTIATE_ARG_REDUCE(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, float, fp8_e5m2)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, __nv_bfloat16, fp8_e5m2_bf16acc)
INSTANTIATE_PROD_DIM(numr_fp8_e5m2, double, fp8_e5m2_fp64acc)
INSTANTIATE_LOGICAL_DIM(numr_fp8_e5m2, float, fp8_e5m2)

} // extern "C"
