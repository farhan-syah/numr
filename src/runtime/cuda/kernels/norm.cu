// Normalization CUDA kernels
// Supports: rms_norm, layer_norm, group_norm
// Types: f32, f64, f16, bf16
// Note: All half-precision variants use FP32 accumulation for numerical stability
//
// LayerNorm uses single-pass Welford algorithm for numerically stable mean+variance
// computation with warp-level merge via __shfl_down_sync.
//
// Shared memory requirements:
// - rms_norm:   blockDim.x * sizeof(T)                   (e.g., 256 * 4 = 1024 bytes for f32)
// - layer_norm: 3 * ceil(blockDim.x / 32) * sizeof(T)    (e.g., 3 * 8 * 4 = 96 bytes for f32)
// - group_norm: 2 * blockDim.x * sizeof(T)                (e.g., 2 * 256 * 4 = 2048 bytes for f32)
//
// The kernel launcher MUST allocate at least this much shared memory via the
// launch configuration's third <<< >>> parameter.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Single-pass RMSNorm variants; the launcher picks them by row width.
#include "rms_norm_regs.cuh"

// ============================================================================
// Welford merge helpers
// ============================================================================

// Welford's online algorithm for numerically stable mean+variance.
// Maintains three accumulators per partition:
//   count: number of elements seen
//   mean:  running mean
//   M2:    sum of squared deviations from the running mean
// Merge formula (combining two partitions a, b):
//   delta    = mean_b - mean_a
//   mean_ab  = mean_a + delta * count_b / (count_a + count_b)
//   M2_ab    = M2_a + M2_b + delta^2 * count_a * count_b / (count_a + count_b)
// This is numerically stable even with extreme value ranges.
__device__ __forceinline__ void welford_merge(
    float count_a, float mean_a, float M2_a,
    float count_b, float mean_b, float M2_b,
    float &count_out, float &mean_out, float &M2_out
) {
    float count = count_a + count_b;
    if (count == 0.0f) {
        count_out = 0.0f;
        mean_out = 0.0f;
        M2_out = 0.0f;
        return;
    }
    float delta = mean_b - mean_a;
    mean_out = mean_a + delta * count_b / count;
    M2_out = M2_a + M2_b + delta * delta * count_a * count_b / count;
    count_out = count;
}

__device__ __forceinline__ void welford_merge_f64(
    double count_a, double mean_a, double M2_a,
    double count_b, double mean_b, double M2_b,
    double &count_out, double &mean_out, double &M2_out
) {
    double count = count_a + count_b;
    if (count == 0.0) {
        count_out = 0.0;
        mean_out = 0.0;
        M2_out = 0.0;
        return;
    }
    double delta = mean_b - mean_a;
    mean_out = mean_a + delta * count_b / count;
    M2_out = M2_a + M2_b + delta * delta * count_a * count_b / count;
    count_out = count;
}

// Warp-level Welford reduction: merges accumulators across 32 warp lanes
// using shuffle instructions (__shfl_down_sync) to avoid shared memory.
// After this function, lane 0 holds the merged result for the entire warp.
__device__ __forceinline__ void welford_warp_reduce(
    float &count, float &mean, float &M2
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        float o_count = __shfl_down_sync(0xffffffff, count, offset);
        float o_mean  = __shfl_down_sync(0xffffffff, mean, offset);
        float o_M2    = __shfl_down_sync(0xffffffff, M2, offset);
        welford_merge(count, mean, M2, o_count, o_mean, o_M2, count, mean, M2);
    }
}

__device__ __forceinline__ void welford_warp_reduce_f64(
    double &count, double &mean, double &M2
) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        double o_count = __shfl_down_sync(0xffffffff, count, offset);
        double o_mean  = __shfl_down_sync(0xffffffff, mean, offset);
        double o_M2    = __shfl_down_sync(0xffffffff, M2, offset);
        welford_merge_f64(count, mean, M2, o_count, o_mean, o_M2, count, mean, M2);
    }
}

// Tree-sums `shared[0 .. blockDim.x)` and leaves the total in `shared[0]`,
// which every thread may read once the call returns.
//
// The first stride is the power of two at or above blockDim.x, so a block whose
// size is not a power of two still folds in the entries a plain `blockDim.x / 2`
// start would step over.
template <typename T>
__device__ __forceinline__ T block_sum_reduce(T* shared) {
    unsigned int n = blockDim.x;
    unsigned int s = 1;
    while (s < n) s <<= 1;
    for (s >>= 1; s > 0; s >>= 1) {
        if (threadIdx.x < s && threadIdx.x + s < n) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }
    return shared[0];
}

extern "C" {

// ============================================================================
// F32 Normalization Operations
// ============================================================================

// RMSNorm: x * rsqrt(mean(x^2) + eps) * weight
// Each block handles one row (hidden_size elements)
__global__ void rms_norm_f32(
    const float* input, const float* weight, float* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Compute sum of squares
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = row_in[i];
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Compute rsqrt(mean + eps)
    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: Normalize and apply weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

// LayerNorm: (x - mean) / sqrt(var + eps) * weight + bias
// Single-pass Welford algorithm with warp-level merge for numerical stability
__global__ void layer_norm_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const float* row_in = input + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so phase 2 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and the CPU kernels shift by the same element.
    float ref = row_in[0];

    // Phase 1: Single-pass Welford accumulation
    float count = 0.0f, mean = 0.0f, M2 = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = row_in[i] - ref;
        count += 1.0f;
        float delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    // Warp-level Welford merge
    welford_warp_reduce(count, mean, M2);

    // Block-level merge via shared memory (one entry per warp)
    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ float shared[];
    // Layout: [count0..countN, mean0..meanN, M2_0..M2_N] where N = num_warps
    float* s_count = shared;
    float* s_mean  = shared + num_warps;
    float* s_M2    = shared + 2 * num_warps;

    if (lane_id == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id]  = mean;
        s_M2[warp_id]    = M2;
    }
    __syncthreads();

    // Final reduction in first warp
    if (warp_id == 0) {
        float r_count = (lane_id < num_warps) ? s_count[lane_id] : 0.0f;
        float r_mean  = (lane_id < num_warps) ? s_mean[lane_id]  : 0.0f;
        float r_M2    = (lane_id < num_warps) ? s_M2[lane_id]    : 0.0f;

        welford_warp_reduce(r_count, r_mean, r_M2);

        if (lane_id == 0) {
            s_mean[0] = r_mean;
            s_M2[0]   = r_M2;
            s_count[0] = r_count;
        }
    }
    __syncthreads();

    float shifted_mean = s_mean[0];
    float inv_std = rsqrtf(s_M2[0] / s_count[0] + eps);

    // Phase 2: Normalize and apply affine transform
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((row_in[i] - ref) - shifted_mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F64 Normalization Operations
// ============================================================================

__global__ void rms_norm_f64(
    const double* input, const double* weight, double* output,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];

    const double* row_in = input + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double val = row_in[i];
        thread_sum += val * val;
    }
    shared_f64[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_f64[threadIdx.x] += shared_f64[threadIdx.x + s];
        }
        __syncthreads();
    }

    double rms_inv = rsqrt(shared_f64[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_in[i] * rms_inv * weight[i];
    }
}

__global__ void layer_norm_f64(
    const double* input, const double* weight, const double* bias, double* output,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const double* row_in = input + row * hidden_size;
    double* row_out = output + row * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so phase 2 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and the CPU kernels shift by the same element.
    double ref = row_in[0];

    // Single-pass Welford
    double count = 0.0, mean = 0.0, M2 = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double x = row_in[i] - ref;
        count += 1.0;
        double delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    // Warp-level merge
    welford_warp_reduce_f64(count, mean, M2);

    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ double shared_f64[];
    double* s_count = shared_f64;
    double* s_mean  = shared_f64 + num_warps;
    double* s_M2    = shared_f64 + 2 * num_warps;

    if (lane_id == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id]  = mean;
        s_M2[warp_id]    = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        double r_count = (lane_id < num_warps) ? s_count[lane_id] : 0.0;
        double r_mean  = (lane_id < num_warps) ? s_mean[lane_id]  : 0.0;
        double r_M2    = (lane_id < num_warps) ? s_M2[lane_id]    : 0.0;

        welford_warp_reduce_f64(r_count, r_mean, r_M2);

        if (lane_id == 0) {
            s_mean[0] = r_mean;
            s_M2[0]   = r_M2;
            s_count[0] = r_count;
        }
    }
    __syncthreads();

    double shifted_mean = s_mean[0];
    double inv_std = rsqrt(s_M2[0] / s_count[0] + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double normalized = ((row_in[i] - ref) - shifted_mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F16 Normalization Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void rms_norm_f16(
    const __half* input, const __half* weight, __half* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __half* row_in = input + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    // Accumulate in FP32 for precision
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __half2float(row_in[i]);
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float result = __half2float(row_in[i]) * rms_inv * __half2float(weight[i]);
        row_out[i] = __float2half(result);
    }
}

__global__ void layer_norm_f16(
    const __half* input, const __half* weight, const __half* bias, __half* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const __half* row_in = input + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so phase 2 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and the CPU kernels shift by the same element.
    float ref = __half2float(row_in[0]);

    // Single-pass Welford with FP32 accumulation
    float count = 0.0f, mean = 0.0f, M2 = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __half2float(row_in[i]) - ref;
        count += 1.0f;
        float delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    welford_warp_reduce(count, mean, M2);

    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ float shared[];
    float* s_count = shared;
    float* s_mean  = shared + num_warps;
    float* s_M2    = shared + 2 * num_warps;

    if (lane_id == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id]  = mean;
        s_M2[warp_id]    = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float r_count = (lane_id < num_warps) ? s_count[lane_id] : 0.0f;
        float r_mean  = (lane_id < num_warps) ? s_mean[lane_id]  : 0.0f;
        float r_M2    = (lane_id < num_warps) ? s_M2[lane_id]    : 0.0f;

        welford_warp_reduce(r_count, r_mean, r_M2);

        if (lane_id == 0) {
            s_mean[0] = r_mean;
            s_M2[0]   = r_M2;
            s_count[0] = r_count;
        }
    }
    __syncthreads();

    float shifted_mean = s_mean[0];
    float inv_std = rsqrtf(s_M2[0] / s_count[0] + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((__half2float(row_in[i]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __half2float(weight[i]) + __half2float(bias[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Normalization Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __nv_bfloat16* row_in = input + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    // Accumulate in FP32 for precision
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(row_in[i]);
        thread_sum += val * val;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] += shared[threadIdx.x + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float result = __bfloat162float(row_in[i]) * rms_inv * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

__global__ void layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    const __nv_bfloat16* row_in = input + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    // Shift every element by the row's first value before accumulating.
    // `x - mean` cancels catastrophically when a row sits far from zero relative
    // to its own spread, and that cancellation, not the reduction, is what costs
    // the mantissa. Welford does NOT help: its mean is still a value of the
    // row's own magnitude, so phase 2 cancels just the same. Mean, variance and
    // the normalized value are all invariant under the shift in exact
    // arithmetic, and the CPU kernels shift by the same element.
    float ref = __bfloat162float(row_in[0]);

    // Single-pass Welford with FP32 accumulation
    float count = 0.0f, mean = 0.0f, M2 = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float x = __bfloat162float(row_in[i]) - ref;
        count += 1.0f;
        float delta = x - mean;
        mean += delta / count;
        M2 += delta * (x - mean);
    }

    welford_warp_reduce(count, mean, M2);

    unsigned int warp_id = threadIdx.x / 32;
    unsigned int lane_id = threadIdx.x % 32;
    unsigned int num_warps = (blockDim.x + 31) / 32;

    extern __shared__ float shared[];
    float* s_count = shared;
    float* s_mean  = shared + num_warps;
    float* s_M2    = shared + 2 * num_warps;

    if (lane_id == 0) {
        s_count[warp_id] = count;
        s_mean[warp_id]  = mean;
        s_M2[warp_id]    = M2;
    }
    __syncthreads();

    if (warp_id == 0) {
        float r_count = (lane_id < num_warps) ? s_count[lane_id] : 0.0f;
        float r_mean  = (lane_id < num_warps) ? s_mean[lane_id]  : 0.0f;
        float r_M2    = (lane_id < num_warps) ? s_M2[lane_id]    : 0.0f;

        welford_warp_reduce(r_count, r_mean, r_M2);

        if (lane_id == 0) {
            s_mean[0] = r_mean;
            s_M2[0]   = r_M2;
            s_count[0] = r_count;
        }
    }
    __syncthreads();

    float shifted_mean = s_mean[0];
    float inv_std = rsqrtf(s_M2[0] / s_count[0] + eps);

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((__bfloat162float(row_in[i]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __bfloat162float(weight[i]) + __bfloat162float(bias[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

// ============================================================================
// F32 GroupNorm Operations
// ============================================================================

// GroupNorm: Divides channels into num_groups, normalizes each group separately
// Each block handles one (batch, group) pair
// Input shape: [batch, channels, spatial...]
__global__ void group_norm_f32(
    const float* input, const float* weight, const float* bias, float* output,
    unsigned int batch, unsigned int channels, unsigned int spatial,
    unsigned int num_groups, unsigned int channels_per_group, float eps
) {
    unsigned int b = blockIdx.x / num_groups;
    unsigned int g = blockIdx.x % num_groups;

    if (b >= batch || g >= num_groups) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    unsigned int group_size = channels_per_group * spatial;
    unsigned int c_start = g * channels_per_group;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    float ref = input[(b * channels + c_start) * spatial];

    // Phase 1: Mean of the shifted values
    float thread_sum = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        thread_sum += input[offset] - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    float shifted_mean = block_sum_reduce(mean_shared) / group_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float diff = (input[offset] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    float inv_std = rsqrtf(block_sum_reduce(var_shared) / group_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine transform
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float normalized = ((input[offset] - ref) - shifted_mean) * inv_std;
        output[offset] = normalized * weight[c] + bias[c];
    }
}

// ============================================================================
// F64 GroupNorm Operations
// ============================================================================

__global__ void group_norm_f64(
    const double* input, const double* weight, const double* bias, double* output,
    unsigned int batch, unsigned int channels, unsigned int spatial,
    unsigned int num_groups, unsigned int channels_per_group, double eps
) {
    unsigned int b = blockIdx.x / num_groups;
    unsigned int g = blockIdx.x % num_groups;

    if (b >= batch || g >= num_groups) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;

    unsigned int group_size = channels_per_group * spatial;
    unsigned int c_start = g * channels_per_group;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    double ref = input[(b * channels + c_start) * spatial];

    // Phase 1: Mean of the shifted values
    double thread_sum = 0.0;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        thread_sum += input[offset] - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    double shifted_mean = block_sum_reduce(mean_shared) / group_size;
    __syncthreads();

    // Phase 2: Compute variance
    double thread_var = 0.0;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        double diff = (input[offset] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    double inv_std = rsqrt(block_sum_reduce(var_shared) / group_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine transform
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        double normalized = ((input[offset] - ref) - shifted_mean) * inv_std;
        output[offset] = normalized * weight[c] + bias[c];
    }
}

// ============================================================================
// F16 GroupNorm Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void group_norm_f16(
    const __half* input, const __half* weight, const __half* bias, __half* output,
    unsigned int batch, unsigned int channels, unsigned int spatial,
    unsigned int num_groups, unsigned int channels_per_group, float eps
) {
    unsigned int b = blockIdx.x / num_groups;
    unsigned int g = blockIdx.x % num_groups;

    if (b >= batch || g >= num_groups) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    unsigned int group_size = channels_per_group * spatial;
    unsigned int c_start = g * channels_per_group;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    float ref = __half2float(input[(b * channels + c_start) * spatial]);

    // Phase 1: Mean of the shifted values (FP32 accumulation)
    float thread_sum = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        thread_sum += __half2float(input[offset]) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    float shifted_mean = block_sum_reduce(mean_shared) / group_size;
    __syncthreads();

    // Phase 2: Compute variance (FP32 accumulation)
    float thread_var = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float diff = (__half2float(input[offset]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    float inv_std = rsqrtf(block_sum_reduce(var_shared) / group_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine transform
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float normalized = ((__half2float(input[offset]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __half2float(weight[c]) + __half2float(bias[c]);
        output[offset] = __float2half(result);
    }
}

// ============================================================================
// BF16 GroupNorm Operations
// Note: Uses FP32 accumulation for numerical stability
// ============================================================================

__global__ void group_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int batch, unsigned int channels, unsigned int spatial,
    unsigned int num_groups, unsigned int channels_per_group, float eps
) {
    unsigned int b = blockIdx.x / num_groups;
    unsigned int g = blockIdx.x % num_groups;

    if (b >= batch || g >= num_groups) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    unsigned int group_size = channels_per_group * spatial;
    unsigned int c_start = g * channels_per_group;

    // Shift every element by the group's first value before accumulating.
    // `x - mean` cancels catastrophically when a group sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    float ref = __bfloat162float(input[(b * channels + c_start) * spatial]);

    // Phase 1: Mean of the shifted values (FP32 accumulation)
    float thread_sum = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        thread_sum += __bfloat162float(input[offset]) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    float shifted_mean = block_sum_reduce(mean_shared) / group_size;
    __syncthreads();

    // Phase 2: Compute variance (FP32 accumulation)
    float thread_var = 0.0f;
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float diff = (__bfloat162float(input[offset]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    float inv_std = rsqrtf(block_sum_reduce(var_shared) / group_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine transform
    for (unsigned int idx = threadIdx.x; idx < group_size; idx += blockDim.x) {
        unsigned int c = c_start + (idx / spatial);
        unsigned int s = idx % spatial;
        unsigned int offset = (b * channels + c) * spatial + s;
        float normalized = ((__bfloat162float(input[offset]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __bfloat162float(weight[c]) + __bfloat162float(bias[c]);
        output[offset] = __float2bfloat16(result);
    }
}

} // extern "C"
