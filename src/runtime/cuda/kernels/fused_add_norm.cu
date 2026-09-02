// Fused Add + Normalization CUDA kernels
// Supports: fused_add_rms_norm, fused_add_layer_norm (forward + backward)
// Types: f32, f64, f16, bf16
// Note: All half-precision variants use FP32 accumulation for numerical stability

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

// Block-wide sum reduction shared with norm.cu.
#include "block_reduce.cuh"

extern "C" {

// ============================================================================
// Helper: atomicAdd for half-precision types via atomicCAS
// ============================================================================

__device__ void atomicAddHalf(__half* address, float val) {
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_us, assumed,
            __half_as_ushort(__float2half(__half2float(__ushort_as_half(assumed)) + val)));
    } while (assumed != old);
}

__device__ void atomicAddBf16(__nv_bfloat16* address, float val) {
    // Use atomicCAS with bit manipulation for BF16
    unsigned short int* address_as_us = (unsigned short int*)address;
    unsigned short int old = *address_as_us, assumed;
    do {
        assumed = old;
        // Extract as uint16, convert to bfloat16, then float, add, convert back
        __nv_bfloat16 old_val;
        unsigned short int* old_val_ptr = (unsigned short int*)&old_val;
        *old_val_ptr = assumed;
        float new_float = __bfloat162float(old_val) + val;
        __nv_bfloat16 new_val = __float2bfloat16(new_float);
        unsigned short int* new_val_ptr = (unsigned short int*)&new_val;
        old = atomicCAS(address_as_us, assumed, *new_val_ptr);
    } while (assumed != old);
}

// ============================================================================
// F32 Fused Add + RMSNorm Forward
// ============================================================================

__global__ void fused_add_rms_norm_f32(
    const float* input, const float* residual, const float* weight,
    float* output, float* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const float* row_in = input + row * hidden_size;
    const float* row_res = residual + row * hidden_size;
    float* row_pn = pre_norm + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Add residual + compute sum of squares
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce
    block_sum_reduce(shared);

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 2: Normalize and apply weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_pn[i] * rms_inv * weight[i];
    }
}

// ============================================================================
// F64 Fused Add + RMSNorm Forward
// ============================================================================

__global__ void fused_add_rms_norm_f64(
    const double* input, const double* residual, const double* weight,
    double* output, double* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];

    const double* row_in = input + row * hidden_size;
    const double* row_res = residual + row * hidden_size;
    double* row_pn = pre_norm + row * hidden_size;
    double* row_out = output + row * hidden_size;

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn * pn;
    }
    shared_f64[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(shared_f64);

    double rms_inv = rsqrt(shared_f64[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        row_out[i] = row_pn[i] * rms_inv * weight[i];
    }
}

// ============================================================================
// F16 Fused Add + RMSNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_f16(
    const __half* input, const __half* residual, const __half* weight,
    __half* output, __half* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __half* row_in = input + row * hidden_size;
    const __half* row_res = residual + row * hidden_size;
    __half* row_pn = pre_norm + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_in[i]) + __half2float(row_res[i]);
        row_pn[i] = __float2half(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(shared);

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_pn[i]);
        float result = pn * rms_inv * __half2float(weight[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Fused Add + RMSNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual, const __nv_bfloat16* weight,
    __nv_bfloat16* output, __nv_bfloat16* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    const __nv_bfloat16* row_in = input + row * hidden_size;
    const __nv_bfloat16* row_res = residual + row * hidden_size;
    __nv_bfloat16* row_pn = pre_norm + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_in[i]) + __bfloat162float(row_res[i]);
        row_pn[i] = __float2bfloat16(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(shared);

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_pn[i]);
        float result = pn * rms_inv * __bfloat162float(weight[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

// ============================================================================
// F32 Fused Add + RMSNorm Backward
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f32(
    const float* grad, const float* pre_norm, const float* weight,
    float* d_input_residual, float* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    // Phase 1: Compute sum_sq and dot = sum(grad * weight * pre_norm)
    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = pre_norm[row * hidden_size + i];
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    // Phase 2: Compute d_input_residual and atomicAdd d_weight
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float pn = pre_norm[row * hidden_size + i];
        d_input_residual[row * hidden_size + i] = (g * w - pn * coeff) * inv_rms;
        atomicAdd(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F64 Fused Add + RMSNorm Backward
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f64(
    const double* grad, const double* pre_norm, const double* weight,
    double* d_input_residual, double* d_weight,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* sum_sq_shared = shared_f64;
    double* dot_shared = shared_f64 + blockDim.x;

    double thread_sq = 0.0, thread_dot = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = pre_norm[row * hidden_size + i];
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    double mean_sq = sum_sq_shared[0] / hidden_size;
    double inv_rms = rsqrt(mean_sq + eps);
    double dot = dot_shared[0];
    double coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double pn = pre_norm[row * hidden_size + i];
        d_input_residual[row * hidden_size + i] = (g * w - pn * coeff) * inv_rms;
        atomicAdd(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F16 Fused Add + RMSNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bwd_f16(
    const __half* grad, const __half* pre_norm, const __half* weight,
    __half* d_input_residual, __half* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(pre_norm[row * hidden_size + i]);
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float pn = __half2float(pre_norm[row * hidden_size + i]);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i] = __float2half(dir);
        atomicAddHalf(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// BF16 Fused Add + RMSNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_rms_norm_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* pre_norm, const __nv_bfloat16* weight,
    __nv_bfloat16* d_input_residual, __nv_bfloat16* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(pre_norm[row * hidden_size + i]);
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float pn = __bfloat162float(pre_norm[row * hidden_size + i]);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i] = __float2bfloat16(dir);
        atomicAddBf16(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// F32 Fused Add + LayerNorm Forward
// ============================================================================

__global__ void fused_add_layer_norm_f32(
    const float* input, const float* residual, const float* weight, const float* bias,
    float* output, float* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const float* row_in = input + row * hidden_size;
    const float* row_res = residual + row * hidden_size;
    float* row_pn = pre_norm + row * hidden_size;
    float* row_out = output + row * hidden_size;

    // Phase 1: Add residual + compute mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    float ref = row_in[0] + row_res[0];

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (row_pn[i] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 3: Normalize and apply affine
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((row_pn[i] - ref) - shifted_mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F64 Fused Add + LayerNorm Forward
// ============================================================================

__global__ void fused_add_layer_norm_f64(
    const double* input, const double* residual, const double* weight, const double* bias,
    double* output, double* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;

    const double* row_in = input + row * hidden_size;
    const double* row_res = residual + row * hidden_size;
    double* row_pn = pre_norm + row * hidden_size;
    double* row_out = output + row * hidden_size;

    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    double ref = row_in[0] + row_res[0];

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double pn = row_in[i] + row_res[i];
        row_pn[i] = pn;
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    double shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    double thread_var = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double diff = (row_pn[i] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    double inv_std = rsqrt(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double normalized = ((row_pn[i] - ref) - shifted_mean) * inv_std;
        row_out[i] = normalized * weight[i] + bias[i];
    }
}

// ============================================================================
// F16 Fused Add + LayerNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_f16(
    const __half* input, const __half* residual, const __half* weight, const __half* bias,
    __half* output, __half* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __half* row_in = input + row * hidden_size;
    const __half* row_res = residual + row * hidden_size;
    __half* row_pn = pre_norm + row * hidden_size;
    __half* row_out = output + row * hidden_size;

    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    float ref = __half2float(row_in[0]) + __half2float(row_res[0]);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __half2float(row_in[i]) + __half2float(row_res[i]);
        row_pn[i] = __float2half(pn);
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (__half2float(row_pn[i]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((__half2float(row_pn[i]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __half2float(weight[i]) + __half2float(bias[i]);
        row_out[i] = __float2half(result);
    }
}

// ============================================================================
// BF16 Fused Add + LayerNorm Forward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* output, __nv_bfloat16* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    const __nv_bfloat16* row_in = input + row * hidden_size;
    const __nv_bfloat16* row_res = residual + row * hidden_size;
    __nv_bfloat16* row_pn = pre_norm + row * hidden_size;
    __nv_bfloat16* row_out = output + row * hidden_size;

    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    float ref = __bfloat162float(row_in[0]) + __bfloat162float(row_res[0]);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = __bfloat162float(row_in[i]) + __bfloat162float(row_res[i]);
        row_pn[i] = __float2bfloat16(pn);
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (__bfloat162float(row_pn[i]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float normalized = ((__bfloat162float(row_pn[i]) - ref) - shifted_mean) * inv_std;
        float result = normalized * __bfloat162float(weight[i]) + __bfloat162float(bias[i]);
        row_out[i] = __float2bfloat16(result);
    }
}

// ============================================================================
// F32 Fused Add + LayerNorm Backward
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f32(
    const float* grad, const float* pre_norm, const float* weight,
    float* d_input_residual, float* d_weight, float* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Phase 1: Compute mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    float ref = pre_norm[row * hidden_size];

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += pre_norm[row * hidden_size + i] - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (pre_norm[row * hidden_size + i] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    // Phase 3: mean_gs and mean_gsn
    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float normalized = ((pre_norm[row * hidden_size + i] - ref) - shifted_mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    // Phase 4: Compute gradients
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = grad[row * hidden_size + i];
        float w = weight[i];
        float normalized = ((pre_norm[row * hidden_size + i] - ref) - shifted_mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = d_ir;
        atomicAdd(&d_weight[i], g * normalized);
        atomicAdd(&d_bias[i], g);
    }
}

// ============================================================================
// F64 Fused Add + LayerNorm Backward
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f64(
    const double* grad, const double* pre_norm, const double* weight,
    double* d_input_residual, double* d_weight, double* d_bias,
    unsigned int batch_size, unsigned int hidden_size, double eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ double shared_f64[];
    double* mean_shared = shared_f64;
    double* var_shared = shared_f64 + blockDim.x;
    double* gs_shared = shared_f64 + 2 * blockDim.x;
    double* gsn_shared = shared_f64 + 3 * blockDim.x;

    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    double ref = pre_norm[row * hidden_size];

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += pre_norm[row * hidden_size + i] - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    double shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    double thread_var = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double diff = (pre_norm[row * hidden_size + i] - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    double var = var_shared[0] / hidden_size;
    double inv_std = rsqrt(var + eps);
    __syncthreads();

    double thread_gs = 0.0, thread_gsn = 0.0;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double normalized = ((pre_norm[row * hidden_size + i] - ref) - shifted_mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);
    double mean_gs = gs_shared[0] / hidden_size;
    double mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        double g = grad[row * hidden_size + i];
        double w = weight[i];
        double normalized = ((pre_norm[row * hidden_size + i] - ref) - shifted_mean) * inv_std;
        double d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = d_ir;
        atomicAdd(&d_weight[i], g * normalized);
        atomicAdd(&d_bias[i], g);
    }
}

// ============================================================================
// F16 Fused Add + LayerNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bwd_f16(
    const __half* grad, const __half* pre_norm, const __half* weight,
    __half* d_input_residual, __half* d_weight, __half* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    float ref = __half2float(pre_norm[row * hidden_size]);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __half2float(pre_norm[row * hidden_size + i]) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (__half2float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float normalized = ((__half2float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __half2float(grad[row * hidden_size + i]);
        float w = __half2float(weight[i]);
        float normalized = ((__half2float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = __float2half(d_ir);
        atomicAddHalf(&d_weight[i], g * normalized);
        atomicAddHalf(&d_bias[i], g);
    }
}

// ============================================================================
// BF16 Fused Add + LayerNorm Backward (FP32 accumulation)
// ============================================================================

__global__ void fused_add_layer_norm_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* pre_norm, const __nv_bfloat16* weight,
    __nv_bfloat16* d_input_residual, __nv_bfloat16* d_weight, __nv_bfloat16* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    float ref = __bfloat162float(pre_norm[row * hidden_size]);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += __bfloat162float(pre_norm[row * hidden_size + i]) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float diff = (__bfloat162float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float var = var_shared[0] / hidden_size;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();

    float thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float normalized = ((__bfloat162float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean) * inv_std;
        thread_gs += g * w;
        thread_gsn += g * w * normalized;
    }
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = __bfloat162float(grad[row * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        float normalized = ((__bfloat162float(pre_norm[row * hidden_size + i]) - ref) - shifted_mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i] = __float2bfloat16(d_ir);
        atomicAddBf16(&d_weight[i], g * normalized);
        atomicAddBf16(&d_bias[i], g);
    }
}

// ============================================================================
// Helper: atomicAdd for FP8 types via 32-bit atomicCAS
// ============================================================================

__device__ void atomicAddFp8E4M3(numr_fp8_e4m3* address, float val) {
    // FP8 is 1 byte — use 32-bit atomicCAS on the containing 4-byte word
    unsigned int* base = (unsigned int*)((size_t)address & ~3ULL);
    unsigned int byte_offset = (unsigned int)((size_t)address & 3);
    unsigned int shift = byte_offset * 8;
    unsigned int old_word = *base, assumed;
    do {
        assumed = old_word;
        uint8_t old_byte = (uint8_t)((assumed >> shift) & 0xFF);
        float old_float = fp8_e4m3_to_f32(old_byte);
        uint8_t new_byte = f32_to_fp8_e4m3(old_float + val);
        unsigned int new_word = (assumed & ~(0xFFu << shift)) | ((unsigned int)new_byte << shift);
        old_word = atomicCAS(base, assumed, new_word);
    } while (assumed != old_word);
}

__device__ void atomicAddFp8E5M2(numr_fp8_e5m2* address, float val) {
    unsigned int* base = (unsigned int*)((size_t)address & ~3ULL);
    unsigned int byte_offset = (unsigned int)((size_t)address & 3);
    unsigned int shift = byte_offset * 8;
    unsigned int old_word = *base, assumed;
    do {
        assumed = old_word;
        uint8_t old_byte = (uint8_t)((assumed >> shift) & 0xFF);
        float old_float = fp8_e5m2_to_f32(old_byte);
        uint8_t new_byte = f32_to_fp8_e5m2(old_float + val);
        unsigned int new_word = (assumed & ~(0xFFu << shift)) | ((unsigned int)new_byte << shift);
        old_word = atomicCAS(base, assumed, new_word);
    } while (assumed != old_word);
}

// ============================================================================
// FP8 E4M3 Fused Add + RMSNorm Forward
// ============================================================================

__global__ void fused_add_rms_norm_fp8_e4m3(
    const numr_fp8_e4m3* input, const numr_fp8_e4m3* residual, const numr_fp8_e4m3* weight,
    numr_fp8_e4m3* output, numr_fp8_e4m3* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(input[row * hidden_size + i].data)
                  + fp8_e4m3_to_f32(residual[row * hidden_size + i].data);
        pre_norm[row * hidden_size + i].data = f32_to_fp8_e4m3(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(shared);

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        output[row * hidden_size + i].data = f32_to_fp8_e4m3(pn * rms_inv * w);
    }
}

__global__ void fused_add_rms_norm_fp8_e5m2(
    const numr_fp8_e5m2* input, const numr_fp8_e5m2* residual, const numr_fp8_e5m2* weight,
    numr_fp8_e5m2* output, numr_fp8_e5m2* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(input[row * hidden_size + i].data)
                  + fp8_e5m2_to_f32(residual[row * hidden_size + i].data);
        pre_norm[row * hidden_size + i].data = f32_to_fp8_e5m2(pn);
        thread_sum += pn * pn;
    }
    shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(shared);

    float rms_inv = rsqrtf(shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        output[row * hidden_size + i].data = f32_to_fp8_e5m2(pn * rms_inv * w);
    }
}

// ============================================================================
// FP8 E4M3 Fused Add + RMSNorm Backward
// ============================================================================

__global__ void fused_add_rms_norm_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* pre_norm, const numr_fp8_e4m3* weight,
    numr_fp8_e4m3* d_input_residual, numr_fp8_e4m3* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float g = fp8_e4m3_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = fp8_e4m3_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i].data = f32_to_fp8_e4m3(dir);
        atomicAddFp8E4M3(&d_weight[i], g * pn * inv_rms);
    }
}

__global__ void fused_add_rms_norm_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* pre_norm, const numr_fp8_e5m2* weight,
    numr_fp8_e5m2* d_input_residual, numr_fp8_e5m2* d_weight,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* sum_sq_shared = shared;
    float* dot_shared = shared + blockDim.x;

    float thread_sq = 0.0f, thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float g = fp8_e5m2_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        thread_sq += pn * pn;
        thread_dot += g * w * pn;
    }
    sum_sq_shared[threadIdx.x] = thread_sq;
    dot_shared[threadIdx.x] = thread_dot;
    __syncthreads();

    block_sum_reduce(sum_sq_shared);
    block_sum_reduce(dot_shared);

    float mean_sq = sum_sq_shared[0] / hidden_size;
    float inv_rms = rsqrtf(mean_sq + eps);
    float dot = dot_shared[0];
    float coeff = dot * inv_rms / (hidden_size * (mean_sq + eps));
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = fp8_e5m2_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float dir = (g * w - pn * coeff) * inv_rms;
        d_input_residual[row * hidden_size + i].data = f32_to_fp8_e5m2(dir);
        atomicAddFp8E5M2(&d_weight[i], g * pn * inv_rms);
    }
}

// ============================================================================
// FP8 E4M3 Fused Add + LayerNorm Forward
// ============================================================================

__global__ void fused_add_layer_norm_fp8_e4m3(
    const numr_fp8_e4m3* input, const numr_fp8_e4m3* residual,
    const numr_fp8_e4m3* weight, const numr_fp8_e4m3* bias,
    numr_fp8_e4m3* output, numr_fp8_e4m3* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    // Phase 1: Add residual + compute mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    float ref = fp8_e4m3_to_f32(input[row * hidden_size].data)
              + fp8_e4m3_to_f32(residual[row * hidden_size].data);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(input[row * hidden_size + i].data)
                  + fp8_e4m3_to_f32(residual[row * hidden_size + i].data);
        pre_norm[row * hidden_size + i].data = f32_to_fp8_e4m3(pn);
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance
    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float diff = (pn - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    // Phase 3: Normalize
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        float b = fp8_e4m3_to_f32(bias[i].data);
        float normalized = ((pn - ref) - shifted_mean) * inv_std;
        output[row * hidden_size + i].data = f32_to_fp8_e4m3(normalized * w + b);
    }
}

__global__ void fused_add_layer_norm_fp8_e5m2(
    const numr_fp8_e5m2* input, const numr_fp8_e5m2* residual,
    const numr_fp8_e5m2* weight, const numr_fp8_e5m2* bias,
    numr_fp8_e5m2* output, numr_fp8_e5m2* pre_norm,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;

    // Shift every element by the row's first pre-norm value before accumulating.
    // Subtracting the mean cancels catastrophically when a row sits far from zero
    // relative to its own spread, and that cancellation, not the reduction, is
    // what costs the mantissa. Mean, variance and the normalized value are all
    // invariant under the shift in exact arithmetic.
    // Taken from the inputs, so it needs no barrier against phase 1's own stores.
    float ref = fp8_e5m2_to_f32(input[row * hidden_size].data)
              + fp8_e5m2_to_f32(residual[row * hidden_size].data);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(input[row * hidden_size + i].data)
                  + fp8_e5m2_to_f32(residual[row * hidden_size + i].data);
        pre_norm[row * hidden_size + i].data = f32_to_fp8_e5m2(pn);
        thread_sum += pn - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float diff = (pn - ref) - shifted_mean;
        thread_var += diff * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    __syncthreads();

    block_sum_reduce(var_shared);
    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        float b = fp8_e5m2_to_f32(bias[i].data);
        float normalized = ((pn - ref) - shifted_mean) * inv_std;
        output[row * hidden_size + i].data = f32_to_fp8_e5m2(normalized * w + b);
    }
}

// ============================================================================
// FP8 E4M3 Fused Add + LayerNorm Backward
// ============================================================================

__global__ void fused_add_layer_norm_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* pre_norm,
    const numr_fp8_e4m3* weight,
    numr_fp8_e4m3* d_input_residual, numr_fp8_e4m3* d_weight, numr_fp8_e4m3* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Phase 1: Compute mean
    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    float ref = fp8_e4m3_to_f32(pre_norm[row * hidden_size].data);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    // Phase 2: Compute variance + dot products
    float thread_var = 0.0f, thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data);
        float g = fp8_e4m3_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        float diff = (pn - ref) - shifted_mean;
        thread_var += diff * diff;
        thread_gs += g * w;
        thread_gsn += g * w * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(var_shared);
    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);

    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = fp8_e4m3_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e4m3_to_f32(weight[i].data);
        float normalized = ((fp8_e4m3_to_f32(pre_norm[row * hidden_size + i].data) - ref) - shifted_mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i].data = f32_to_fp8_e4m3(d_ir);
        atomicAddFp8E4M3(&d_weight[i], g * normalized);
        atomicAddFp8E4M3(&d_bias[i], g);
    }
}

__global__ void fused_add_layer_norm_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* pre_norm,
    const numr_fp8_e5m2* weight,
    numr_fp8_e5m2* d_input_residual, numr_fp8_e5m2* d_weight, numr_fp8_e5m2* d_bias,
    unsigned int batch_size, unsigned int hidden_size, float eps
) {
    unsigned int row = blockIdx.x;
    if (row >= batch_size) return;

    extern __shared__ float shared[];
    float* mean_shared = shared;
    float* var_shared = shared + blockDim.x;
    float* gs_shared = shared + 2 * blockDim.x;
    float* gsn_shared = shared + 3 * blockDim.x;

    // Shift every element by the row's first pre-norm value before accumulating.
    // This pass RECOMPUTES the mean and variance from `pre_norm` rather than
    // consuming a saved statistic, so it carries the forward pass's cancellation
    // exactly: subtracting the mean loses the mantissa whenever a row sits far
    // from zero relative to its own spread. Mean, variance and the normalized
    // value are all invariant under the shift in exact arithmetic. `mean_gs` and
    // `mean_gsn` need no shift: they average `g * w` and an already-shifted
    // normalized value.
    float ref = fp8_e5m2_to_f32(pre_norm[row * hidden_size].data);

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        thread_sum += fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data) - ref;
    }
    mean_shared[threadIdx.x] = thread_sum;
    __syncthreads();

    block_sum_reduce(mean_shared);
    float shifted_mean = mean_shared[0] / hidden_size;
    __syncthreads();

    float thread_var = 0.0f, thread_gs = 0.0f, thread_gsn = 0.0f;
    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float pn = fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data);
        float g = fp8_e5m2_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        float diff = (pn - ref) - shifted_mean;
        thread_var += diff * diff;
        thread_gs += g * w;
        thread_gsn += g * w * diff;
    }
    var_shared[threadIdx.x] = thread_var;
    gs_shared[threadIdx.x] = thread_gs;
    gsn_shared[threadIdx.x] = thread_gsn;
    __syncthreads();

    block_sum_reduce(var_shared);
    block_sum_reduce(gs_shared);
    block_sum_reduce(gsn_shared);

    float inv_std = rsqrtf(var_shared[0] / hidden_size + eps);
    float mean_gs = gs_shared[0] / hidden_size;
    float mean_gsn = gsn_shared[0] / hidden_size;
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < hidden_size; i += blockDim.x) {
        float g = fp8_e5m2_to_f32(grad[row * hidden_size + i].data);
        float w = fp8_e5m2_to_f32(weight[i].data);
        float normalized = ((fp8_e5m2_to_f32(pre_norm[row * hidden_size + i].data) - ref) - shifted_mean) * inv_std;
        float d_ir = inv_std * (g * w - mean_gs - normalized * mean_gsn);
        d_input_residual[row * hidden_size + i].data = f32_to_fp8_e5m2(d_ir);
        atomicAddFp8E5M2(&d_weight[i], g * normalized);
        atomicAddFp8E5M2(&d_bias[i], g);
    }
}

} // extern "C"
