// Softmax CUDA kernels (forward + backward)
// Supports: softmax (last-dim), softmax_dim (non-last-dim), softmax_bwd, softmax_bwd_dim
// Types: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "dtype_traits.cuh"

extern "C" {

// ============================================================================
// Softmax Forward (Last Dimension)
// ============================================================================

__global__ void softmax_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const float* row_in = input + outer_idx * dim_size;
    float* row_out = output + outer_idx * dim_size;

    // Phase 1: Find max value for numerical stability
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, row_in[i]);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    // Reduce max across threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: Compute exp(x - max) and sum
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(row_in[i] - row_max);
        row_out[i] = val;  // Temporarily store exp values
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduce sum across threads
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: Normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

__global__ void softmax_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ double shared_f64[];
    double* max_val = shared_f64;
    double* sum_exp = shared_f64 + blockDim.x;

    const double* row_in = input + outer_idx * dim_size;
    double* row_out = output + outer_idx * dim_size;

    double thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmax(thread_max, row_in[i]);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmax(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    double row_max = max_val[0];
    __syncthreads();

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double val = exp(row_in[i] - row_max);
        row_out[i] = val;
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    double row_sum = sum_exp[0];
    __syncthreads();

    double inv_sum = 1.0 / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

__global__ void softmax_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const __half* row_in = input + outer_idx * dim_size;
    __half* row_out = output + outer_idx * dim_size;

    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __half2float(row_in[i]));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__half2float(row_in[i]) - row_max);
        row_out[i] = __float2half(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2half(__half2float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const __nv_bfloat16* row_in = input + outer_idx * dim_size;
    __nv_bfloat16* row_out = output + outer_idx * dim_size;

    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, __bfloat162float(row_in[i]));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__bfloat162float(row_in[i]) - row_max);
        row_out[i] = __float2bfloat16(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2bfloat16(__bfloat162float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const numr_fp8_e4m3* row_in = input + outer_idx * dim_size;
    numr_fp8_e4m3* row_out = output + outer_idx * dim_size;

    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, fp8_e4m3_to_f32(row_in[i].data));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(fp8_e4m3_to_f32(row_in[i].data) - row_max);
        row_out[i] = numr_fp8_e4m3(f32_to_fp8_e4m3(val));
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = numr_fp8_e4m3(f32_to_fp8_e4m3(fp8_e4m3_to_f32(row_out[i].data) * inv_sum));
    }
}

__global__ void softmax_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];
    float* max_val = shared;
    float* sum_exp = shared + blockDim.x;

    const numr_fp8_e5m2* row_in = input + outer_idx * dim_size;
    numr_fp8_e5m2* row_out = output + outer_idx * dim_size;

    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_max = fmaxf(thread_max, fp8_e5m2_to_f32(row_in[i].data));
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(fp8_e5m2_to_f32(row_in[i].data) - row_max);
        row_out[i] = numr_fp8_e5m2(f32_to_fp8_e5m2(val));
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        }
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = numr_fp8_e5m2(f32_to_fp8_e5m2(fp8_e5m2_to_f32(row_out[i].data) * inv_sum));
    }
}

// ============================================================================
// Softmax Forward (Non-Last Dimension)
// For shape [A, B, C] with softmax over dim=1:
// outer_size = A, dim_size = B, inner_size = C
// ============================================================================

__global__ void softmax_dim_f32(
    const float* input, float* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float max_val = input[base];
    float sum = 1.0f;
    for (unsigned int i = 1; i < dim_size; i++) {
        float val = input[base + i * stride];
        if (val > max_val) {
            sum = sum * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum += expf(val - max_val);
        }
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = expf(input[base + i * stride] - max_val) * inv_sum;
    }
}

__global__ void softmax_dim_f64(
    const double* input, double* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    double max_val = input[base];
    double sum = 1.0;
    for (unsigned int i = 1; i < dim_size; i++) {
        double val = input[base + i * stride];
        if (val > max_val) {
            sum = sum * exp(max_val - val) + 1.0;
            max_val = val;
        } else {
            sum += exp(val - max_val);
        }
    }

    double inv_sum = 1.0 / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        output[base + i * stride] = exp(input[base + i * stride] - max_val) * inv_sum;
    }
}

__global__ void softmax_dim_f16(
    const __half* input, __half* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float max_val = __half2float(input[base]);
    float sum = 1.0f;
    for (unsigned int i = 1; i < dim_size; i++) {
        float val = __half2float(input[base + i * stride]);
        if (val > max_val) {
            sum = sum * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum += expf(val - max_val);
        }
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = __half2float(input[base + i * stride]);
        output[base + i * stride] = __float2half(expf(val - max_val) * inv_sum);
    }
}

__global__ void softmax_dim_bf16(
    const __nv_bfloat16* input, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float max_val = __bfloat162float(input[base]);
    float sum = 1.0f;
    for (unsigned int i = 1; i < dim_size; i++) {
        float val = __bfloat162float(input[base + i * stride]);
        if (val > max_val) {
            sum = sum * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum += expf(val - max_val);
        }
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = __bfloat162float(input[base + i * stride]);
        output[base + i * stride] = __float2bfloat16(expf(val - max_val) * inv_sum);
    }
}

__global__ void softmax_dim_fp8_e4m3(
    const numr_fp8_e4m3* input, numr_fp8_e4m3* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float max_val = fp8_e4m3_to_f32(input[base].data);
    float sum = 1.0f;
    for (unsigned int i = 1; i < dim_size; i++) {
        float val = fp8_e4m3_to_f32(input[base + i * stride].data);
        if (val > max_val) {
            sum = sum * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum += expf(val - max_val);
        }
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = fp8_e4m3_to_f32(input[base + i * stride].data);
        output[base + i * stride] = numr_fp8_e4m3(f32_to_fp8_e4m3(expf(val - max_val) * inv_sum));
    }
}

__global__ void softmax_dim_fp8_e5m2(
    const numr_fp8_e5m2* input, numr_fp8_e5m2* output,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float max_val = fp8_e5m2_to_f32(input[base].data);
    float sum = 1.0f;
    for (unsigned int i = 1; i < dim_size; i++) {
        float val = fp8_e5m2_to_f32(input[base + i * stride].data);
        if (val > max_val) {
            sum = sum * expf(max_val - val) + 1.0f;
            max_val = val;
        } else {
            sum += expf(val - max_val);
        }
    }

    float inv_sum = 1.0f / sum;
    for (unsigned int i = 0; i < dim_size; i++) {
        float val = fp8_e5m2_to_f32(input[base + i * stride].data);
        output[base + i * stride] = numr_fp8_e5m2(f32_to_fp8_e5m2(expf(val - max_val) * inv_sum));
    }
}

// ============================================================================
// Softmax Backward (Last Dimension)
// d_input = output * (grad - dot), where dot = sum(grad * output)
// ============================================================================

__global__ void softmax_bwd_f32(
    const float* grad, const float* output, float* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared[];

    const float* g_row = grad + outer_idx * dim_size;
    const float* o_row = output + outer_idx * dim_size;
    float* d_row = d_input + outer_idx * dim_size;

    float thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += g_row[i] * o_row[i];
    }
    shared[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared[threadIdx.x] += shared[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        d_row[i] = o_row[i] * (g_row[i] - dot);
    }
}

__global__ void softmax_bwd_f64(
    const double* grad, const double* output, double* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ double shared_d[];

    const double* g_row = grad + outer_idx * dim_size;
    const double* o_row = output + outer_idx * dim_size;
    double* d_row = d_input + outer_idx * dim_size;

    double thread_dot = 0.0;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += g_row[i] * o_row[i];
    }
    shared_d[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_d[threadIdx.x] += shared_d[threadIdx.x + s];
        __syncthreads();
    }
    double dot = shared_d[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        d_row[i] = o_row[i] * (g_row[i] - dot);
    }
}

__global__ void softmax_bwd_f16(
    const __half* grad, const __half* output, __half* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_f16[];

    const __half* g_row = grad + outer_idx * dim_size;
    const __half* o_row = output + outer_idx * dim_size;
    __half* d_row = d_input + outer_idx * dim_size;

    float thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += __half2float(g_row[i]) * __half2float(o_row[i]);
    }
    shared_f16[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_f16[threadIdx.x] += shared_f16[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared_f16[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float g = __half2float(g_row[i]);
        float o = __half2float(o_row[i]);
        d_row[i] = __float2half(o * (g - dot));
    }
}

__global__ void softmax_bwd_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* output, __nv_bfloat16* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_bf16[];

    const __nv_bfloat16* g_row = grad + outer_idx * dim_size;
    const __nv_bfloat16* o_row = output + outer_idx * dim_size;
    __nv_bfloat16* d_row = d_input + outer_idx * dim_size;

    float thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += __bfloat162float(g_row[i]) * __bfloat162float(o_row[i]);
    }
    shared_bf16[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_bf16[threadIdx.x] += shared_bf16[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared_bf16[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float g = __bfloat162float(g_row[i]);
        float o = __bfloat162float(o_row[i]);
        d_row[i] = __float2bfloat16(o * (g - dot));
    }
}

__global__ void softmax_bwd_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* output, numr_fp8_e4m3* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_fp8[];

    const numr_fp8_e4m3* g_row = grad + outer_idx * dim_size;
    const numr_fp8_e4m3* o_row = output + outer_idx * dim_size;
    numr_fp8_e4m3* d_row = d_input + outer_idx * dim_size;

    float thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += fp8_e4m3_to_f32(g_row[i].data) * fp8_e4m3_to_f32(o_row[i].data);
    }
    shared_fp8[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_fp8[threadIdx.x] += shared_fp8[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared_fp8[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float g = fp8_e4m3_to_f32(g_row[i].data);
        float o = fp8_e4m3_to_f32(o_row[i].data);
        d_row[i] = numr_fp8_e4m3(f32_to_fp8_e4m3(o * (g - dot)));
    }
}

__global__ void softmax_bwd_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* output, numr_fp8_e5m2* d_input,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_fp8e5[];

    const numr_fp8_e5m2* g_row = grad + outer_idx * dim_size;
    const numr_fp8_e5m2* o_row = output + outer_idx * dim_size;
    numr_fp8_e5m2* d_row = d_input + outer_idx * dim_size;

    float thread_dot = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        thread_dot += fp8_e5m2_to_f32(g_row[i].data) * fp8_e5m2_to_f32(o_row[i].data);
    }
    shared_fp8e5[threadIdx.x] = thread_dot;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) shared_fp8e5[threadIdx.x] += shared_fp8e5[threadIdx.x + s];
        __syncthreads();
    }
    float dot = shared_fp8e5[0];
    __syncthreads();

    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float g = fp8_e5m2_to_f32(g_row[i].data);
        float o = fp8_e5m2_to_f32(o_row[i].data);
        d_row[i] = numr_fp8_e5m2(f32_to_fp8_e5m2(o * (g - dot)));
    }
}

// ============================================================================
// Softmax Backward (Non-Last Dimension)
// ============================================================================

__global__ void softmax_bwd_dim_f32(
    const float* grad, const float* output, float* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float dot = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += grad[base + i * stride] * output[base + i * stride];
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = output[idx] * (grad[idx] - dot);
    }
}

__global__ void softmax_bwd_dim_f64(
    const double* grad, const double* output, double* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    double dot = 0.0;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += grad[base + i * stride] * output[base + i * stride];
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = output[idx] * (grad[idx] - dot);
    }
}

__global__ void softmax_bwd_dim_f16(
    const __half* grad, const __half* output, __half* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float dot = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += __half2float(grad[base + i * stride]) * __half2float(output[base + i * stride]);
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = __float2half(__half2float(output[idx]) * (__half2float(grad[idx]) - dot));
    }
}

__global__ void softmax_bwd_dim_bf16(
    const __nv_bfloat16* grad, const __nv_bfloat16* output, __nv_bfloat16* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float dot = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += __bfloat162float(grad[base + i * stride]) * __bfloat162float(output[base + i * stride]);
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = __float2bfloat16(__bfloat162float(output[idx]) * (__bfloat162float(grad[idx]) - dot));
    }
}

__global__ void softmax_bwd_dim_fp8_e4m3(
    const numr_fp8_e4m3* grad, const numr_fp8_e4m3* output, numr_fp8_e4m3* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float dot = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += fp8_e4m3_to_f32(grad[base + i * stride].data) * fp8_e4m3_to_f32(output[base + i * stride].data);
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = numr_fp8_e4m3(f32_to_fp8_e4m3(fp8_e4m3_to_f32(output[idx].data) * (fp8_e4m3_to_f32(grad[idx].data) - dot)));
    }
}

__global__ void softmax_bwd_dim_fp8_e5m2(
    const numr_fp8_e5m2* grad, const numr_fp8_e5m2* output, numr_fp8_e5m2* d_input,
    unsigned int outer_size, unsigned int dim_size, unsigned int inner_size
) {
    // Flat 1-D grid of full blocks: inner_idx is the fast-varying axis of tid,
    // so neighbouring threads read adjacent addresses and coalesce.
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= outer_size * inner_size) return;

    unsigned int outer_idx = tid / inner_size;
    unsigned int inner_idx = tid - outer_idx * inner_size;

    unsigned int base = outer_idx * dim_size * inner_size + inner_idx;
    unsigned int stride = inner_size;

    float dot = 0.0f;
    for (unsigned int i = 0; i < dim_size; i++) {
        dot += fp8_e5m2_to_f32(grad[base + i * stride].data) * fp8_e5m2_to_f32(output[base + i * stride].data);
    }
    for (unsigned int i = 0; i < dim_size; i++) {
        unsigned int idx = base + i * stride;
        d_input[idx] = numr_fp8_e5m2(f32_to_fp8_e5m2(fp8_e5m2_to_f32(output[idx].data) * (fp8_e5m2_to_f32(grad[idx].data) - dot)));
    }
}

// ============================================================================
// Softmax-with-Bias (Last Dimension, Fused)
//
// Computes softmax(a + bias, last_dim) in a single pass.
// bias shape [batch, 1, 1, dim_size] broadcasts over [batch, heads, seq, dim_size].
// `outer_size` = batch * heads * seq  (number of rows in `a`)
// `dim_size`   = size of last dim (== softmax dim)
// `bias_stride`= stride between bias rows = 1 (the inner-most dim of bias,
//                bias cycles every `dim_size` elements)
// The bias element for column `col` of any row is simply bias[col].
// ============================================================================

__global__ void softmax_bias_f32(
    const float* input, const float* bias, float* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_sb_f32[];
    float* max_val = shared_sb_f32;
    float* sum_exp = shared_sb_f32 + blockDim.x;

    const float* row_in = input + outer_idx * dim_size;
    float* row_out = output + outer_idx * dim_size;

    // Phase 1: find max of (a + bias)
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float v = row_in[i] + bias[i];
        thread_max = fmaxf(thread_max, v);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: exp(a + bias - max) and sum
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(row_in[i] + bias[i] - row_max);
        row_out[i] = val;
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: normalize
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

__global__ void softmax_bias_f16(
    const __half* input, const __half* bias, __half* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_sb_f16[];
    float* max_val = shared_sb_f16;
    float* sum_exp = shared_sb_f16 + blockDim.x;

    const __half* row_in = input + outer_idx * dim_size;
    __half* row_out = output + outer_idx * dim_size;

    // Phase 1: max of (a + bias) in F32
    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float v = __half2float(row_in[i]) + __half2float(bias[i]);
        thread_max = fmaxf(thread_max, v);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    // Phase 2: exp(a + bias - max) in F32, store as F16
    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__half2float(row_in[i]) + __half2float(bias[i]) - row_max);
        row_out[i] = __float2half(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    // Phase 3: normalize in F32, store as F16
    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2half(__half2float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_bias_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* bias, __nv_bfloat16* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ float shared_sb_bf16[];
    float* max_val = shared_sb_bf16;
    float* sum_exp = shared_sb_bf16 + blockDim.x;

    const __nv_bfloat16* row_in = input + outer_idx * dim_size;
    __nv_bfloat16* row_out = output + outer_idx * dim_size;

    float thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float v = __bfloat162float(row_in[i]) + __bfloat162float(bias[i]);
        thread_max = fmaxf(thread_max, v);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) max_val[threadIdx.x] = fmaxf(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        __syncthreads();
    }
    float row_max = max_val[0];
    __syncthreads();

    float thread_sum = 0.0f;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        float val = expf(__bfloat162float(row_in[i]) + __bfloat162float(bias[i]) - row_max);
        row_out[i] = __float2bfloat16(val);
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        __syncthreads();
    }
    float row_sum = sum_exp[0];
    __syncthreads();

    float inv_sum = 1.0f / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] = __float2bfloat16(__bfloat162float(row_out[i]) * inv_sum);
    }
}

__global__ void softmax_bias_f64(
    const double* input, const double* bias, double* output,
    unsigned int outer_size, unsigned int dim_size
) {
    unsigned int outer_idx = blockIdx.x;
    if (outer_idx >= outer_size) return;

    extern __shared__ double shared_sb_f64[];
    double* max_val = shared_sb_f64;
    double* sum_exp = shared_sb_f64 + blockDim.x;

    const double* row_in = input + outer_idx * dim_size;
    double* row_out = output + outer_idx * dim_size;

    double thread_max = -INFINITY;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double v = row_in[i] + bias[i];
        thread_max = fmax(thread_max, v);
    }
    max_val[threadIdx.x] = thread_max;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) max_val[threadIdx.x] = fmax(max_val[threadIdx.x], max_val[threadIdx.x + s]);
        __syncthreads();
    }
    double row_max = max_val[0];
    __syncthreads();

    double thread_sum = 0.0;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        double val = exp(row_in[i] + bias[i] - row_max);
        row_out[i] = val;
        thread_sum += val;
    }
    sum_exp[threadIdx.x] = thread_sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sum_exp[threadIdx.x] += sum_exp[threadIdx.x + s];
        __syncthreads();
    }
    double row_sum = sum_exp[0];
    __syncthreads();

    double inv_sum = 1.0 / row_sum;
    for (unsigned int i = threadIdx.x; i < dim_size; i += blockDim.x) {
        row_out[i] *= inv_sum;
    }
}

} // extern "C"
