// Fused GEMM epilogue kernels:
// - gemm_bias_act: C = activation(A @ B + bias)
// - gemm_bias_residual: C = A @ B + bias + residual
//
// activation_type: 0=None, 1=ReLU, 2=GELU, 3=SiLU, 4=Sigmoid, 5=Tanh

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "matmul_f32_tiled.cuh"

// ============================================================================
// Activation helpers (device functions)
// ============================================================================

__device__ __forceinline__ float apply_activation_f32(float x, unsigned int act_type) {
    switch (act_type) {
        case 0: return x; // None
        case 1: return fmaxf(x, 0.0f); // ReLU
        case 2: { // GELU
            const float sqrt_2_over_pi = 0.7978845608f;
            const float coef = 0.044715f;
            float inner = sqrt_2_over_pi * (x + coef * x * x * x);
            return 0.5f * x * (1.0f + tanhf(inner));
        }
        case 3: { // SiLU
            return x / (1.0f + expf(-x));
        }
        case 4: { // Sigmoid
            return 1.0f / (1.0f + expf(-x));
        }
        case 5: { // Tanh
            return tanhf(x);
        }
        default: return x;
    }
}

__device__ __forceinline__ double apply_activation_f64(double x, unsigned int act_type) {
    switch (act_type) {
        case 0: return x;
        case 1: return fmax(x, 0.0);
        case 2: {
            const double sqrt_2_over_pi = 0.7978845608028654;
            const double coef = 0.044715;
            double inner = sqrt_2_over_pi * (x + coef * x * x * x);
            return 0.5 * x * (1.0 + tanh(inner));
        }
        case 3: return x / (1.0 + exp(-x));
        case 4: return 1.0 / (1.0 + exp(-x));
        case 5: return tanh(x);
        default: return x;
    }
}

// ============================================================================
// GEMM + bias + activation: F32
// ============================================================================

extern "C" __global__ void gemm_bias_act_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + activation
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + bias[global_col];
                    C[global_row * N + global_col] = apply_activation_f32(val, activation_type);
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_act_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const float* A_batch = A + batch * M * K;
    const float* B_batch = B + batch * K * N;
    float* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + bias[global_col];
                    C_batch[global_row * N + global_col] = apply_activation_f32(val, activation_type);
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + activation: F64
// ============================================================================

extern "C" __global__ void gemm_bias_act_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0;

    double reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    double val = reg_c[i][j] + bias[global_col];
                    C[global_row * N + global_col] = apply_activation_f64(val, activation_type);
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_act_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const double* A_batch = A + batch * M * K;
    const double* B_batch = B + batch * K * N;
    double* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0;

    double reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    double val = reg_c[i][j] + bias[global_col];
                    C_batch[global_row * N + global_col] = apply_activation_f64(val, activation_type);
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + residual: F32
// ============================================================================

extern "C" __global__ void gemm_bias_residual_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + residual
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    C[idx] = reg_c[i][j] + bias[global_col] + residual[idx];
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_residual_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const float* A_batch = A + batch * M * K;
    const float* B_batch = B + batch * K * N;
    const float* res_batch = residual + batch * M * N;
    float* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    C_batch[idx] = reg_c[i][j] + bias[global_col] + res_batch[idx];
                }
            }
        }
    }
}
// ============================================================================
// GEMM + bias + residual: F64
// ============================================================================

extern "C" __global__ void gemm_bias_residual_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    const double* __restrict__ residual,
    double* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0;

    double reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    C[idx] = reg_c[i][j] + bias[global_col] + residual[idx];
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_residual_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    const double* __restrict__ residual,
    double* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const double* A_batch = A + batch * M * K;
    const double* B_batch = B + batch * K * N;
    const double* res_batch = residual + batch * M * N;
    double* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0;

    double reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? A_batch[gr * K + gc] : 0.0;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? B_batch[gr * N + gc] : 0.0;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    C_batch[idx] = reg_c[i][j] + bias[global_col] + res_batch[idx];
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + activation: F16
// ============================================================================

extern "C" __global__ void gemm_bias_act_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem_f16[];
    float* As = shared_mem_f16;
    float* Bs = shared_mem_f16 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __half2float(A[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __half2float(B[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + activation
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + __half2float(bias[global_col]);
                    C[global_row * N + global_col] = __float2half(apply_activation_f32(val, activation_type));
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_act_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem_f16[];
    float* As = shared_mem_f16;
    float* Bs = shared_mem_f16 + block_m * block_k;

    const __half* A_batch = A + batch * M * K;
    const __half* B_batch = B + batch * K * N;
    __half* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __half2float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __half2float(B_batch[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + __half2float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2half(apply_activation_f32(val, activation_type));
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + residual: F16
// ============================================================================

extern "C" __global__ void gemm_bias_residual_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem_f16[];
    float* As = shared_mem_f16;
    float* Bs = shared_mem_f16 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __half2float(A[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __half2float(B[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + residual
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    float val = reg_c[i][j] + __half2float(bias[global_col]) + __half2float(residual[idx]);
                    C[idx] = __float2half(val);
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_residual_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem_f16[];
    float* As = shared_mem_f16;
    float* Bs = shared_mem_f16 + block_m * block_k;

    const __half* A_batch = A + batch * M * K;
    const __half* B_batch = B + batch * K * N;
    const __half* res_batch = residual + batch * M * N;
    __half* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __half2float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __half2float(B_batch[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    float val = reg_c[i][j] + __half2float(bias[global_col]) + __half2float(res_batch[idx]);
                    C_batch[idx] = __float2half(val);
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + activation: BF16
// ============================================================================

extern "C" __global__ void gemm_bias_act_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem_bf16[];
    float* As = shared_mem_bf16;
    float* Bs = shared_mem_bf16 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        #pragma unroll
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __bfloat162float(A[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __bfloat162float(B[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + activation
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + __bfloat162float(bias[global_col]);
                    C[global_row * N + global_col] = __float2bfloat16(apply_activation_f32(val, activation_type));
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_act_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem_bf16[];
    float* As = shared_mem_bf16;
    float* Bs = shared_mem_bf16 + block_m * block_k;

    const __nv_bfloat16* A_batch = A + batch * M * K;
    const __nv_bfloat16* B_batch = B + batch * K * N;
    __nv_bfloat16* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __bfloat162float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __bfloat162float(B_batch[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float val = reg_c[i][j] + __bfloat162float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2bfloat16(apply_activation_f32(val, activation_type));
                }
            }
        }
    }
}

// ============================================================================
// GEMM + bias + residual: BF16
// ============================================================================

extern "C" __global__ void gemm_bias_residual_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    extern __shared__ float shared_mem_bf16[];
    float* As = shared_mem_bf16;
    float* Bs = shared_mem_bf16 + block_m * block_k;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __bfloat162float(A[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __bfloat162float(B[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    // EPILOGUE: bias + residual
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    float val = reg_c[i][j] + __bfloat162float(bias[global_col]) + __bfloat162float(residual[idx]);
                    C[idx] = __float2bfloat16(val);
                }
            }
        }
    }
}

extern "C" __global__ void gemm_bias_residual_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    extern __shared__ float shared_mem_bf16[];
    float* As = shared_mem_bf16;
    float* Bs = shared_mem_bf16 + block_m * block_k;

    const __nv_bfloat16* A_batch = A + batch * M * K;
    const __nv_bfloat16* B_batch = B + batch * K * N;
    const __nv_bfloat16* res_batch = residual + batch * M * N;
    __nv_bfloat16* C_batch = C + batch * M * N;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;
    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++)
            reg_c[i][j] = 0.0f;

    float reg_a[8], reg_b[8];
    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_k, lc = load_idx % block_k;
            const unsigned int gr = block_row + lr, gc = k_offset + lc;
            As[lr * block_k + lc] = (gr < M && gc < K) ? __bfloat162float(A_batch[gr * K + gc]) : 0.0f;
        }
        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int lr = load_idx / block_n, lc = load_idx % block_n;
            const unsigned int gr = k_offset + lr, gc = block_col + lc;
            Bs[lr * block_n + lc] = (gr < K && gc < N) ? __bfloat162float(B_batch[gr * N + gc]) : 0.0f;
        }
        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) reg_a[i] = As[(thread_row + i) * block_k + k];
            for (unsigned int j = 0; j < thread_n; j++) reg_b[j] = Bs[k * block_n + thread_col + j];
            for (unsigned int i = 0; i < thread_m; i++)
                for (unsigned int j = 0; j < thread_n; j++)
                    reg_c[i][j] += reg_a[i] * reg_b[j];
        }
        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    unsigned int idx = global_row * N + global_col;
                    float val = reg_c[i][j] + __bfloat162float(bias[global_col]) + __bfloat162float(res_batch[idx]);
                    C_batch[idx] = __float2bfloat16(val);
                }
            }
        }
    }
}

// ============================================================================
// Compile-Time-Tiled FP32 epilogues
// ============================================================================
//
// Same GEMM core as matmul_f32_tiled_128x128x8_8x8 (matmul_f32_tiled.cuh),
// with the epilogue supplied as a compile-time functor.  The generic kernels
// above stay as the fallback for tiles and dtypes without an instantiation.
//
// The epilogue runs only for in-range (row, col) pairs, so bias[col] and
// residual[idx] are never read past the end on a ragged tile.
//
// Configs dispatched from Rust (gemm_epilogue/tiled_f32.rs):
//   128x128x8_8x8   BM=128 BN=128 BK=8  TM=8 TN=8 -> 256 threads
//   64x64x32_8x4    BM=64  BN=64  BK=32 TM=8 TN=4 -> 128 threads
// ============================================================================

// C = activation(A @ B + bias).  Same activation codes and same math as the
// generic gemm_bias_act_f32 kernel: both call apply_activation_f32.
struct GemmEpilogueBiasAct {
    const float* __restrict__ bias;
    unsigned int activation_type;

    __device__ __forceinline__ float apply(float acc, unsigned int, unsigned int col) const {
        return apply_activation_f32(acc + bias[col], activation_type);
    }
};

// C = A @ B + bias + residual.  The residual is elementwise over the output,
// indexed by the flat output offset, matching gemm_bias_residual_f32.
struct GemmEpilogueBiasResidual {
    const float* __restrict__ bias;
    const float* __restrict__ residual;

    __device__ __forceinline__ float apply(float acc, unsigned int idx, unsigned int col) const {
        return acc + bias[col] + residual[idx];
    }
};

// The activation epilogue costs registers: without a bound ptxas gives the
// 128x128 tile 149 registers, and 256 threads x 149 exceeds half the 65 536
// registers per SM, so only one block resides at a time. __launch_bounds__
// caps it at 128 registers with no spill (verified with ptxas -v), restoring
// two resident blocks. The 64x64x32 tile fits without a bound.
extern "C" __global__ __launch_bounds__(256, 2) void gemm_bias_act_f32_tiled_128x128x8_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    GemmEpilogueBiasAct epi{bias, activation_type};
    matmul_f32_tiled_impl<128, 128, 8, 8, 8>(A, B, C, M, N, K, epi);
}

extern "C" __global__ void gemm_bias_act_f32_tiled_64x64x32_8x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    GemmEpilogueBiasAct epi{bias, activation_type};
    matmul_f32_tiled_impl<64, 64, 32, 8, 4>(A, B, C, M, N, K, epi);
}

// Batched: blockIdx.z selects the batch.  A, B and C advance by one full
// matrix per batch (no broadcast operand in this path, matching
// gemm_bias_act_batched_f32); the bias is shared by every batch.
extern "C" __global__ __launch_bounds__(256, 2) void gemm_bias_act_batched_f32_tiled_128x128x8_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    GemmEpilogueBiasAct epi{bias, activation_type};
    matmul_f32_tiled_impl<128, 128, 8, 8, 8>(
        A + batch * (M * K), B + batch * (K * N), C + batch * (M * N), M, N, K, epi);
}

extern "C" __global__ void gemm_bias_act_batched_f32_tiled_64x64x32_8x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    GemmEpilogueBiasAct epi{bias, activation_type};
    matmul_f32_tiled_impl<64, 64, 32, 8, 4>(
        A + batch * (M * K), B + batch * (K * N), C + batch * (M * N), M, N, K, epi);
}

extern "C" __global__ void gemm_bias_residual_f32_tiled_128x128x8_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    GemmEpilogueBiasResidual epi{bias, residual};
    matmul_f32_tiled_impl<128, 128, 8, 8, 8>(A, B, C, M, N, K, epi);
}

extern "C" __global__ void gemm_bias_residual_f32_tiled_64x64x32_8x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    GemmEpilogueBiasResidual epi{bias, residual};
    matmul_f32_tiled_impl<64, 64, 32, 8, 4>(A, B, C, M, N, K, epi);
}

// Batched: the residual advances one full M×N matrix per batch, matching
// gemm_bias_residual_batched_f32.
extern "C" __global__ void gemm_bias_residual_batched_f32_tiled_128x128x8_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    GemmEpilogueBiasResidual epi{bias, residual + batch * (M * N)};
    matmul_f32_tiled_impl<128, 128, 8, 8, 8>(
        A + batch * (M * K), B + batch * (K * N), C + batch * (M * N), M, N, K, epi);
}

extern "C" __global__ void gemm_bias_residual_batched_f32_tiled_64x64x32_8x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    const float* __restrict__ residual,
    float* __restrict__ C,
    unsigned int batch_count,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int batch = blockIdx.z;
    if (batch >= batch_count) return;

    GemmEpilogueBiasResidual epi{bias, residual + batch * (M * N)};
    matmul_f32_tiled_impl<64, 64, 32, 8, 4>(
        A + batch * (M * K), B + batch * (K * N), C + batch * (M * N), M, N, K, epi);
}
