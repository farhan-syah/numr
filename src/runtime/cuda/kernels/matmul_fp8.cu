// Compile-Time-Tiled FP8 GEMM (FP8E4M3, FP8E5M2)
//
// Same dtype in, same dtype out: C[M,N] = A[M,K] @ B[K,N] with FP8 storage.
// The mixed-precision kernels in `fp8_matmul.cu` are a different operation -
// FP8 inputs with an F32/F16/BF16 output, reached through `Fp8MatmulOps` - and
// cannot serve this path.
//
// Accumulation is in `float`, never in the element type. This is what CPU does
// (`matmul_scalar_acc::<T, f32>` in `runtime/cpu/kernels/matmul/kernel.rs`,
// selected by `is_narrow_float`): an E4M3 dot product saturates at 448 after a
// handful of terms, so accumulating in FP8 would report a different number than
// the CPU reference. Each output element is narrowed exactly once, at the store.
//
// Shared memory holds F32, not FP8, so the micro-kernel is a plain FMA chain and
// each element is decoded once per tile instead of once per multiply. This
// mirrors the F16/BF16 kernels in `matmul.cu`.
//
// One tile shape per dtype: BM=BN=64, BK=8, TM=TN=4, giving (64/4) x (64/4) =
// 256 threads per block, 16 float accumulators per thread, and a static 4 KB of
// shared memory. The 64-wide block tile keeps the waste bounded on the small
// matrices FP8 matmul is typically called with, and stays well inside the
// 48 KB per-block shared-memory default.

#include "dtype_traits.cuh"

// Conversion policies. Each names one FP8 format's encode/decode pair, so the
// tiled kernel below is written once.
struct Fp8E4M3Conv {
    static __device__ __forceinline__ float to_f32(uint8_t v) { return fp8_e4m3_to_f32(v); }
    static __device__ __forceinline__ uint8_t from_f32(float v) { return f32_to_fp8_e4m3(v); }
};

struct Fp8E5M2Conv {
    static __device__ __forceinline__ float to_f32(uint8_t v) { return fp8_e5m2_to_f32(v); }
    static __device__ __forceinline__ uint8_t from_f32(float v) { return f32_to_fp8_e5m2(v); }
};

template<typename Conv, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_fp8_tiled_impl(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // Static shared memory in F32: compile-time sizes constant-fold the index
    // arithmetic and let the loads below unroll.
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const unsigned int tx = threadIdx.x;      // [0, BN/TN)
    const unsigned int ty = threadIdx.y;      // [0, BM/TM)
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    const unsigned int thread_row = ty * TM;
    const unsigned int thread_col = tx * TN;

    float reg_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const unsigned int num_k_tiles = (K + BK - 1) / BK;
    const unsigned int thread_id = ty * (BN / TN) + tx;
    const unsigned int num_threads = (BM / TM) * (BN / TN);

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * BK;

        // Out-of-range elements load as zero, so the micro-kernel below runs the
        // full BK trip count on ragged tiles without changing the result.
        for (unsigned int load_idx = thread_id; load_idx < BM * BK; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BK;
            const unsigned int load_col = load_idx % BK;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = Conv::to_f32(A[global_row * K + global_col]);
            }
            As[load_row][load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < BK * BN; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BN;
            const unsigned int load_col = load_idx % BN;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = Conv::to_f32(B[global_row * N + global_col]);
            }
            Bs[load_row][load_col] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[thread_row + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[k][thread_col + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = Conv::from_f32(reg_c[i][j]);
                }
            }
        }
    }
}

// Batched: one grid z slice per batch element. `a_batch_count` / `b_batch_count`
// carry broadcasting, so a shared operand is read by every slice.
template<typename Conv, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_batched_fp8_tiled_impl(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    matmul_fp8_tiled_impl<Conv, BM, BN, BK, TM, TN>(
        A + (size_t)(b % a_batch_count) * M * K,
        B + (size_t)(b % b_batch_count) * K * N,
        C + (size_t)b * M * N,
        M, N, K
    );
}

// ---------------------------------------------------------------------------
// Extern "C" entry points (one per instantiation)
//
// Grid: (ceil(N/64), ceil(M/64), batch)   Block: (16, 16, 1)
// Shared memory is static, so these launch with zero dynamic shared memory.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_fp8_e4m3_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_fp8_tiled_impl<Fp8E4M3Conv, 64, 64, 8, 4, 4>(A, B, C, M, N, K);
}

extern "C" __global__ void matmul_fp8_e5m2_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_fp8_tiled_impl<Fp8E5M2Conv, 64, 64, 8, 4, 4>(A, B, C, M, N, K);
}

extern "C" __global__ void matmul_batched_fp8_e4m3_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    matmul_batched_fp8_tiled_impl<Fp8E4M3Conv, 64, 64, 8, 4, 4>(
        A, B, C, batch, M, N, K, a_batch_count, b_batch_count
    );
}

extern "C" __global__ void matmul_batched_fp8_e5m2_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    matmul_batched_fp8_tiled_impl<Fp8E5M2Conv, 64, 64, 8, 4, 4>(
        A, B, C, batch, M, N, K, a_batch_count, b_batch_count
    );
}

// ---------------------------------------------------------------------------
// Fused GEMM with bias: C = A @ B + bias
//
// The bias seeds the F32 accumulator rather than being added to a narrowed
// result. CPU does the same (`matmul_bias_scalar_acc::<T, f32>`), so a
// matmul-then-add composition would narrow twice and report a different number.
// ---------------------------------------------------------------------------

template<typename Conv, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_bias_fp8_tiled_impl(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    const unsigned int thread_row = ty * TM;
    const unsigned int thread_col = tx * TN;

    // Seed each accumulator with its output column's bias.
    float reg_c[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            const unsigned int global_col = block_col + thread_col + j;
            reg_c[i][j] = (global_col < N) ? Conv::to_f32(bias[global_col]) : 0.0f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const unsigned int num_k_tiles = (K + BK - 1) / BK;
    const unsigned int thread_id = ty * (BN / TN) + tx;
    const unsigned int num_threads = (BM / TM) * (BN / TN);

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * BK;

        for (unsigned int load_idx = thread_id; load_idx < BM * BK; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BK;
            const unsigned int load_col = load_idx % BK;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = Conv::to_f32(A[global_row * K + global_col]);
            }
            As[load_row][load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < BK * BN; load_idx += num_threads) {
            const unsigned int load_row = load_idx / BN;
            const unsigned int load_col = load_idx % BN;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = Conv::to_f32(B[global_row * N + global_col]);
            }
            Bs[load_row][load_col] = val;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; k++) {
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                reg_a[i] = As[thread_row + i][k];
            }
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                reg_b[j] = Bs[k][thread_col + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; i++) {
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = Conv::from_f32(reg_c[i][j]);
                }
            }
        }
    }
}

// Bias is [N] and shared by every batch slice, so it is not offset here.
template<typename Conv, int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_bias_batched_fp8_tiled_impl(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    matmul_bias_fp8_tiled_impl<Conv, BM, BN, BK, TM, TN>(
        A + (size_t)(b % a_batch_count) * M * K,
        B + (size_t)(b % b_batch_count) * K * N,
        bias,
        C + (size_t)b * M * N,
        M, N, K
    );
}

extern "C" __global__ void matmul_bias_fp8_e4m3_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_bias_fp8_tiled_impl<Fp8E4M3Conv, 64, 64, 8, 4, 4>(A, B, bias, C, M, N, K);
}

extern "C" __global__ void matmul_bias_fp8_e5m2_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_bias_fp8_tiled_impl<Fp8E5M2Conv, 64, 64, 8, 4, 4>(A, B, bias, C, M, N, K);
}

extern "C" __global__ void matmul_bias_batched_fp8_e4m3_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    matmul_bias_batched_fp8_tiled_impl<Fp8E4M3Conv, 64, 64, 8, 4, 4>(
        A, B, bias, C, batch, M, N, K, a_batch_count, b_batch_count
    );
}

extern "C" __global__ void matmul_bias_batched_fp8_e5m2_tiled_64x64x8_4x4(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    const uint8_t* __restrict__ bias,
    uint8_t* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    matmul_bias_batched_fp8_tiled_impl<Fp8E5M2Conv, 64, 64, 8, 4, 4>(
        A, B, bias, C, batch, M, N, K, a_batch_count, b_batch_count
    );
}
