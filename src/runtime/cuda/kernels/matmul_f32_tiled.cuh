// Compile-time-tiled FP32 GEMM core with a pluggable epilogue.
//
// WHY: the dynamic matmul_f32 / matmul_bias_f32 / gemm_bias_act_f32 kernels
// pass BM/BN/BK/TM/TN as runtime unsigned-int arguments. NVCC cannot unroll
// the micro-kernel loops when the bounds are runtime values, so reg_c[TM][TN]
// spills to local memory and every FMA becomes a pair of local-memory
// loads/stores.
//
// FIX: make tile sizes compile-time C++ template parameters so NVCC can:
//   1. Unroll all inner loops with #pragma unroll (no loop overhead).
//   2. Keep accum[TM][TN], reg_a[TM], reg_b[TN] in registers (no spill).
//   3. Emit back-to-back FMA instructions filling the FMA pipe.
//
// Structure:
//   - Double-buffered shared memory (two ping-pong slots).
//   - Float4-vectorised cooperative tile loads, scalar fallback for ragged edges.
//   - Full bounds checks on A/B loads (out-of-range -> 0) and C stores.
//   - Grid: (ceil(N/BN), ceil(M/BM), 1)   Block: (BN/TN, BM/TM, 1)
//
// The epilogue is a compile-time functor parameter, so bias, activation and
// residual variants reuse this one GEMM core instead of copying it. A functor
// carries its own pointers, so the core needs no extra arguments per variant.
//
// This header holds the device code only. Each translation unit that includes
// it declares its own extern "C" entry points, because a CUDA module only
// exposes the kernels compiled into it.

#ifndef NUMR_MATMUL_F32_TILED_CUH
#define NUMR_MATMUL_F32_TILED_CUH

// Storage dtype is a template parameter defaulting to `float`, so every
// existing call site is unchanged and its generated code identical: for
// `float`, AccumTraits load/store are `p[i]` and `p[i] = v`, and the
// float4 path is selected at compile time exactly as before.
//
// Shared memory and the accumulator stay FP32 whatever the storage dtype, so a
// half-precision GEMM here accumulates in single precision rather than in half.
#include "dtype_traits.cuh"

// Scalar tile loads, shared by the vectorised and non-vectorised paths so the
// bounds logic is written once.
template<typename T, int BM, int BK>
__device__ __forceinline__ void ct_load_a_scalar(
    const T* __restrict__ A,
    float smem_A[BM][BK],
    unsigned int block_row,
    unsigned int k_offset,
    unsigned int M,
    unsigned int K,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BM * BK;
    for (unsigned int idx = thread_id; idx < tile_elems; idx += num_threads) {
        const unsigned int load_row = idx / BK;
        const unsigned int load_col = idx % BK;
        const unsigned int global_row = block_row + load_row;
        const unsigned int global_col = k_offset + load_col;
        float val = 0.f;
        if (global_row < M && global_col < K) {
            val = AccumTraits<T, float>::load(A, (int)(global_row * K + global_col));
        }
        smem_A[load_row][load_col] = val;
    }
}

template<typename T, int BK, int BN>
__device__ __forceinline__ void ct_load_b_scalar(
    const T* __restrict__ B,
    float smem_B[BK][BN],
    unsigned int block_col,
    unsigned int k_offset,
    unsigned int K,
    unsigned int N,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BK * BN;
    for (unsigned int idx = thread_id; idx < tile_elems; idx += num_threads) {
        const unsigned int load_row = idx / BN;
        const unsigned int load_col = idx % BN;
        const unsigned int global_row = k_offset + load_row;
        const unsigned int global_col = block_col + load_col;
        float val = 0.f;
        if (global_row < K && global_col < N) {
            val = AccumTraits<T, float>::load(B, (int)(global_row * N + global_col));
        }
        smem_B[load_row][load_col] = val;
    }
}

// ---------------------------------------------------------------------------
// Tile-load helpers (compile-time inner dims so the compiler can vectorise)
// ---------------------------------------------------------------------------

// Load A tile [BM × BK] cooperatively.  Float4 when BK%4==0 && K%4==0.
template<typename T, int BM, int BK>
__device__ __forceinline__ void ct_load_a(
    const T* __restrict__ A,
    float smem_A[BM][BK],
    unsigned int block_row,
    unsigned int k_offset,
    unsigned int M,
    unsigned int K,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BM * BK;
    // The float4 path reinterprets the source pointer, so it is only valid for
    // 32-bit storage; narrower dtypes take the scalar loads.
    if constexpr (!std::is_same_v<T, float>) {
        ct_load_a_scalar<T, BM, BK>(A, smem_A, block_row, k_offset, M, K,
                                    thread_id, num_threads);
    } else if ((BK & 3) == 0 && (K & 3u) == 0u) {
        // Vectorised path: load float4 (4 × float) per step.
        const unsigned int vec_elems = tile_elems >> 2;
        for (unsigned int vi = thread_id; vi < vec_elems; vi += num_threads) {
            const unsigned int load_idx = vi << 2;
            const unsigned int load_row = load_idx / BK;
            const unsigned int load_col = load_idx % BK;      // multiple of 4
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (global_row < M && global_col + 3 < K) {
                v = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
            } else if (global_row < M && global_col < K) {
                v.x = (global_col     < K) ? A[global_row * K + global_col    ] : 0.f;
                v.y = (global_col + 1 < K) ? A[global_row * K + global_col + 1] : 0.f;
                v.z = (global_col + 2 < K) ? A[global_row * K + global_col + 2] : 0.f;
                v.w = (global_col + 3 < K) ? A[global_row * K + global_col + 3] : 0.f;
            }
            // Scatter back to [row][col] in smem (row-major layout).
            float* dst = &smem_A[load_row][load_col];
            *reinterpret_cast<float4*>(dst) = v;
        }
    } else {
        ct_load_a_scalar<T, BM, BK>(A, smem_A, block_row, k_offset, M, K,
                                    thread_id, num_threads);
    }
}

// Load B tile [BK × BN] cooperatively.  Float4 when BN%4==0 && N%4==0.
template<typename T, int BK, int BN>
__device__ __forceinline__ void ct_load_b(
    const T* __restrict__ B,
    float smem_B[BK][BN],
    unsigned int block_col,
    unsigned int k_offset,
    unsigned int K,
    unsigned int N,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BK * BN;
    if constexpr (!std::is_same_v<T, float>) {
        ct_load_b_scalar<T, BK, BN>(B, smem_B, block_col, k_offset, K, N,
                                    thread_id, num_threads);
    } else if ((BN & 3) == 0 && (N & 3u) == 0u) {
        const unsigned int vec_elems = tile_elems >> 2;
        for (unsigned int vi = thread_id; vi < vec_elems; vi += num_threads) {
            const unsigned int load_idx = vi << 2;
            const unsigned int load_row = load_idx / BN;
            const unsigned int load_col = load_idx % BN;      // multiple of 4
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (global_row < K && global_col + 3 < N) {
                v = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
            } else if (global_row < K && global_col < N) {
                v.x = (global_col     < N) ? B[global_row * N + global_col    ] : 0.f;
                v.y = (global_col + 1 < N) ? B[global_row * N + global_col + 1] : 0.f;
                v.z = (global_col + 2 < N) ? B[global_row * N + global_col + 2] : 0.f;
                v.w = (global_col + 3 < N) ? B[global_row * N + global_col + 3] : 0.f;
            }
            float* dst = &smem_B[load_row][load_col];
            *reinterpret_cast<float4*>(dst) = v;
        }
    } else {
        ct_load_b_scalar<T, BK, BN>(B, smem_B, block_col, k_offset, K, N,
                                    thread_id, num_threads);
    }
}

// ---------------------------------------------------------------------------
// Epilogue functors
// ---------------------------------------------------------------------------
//
// `apply` receives the accumulated value, the flat output index
// (global_row * N + global_col) and the output column, and returns the value
// to store.  It runs only for in-range (row, col) pairs, so an epilogue may
// index bias[col] and residual[idx] without its own bounds check.
//
// MatmulEpilogueNone is the plain-GEMM store and the template default.

struct MatmulEpilogueNone {
    __device__ __forceinline__ float apply(float acc, unsigned int, unsigned int) const {
        return acc;
    }
};

// ---------------------------------------------------------------------------
// Main templated kernel
// ---------------------------------------------------------------------------
//
// Template parameters:
//   BM, BN   - block tile rows/cols (shared memory footprint)
//   BK       - K-depth of tile (controls inner-loop trip count)
//   TM, TN   - per-thread register micro-tile
//   Epilogue - functor applied to each accumulator before the store
//
// Thread block: (BN/TN) × (BM/TM)   [x × y]
// Grid:         (ceil(N/BN), ceil(M/BM), 1)
// Shared memory: 2 × (BM×BK + BK×BN) × 4 bytes  (double-buffered, static)
//
// NOTE: smem uses static 2-D arrays so the index arithmetic is compile-time.
// The double-buffer is realised as two separate static arrays (buf0/buf1) rather
// than a runtime-indexed extern __shared__ pointer so the compiler sees fixed
// strides and can pipeline the loads.
// ---------------------------------------------------------------------------
template<int BM, int BN, int BK, int TM, int TN, class Epilogue = MatmulEpilogueNone,
         typename T = float>
__device__ __forceinline__ void matmul_f32_tiled_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    T* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    Epilogue epi = Epilogue()
) {
    // Static shared memory — compile-time sizes enable unrolling.
    __shared__ float As0[BM][BK];
    __shared__ float Bs0[BK][BN];
    __shared__ float As1[BM][BK];
    __shared__ float Bs1[BK][BN];

    const unsigned int tx = threadIdx.x;      // [0, BN/TN)
    const unsigned int ty = threadIdx.y;      // [0, BM/TM)
    const unsigned int block_row = blockIdx.y * BM;
    const unsigned int block_col = blockIdx.x * BN;
    const unsigned int thread_row = ty * TM;  // start row within block tile
    const unsigned int thread_col = tx * TN;  // start col within block tile

    // Register accumulator — stays in registers because TM/TN are compile-time.
    float accum[TM][TN];
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        #pragma unroll
        for (int j = 0; j < TN; j++) {
            accum[i][j] = 0.f;
        }
    }

    float reg_a[TM];
    float reg_b[TN];

    const unsigned int num_k_tiles = (K + BK - 1) / BK;
    const unsigned int thread_id  = ty * (BN / TN) + tx;
    const unsigned int num_threads = (BM / TM) * (BN / TN);

    // K == 0 leaves the accumulator at zero and falls straight to the
    // epilogue, so C is still written (the generic kernels do the same).
    // The branch is uniform across the block, so the __syncthreads calls
    // inside stay collective.
    if (num_k_tiles > 0) {

        // Preload tile 0 into buffer 0.
        ct_load_a<T, BM, BK>(A, As0, block_row, 0u, M, K, thread_id, num_threads);
        ct_load_b<T, BK, BN>(B, Bs0, block_col, 0u, K, N, thread_id, num_threads);
        __syncthreads();

        for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
            // Select current / next ping-pong buffers.
            float (*As_cur)[BK] = (bk & 1u) ? As1 : As0;
            float (*Bs_cur)[BN] = (bk & 1u) ? Bs1 : Bs0;
            float (*As_nxt)[BK] = (bk & 1u) ? As0 : As1;
            float (*Bs_nxt)[BN] = (bk & 1u) ? Bs0 : Bs1;

            // Prefetch next tile (if any) while computing current tile.
            const unsigned int next_k = (bk + 1) * BK;
            if (bk + 1 < num_k_tiles) {
                ct_load_a<T, BM, BK>(A, As_nxt, block_row, next_k, M, K, thread_id, num_threads);
                ct_load_b<T, BK, BN>(B, Bs_nxt, block_col, next_k, K, N, thread_id, num_threads);
            }

            // Micro-kernel: fully unrolled, accumulators stay in registers.
            #pragma unroll
            for (int k = 0; k < BK; k++) {
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    reg_a[i] = As_cur[thread_row + i][k];
                }
                #pragma unroll
                for (int j = 0; j < TN; j++) {
                    reg_b[j] = Bs_cur[k][thread_col + j];
                }
                #pragma unroll
                for (int i = 0; i < TM; i++) {
                    #pragma unroll
                    for (int j = 0; j < TN; j++) {
                        accum[i][j] += reg_a[i] * reg_b[j];
                    }
                }
            }

            // Sync so next prefetch completes before it is read as cur.
            if (bk + 1 < num_k_tiles) {
                __syncthreads();
            }
        }

    }  // num_k_tiles > 0

    // Epilogue: write results with full bounds check.
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    const unsigned int idx = global_row * N + global_col;
                    AccumTraits<T, float>::store(
                        C, (int)idx, epi.apply(accum[i][j], idx, global_col));
                }
            }
        }
    }
}

#endif // NUMR_MATMUL_F32_TILED_CUH
