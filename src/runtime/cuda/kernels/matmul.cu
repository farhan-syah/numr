// Native Register-Tiled Matrix Multiplication CUDA Kernels
// Implements C = A @ B using the universal tiled GEMM algorithm:
//
// Level 1: Block Tiling (Shared Memory)
//   - BLOCK_M × BLOCK_N output tiles loaded to shared memory
// Level 2: Register Tiling (Per-Thread Micro-Tiles)
//   - Each thread computes THREAD_M × THREAD_N output elements
// Level 3: FMA Micro-Kernel
//   - Outer product of register arrays
//
// TILE SIZES ARE NOT HARDCODED - passed as kernel parameters
// Optimal values vary by device:
//   - Apple M1/M2: 32×32 or 64×64 tiles
//   - NVIDIA Desktop: 128×128 with 8×8 thread tiles
//   - Mobile/Low-end: Smaller tiles (32×32 with 4×4 thread tiles)

#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ============================================================================
// Configurable Register-Tiled GEMM Kernel (F32)
// ============================================================================
// Tile configuration passed as kernel parameters:
//   block_m, block_n, block_k - Shared memory tile sizes
//   thread_m, thread_n - Per-thread register tile sizes
//
// Dynamic shared memory layout:
//   As[block_m][block_k] followed by Bs[block_k][block_n]

// ============================================================================
// Load helpers: cooperative load of a sub-tile into smem with float4
// vectorization when the innermost dimension is a multiple of 4.
// For A: inner dim = block_k (load_col index).
// For B: inner dim = block_n (load_col index).
// Falls back to scalar for any ragged edge — correctness over speed.
// ============================================================================

// Load A tile [block_m × block_k] into smem_A (no boundary check needed when
// the tile is perfectly aligned; use safe fallback otherwise).
__device__ __forceinline__ void load_a_tile_f32(
    const float* __restrict__ A,
    float* smem_A,
    unsigned int block_row,
    unsigned int k_offset,
    unsigned int M,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_k,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = block_m * block_k;
    // Vectorized path: block_k divisible by 4, tile_elems divisible by 4,
    // AND K divisible by 4.  K%4==0 guarantees that every row-start offset
    // (global_row * K) is 4-float aligned, making the float4 load safe.
    // Without this, batch-slice bases with odd M*K shift every row out of
    // 16-byte alignment, causing CUDA_ERROR_MISALIGNED_ADDRESS.
    if ((block_k & 3u) == 0u && (tile_elems & 3u) == 0u && (K & 3u) == 0u) {
        const unsigned int vec_elems = tile_elems >> 2;
        for (unsigned int vi = thread_id; vi < vec_elems; vi += num_threads) {
            const unsigned int load_idx = vi << 2;
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (global_row < M && global_col + 3 < K) {
                v = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
            } else if (global_row < M && global_col < K) {
                // Scalar fallback for boundary columns
                v.x = (global_col     < K) ? A[global_row * K + global_col    ] : 0.f;
                v.y = (global_col + 1 < K) ? A[global_row * K + global_col + 1] : 0.f;
                v.z = (global_col + 2 < K) ? A[global_row * K + global_col + 2] : 0.f;
                v.w = (global_col + 3 < K) ? A[global_row * K + global_col + 3] : 0.f;
            }
            *reinterpret_cast<float4*>(&smem_A[load_idx]) = v;
        }
    } else {
        for (unsigned int load_idx = thread_id; load_idx < tile_elems; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;
            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            smem_A[load_idx] = val;
        }
    }
}

// Load B tile [block_k × block_n] into smem_B.
__device__ __forceinline__ void load_b_tile_f32(
    const float* __restrict__ B,
    float* smem_B,
    unsigned int block_col,
    unsigned int k_offset,
    unsigned int K,
    unsigned int N,
    unsigned int block_k,
    unsigned int block_n,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = block_k * block_n;
    // N%4==0 ensures every row-start (global_row * N) and column offset
    // (block_col + load_col, where both are multiples of block_n and 4) are
    // 4-float aligned, making the float4 load safe.
    if ((block_n & 3u) == 0u && (tile_elems & 3u) == 0u && (N & 3u) == 0u) {
        const unsigned int vec_elems = tile_elems >> 2;
        for (unsigned int vi = thread_id; vi < vec_elems; vi += num_threads) {
            const unsigned int load_idx = vi << 2;
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
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
            *reinterpret_cast<float4*>(&smem_B[load_idx]) = v;
        }
    } else {
        for (unsigned int load_idx = thread_id; load_idx < tile_elems; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;
            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            smem_B[load_idx] = val;
        }
    }
}

extern "C" __global__ void matmul_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
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
    // Double-buffered shared memory layout:
    //   smem[0]: As0[block_m * block_k], Bs0[block_k * block_n]  (buffer 0)
    //   smem[1]: As1[block_m * block_k], Bs1[block_k * block_n]  (buffer 1)
    // Allocated at launch as 2 * (block_m*block_k + block_k*block_n) floats.
    extern __shared__ float shared_mem[];
    const unsigned int tile_a = block_m * block_k;
    const unsigned int tile_b = block_k * block_n;
    const unsigned int buf_stride = tile_a + tile_b;
    // Buffer 0
    float* As0 = shared_mem;
    float* Bs0 = shared_mem + tile_a;
    // Buffer 1
    float* As1 = shared_mem + buf_stride;
    float* Bs1 = shared_mem + buf_stride + tile_a;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    if (num_k_tiles == 0) return;

    // Preload tile 0 into buffer 0
    load_a_tile_f32(A, As0, block_row, 0, M, K, block_m, block_k, thread_id, num_threads);
    load_b_tile_f32(B, Bs0, block_col, 0, K, N, block_k, block_n, thread_id, num_threads);
    __syncthreads();

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        // Pointers to current (cur) and next (nxt) buffers
        float* As_cur = (bk & 1u) ? As1 : As0;
        float* Bs_cur = (bk & 1u) ? Bs1 : Bs0;
        float* As_nxt = (bk & 1u) ? As0 : As1;
        float* Bs_nxt = (bk & 1u) ? Bs0 : Bs1;

        // Prefetch next tile while computing current (if there is a next tile)
        const unsigned int next_k_offset = (bk + 1) * block_k;
        if (bk + 1 < num_k_tiles) {
            load_a_tile_f32(A, As_nxt, block_row, next_k_offset, M, K, block_m, block_k, thread_id, num_threads);
            load_b_tile_f32(B, Bs_nxt, block_col, next_k_offset, K, N, block_k, block_n, thread_id, num_threads);
        }

        // Compute partial products from current buffer
        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As_cur[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs_cur[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        // Sync: next tile loads (if any) must be complete before the next
        // iteration reads from the nxt buffer (which becomes cur).
        if (bk + 1 < num_k_tiles) {
            __syncthreads();
        }
    }

    // Write register tile to global memory
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable Batched GEMM (F32) — double-buffered + float4 vectorized loads
// ============================================================================

extern "C" __global__ void matmul_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    // Double-buffered shared memory: 2 * (tile_a + tile_b) floats allocated at launch.
    extern __shared__ float shared_mem[];
    const unsigned int tile_a = block_m * block_k;
    const unsigned int tile_b = block_k * block_n;
    const unsigned int buf_stride = tile_a + tile_b;
    float* As0 = shared_mem;
    float* Bs0 = shared_mem + tile_a;
    float* As1 = shared_mem + buf_stride;
    float* Bs1 = shared_mem + buf_stride + tile_a;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const float* A_batch = A + (b % a_batch_count) * stride_a;
    const float* B_batch = B + (b % b_batch_count) * stride_b;
    float* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    if (num_k_tiles == 0) return;

    // Helper lambdas as inline cooperative loads (reuse device functions from non-batched path
    // but operating on A_batch / B_batch pointers).
    // --- Preload tile 0 into buffer 0 ---
    load_a_tile_f32(A_batch, As0, block_row, 0, M, K, block_m, block_k, thread_id, num_threads);
    load_b_tile_f32(B_batch, Bs0, block_col, 0, K, N, block_k, block_n, thread_id, num_threads);
    __syncthreads();

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        float* As_cur = (bk & 1u) ? As1 : As0;
        float* Bs_cur = (bk & 1u) ? Bs1 : Bs0;
        float* As_nxt = (bk & 1u) ? As0 : As1;
        float* Bs_nxt = (bk & 1u) ? Bs0 : Bs1;

        const unsigned int next_k_offset = (bk + 1) * block_k;
        if (bk + 1 < num_k_tiles) {
            load_a_tile_f32(A_batch, As_nxt, block_row, next_k_offset, M, K, block_m, block_k, thread_id, num_threads);
            load_b_tile_f32(B_batch, Bs_nxt, block_col, next_k_offset, K, N, block_k, block_n, thread_id, num_threads);
        }

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As_cur[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs_cur[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        if (bk + 1 < num_k_tiles) {
            __syncthreads();
        }
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (F64)
// ============================================================================

extern "C" __global__ void matmul_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const double* A_batch = A + (b % a_batch_count) * stride_a;
    const double* B_batch = B + (b % b_batch_count) * stride_b;
    double* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j];
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (F16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
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
    // F16 uses F32 shared memory and accumulation for numerical stability
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = __float2half(reg_c[i][j]);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __half* A_batch = A + (b % a_batch_count) * stride_a;
    const __half* B_batch = B + (b % b_batch_count) * stride_b;
    __half* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = __float2half(reg_c[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// Configurable GEMM (BF16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = __float2bfloat16(reg_c[i][j]);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __nv_bfloat16* A_batch = A + (b % a_batch_count) * stride_a;
    const __nv_bfloat16* B_batch = B + (b % b_batch_count) * stride_b;
    __nv_bfloat16* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = __float2bfloat16(reg_c[i][j]);
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias Addition: C = A @ B + bias
// ============================================================================
// Same algorithm as matmul kernels above, but bias[j] is added in the epilogue
// before writing to global memory. This fuses the bias addition to avoid
// an extra memory round-trip.

extern "C" __global__ void matmul_bias_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
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
    for (int i = 0; i < 8; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    // Fused bias addition: C[i][j] = matmul_result + bias[j]
                    C[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const float* __restrict__ bias,
    float* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const float* A_batch = A + (b % a_batch_count) * stride_a;
    const float* B_batch = B + (b % b_batch_count) * stride_b;
    float* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition (same bias for all batches)
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (F64)
// ============================================================================

extern "C" __global__ void matmul_bias_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f64(
    const double* __restrict__ A,
    const double* __restrict__ B,
    const double* __restrict__ bias,
    double* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ double shared_mem_f64[];
    double* As = shared_mem_f64;
    double* Bs = shared_mem_f64 + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const double* A_batch = A + (b % a_batch_count) * stride_a;
    const double* B_batch = B + (b % b_batch_count) * stride_b;
    double* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    double reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0;
        }
    }

    double reg_a[8];
    double reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            double val = 0.0;
            if (global_row < M && global_col < K) {
                val = A_batch[global_row * K + global_col];
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            double val = 0.0;
            if (global_row < K && global_col < N) {
                val = B_batch[global_row * N + global_col];
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C_batch[global_row * N + global_col] = reg_c[i][j] + bias[global_col];
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (F16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bias_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition (convert bias to F32 for add)
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __half2float(bias[global_col]);
                    C[global_row * N + global_col] = __float2half(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __half* A_batch = A + (b % a_batch_count) * stride_a;
    const __half* B_batch = B + (b % b_batch_count) * stride_b;
    __half* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __half2float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __half2float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __half2float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2half(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

// ============================================================================
// Fused GEMM with Bias (BF16 with F32 Accumulation)
// ============================================================================

extern "C" __global__ void matmul_bias_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
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
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __bfloat162float(bias[global_col]);
                    C[global_row * N + global_col] = __float2bfloat16(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

extern "C" __global__ void matmul_bias_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int block_m,
    unsigned int block_n,
    unsigned int block_k,
    unsigned int thread_m,
    unsigned int thread_n,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    extern __shared__ float shared_mem[];
    float* As = shared_mem;
    float* Bs = shared_mem + block_m * block_k;

    const unsigned int b = blockIdx.z;
    if (b >= batch) return;

    const unsigned int stride_a = M * K;
    const unsigned int stride_b = K * N;
    const unsigned int stride_c = M * N;

    const __nv_bfloat16* A_batch = A + (b % a_batch_count) * stride_a;
    const __nv_bfloat16* B_batch = B + (b % b_batch_count) * stride_b;
    __nv_bfloat16* C_batch = C + b * stride_c;

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int threads_x = block_n / thread_n;

    const unsigned int block_row = blockIdx.y * block_m;
    const unsigned int block_col = blockIdx.x * block_n;
    const unsigned int thread_row = ty * thread_m;
    const unsigned int thread_col = tx * thread_n;

    float reg_c[8][8];
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            reg_c[i][j] = 0.0f;
        }
    }

    float reg_a[8];
    float reg_b[8];

    const unsigned int num_k_tiles = (K + block_k - 1) / block_k;
    const unsigned int thread_id = ty * threads_x + tx;
    const unsigned int num_threads = blockDim.x * blockDim.y;

    for (unsigned int bk = 0; bk < num_k_tiles; bk++) {
        const unsigned int k_offset = bk * block_k;

        for (unsigned int load_idx = thread_id; load_idx < block_m * block_k; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_k;
            const unsigned int load_col = load_idx % block_k;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;

            float val = 0.0f;
            if (global_row < M && global_col < K) {
                val = __bfloat162float(A_batch[global_row * K + global_col]);
            }
            As[load_row * block_k + load_col] = val;
        }

        for (unsigned int load_idx = thread_id; load_idx < block_k * block_n; load_idx += num_threads) {
            const unsigned int load_row = load_idx / block_n;
            const unsigned int load_col = load_idx % block_n;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;

            float val = 0.0f;
            if (global_row < K && global_col < N) {
                val = __bfloat162float(B_batch[global_row * N + global_col]);
            }
            Bs[load_row * block_n + load_col] = val;
        }

        __syncthreads();

        for (unsigned int k = 0; k < block_k; k++) {
            for (unsigned int i = 0; i < thread_m; i++) {
                reg_a[i] = As[(thread_row + i) * block_k + k];
            }
            for (unsigned int j = 0; j < thread_n; j++) {
                reg_b[j] = Bs[k * block_n + thread_col + j];
            }
            for (unsigned int i = 0; i < thread_m; i++) {
                for (unsigned int j = 0; j < thread_n; j++) {
                    reg_c[i][j] += reg_a[i] * reg_b[j];
                }
            }
        }

        __syncthreads();
    }

    // EPILOGUE: Write with fused bias addition
    for (unsigned int i = 0; i < thread_m; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            for (unsigned int j = 0; j < thread_n; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    float bias_val = __bfloat162float(bias[global_col]);
                    C_batch[global_row * N + global_col] = __float2bfloat16(reg_c[i][j] + bias_val);
                }
            }
        }
    }
}

// ============================================================================
// Compile-Time-Tiled FP32 GEMM  (template specialisations)
// ============================================================================
//
// WHY: The dynamic matmul_f32 kernel above passes BM/BN/BK/TM/TN as runtime
// unsigned-int arguments. NVCC cannot unroll the micro-kernel loops when the
// bounds are runtime values, so reg_c[TM][TN] spills to local memory and every
// FMA becomes a pair of local-memory loads/stores → ~61 GFLOP/s (0.5% of peak).
//
// FIX: Make tile sizes compile-time C++ template parameters so NVCC can:
//   1. Unroll all inner loops with #pragma unroll (no loop overhead).
//   2. Keep reg_c[TM][TN], reg_a[TM], reg_b[TN] in registers (no spill).
//   3. Emit back-to-back FMA instructions filling the FMA pipe.
//
// The kernel is otherwise structurally identical to matmul_f32:
//   - Double-buffered shared memory (two ping-pong slots).
//   - Float4-vectorised cooperative tile loads, scalar fallback for ragged edges.
//   - Full bounds checks on A/B loads (out-of-range → 0) and C stores.
//   - Grid: (ceil(N/BN), ceil(M/BM), 1)   Block: (BN/TN, BM/TM, 1)
//
// Extern "C" entry points are instantiated below for the two configs dispatched
// from Rust:
//   matmul_f32_tiled_128x128x8_8x8   BM=128 BN=128 BK=8  TM=8 TN=8 → 256 threads
//   matmul_f32_tiled_64x64x32_8x4    BM=64  BN=64  BK=32 TM=8 TN=4 → 128 threads
// ============================================================================

// ---------------------------------------------------------------------------
// Tile-load helpers (compile-time inner dims so the compiler can vectorise)
// ---------------------------------------------------------------------------

// Load A tile [BM × BK] cooperatively.  Float4 when BK%4==0 && K%4==0.
template<int BM, int BK>
__device__ __forceinline__ void ct_load_a(
    const float* __restrict__ A,
    float smem_A[BM][BK],
    unsigned int block_row,
    unsigned int k_offset,
    unsigned int M,
    unsigned int K,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BM * BK;
    if ((BK & 3) == 0 && (K & 3u) == 0u) {
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
        // Scalar fallback.
        for (unsigned int idx = thread_id; idx < tile_elems; idx += num_threads) {
            const unsigned int load_row = idx / BK;
            const unsigned int load_col = idx % BK;
            const unsigned int global_row = block_row + load_row;
            const unsigned int global_col = k_offset + load_col;
            float val = 0.f;
            if (global_row < M && global_col < K) val = A[global_row * K + global_col];
            smem_A[load_row][load_col] = val;
        }
    }
}

// Load B tile [BK × BN] cooperatively.  Float4 when BN%4==0 && N%4==0.
template<int BK, int BN>
__device__ __forceinline__ void ct_load_b(
    const float* __restrict__ B,
    float smem_B[BK][BN],
    unsigned int block_col,
    unsigned int k_offset,
    unsigned int K,
    unsigned int N,
    unsigned int thread_id,
    unsigned int num_threads
) {
    const unsigned int tile_elems = BK * BN;
    if ((BN & 3) == 0 && (N & 3u) == 0u) {
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
        for (unsigned int idx = thread_id; idx < tile_elems; idx += num_threads) {
            const unsigned int load_row = idx / BN;
            const unsigned int load_col = idx % BN;
            const unsigned int global_row = k_offset + load_row;
            const unsigned int global_col = block_col + load_col;
            float val = 0.f;
            if (global_row < K && global_col < N) val = B[global_row * N + global_col];
            smem_B[load_row][load_col] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Main templated kernel
// ---------------------------------------------------------------------------
//
// Template parameters:
//   BM, BN - block tile rows/cols (shared memory footprint)
//   BK     - K-depth of tile (controls inner-loop trip count)
//   TM, TN - per-thread register micro-tile
//
// Thread block: (BN/TN) × (BM/TM)   [x × y]
// Grid:         (ceil(N/BN), ceil(M/BM), 1)
// Shared memory: 2 × (BM×BK + BK×BN) × 4 bytes  (double-buffered)
//
// NOTE: smem uses static 2-D arrays so the index arithmetic is compile-time.
// The double-buffer is realised as two separate static arrays (buf0/buf1) rather
// than a runtime-indexed extern __shared__ pointer so the compiler sees fixed
// strides and can pipeline the loads.
// ---------------------------------------------------------------------------
template<int BM, int BN, int BK, int TM, int TN>
__device__ __forceinline__ void matmul_f32_tiled_impl(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
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

    if (num_k_tiles == 0) return;

    // Preload tile 0 into buffer 0.
    ct_load_a<BM, BK>(A, As0, block_row, 0u, M, K, thread_id, num_threads);
    ct_load_b<BK, BN>(B, Bs0, block_col, 0u, K, N, thread_id, num_threads);
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
            ct_load_a<BM, BK>(A, As_nxt, block_row, next_k, M, K, thread_id, num_threads);
            ct_load_b<BK, BN>(B, Bs_nxt, block_col, next_k, K, N, thread_id, num_threads);
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

    // Epilogue: write results with full bounds check.
    #pragma unroll
    for (int i = 0; i < TM; i++) {
        const unsigned int global_row = block_row + thread_row + i;
        if (global_row < M) {
            #pragma unroll
            for (int j = 0; j < TN; j++) {
                const unsigned int global_col = block_col + thread_col + j;
                if (global_col < N) {
                    C[global_row * N + global_col] = accum[i][j];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Extern "C" entry points (one per instantiation)
// ---------------------------------------------------------------------------
//
// Config 1: BM=128, BN=128, BK=8,  TM=8, TN=8  → 256 threads/block
//   Smem per buffer: (128*8 + 8*128)*4 = 8 192 bytes.  Two buffers = 16 KB.
//   Optimal for large square matmuls (≥128×128).
//
// Config 2: BM=64,  BN=64,  BK=32, TM=8, TN=4  → 128 threads/block
//   Smem per buffer: (64*32 + 32*64)*4 = 16 384 bytes.  Two buffers = 32 KB.
//   Optimal for small-N / small-M shapes (attention score/context paths).
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_f32_tiled_128x128x8_8x8(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_f32_tiled_impl<128, 128, 8, 8, 8>(A, B, C, M, N, K);
}

extern "C" __global__ void matmul_f32_tiled_64x64x32_8x4(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    matmul_f32_tiled_impl<64, 64, 32, 8, 4>(A, B, C, M, N, K);
}
