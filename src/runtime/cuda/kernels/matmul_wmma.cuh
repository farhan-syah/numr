// WMMA (Tensor-Core) GEMM kernel body, shared by every F16/BF16 tensor-core
// kernel in matmul_wmma.cu. The instantiations live there; this header holds
// the tile geometry, the epilogue transforms, and the kernel body macro.
//
//
// Uses nvcuda::wmma 16x16x16 fragments with F32 accumulation.
// Shared-memory staging of A and B tiles means global addresses never need
// to satisfy the fragment-alignment requirement (sidesteps the
// CUDA_ERROR_MISALIGNED_ADDRESS class of bug that float4 global loads hit).
//
// Warp tiling (128x128 block tile, 4x4 warp grid, 16 warps, 512 threads):
//   warp_row = warp_id / WARP_COLS  (0..3)
//   warp_col = warp_id % WARP_COLS  (0..3)
//   Each warp computes WARP_M×WARP_N = 2×2 fragments (32×32 outputs).
//   BLOCK_TILE_M = WARP_ROWS*WARP_M*16 = 4*2*16 = 128 ✓
//   BLOCK_TILE_N = WARP_COLS*WARP_N*16 = 4*2*16 = 128 ✓
//   16 warps × 32×32 = 128×128 block ✓
//
// Epilogue: per-warp float scratch in shared memory.
//   store_matrix_sync requires float* for a float accumulator fragment; we
//   cannot store an F32 fragment directly into an F16/BF16 global buffer.
//   Each warp stores its F32 fragment into a dedicated 16×16 float scratch
//   region, then each lane converts + writes elements to global C via STORE_FN
//   (float→HALF_T). No cross-warp collisions.
//
// Static smem (2-stage ping-pong staging, aliased with the epilogue scratch):
//   staging (per stage): smem_A 128 × 24 × 2 = 6 144 bytes
//                        smem_B  16 × 136 × 2 = 4 352 bytes
//   staging (two stages):                        20 992 bytes
//   scratch: 16 warps × 256 × 4 bytes =         16 384 bytes
//   The scratch is written only after the K-loop ends, so it overlays the
//   staging buffers. Total = max(20 992, 16 384) = 20 992 bytes.
//
// Scalar staging (bounds-checked zero-pad loops):
//   Each thread iterates over its share of tile elements with strided loops,
//   loading one element at a time with explicit bounds checks. Out-of-bounds
//   positions are zero-padded. This is deterministic and correct for all shapes.
//
// Caller must guarantee M, N, K are all multiples of 16 before dispatching here.
// FMA fallback handles all other shapes. Given that guarantee, the K-direction
// zero-pad in the staging loops below never triggers — it is currently
// unreachable, not tested. Only the M/N-direction zero-pad (block tiles that
// overhang the matrix) is live.
//
// Double-buffering is synchronous: the global loads for K-tile k+1 are issued
// before the mma work for tile k, so their latency overlaps the compute. One
// __syncthreads() per K-tile.
//
// cp.async double-buffering is DISABLED (documented nondeterminism dead-end).
// Do NOT enable it. The synchronous path is deterministic and correct.

#ifndef NUMR_MATMUL_WMMA_CUH
#define NUMR_MATMUL_WMMA_CUH

// The `#if __CUDA_ARCH__ >= 700` guard lives in the including .cu file.

#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "gemm_activation.cuh"

using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// Warp grid: 4 rows × 4 cols = 16 warps = 512 threads.
#define WARP_ROWS 4
#define WARP_COLS 4

// Warp tile: each warp computes WARP_M × WARP_N fragments.
// WARP_M=2, WARP_N=2 → 4 mma_syncs/warp/K-step.
#define WARP_M 2
#define WARP_N 2

// NOTE: BLOCK_K is a free parameter. The K-loop below steps by WMMA_K, so
// BLOCK_K > WMMA_K just adds more mma_sync calls per K-tile, no rewrite
// needed. An earlier version of this note described a correctness
// regression at BLOCK_K=32; that regression belonged to a since-removed
// cp.async double-buffered kernel, not this synchronous scalar-staging one.
// Raising BLOCK_K here is bounded by the static shared-memory budget above
// (two staging stages now, so the cost per BLOCK_K step doubles) and reduces
// the number of blocks resident per SM.
#define BLOCK_K   16

// Occupancy cap. 16 warps = 512 threads per block; 2 resident blocks fills the
// 64K-register file exactly at 64 registers per thread. Without the cap ptxas
// picks up to 72 registers on some arches, which drops those kernels to one
// resident block. Applied to every kernel in matmul_wmma.cu.
#define WMMA_LAUNCH_BOUNDS __launch_bounds__(WARP_ROWS * WARP_COLS * 32, 2)

// Block tile:
//   BLOCK_TILE_M = WARP_ROWS * WARP_M * WMMA_M = 4 * 2 * 16 = 128 ✓
//   BLOCK_TILE_N = WARP_COLS * WARP_N * WMMA_N = 4 * 2 * 16 = 128 ✓
#define BLOCK_TILE_M  (WARP_ROWS * WARP_M * WMMA_M)   // 128
#define BLOCK_TILE_N  (WARP_COLS * WARP_N * WMMA_N)   // 128

// Smem strides with +8 padding to avoid 32-bank conflicts.
#define SMEM_STRIDE_A (BLOCK_K       + 8)   // 24 halves
#define SMEM_STRIDE_B (BLOCK_TILE_N  + 8)   // 136 halves

// Element counts for one staging stage and for the epilogue scratch.
#define SMEM_A_ELEMS  (BLOCK_TILE_M * SMEM_STRIDE_A)                 // 3072
#define SMEM_B_ELEMS  (BLOCK_K      * SMEM_STRIDE_B)                 // 2176
#define SMEM_SCRATCH_ELEMS ((WARP_ROWS * WARP_COLS) * WMMA_M * WMMA_N)  // 4096

// One byte array backs both the two staging stages and the epilogue scratch.
// The four staging bases sit at byte offsets 0, 6144, 10496 and 16640, all
// multiples of 16, so every wmma fragment pointer keeps the 16-byte alignment
// load_matrix_sync/store_matrix_sync require.
#define SMEM_STAGE_BYTES(HALF_T) \
    (2 * (SMEM_A_ELEMS + SMEM_B_ELEMS) * sizeof(HALF_T))
#define SMEM_SCRATCH_BYTES (SMEM_SCRATCH_ELEMS * sizeof(float))
#define SMEM_TOTAL_BYTES(HALF_T)                                             \
    (SMEM_STAGE_BYTES(HALF_T) > SMEM_SCRATCH_BYTES                           \
        ? SMEM_STAGE_BYTES(HALF_T) : SMEM_SCRATCH_BYTES)


/* Stage one K-tile into the given A/B buffers. Bounds-checked: positions    \
   past the M/N/K edge are zero-padded. */                                   \
#define WMMA_STAGE_TILE(HALF_T, ZERO_EXPR, A_ptr, B_ptr, DST_A, DST_B, K_OFF) \
{                                                                            \
    /* A tile [BLOCK_TILE_M x BLOCK_K]. */                                   \
    for (unsigned int idx = tid;                                             \
         idx < BLOCK_TILE_M * BLOCK_K; idx += num_threads) {                 \
        unsigned int r  = idx / BLOCK_K;                                     \
        unsigned int c  = idx % BLOCK_K;                                     \
        unsigned int gr = block_row + r;                                     \
        unsigned int gc = (K_OFF) + c;                                       \
        HALF_T val = ZERO_EXPR;                                              \
        if (gr < M && gc < K) val = (A_ptr)[gr * K + gc];                    \
        (DST_A)[r * SMEM_STRIDE_A + c] = val;                                \
    }                                                                        \
    /* B tile [BLOCK_K x BLOCK_TILE_N]. */                                   \
    for (unsigned int idx = tid;                                             \
         idx < BLOCK_K * BLOCK_TILE_N; idx += num_threads) {                 \
        unsigned int r  = idx / BLOCK_TILE_N;                                \
        unsigned int c  = idx % BLOCK_TILE_N;                                \
        unsigned int gr = (K_OFF) + r;                                       \
        unsigned int gc = block_col + c;                                     \
        HALF_T val = ZERO_EXPR;                                              \
        if (gr < K && gc < N) val = (B_ptr)[gr * N + gc];                    \
        (DST_B)[r * SMEM_STRIDE_B + c] = val;                                \
    }                                                                        \
}

// ---------------------------------------------------------------------------
// Kernel body macro — instantiated for F16 (non-batched), F16 (batched),
// BF16 (non-batched), BF16 (batched), each with and without a fused bias.
//
// HALF_T  : __half | __nv_bfloat16
// ZERO    : __float2half(0.0f) | __float2bfloat16(0.0f)
// STORE   : __float2half | __float2bfloat16
// EPI_FN  : epilogue value transform,
//           EPI_FN(f32_accumulator, global_row, global_col) → float.
//           WMMA_EPILOGUE_PLAIN is the identity; every other form adds
//           bias[global_col] — and, where named, the activation and the
//           residual — in F32 BEFORE STORE_FN narrows, matching the CPU
//           reference (cpu/kernels/simd/matmul/half_convert.rs).
//
// EPI_FN is a macro parameter rather than a runtime `bias != nullptr` branch:
// the no-bias kernels then expand to the exact code they had before, with no
// extra pointer in the register file and no branch in the epilogue loop.
//
// For the batched variant, the caller sets up A_ptr/B_ptr/C_ptr — and, for the
// residual forms, RES_ptr — from the batch-slice offset before entering the
// macro. The bias is indexed by global column only, so it broadcasts across
// rows and across batch slices — the same semantics as `matmul_bias_batched_*`
// in matmul.cu. The residual is elementwise over [M,N] and its slice advances
// with the batch, matching `gemm_bias_residual_batched_*` in gemm_epilogue.cu.
// ---------------------------------------------------------------------------

#define WMMA_EPILOGUE_PLAIN(VAL, ROW, COL)      (VAL)
#define WMMA_EPILOGUE_BIAS_F16(VAL, ROW, COL)   ((VAL) + __half2float(bias[(COL)]))
#define WMMA_EPILOGUE_BIAS_BF16(VAL, ROW, COL)  ((VAL) + __bfloat162float(bias[(COL)]))

/* bias + activation. `activation_type` is the kernel parameter carrying the
   code `activation_to_u32` emits (gemm_epilogue/launcher.rs); the math itself
   is apply_activation_f32 from gemm_activation.cuh, the same function the
   generic gemm_bias_act_* kernels call.

   The transcendental in the GELU/SiLU/Sigmoid/Tanh cases makes the front end
   give the accumulator array a 128-byte local depot: ptxas -v reports a
   128-byte stack frame and 0 spills. The K loop still keeps the accumulators
   in registers — the local traffic is one store and one load per thread, in
   the epilogue only. WMMA_LAUNCH_BOUNDS caps registers for these kernels the
   same way it does for the rest. */
#define WMMA_EPILOGUE_BIAS_ACT_F16(VAL, ROW, COL)                            \
    apply_activation_f32((VAL) + __half2float(bias[(COL)]), activation_type)
#define WMMA_EPILOGUE_BIAS_ACT_BF16(VAL, ROW, COL)                           \
    apply_activation_f32((VAL) + __bfloat162float(bias[(COL)]), activation_type)

/* bias + residual. `res_ptr` is the residual slice this kernel instance reads,
   indexed by the flat output offset row * N + col. */
#define WMMA_EPILOGUE_BIAS_RESIDUAL_F16(VAL, ROW, COL)                       \
    ((VAL) + __half2float(bias[(COL)])                                       \
           + __half2float(res_ptr[(ROW) * N + (COL)]))
#define WMMA_EPILOGUE_BIAS_RESIDUAL_BF16(VAL, ROW, COL)                      \
    ((VAL) + __bfloat162float(bias[(COL)])                                   \
           + __bfloat162float(res_ptr[(ROW) * N + (COL)]))

/* No-bias body: the epilogue transform is the identity. */
#define WMMA_KERNEL_BODY(HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr)   \
    WMMA_KERNEL_BODY_EPI(HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr,   \
                         WMMA_EPILOGUE_PLAIN)

#define WMMA_KERNEL_BODY_EPI(HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr, EPI_FN) \
{                                                                            \
    const unsigned int warp_id  = threadIdx.x / 32;                          \
    const unsigned int warp_row = warp_id / WARP_COLS;                       \
    const unsigned int warp_col = warp_id % WARP_COLS;                       \
    const unsigned int block_row = blockIdx.y * BLOCK_TILE_M;                \
    const unsigned int block_col = blockIdx.x * BLOCK_TILE_N;                \
    const unsigned int num_threads = blockDim.x;                             \
    const unsigned int tid = threadIdx.x;                                    \
                                                                             \
    /* One backing array for the two staging stages and the epilogue scratch. \
       The scratch is written only after the last staging read, so the two   \
       uses never overlap in time; the __syncthreads() before the epilogue   \
       enforces that. __align__(16) keeps every wmma fragment pointer aligned. */ \
    __shared__ __align__(16) char smem_raw[SMEM_TOTAL_BYTES(HALF_T)];        \
                                                                             \
    HALF_T* const smem_A0 = reinterpret_cast<HALF_T*>(smem_raw);             \
    HALF_T* const smem_B0 = smem_A0 + SMEM_A_ELEMS;                          \
    HALF_T* const smem_A1 = smem_B0 + SMEM_B_ELEMS;                          \
    HALF_T* const smem_B1 = smem_A1 + SMEM_A_ELEMS;                          \
    /* Per-warp F32 epilogue scratch: 16 warps x 16x16 floats.               \
       Each warp owns scratch[warp_id][256]; no cross-warp collision. */     \
    float* const smem_scratch = reinterpret_cast<float*>(smem_raw);          \
                                                                             \
    /* Warp-tiled accumulators: WARP_M x WARP_N fragments per warp. */       \
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[WARP_M][WARP_N]; \
    for (unsigned int wi = 0; wi < WARP_M; wi++) {                           \
        for (unsigned int wj = 0; wj < WARP_N; wj++) {                       \
            fill_fragment(frag_c[wi][wj], 0.0f);                             \
        }                                                                    \
    }                                                                        \
                                                                             \
    const unsigned int num_k_tiles = (K + BLOCK_K - 1) / BLOCK_K;            \
                                                                             \
    /* K == 0 leaves the accumulators at zero and falls straight to the      \
       epilogue, so C is still written. The branch is block-uniform, so the  \
       __syncthreads() calls inside stay collective. */                      \
    if (num_k_tiles > 0) {                                                   \
        /* Prologue: stage K-tile 0 into stage 0. */                         \
        WMMA_STAGE_TILE(HALF_T, ZERO_EXPR, A_ptr, B_ptr, smem_A0, smem_B0, 0u) \
        __syncthreads();                                                     \
                                                                             \
        for (unsigned int bk = 0; bk < num_k_tiles; bk++) {                  \
            const HALF_T* smem_A_cur = (bk & 1u) ? smem_A1 : smem_A0;        \
            const HALF_T* smem_B_cur = (bk & 1u) ? smem_B1 : smem_B0;        \
            HALF_T* smem_A_nxt = (bk & 1u) ? smem_A0 : smem_A1;              \
            HALF_T* smem_B_nxt = (bk & 1u) ? smem_B0 : smem_B1;              \
                                                                             \
            /* Issue the global loads for tile bk+1 into the inactive stage  \
               BEFORE the mma work below, so their latency overlaps it. The  \
               stage written here was last read at iteration bk-1, ahead of  \
               that iteration's closing __syncthreads(). */                  \
            if (bk + 1 < num_k_tiles) {                                      \
                WMMA_STAGE_TILE(HALF_T, ZERO_EXPR, A_ptr, B_ptr,             \
                                smem_A_nxt, smem_B_nxt, (bk + 1) * BLOCK_K)  \
            }                                                                \
                                                                             \
            /* WMMA compute: load frag_a[WARP_M] and frag_b[WARP_N], then    \
               WARP_M x WARP_N = 4 mma_syncs. */                             \
            for (unsigned int k = 0; k < BLOCK_K; k += WMMA_K) {             \
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> \
                    frag_a[WARP_M];                                          \
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> \
                    frag_b[WARP_N];                                          \
                                                                             \
                /* Load WARP_M A fragments (one per M-row of this warp). */  \
                for (unsigned int wi = 0; wi < WARP_M; wi++) {               \
                    unsigned int row = (warp_row * WARP_M + wi) * WMMA_M;    \
                    const HALF_T* a_ptr = smem_A_cur + row * SMEM_STRIDE_A + k; \
                    load_matrix_sync(frag_a[wi], a_ptr, SMEM_STRIDE_A);      \
                }                                                            \
                /* Load WARP_N B fragments (one per N-col of this warp). */  \
                for (unsigned int wj = 0; wj < WARP_N; wj++) {               \
                    unsigned int col = (warp_col * WARP_N + wj) * WMMA_N;    \
                    const HALF_T* b_ptr = smem_B_cur + k * SMEM_STRIDE_B + col; \
                    load_matrix_sync(frag_b[wj], b_ptr, SMEM_STRIDE_B);      \
                }                                                            \
                for (unsigned int wi = 0; wi < WARP_M; wi++) {               \
                    for (unsigned int wj = 0; wj < WARP_N; wj++) {           \
                        mma_sync(frag_c[wi][wj], frag_a[wi], frag_b[wj],     \
                                 frag_c[wi][wj]);                            \
                    }                                                        \
                }                                                            \
            }                                                                \
                                                                             \
            /* Publish the tile staged above before it is read as cur. */    \
            if (bk + 1 < num_k_tiles) {                                      \
                __syncthreads();                                             \
            }                                                                \
        }                                                                    \
    }                                                                        \
                                                                             \
    /* The epilogue scratch aliases the staging buffers. This barrier is what \
       makes that safe: it orders every load_matrix_sync read of the staging \
       buffers above against the first scratch write below. Without it a warp \
       still reading A/B fragments would race the store_matrix_sync of a warp \
       that has already reached the epilogue. */                             \
    __syncthreads();                                                         \
                                                                             \
    /* Epilogue: F32 frag -> per-warp float scratch -> STORE_FN -> global C. \
       store_matrix_sync requires float* for a float accumulator fragment;   \
       we cannot pass C_ptr (HALF_T*) directly. Each warp uses its own       \
       16x16 scratch region (indexed by warp_id) so warps never collide.     \
       __syncwarp() orders the warp-collective store vs the per-lane reads   \
       (and again after the per-lane writes before scratch is reused). */    \
    float* warp_scratch = smem_scratch + warp_id * (WMMA_M * WMMA_N);        \
    const unsigned int lane = threadIdx.x % 32;                              \
    for (unsigned int wi = 0; wi < WARP_M; wi++) {                           \
        for (unsigned int wj = 0; wj < WARP_N; wj++) {                       \
            unsigned int gr = block_row                                      \
                + (warp_row * WARP_M + wi) * WMMA_M;                         \
            unsigned int gc = block_col                                      \
                + (warp_col * WARP_N + wj) * WMMA_N;                         \
            /* Store F32 fragment into warp's float scratch (stride=16). */  \
            store_matrix_sync(warp_scratch, frag_c[wi][wj],                  \
                              WMMA_N, mem_row_major);                        \
            __syncwarp();                                                    \
            /* Each lane converts and writes its share of 256 elements. */   \
            for (unsigned int idx = lane; idx < WMMA_M * WMMA_N; idx += 32) { \
                unsigned int r = idx / WMMA_N;                               \
                unsigned int c = idx % WMMA_N;                               \
                if ((gr + r) < M && (gc + c) < N) {                          \
                    (C_ptr)[(gr + r) * N + (gc + c)] = STORE_FN(             \
                        EPI_FN(warp_scratch[r * WMMA_N + c],                 \
                               gr + r, gc + c));                             \
                }                                                            \
            }                                                                \
            __syncwarp();   /* before reusing scratch for next fragment */   \
        }                                                                    \
    }                                                                        \
}

#endif  // NUMR_MATMUL_WMMA_CUH
