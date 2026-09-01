// WMMA (Tensor-Core) GEMM kernel body, shared by every F16/BF16 tensor-core
// kernel in matmul_wmma.cu. The instantiations live there; this header holds
// the tile geometry, the epilogue transforms, and the kernel body macro.
//
//
// Uses nvcuda::wmma 16x16x16 fragments with F32 accumulation.
// Shared-memory staging of A and B tiles means the wmma fragment pointers are
// always shared-memory addresses, so global addresses never need to satisfy
// the fragment-alignment requirement. The staging loads themselves are 128-bit
// where the operand allows it, and fall back to element-at-a-time otherwise;
// see WMMA_VEC_OK below for why that fallback is a correctness path.
//
// Warp tiling. The warp grid is fixed at WARP_ROWS x WARP_COLS = 4x4 warps
// (16 warps, 512 threads); the block tile is a parameter of the kernel body
// macro, carried as the per-warp fragment counts WM x WN:
//   warp_row = warp_id / WARP_COLS  (0..3)
//   warp_col = warp_id % WARP_COLS  (0..3)
//   Each warp computes WM x WN fragments of 16x16 outputs.
//   BLOCK_TILE_M = WARP_ROWS * WM * 16
//   BLOCK_TILE_N = WARP_COLS * WN * 16
// Two tiles are instantiated in matmul_wmma.cu:
//   WM=2, WN=2 -> 128x128 block tile, 4 mma_syncs per warp per K-step
//   WM=1, WN=1 ->   64x64 block tile, 1 mma_sync  per warp per K-step
// The tile is part of every kernel symbol name (matmul_wmma_f16_128x128,
// matmul_wmma_f16_64x64, ...), the same convention matmul_f32_tiled.cu uses.
// The host picks one per launch; see
// src/runtime/cuda/kernels/loader/matmul_wmma.rs.
//
// Epilogue: per-warp float scratch in shared memory.
//   store_matrix_sync requires float* for a float accumulator fragment; we
//   cannot store an F32 fragment directly into an F16/BF16 global buffer.
//   Each warp stores its F32 fragment into a dedicated 16x16 float scratch
//   region, then each lane converts + writes elements to global C via STORE_FN
//   (float->HALF_T). No cross-warp collisions. One fragment is in flight per
//   warp at a time, so the scratch is 16 warps x 16x16 floats = 16 384 bytes
//   for every tile shape, independent of WM and WN.
//
// Static smem (WMMA_STAGES-stage ring, aliased with the epilogue scratch).
// WMMA_STAGES is 2 today (classic ping-pong); the ring shape is what would
// let a later multi-stage pipeline raise it without reshaping this staging
// code. BLOCK_K (below) is chosen per architecture, so the staging footprint
// varies by build target. Per stage the staging holds
//   smem_A: BLOCK_TILE_M x SMEM_STRIDE_A halves
//   smem_B: BLOCK_K      x SMEM_STRIDE_B halves
// and the total is max(WMMA_STAGES stages, scratch), because the scratch is
// written only after the K-loop ends and therefore overlays the staging
// buffers:
//   BLOCK_K=16 (SMEM_STRIDE_A=24): 128x128 -> 20 992 B, 64x64 -> 16 384 B
//   BLOCK_K=32 (SMEM_STRIDE_A=40): 128x128 -> 37 888 B, 64x64 -> 19 456 B
// The 64x64 tile never needs more smem than the 128x128 one, so it never
// lowers the resident-block count.
//
// Staging (bounds-checked zero-pad loops, vectorised where legal):
//   Threads cooperatively copy the tile with strided loops. The fast path
//   moves eight halves (128 bits) per access, so the loop trip count and the
//   address arithmetic and branching that go with it drop by 8x; this is where
//   nearly all of the kernel's non-mma instructions used to go. It is taken
//   only when the global operand is 16-byte aligned with a row stride that is
//   a multiple of eight halves, and the tile does not overhang the contraction
//   (A) or N (B) edge. Everything else takes the element-at-a-time path.
//   Out-of-bounds positions are zero-padded identically on both paths, so the
//   two agree bit-for-bit. Both are deterministic and correct for all shapes.
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

// Warp grid: 4 rows × 4 cols = 16 warps = 512 threads. Fixed for every tile;
// the tile is varied through the per-warp fragment counts WM and WN instead,
// so the thread count and the launch bounds stay the same across tiles.
#define WARP_ROWS 4
#define WARP_COLS 4

// NOTE: BLOCK_K is a free parameter. The K-loop below steps by WMMA_K, so
// BLOCK_K > WMMA_K just adds more mma_sync calls per K-tile, no rewrite
// needed. An earlier version of this note described a correctness
// regression at BLOCK_K=32; that regression belonged to a since-removed
// cp.async double-buffered kernel, not this synchronous scalar-staging one.
//
// BLOCK_K sets the WMMA_STAGES-stage staging footprint (see the smem
// accounting above), which bounds how many blocks can be resident per SM at
// once — and the shared-memory budget per SM differs by architecture. A
// single BLOCK_K cannot be optimal everywhere, so it is chosen per
// architecture: this file compiles once per target arch into the fatbin, and
// __CUDA_ARCH__ is only defined during device compilation, so this stays a
// compile-time constant with no host-side effect.
#if __CUDA_ARCH__ >= 800
#define BLOCK_K   32
#else
#define BLOCK_K   16
#endif

// Occupancy cap. 16 warps = 512 threads per block; 2 resident blocks fills the
// 64K-register file exactly at 64 registers per thread. Without the cap ptxas
// picks up to 72 registers on some arches, which drops those kernels to one
// resident block. Applied to every kernel in matmul_wmma.cu.
#define WMMA_LAUNCH_BOUNDS __launch_bounds__(WARP_ROWS * WARP_COLS * 32, 2)

// Number of shared-memory staging buffers in the ring. 2 today: while the
// K-loop computes on stage bk % WMMA_STAGES, the global loads for the next
// K-tile land in stage (bk+1) % WMMA_STAGES. Kept as its own constant, and
// threaded through the smem sizing below, so a later multi-stage pipeline
// changes only this value instead of reshaping the staging code.
#define WMMA_STAGES 2
static_assert(WMMA_STAGES >= 2,
              "ring needs >= 2 distinct stages so stage bk and stage bk+1 "
              "never alias the same buffer while bk is still being read");

/* Tile-derived compile-time constants, declared inside the kernel body so the
   tile can vary per instantiation. Smem strides carry +8 padding to avoid
   32-bank conflicts; both stay multiples of 8 halves (16 bytes), the alignment
   load_matrix_sync/store_matrix_sync require of their leading dimension. */
#define WMMA_TILE_CONSTANTS(WM, WN)                                          \
    constexpr unsigned int BLOCK_TILE_M = WARP_ROWS * (WM) * WMMA_M;         \
    constexpr unsigned int BLOCK_TILE_N = WARP_COLS * (WN) * WMMA_N;         \
    constexpr unsigned int SMEM_STRIDE_A = BLOCK_K + 8;                      \
    constexpr unsigned int SMEM_STRIDE_B = BLOCK_TILE_N + 8;                 \
    constexpr unsigned int SMEM_A_ELEMS = BLOCK_TILE_M * SMEM_STRIDE_A;      \
    constexpr unsigned int SMEM_B_ELEMS = BLOCK_K * SMEM_STRIDE_B;           \
    /* One 16x16 float region per warp; one fragment is in flight at a time. */ \
    constexpr unsigned int SMEM_SCRATCH_ELEMS =                              \
        (WARP_ROWS * WARP_COLS) * WMMA_M * WMMA_N;                           \
    /* The 128-bit staging path below writes shared memory at                \
       r * SMEM_STRIDE_* + c with c a multiple of WMMA_VEC_HALVES, so both   \
       strides must be multiples of WMMA_VEC_HALVES for the destination to   \
       stay 16-byte aligned. The same holds for the WMMA_STAGES staging      \
       bases, which sit at multiples of SMEM_A_ELEMS / SMEM_B_ELEMS. Changing\
       BLOCK_K or the tile without keeping these true faults the kernel. */  \
    static_assert(SMEM_STRIDE_A % 8 == 0 && SMEM_STRIDE_B % 8 == 0,          \
                  "smem strides must be multiples of 8 halves");             \
    static_assert(SMEM_A_ELEMS % 8 == 0 && SMEM_B_ELEMS % 8 == 0,            \
                  "smem staging bases must be 16-byte aligned");             \
    static_assert(BLOCK_K % 8 == 0 && BLOCK_TILE_N % 8 == 0,                 \
                  "tile inner dims must be whole 128-bit groups");

/* One byte array backs both the WMMA_STAGES staging buffers and the epilogue
   scratch. The WMMA_STAGES staging bases sit at multiples of 16 bytes for
   every instantiated tile, so every wmma fragment pointer keeps the 16-byte
   alignment load_matrix_sync/store_matrix_sync require. */
#define SMEM_STAGE_BYTES(HALF_T) \
    (WMMA_STAGES * (SMEM_A_ELEMS + SMEM_B_ELEMS) * sizeof(HALF_T))
#define SMEM_SCRATCH_BYTES (SMEM_SCRATCH_ELEMS * sizeof(float))
#define SMEM_TOTAL_BYTES(HALF_T)                                             \
    (SMEM_STAGE_BYTES(HALF_T) > SMEM_SCRATCH_BYTES                           \
        ? SMEM_STAGE_BYTES(HALF_T) : SMEM_SCRATCH_BYTES)


// Staging macros (WMMA_VEC_HALVES, WMMA_VEC_OK, WMMA_VEC_STAGE,
// WMMA_STAGE_TILE) live in matmul_wmma_stage.cuh, split out to keep this
// file under the line cap. They expand against the tile constants declared
// above (SMEM_STRIDE_A, SMEM_STRIDE_B, BLOCK_K, ...).
#include "matmul_wmma_stage.cuh"

// ---------------------------------------------------------------------------
// Kernel body macro — instantiated for F16 (non-batched), F16 (batched),
// BF16 (non-batched), BF16 (batched), each with and without a fused bias, and
// each at both block tiles.
//
// WM, WN  : per-warp fragment counts; the block tile is
//           WARP_ROWS*WM*16 by WARP_COLS*WN*16.
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
   give the accumulator array a local depot: ptxas -v reports a stack frame and
   0 spills. The K loop still keeps the accumulators in registers — the local
   traffic is one store and one load per thread, in the epilogue only.
   WMMA_LAUNCH_BOUNDS caps registers for these kernels the same way it does for
   the rest. */
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
#define WMMA_KERNEL_BODY(WM, WN, HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr) \
    WMMA_KERNEL_BODY_EPI(WM, WN, HALF_T, ZERO_EXPR, STORE_FN,                \
                         A_ptr, B_ptr, C_ptr, WMMA_EPILOGUE_PLAIN)

#define WMMA_KERNEL_BODY_EPI(WM, WN, HALF_T, ZERO_EXPR, STORE_FN, A_ptr, B_ptr, C_ptr, EPI_FN) \
{                                                                            \
    WMMA_TILE_CONSTANTS(WM, WN)                                              \
                                                                             \
    const unsigned int warp_id  = threadIdx.x / 32;                          \
    const unsigned int warp_row = warp_id / WARP_COLS;                       \
    const unsigned int warp_col = warp_id % WARP_COLS;                       \
    const unsigned int block_row = blockIdx.y * BLOCK_TILE_M;                \
    const unsigned int block_col = blockIdx.x * BLOCK_TILE_N;                \
    const unsigned int num_threads = blockDim.x;                             \
    const unsigned int tid = threadIdx.x;                                    \
                                                                             \
    /* One backing array for the WMMA_STAGES staging buffers and the epilogue \
       scratch. The scratch is written only after the last staging read, so  \
       the two uses never overlap in time; the __syncthreads() before the    \
       epilogue enforces that. __align__(16) keeps every wmma fragment       \
       pointer aligned. */                                                   \
    __shared__ __align__(16) char smem_raw[SMEM_TOTAL_BYTES(HALF_T)];        \
    /* Guards the SELECTION below, not sizing: SMEM_TOTAL_BYTES already takes \
       a max over the stage footprint, so a size assert could never fail. */  \
    static_assert(WMMA_STAGES == 2,                                           \
        "WMMA_STAGES > 2 needs more named bases and more selection cases");   \
                                                                             \
    /* WMMA_STAGES named stage bases rather than an indexed array: a local   \
       array read with a runtime index (bk % WMMA_STAGES) does not stay in   \
       registers, it spills to local memory on this hot path. Named bases    \
       plus the compile-time-foldable ternary below select in registers,     \
       exactly as the fixed 2-buffer ping-pong did. Extending WMMA_STAGES    \
       past 2 means adding a base here and a case to the selection below. */ \
    HALF_T* const smem_A0 = reinterpret_cast<HALF_T*>(smem_raw);             \
    HALF_T* const smem_B0 = smem_A0 + SMEM_A_ELEMS;                          \
    HALF_T* const smem_A1 = smem_B0 + SMEM_B_ELEMS;                          \
    HALF_T* const smem_B1 = smem_A1 + SMEM_A_ELEMS;                          \
    /* Per-warp F32 epilogue scratch: 16 warps x 16x16 floats.               \
       Each warp owns scratch[warp_id][256]; no cross-warp collision. */     \
    float* const smem_scratch = reinterpret_cast<float*>(smem_raw);          \
                                                                             \
    /* Warp-tiled accumulators: WM x WN fragments per warp. */               \
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[WM][WN];     \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
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
        /* Prologue: stage K-tile 0 into ring slot 0. */                     \
        WMMA_STAGE_TILE(HALF_T, ZERO_EXPR, A_ptr, B_ptr,                     \
                        smem_A0, smem_B0, 0u)                                \
        __syncthreads();                                                     \
                                                                             \
        for (unsigned int bk = 0; bk < num_k_tiles; bk++) {                  \
            const unsigned int stage_cur = bk % WMMA_STAGES;                 \
            const unsigned int stage_nxt = (bk + 1) % WMMA_STAGES;           \
            const HALF_T* smem_A_cur = (stage_cur == 0) ? smem_A0 : smem_A1; \
            const HALF_T* smem_B_cur = (stage_cur == 0) ? smem_B0 : smem_B1; \
            HALF_T* smem_A_nxt = (stage_nxt == 0) ? smem_A0 : smem_A1;       \
            HALF_T* smem_B_nxt = (stage_nxt == 0) ? smem_B0 : smem_B1;       \
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
            /* WMMA compute: load frag_a[WM] and frag_b[WN], then            \
               WM x WN mma_syncs. */                                         \
            for (unsigned int k = 0; k < BLOCK_K; k += WMMA_K) {             \
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> \
                    frag_a[WM];                                              \
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, HALF_T, row_major> \
                    frag_b[WN];                                              \
                                                                             \
                /* Load WM A fragments (one per M-row of this warp). */      \
                for (unsigned int wi = 0; wi < (WM); wi++) {                 \
                    unsigned int row = (warp_row * (WM) + wi) * WMMA_M;      \
                    const HALF_T* a_ptr = smem_A_cur + row * SMEM_STRIDE_A + k; \
                    load_matrix_sync(frag_a[wi], a_ptr, SMEM_STRIDE_A);      \
                }                                                            \
                /* Load WN B fragments (one per N-col of this warp). */      \
                for (unsigned int wj = 0; wj < (WN); wj++) {                 \
                    unsigned int col = (warp_col * (WN) + wj) * WMMA_N;      \
                    const HALF_T* b_ptr = smem_B_cur + k * SMEM_STRIDE_B + col; \
                    load_matrix_sync(frag_b[wj], b_ptr, SMEM_STRIDE_B);      \
                }                                                            \
                for (unsigned int wi = 0; wi < (WM); wi++) {                 \
                    for (unsigned int wj = 0; wj < (WN); wj++) {             \
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
       16x16 scratch region (indexed by warp_id) so warps never collide,     \
       for every tile shape. __syncwarp() orders the warp-collective store   \
       vs the per-lane reads (and again after the per-lane writes before     \
       scratch is reused). */                                                \
    float* warp_scratch = smem_scratch + warp_id * (WMMA_M * WMMA_N);        \
    const unsigned int lane = threadIdx.x % 32;                              \
    for (unsigned int wi = 0; wi < (WM); wi++) {                             \
        for (unsigned int wj = 0; wj < (WN); wj++) {                         \
            unsigned int gr = block_row                                      \
                + (warp_row * (WM) + wi) * WMMA_M;                           \
            unsigned int gc = block_col                                      \
                + (warp_col * (WN) + wj) * WMMA_N;                           \
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
