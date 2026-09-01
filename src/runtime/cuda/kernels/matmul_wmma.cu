// WMMA (Tensor-Core) GEMM kernels for F16 and BF16.
//
// Instantiations only: the tile geometry, the epilogue transforms, and the
// kernel body macro live in matmul_wmma.cuh. Each kernel below picks a dtype,
// a batched or 2-D operand setup, one epilogue transform, and one block tile.
//
//   matmul_wmma_*             C = A @ B
//   matmul_bias_wmma_*        C = A @ B + bias
//   gemm_bias_act_wmma_*      C = activation(A @ B + bias)
//   gemm_bias_residual_wmma_* C = A @ B + bias + residual
//
// Every family is instantiated at three block tiles, 128x128, 128x64 and
// 64x64, named by the tile suffix on the symbol (matmul_wmma_f16_128x128,
// matmul_wmma_f16_128x64, matmul_wmma_f16_64x64, ...) the way
// matmul_f32_tiled.cu names its tiles. The host chooses per launch; see
// loader/matmul_wmma.rs. The tiles differ only in the per-warp fragment
// counts WM x WN passed to the body macro: 2x2 for 128x128, 2x1 for 128x64,
// 1x1 for 64x64. Thread count, launch bounds, staging loops and epilogue are
// shared.
//
// Caller must guarantee M, N, K are all multiples of 16 before dispatching
// here. The FMA fallback handles all other shapes.

#if __CUDA_ARCH__ >= 700

#include "matmul_wmma.cuh"

// ---------------------------------------------------------------------------
// cp.async double-buffering is DISABLED: it has a data race that corrupts GEMM
// output nondeterministically (observed as reranker recall@10 flipping 0.0<->1.0
// run-to-run). A WAR barrier fix (sync before buf swap) was necessary but NOT
// sufficient — at least one more hazard remains in the async path. The synchronous
// path is deterministic AND equally fast here, so we use it on all arches. Do NOT
// re-enable the async path until the race is fully fixed and verified by REPEATED-RUN
// determinism on the real workload (parity tests with clean shapes did not catch it).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Entry-point generators. NAME carries the dtype and the tile
// (e.g. f16_128x128); WM/WN carry the tile itself.
//
// bias is [N] and broadcasts across rows, and across batch slices in the
// batched forms. It is added in F32, before the narrowing store, so the result
// matches the CPU reference. The residual is [M,N] elementwise, read at the
// flat output offset — the same indexing as `gemm_bias_residual_*` in
// gemm_epilogue.cu — and its slice advances with the batch. `activation_type`
// carries the code from `activation_to_u32` (gemm_epilogue/launcher.rs).
// ---------------------------------------------------------------------------

#define DEFINE_WMMA_MATMUL(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void matmul_wmma_##NAME(             \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    HALF_T*       __restrict__ C,                                             \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    WMMA_KERNEL_BODY(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, A, B, C) \
}

#define DEFINE_WMMA_MATMUL_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void matmul_wmma_batched_##NAME(     \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    HALF_T*       __restrict__ C,                                             \
    unsigned int batch,                                                       \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K,                                                           \
    unsigned int a_batch_count,                                               \
    unsigned int b_batch_count                                                \
) {                                                                           \
    const unsigned int b = blockIdx.z;                                        \
    if (b >= batch) return;                                                   \
    const HALF_T* A_b = A + (b % a_batch_count) * (M * K);                    \
    const HALF_T* B_b = B + (b % b_batch_count) * (K * N);                    \
    HALF_T*       C_b = C + b * (M * N);                                      \
    WMMA_KERNEL_BODY(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,        \
                     A_b, B_b, C_b)                                           \
}

#define DEFINE_WMMA_MATMUL_BIAS(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void matmul_bias_wmma_##NAME(        \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    HALF_T*       __restrict__ C,                                             \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A, B, C, EPI_FN)                                     \
}

#define DEFINE_WMMA_MATMUL_BIAS_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void matmul_bias_wmma_batched_##NAME( \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    HALF_T*       __restrict__ C,                                             \
    unsigned int batch,                                                       \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K,                                                           \
    unsigned int a_batch_count,                                               \
    unsigned int b_batch_count                                                \
) {                                                                           \
    const unsigned int b = blockIdx.z;                                        \
    if (b >= batch) return;                                                   \
    const HALF_T* A_b = A + (b % a_batch_count) * (M * K);                    \
    const HALF_T* B_b = B + (b % b_batch_count) * (K * N);                    \
    HALF_T*       C_b = C + b * (M * N);                                      \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A_b, B_b, C_b, EPI_FN)                               \
}

#define DEFINE_WMMA_GEMM_BIAS_ACT(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void gemm_bias_act_wmma_##NAME(      \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    HALF_T*       __restrict__ C,                                             \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K,                                                           \
    unsigned int activation_type                                              \
) {                                                                           \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A, B, C, EPI_FN)                                     \
}

#define DEFINE_WMMA_GEMM_BIAS_ACT_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void gemm_bias_act_wmma_batched_##NAME( \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    HALF_T*       __restrict__ C,                                             \
    unsigned int batch,                                                       \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K,                                                           \
    unsigned int activation_type                                              \
) {                                                                           \
    const unsigned int b = blockIdx.z;                                        \
    if (b >= batch) return;                                                   \
    const HALF_T* A_b = A + b * (M * K);                                      \
    const HALF_T* B_b = B + b * (K * N);                                      \
    HALF_T*       C_b = C + b * (M * N);                                      \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A_b, B_b, C_b, EPI_FN)                               \
}

#define DEFINE_WMMA_GEMM_BIAS_RESIDUAL(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void gemm_bias_residual_wmma_##NAME( \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    const HALF_T* __restrict__ residual,                                      \
    HALF_T*       __restrict__ C,                                             \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    const HALF_T* res_ptr = residual;                                         \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A, B, C, EPI_FN)                                     \
}

#define DEFINE_WMMA_GEMM_BIAS_RESIDUAL_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, EPI_FN) \
extern "C" __global__ WMMA_LAUNCH_BOUNDS void gemm_bias_residual_wmma_batched_##NAME( \
    const HALF_T* __restrict__ A,                                             \
    const HALF_T* __restrict__ B,                                             \
    const HALF_T* __restrict__ bias,                                          \
    const HALF_T* __restrict__ residual,                                      \
    HALF_T*       __restrict__ C,                                             \
    unsigned int batch,                                                       \
    unsigned int M,                                                           \
    unsigned int N,                                                           \
    unsigned int K                                                            \
) {                                                                           \
    const unsigned int b = blockIdx.z;                                        \
    if (b >= batch) return;                                                   \
    const HALF_T* A_b = A + b * (M * K);                                      \
    const HALF_T* B_b = B + b * (K * N);                                      \
    const HALF_T* res_ptr = residual + b * (M * N);                           \
    HALF_T*       C_b = C + b * (M * N);                                      \
    WMMA_KERNEL_BODY_EPI(WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN,    \
                         A_b, B_b, C_b, EPI_FN)                               \
}

/* All eight families at one dtype, one tile, one BLOCK_K. */
#define DEFINE_WMMA_FAMILY(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN, \
                           EPI_BIAS, EPI_ACT, EPI_RESIDUAL)                    \
    DEFINE_WMMA_MATMUL(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR, STORE_FN) \
    DEFINE_WMMA_MATMUL_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR,   \
                               STORE_FN)                                      \
    DEFINE_WMMA_MATMUL_BIAS(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR,      \
                            STORE_FN, EPI_BIAS)                               \
    DEFINE_WMMA_MATMUL_BIAS_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T,         \
                                    ZERO_EXPR, STORE_FN, EPI_BIAS)             \
    DEFINE_WMMA_GEMM_BIAS_ACT(NAME, WM, WN, BLOCK_K_VAL, HALF_T, ZERO_EXPR,     \
                              STORE_FN, EPI_ACT)                              \
    DEFINE_WMMA_GEMM_BIAS_ACT_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T,       \
                                      ZERO_EXPR, STORE_FN, EPI_ACT)            \
    DEFINE_WMMA_GEMM_BIAS_RESIDUAL(NAME, WM, WN, BLOCK_K_VAL, HALF_T,          \
                                   ZERO_EXPR, STORE_FN, EPI_RESIDUAL)          \
    DEFINE_WMMA_GEMM_BIAS_RESIDUAL_BATCHED(NAME, WM, WN, BLOCK_K_VAL, HALF_T,   \
                                           ZERO_EXPR, STORE_FN, EPI_RESIDUAL)

// ---------------------------------------------------------------------------
// F16, all three tiles.
// ---------------------------------------------------------------------------

DEFINE_WMMA_FAMILY(f16_128x128, 2, 2, WMMA_BLOCK_K_DEFAULT, __half,
                   __float2half(0.0f), __float2half,
                   WMMA_EPILOGUE_BIAS_F16, WMMA_EPILOGUE_BIAS_ACT_F16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_F16)

DEFINE_WMMA_FAMILY(f16_128x64, 2, 1, WMMA_BLOCK_K_DEFAULT, __half,
                   __float2half(0.0f), __float2half,
                   WMMA_EPILOGUE_BIAS_F16, WMMA_EPILOGUE_BIAS_ACT_F16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_F16)

DEFINE_WMMA_FAMILY(f16_64x64, 1, 1, WMMA_BLOCK_K_DEFAULT, __half,
                   __float2half(0.0f), __float2half,
                   WMMA_EPILOGUE_BIAS_F16, WMMA_EPILOGUE_BIAS_ACT_F16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_F16)

// ---------------------------------------------------------------------------
// BF16, all three tiles.
//
// BF16 WMMA fragments (nvcuda::wmma::fragment<..., __nv_bfloat16, ...>) are
// only a complete type from sm_80. Below that, `mma.h` declares them as an
// incomplete type and the kernel fails to compile. Guard so these symbols are
// absent by design on sm_75 fatbin slices; the launcher
// (src/runtime/cuda/kernels/loader/matmul_wmma.rs) must not request them on
// a device that lacks `caps.bf16`.
// ---------------------------------------------------------------------------

#if __CUDA_ARCH__ >= 800

DEFINE_WMMA_FAMILY(bf16_128x128, 2, 2, WMMA_BLOCK_K_DEFAULT, __nv_bfloat16,
                   __float2bfloat16(0.0f),
                   __float2bfloat16, WMMA_EPILOGUE_BIAS_BF16,
                   WMMA_EPILOGUE_BIAS_ACT_BF16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_BF16)

DEFINE_WMMA_FAMILY(bf16_128x64, 2, 1, WMMA_BLOCK_K_DEFAULT, __nv_bfloat16,
                   __float2bfloat16(0.0f),
                   __float2bfloat16, WMMA_EPILOGUE_BIAS_BF16,
                   WMMA_EPILOGUE_BIAS_ACT_BF16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_BF16)

DEFINE_WMMA_FAMILY(bf16_64x64, 1, 1, WMMA_BLOCK_K_DEFAULT, __nv_bfloat16,
                   __float2bfloat16(0.0f),
                   __float2bfloat16, WMMA_EPILOGUE_BIAS_BF16,
                   WMMA_EPILOGUE_BIAS_ACT_BF16,
                   WMMA_EPILOGUE_BIAS_RESIDUAL_BF16)

#endif  // __CUDA_ARCH__ >= 800

#endif  // __CUDA_ARCH__ >= 700
