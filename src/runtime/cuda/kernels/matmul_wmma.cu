// WMMA (Tensor-Core) GEMM kernels for F16 and BF16.
//
// Instantiations only: the tile geometry, the epilogue transforms, and the
// kernel body macro live in matmul_wmma.cuh. Each kernel below picks a dtype,
// a batched or 2-D operand setup, and one epilogue transform.
//
//   matmul_wmma_*             C = A @ B
//   matmul_bias_wmma_*        C = A @ B + bias
//   gemm_bias_act_wmma_*      C = activation(A @ B + bias)
//   gemm_bias_residual_wmma_* C = A @ B + bias + residual
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
// F16 non-batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY(__half, __float2half(0.0f), __float2half, A, B, C)
}

// ---------------------------------------------------------------------------
// F16 batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __half* A_b = A + (b % a_batch_count) * (M * K);
    const __half* B_b = B + (b % b_batch_count) * (K * N);
    __half*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY(__half, __float2half(0.0f), __float2half, A_b, B_b, C_b)
}

// ---------------------------------------------------------------------------
// F16 non-batched, fused bias
//
// bias is [N] and broadcasts across rows. It is added in F32, before the
// narrowing store, so the result matches the CPU reference.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_bias_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A, B, C,
                         WMMA_EPILOGUE_BIAS_F16)
}

// ---------------------------------------------------------------------------
// F16 batched, fused bias
//
// bias is [N] and broadcasts across rows AND batch slices.
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_bias_wmma_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __half* A_b = A + (b % a_batch_count) * (M * K);
    const __half* B_b = B + (b % b_batch_count) * (K * N);
    __half*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_F16)
}

// ---------------------------------------------------------------------------
// F16 non-batched, fused bias + activation
//
// C = activation(A @ B + bias). The bias add and the activation both run in
// F32 before the narrowing store. `activation_type` carries the code from
// `activation_to_u32` (gemm_epilogue/launcher.rs).
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_act_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A, B, C,
                         WMMA_EPILOGUE_BIAS_ACT_F16)
}

// ---------------------------------------------------------------------------
// F16 batched, fused bias + activation
//
// A, B and C advance one [M,K] / [K,N] / [M,N] slice per batch index; the bias
// is [N] and broadcasts across rows and batch slices.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_act_wmma_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    __half*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __half* A_b = A + b * (M * K);
    const __half* B_b = B + b * (K * N);
    __half*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_ACT_F16)
}

// ---------------------------------------------------------------------------
// F16 non-batched, fused bias + residual
//
// C = A @ B + bias + residual. The residual is [M,N], elementwise, read at the
// flat output offset — the same indexing as `gemm_bias_residual_f16` in
// gemm_epilogue.cu.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_residual_wmma_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const __half* res_ptr = residual;
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A, B, C,
                         WMMA_EPILOGUE_BIAS_RESIDUAL_F16)
}

// ---------------------------------------------------------------------------
// F16 batched, fused bias + residual
//
// The residual carries one [M,N] slice per batch index, like C.
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_residual_wmma_batched_f16(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    const __half* __restrict__ bias,
    const __half* __restrict__ residual,
    __half*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __half* A_b = A + b * (M * K);
    const __half* B_b = B + b * (K * N);
    const __half* res_ptr = residual + b * (M * N);
    __half*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__half, __float2half(0.0f), __float2half, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_RESIDUAL_F16)
}

// ---------------------------------------------------------------------------
// BF16 non-batched
//
// BF16 WMMA fragments (nvcuda::wmma::fragment<..., __nv_bfloat16, ...>) are
// only a complete type from sm_80. Below that, `mma.h` declares them as an
// incomplete type and the kernel fails to compile. Guard so these two
// symbols are absent by design on sm_75 fatbin slices; the launcher
// (src/runtime/cuda/kernels/loader/matmul_wmma.rs) must not request them on
// a device that lacks `caps.bf16`.
// ---------------------------------------------------------------------------

#if __CUDA_ARCH__ >= 800

extern "C" __global__ void matmul_wmma_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A, B, C)
}

// ---------------------------------------------------------------------------
// BF16 batched
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_wmma_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __nv_bfloat16* A_b = A + (b % a_batch_count) * (M * K);
    const __nv_bfloat16* B_b = B + (b % b_batch_count) * (K * N);
    __nv_bfloat16*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A_b, B_b, C_b)
}

// ---------------------------------------------------------------------------
// BF16 non-batched, fused bias
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_bias_wmma_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A, B, C,
                         WMMA_EPILOGUE_BIAS_BF16)
}

// ---------------------------------------------------------------------------
// BF16 batched, fused bias
// ---------------------------------------------------------------------------

extern "C" __global__ void matmul_bias_wmma_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int a_batch_count,
    unsigned int b_batch_count
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __nv_bfloat16* A_b = A + (b % a_batch_count) * (M * K);
    const __nv_bfloat16* B_b = B + (b % b_batch_count) * (K * N);
    __nv_bfloat16*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_BF16)
}

// ---------------------------------------------------------------------------
// BF16 non-batched, fused bias + activation
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_act_wmma_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A, B, C,
                         WMMA_EPILOGUE_BIAS_ACT_BF16)
}

// ---------------------------------------------------------------------------
// BF16 batched, fused bias + activation
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_act_wmma_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int activation_type
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __nv_bfloat16* A_b = A + b * (M * K);
    const __nv_bfloat16* B_b = B + b * (K * N);
    __nv_bfloat16*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_ACT_BF16)
}

// ---------------------------------------------------------------------------
// BF16 non-batched, fused bias + residual
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_residual_wmma_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16*       __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const __nv_bfloat16* res_ptr = residual;
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A, B, C,
                         WMMA_EPILOGUE_BIAS_RESIDUAL_BF16)
}

// ---------------------------------------------------------------------------
// BF16 batched, fused bias + residual
// ---------------------------------------------------------------------------

extern "C" __global__ void gemm_bias_residual_wmma_batched_bf16(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    const __nv_bfloat16* __restrict__ bias,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16*       __restrict__ C,
    unsigned int batch,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int b = blockIdx.z;
    if (b >= batch) return;
    const __nv_bfloat16* A_b = A + b * (M * K);
    const __nv_bfloat16* B_b = B + b * (K * N);
    const __nv_bfloat16* res_ptr = residual + b * (M * N);
    __nv_bfloat16*       C_b = C + b * (M * N);
    WMMA_KERNEL_BODY_EPI(__nv_bfloat16, __float2bfloat16(0.0f), __float2bfloat16, A_b, B_b, C_b,
                         WMMA_EPILOGUE_BIAS_RESIDUAL_BF16)
}

#endif  // __CUDA_ARCH__ >= 800

#endif  // __CUDA_ARCH__ >= 700
