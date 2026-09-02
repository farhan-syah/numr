// Grouped FP32 GEMM: one independent matmul per group, row counts on device.
//
//   C[offsets[g] .. offsets[g+1]] = A[offsets[g] .. offsets[g+1]] @ B[g]
//
// A is [total_rows, K], B is [num_groups, K, N], C is [total_rows, N], and
// `offsets` is [num_groups + 1] on the DEVICE. Keeping the boundaries on the
// device is the whole point: a caller that knew them on the host could just
// launch one matmul per group.
//
// The launcher cannot size grid.y from a group's row count for that same
// reason, so it uses the total row count for every group and each block drops
// out if its row tile is past its own group. That test is block-uniform, so it
// costs one comparison and skips the whole K loop.
//
// The GEMM itself is `matmul_f32_tiled_impl` unchanged — the same
// compile-time-tiled, double-buffered, register-blocked core the dense matmul
// entry points use. This file only slices the pointers and picks the group.
//
// Storage may be F32, F16 or BF16; the core accumulates in F32 either way.

#include "gemm_activation.cuh"
#include "matmul_f32_tiled.cuh"

// C = activation(A @ B). Same activation codes and same math as every other
// epilogue in the crate: all of them call apply_activation_f32.
struct GroupedMatmulEpilogueAct {
    unsigned int activation_type;

    __device__ __forceinline__ float apply(float acc, unsigned int, unsigned int) const {
        return apply_activation_f32(acc, activation_type);
    }
};

template<typename T, int BM, int BN, int BK, int TM, int TN, class Epilogue>
__device__ __forceinline__ void grouped_matmul_impl(
    const T* __restrict__ A,
    const T* __restrict__ B,
    const int* __restrict__ offsets,
    T* __restrict__ C,
    unsigned int N,
    unsigned int K,
    int num_groups,
    Epilogue epi
) {
    const int group = blockIdx.z;
    if (group >= num_groups) return;

    const int start = offsets[group];
    const int count = offsets[group + 1] - start;
    if (count <= 0) return;

    // Block-uniform: blockIdx.y and the group's count are both uniform.
    if ((long long)blockIdx.y * BM >= (long long)count) return;

    matmul_f32_tiled_impl<BM, BN, BK, TM, TN, Epilogue, T>(
        A + (size_t)start * K,
        B + (size_t)group * K * N,
        C + (size_t)start * N,
        (unsigned int)count,
        N,
        K,
        epi
    );
}

#define GROUPED_MATMUL_ENTRY(DT, T, SUFFIX, BM, BN, BK, TM, TN)              \
extern "C" __global__ void grouped_matmul_##DT##_##SUFFIX(                   \
    const T* __restrict__ A,                                                 \
    const T* __restrict__ B,                                                 \
    const int* __restrict__ offsets,                                         \
    T* __restrict__ C,                                                       \
    unsigned int N,                                                          \
    unsigned int K,                                                          \
    int num_groups                                                           \
) {                                                                          \
    grouped_matmul_impl<T, BM, BN, BK, TM, TN, MatmulEpilogueNone>(          \
        A, B, offsets, C, N, K, num_groups, MatmulEpilogueNone());           \
}                                                                            \
                                                                             \
extern "C" __global__ void grouped_matmul_act_##DT##_##SUFFIX(               \
    const T* __restrict__ A,                                                 \
    const T* __restrict__ B,                                                 \
    const int* __restrict__ offsets,                                         \
    T* __restrict__ C,                                                       \
    unsigned int N,                                                          \
    unsigned int K,                                                          \
    int num_groups,                                                          \
    unsigned int activation_type                                             \
) {                                                                          \
    GroupedMatmulEpilogueAct epi{activation_type};                           \
    grouped_matmul_impl<T, BM, BN, BK, TM, TN, GroupedMatmulEpilogueAct>(    \
        A, B, offsets, C, N, K, num_groups, epi);                            \
}

// Same two tiles the dense F32 path specialises, so the group case inherits
// whatever the dense one was tuned to.
#define GROUPED_MATMUL_TILES(DT, T)                                          \
GROUPED_MATMUL_ENTRY(DT, T, 128x128x8_8x8, 128, 128, 8, 8, 8)                \
GROUPED_MATMUL_ENTRY(DT, T, 64x64x32_8x4, 64, 64, 32, 8, 4)

GROUPED_MATMUL_TILES(f32, float)
GROUPED_MATMUL_TILES(f16, __half)
GROUPED_MATMUL_TILES(bf16, __nv_bfloat16)
