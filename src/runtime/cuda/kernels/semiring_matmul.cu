// Semiring matrix multiplication CUDA kernels.
//
//   C[i,j] = reduce_k( combine(A[i,k], B[k,j]) )
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2, i32, i64, u8.
//
// The dtype list follows `SemiringOp::validate_dtype` in src/ops/semiring.rs:
// every semiring except OrAnd admits F32, F64, I32, I64 and the feature-gated
// float types, and OrAnd admits Bool and U8 only. Bool reaches the u8 row,
// which is why there is no separate bool kernel.
//
// Kernel naming matches the names the Rust launchers build in
// src/runtime/cuda/kernels/loader.rs from dtype_suffix():
//   semiring_matmul_{suffix}
//   semiring_matmul_batched_{suffix}
//
// The semiring codes, the storage policies, and the two kernel bodies all live
// in semiring_matmul_ops.cuh.

#include "semiring_matmul_ops.cuh"

// One dtype's two kernels. `P` is the storage policy, `S` the element type in
// the signature, `SUF` the kernel-name suffix.
#define NUMR_SEMIRING_ROW(P, S, SUF)                                            \
    __global__ void semiring_matmul_##SUF(                                      \
        const S* __restrict__ A, const S* __restrict__ B, S* __restrict__ C,    \
        unsigned int M, unsigned int N, unsigned int K, unsigned int op         \
    ) { semiring_matmul_impl<P>(A, B, C, M, N, K, op); }                        \
    __global__ void semiring_matmul_batched_##SUF(                              \
        const S* __restrict__ A, const S* __restrict__ B, S* __restrict__ C,    \
        unsigned int M, unsigned int N, unsigned int K, unsigned int op,        \
        unsigned int batch_size, unsigned int a_batch_count,                    \
        unsigned int b_batch_count                                              \
    ) {                                                                         \
        semiring_matmul_batched_impl<P>(A, B, C, M, N, K, op, batch_size,       \
                                        a_batch_count, b_batch_count);          \
    }

extern "C" {

NUMR_SEMIRING_ROW(SrF32, float, f32)
NUMR_SEMIRING_ROW(SrF64, double, f64)
NUMR_SEMIRING_ROW(SrF16, __half, f16)
NUMR_SEMIRING_ROW(SrBF16, __nv_bfloat16, bf16)
NUMR_SEMIRING_ROW(SrFp8E4M3, numr_fp8_e4m3, fp8_e4m3)
NUMR_SEMIRING_ROW(SrFp8E5M2, numr_fp8_e5m2, fp8_e5m2)
NUMR_SEMIRING_ROW(SrI32, int, i32)
NUMR_SEMIRING_ROW(SrI64, long long, i64)
NUMR_SEMIRING_ROW(SrU8, unsigned char, u8)

} // extern "C"
