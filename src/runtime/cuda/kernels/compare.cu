// Comparison CUDA kernels
//
// Operations: eq, ne, lt, le, gt, ge
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8
//
// Output is the SAME dtype as the input (1 for true, 0 for false), not a bool
// tensor. That is intentional and matches the CPU reference: a mask can be fed
// straight into arithmetic (mask * tensor) without a dtype conversion.
//
// Kernel naming, matching the names the Rust launchers build in
// src/runtime/cuda/kernels/compare.rs from dtype_suffix() in loader.rs:
//   {op}_{suffix}            element-wise, same-shape operands
//   {op}_broadcast_{suffix}  broadcast, strides in device memory
//
// Comparison never overflows, so unlike the arithmetic kernels in binary.cu the
// integer dtypes need no wrapping treatment: one template covers all eight.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Per-operation device functions
// ============================================================================

#define NUMR_CMP_PRIMARY(NAME, OP)                                              \
    template<typename T>                                                        \
    __device__ __forceinline__ T compare_##NAME(T a, T b) {                     \
        return (a OP b) ? (T)1 : (T)0;                                          \
    }

NUMR_CMP_PRIMARY(eq, ==)
NUMR_CMP_PRIMARY(ne, !=)
NUMR_CMP_PRIMARY(lt, <)
NUMR_CMP_PRIMARY(le, <=)
NUMR_CMP_PRIMARY(gt, >)
NUMR_CMP_PRIMARY(ge, >=)

// F16 and BF16 have no integer conversion, so 1/0 is built from a float. BF16
// comparison intrinsics need SM 8.0+; below that the operands round-trip
// through F32, which is what the pre-Ampere path has always done.
#if __CUDA_ARCH__ >= 800
#define NUMR_BF16_CMP(INTRIN, OP, a, b) INTRIN(a, b)
#else
#define NUMR_BF16_CMP(INTRIN, OP, a, b) (__bfloat162float(a) OP __bfloat162float(b))
#endif

#define NUMR_CMP_HALF_SPEC(NAME, INTRIN, OP)                                    \
    template<>                                                                  \
    __device__ __forceinline__ __half compare_##NAME(__half a, __half b) {      \
        return INTRIN(a, b) ? __float2half(1.0f) : __float2half(0.0f);          \
    }                                                                           \
    template<>                                                                  \
    __device__ __forceinline__ __nv_bfloat16                                    \
    compare_##NAME(__nv_bfloat16 a, __nv_bfloat16 b) {                          \
        return NUMR_BF16_CMP(INTRIN, OP, a, b) ? __float2bfloat16(1.0f)         \
                                               : __float2bfloat16(0.0f);        \
    }

NUMR_CMP_HALF_SPEC(eq, __heq, ==)
NUMR_CMP_HALF_SPEC(ne, __hne, !=)
NUMR_CMP_HALF_SPEC(lt, __hlt, <)
NUMR_CMP_HALF_SPEC(le, __hle, <=)
NUMR_CMP_HALF_SPEC(gt, __hgt, >)
NUMR_CMP_HALF_SPEC(ge, __hge, >=)

// FP8 compares the decoded F32 values: the raw byte order is not the value
// order. The 1/0 result is re-encoded, both of which are exact in FP8.
#define NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, NAME, OP)                        \
    template<>                                                                  \
    __device__ __forceinline__ T compare_##NAME(T a, T b) {                     \
        return T(FROM_F32((TO_F32(a.data) OP TO_F32(b.data)) ? 1.0f : 0.0f));   \
    }

#define NUMR_CMP_FP8_ALL(T, TO_F32, FROM_F32)                                   \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, eq, ==)                              \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, ne, !=)                              \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, lt, <)                               \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, le, <=)                              \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, gt, >)                               \
    NUMR_CMP_FP8_SPEC(T, TO_F32, FROM_F32, ge, >=)

NUMR_CMP_FP8_ALL(numr_fp8_e4m3, fp8_e4m3_to_f32, f32_to_fp8_e4m3)
NUMR_CMP_FP8_ALL(numr_fp8_e5m2, fp8_e5m2_to_f32, f32_to_fp8_e5m2)

// ============================================================================
// Kernel body templates
// ============================================================================

template<typename T, typename CompFunc>
__device__ __forceinline__ void compare_elementwise_impl(
    const T* a, const T* b, T* out, unsigned int n, CompFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = op(a[idx], b[idx]);
    }
}

template<typename T, typename CompFunc>
__device__ void compare_broadcast_kernel_impl(
    const T* a, const T* b, T* out,
    const unsigned int* a_strides, const unsigned int* b_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int n,
    CompFunc op
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    unsigned int remaining = idx;
    unsigned int a_offset = 0, b_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        a_offset += coord * a_strides[d];
        b_offset += coord * b_strides[d];
    }

    out[idx] = op(a[a_offset], b[b_offset]);
}

// ============================================================================
// Instantiation macros
// ============================================================================
// One (operation, dtype) pair emits the element-wise kernel and the broadcast
// kernel. out_strides is unused by the broadcast body but stays in the
// signature: the host passes it.

#define NUMR_COMPARE_OP(T, S, OP)                                               \
    __global__ void OP##_##S(const T* a, const T* b, T* out, unsigned int n) {  \
        compare_elementwise_impl<T>(a, b, out, n, compare_##OP<T>);             \
    }                                                                           \
    __global__ void OP##_broadcast_##S(                                         \
        const T* a, const T* b, T* out,                                         \
        const unsigned int* a_strides, const unsigned int* b_strides,           \
        const unsigned int* shape, unsigned int ndim, unsigned int n) {         \
        compare_broadcast_kernel_impl<T>(a, b, out, a_strides, b_strides,       \
                                         shape, ndim, n, compare_##OP<T>);      \
    }

#define NUMR_COMPARE_ROW(T, S)                                                  \
    NUMR_COMPARE_OP(T, S, eq) NUMR_COMPARE_OP(T, S, ne)                         \
    NUMR_COMPARE_OP(T, S, lt) NUMR_COMPARE_OP(T, S, le)                         \
    NUMR_COMPARE_OP(T, S, gt) NUMR_COMPARE_OP(T, S, ge)

extern "C" {

NUMR_COMPARE_ROW(float, f32)
NUMR_COMPARE_ROW(double, f64)
NUMR_COMPARE_ROW(__half, f16)
NUMR_COMPARE_ROW(__nv_bfloat16, bf16)
NUMR_COMPARE_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_COMPARE_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_COMPARE_ROW(int64_t, i64)
NUMR_COMPARE_ROW(int32_t, i32)
NUMR_COMPARE_ROW(int16_t, i16)
NUMR_COMPARE_ROW(int8_t, i8)
NUMR_COMPARE_ROW(uint64_t, u64)
NUMR_COMPARE_ROW(uint32_t, u32)
NUMR_COMPARE_ROW(uint16_t, u16)
NUMR_COMPARE_ROW(uint8_t, u8)

} // extern "C"
