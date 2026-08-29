// Ternary CUDA kernels
//
// Operation: where (conditional select) — where(cond, x, y) = cond ? x : y
//
// Value dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//               i64, i32, i16, i8, u64, u32, u16, u8
//
// Kernel naming, matching the names the Rust launchers build in
// src/runtime/cuda/kernels/ternary.rs from dtype_suffix() in loader.rs:
//   where_{suffix}                          U8 condition, same-shape operands
//   where_broadcast_{suffix}                U8 condition, strides in device memory
//   where_cond_{cond}_{out}                 non-U8 condition, same shape
//   where_broadcast_cond_{cond}_{out}       non-U8 condition, broadcast
//
// A U8 condition tests the byte directly; every other condition dtype tests
// "not zero", which for a float means the numeric value and not the bit
// pattern (-0.0 is false). Selection copies an operand through unchanged, so
// unlike the arithmetic kernels there is nothing here that can overflow: one
// template covers every dtype, integers included.
//
// The non-U8 condition pairs are an explicit list, not a cross product. A pair
// with no instantiation is reported as an unsupported CONDITION dtype by
// launch_where_generic_op, which is why the list is spelled out rather than
// generated.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"

// ============================================================================
// Non-zero check per condition dtype
// ============================================================================
// Declared without a body: a condition dtype with no specialization fails to
// link rather than falling back to a wrong comparison.

template<typename C>
__device__ __forceinline__ bool is_nonzero(C val);

#define NUMR_IS_NONZERO_INT(C)                                                  \
    template<>                                                                  \
    __device__ __forceinline__ bool is_nonzero<C>(C val) { return val != (C)0; }

NUMR_IS_NONZERO_INT(int8_t)
NUMR_IS_NONZERO_INT(int16_t)
NUMR_IS_NONZERO_INT(int32_t)
NUMR_IS_NONZERO_INT(int64_t)
NUMR_IS_NONZERO_INT(uint8_t)
NUMR_IS_NONZERO_INT(uint16_t)
NUMR_IS_NONZERO_INT(uint32_t)
NUMR_IS_NONZERO_INT(uint64_t)

#undef NUMR_IS_NONZERO_INT

template<>
__device__ __forceinline__ bool is_nonzero<float>(float val) {
    return val != 0.0f;
}

template<>
__device__ __forceinline__ bool is_nonzero<double>(double val) {
    return val != 0.0;
}

// F16, BF16 and FP8 decode to F32 first: the raw bits of -0.0 are non-zero, but
// the value is not.
template<>
__device__ __forceinline__ bool is_nonzero<__half>(__half val) {
    return __half2float(val) != 0.0f;
}

template<>
__device__ __forceinline__ bool is_nonzero<__nv_bfloat16>(__nv_bfloat16 val) {
    return __bfloat162float(val) != 0.0f;
}

template<>
__device__ __forceinline__ bool is_nonzero<numr_fp8_e4m3>(numr_fp8_e4m3 val) {
    return fp8_e4m3_to_f32(val.data) != 0.0f;
}

template<>
__device__ __forceinline__ bool is_nonzero<numr_fp8_e5m2>(numr_fp8_e5m2 val) {
    return fp8_e5m2_to_f32(val.data) != 0.0f;
}

// ============================================================================
// Kernel body templates
// ============================================================================

// Generic where with any condition type
template<typename C, typename T>
__device__ __forceinline__ T where_impl_generic(C cond, T x, T y) {
    return is_nonzero(cond) ? x : y;
}

// Optimized where for u8 condition (backward compatible)
template<typename T>
__device__ __forceinline__ T where_impl(unsigned char cond, T x, T y) {
    return cond ? x : y;
}

// Broadcasting walk shared by both condition forms: strides and shape live in
// device memory, one linear index becomes three operand offsets.
template<typename C, typename T, typename SelFunc>
__device__ void where_broadcast_walk(
    const C* cond, const T* x, const T* y, T* out,
    const unsigned int* cond_strides, const unsigned int* x_strides, const unsigned int* y_strides,
    const unsigned int* shape, unsigned int ndim, unsigned int idx,
    SelFunc sel
) {
    unsigned int remaining = idx;
    unsigned int cond_offset = 0;
    unsigned int x_offset = 0;
    unsigned int y_offset = 0;

    for (int d = ndim - 1; d >= 0; d--) {
        unsigned int coord = remaining % shape[d];
        remaining /= shape[d];
        cond_offset += coord * cond_strides[d];
        x_offset += coord * x_strides[d];
        y_offset += coord * y_strides[d];
    }

    out[idx] = sel(cond[cond_offset], x[x_offset], y[y_offset]);
}

// ============================================================================
// Kernel-parameter macros
// ============================================================================

#define WHERE_BROADCAST_ARGS(C, T)                                              \
    const C* cond, const T* x, const T* y, T* out,                              \
    const unsigned int* cond_strides, const unsigned int* x_strides,            \
    const unsigned int* y_strides,                                              \
    const unsigned int* shape, unsigned int ndim, unsigned int n

#define WHERE_BROADCAST_CALL                                                    \
    cond, x, y, out, cond_strides, x_strides, y_strides, shape, ndim, idx

// ============================================================================
// Instantiation macros
// ============================================================================

// One value dtype with a U8 condition: the element-wise kernel and the
// broadcast kernel.
#define NUMR_WHERE_ROW(T, SUF)                                                  \
    __global__ void where_##SUF(                                                \
        const unsigned char* cond, const T* x, const T* y,                      \
        T* out, unsigned int n) {                                               \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = where_impl<T>(cond[idx], x[idx], y[idx]);                \
        }                                                                       \
    }                                                                           \
    __global__ void where_broadcast_##SUF(                                      \
        WHERE_BROADCAST_ARGS(unsigned char, T)) {                               \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            where_broadcast_walk<unsigned char, T>(                             \
                WHERE_BROADCAST_CALL, where_impl<T>);                           \
        }                                                                       \
    }

// One (condition dtype, value dtype) pair, element-wise.
#define NUMR_WHERE_COND(C, T, CSUF, TSUF)                                       \
    __global__ void where_cond_##CSUF##_##TSUF(                                 \
        const C* cond, const T* x, const T* y, T* out, unsigned int n) {        \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = where_impl_generic<C, T>(cond[idx], x[idx], y[idx]);     \
        }                                                                       \
    }

// One (condition dtype, value dtype) pair, broadcast.
#define NUMR_WHERE_COND_BROADCAST(C, T, CSUF, TSUF)                             \
    __global__ void where_broadcast_cond_##CSUF##_##TSUF(                       \
        WHERE_BROADCAST_ARGS(C, T)) {                                           \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            where_broadcast_walk<C, T>(                                         \
                WHERE_BROADCAST_CALL, where_impl_generic<C, T>);                \
        }                                                                       \
    }

extern "C" {

// ============================================================================
// U8 condition: one row per value dtype
// ============================================================================

NUMR_WHERE_ROW(float, f32)
NUMR_WHERE_ROW(double, f64)
NUMR_WHERE_ROW(__half, f16)
NUMR_WHERE_ROW(__nv_bfloat16, bf16)
NUMR_WHERE_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_WHERE_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_WHERE_ROW(int64_t, i64)
NUMR_WHERE_ROW(int32_t, i32)
NUMR_WHERE_ROW(int16_t, i16)
NUMR_WHERE_ROW(int8_t, i8)
NUMR_WHERE_ROW(uint64_t, u64)
NUMR_WHERE_ROW(uint32_t, u32)
NUMR_WHERE_ROW(uint16_t, u16)
NUMR_WHERE_ROW(uint8_t, u8)

// ============================================================================
// Non-U8 condition, element-wise
// ============================================================================

NUMR_WHERE_COND(float, float, f32, f32)
NUMR_WHERE_COND(double, double, f64, f64)
NUMR_WHERE_COND(float, double, f32, f64)
NUMR_WHERE_COND(double, float, f64, f32)
NUMR_WHERE_COND(int32_t, float, i32, f32)
NUMR_WHERE_COND(int32_t, double, i32, f64)
NUMR_WHERE_COND(int64_t, float, i64, f32)
NUMR_WHERE_COND(int64_t, double, i64, f64)
NUMR_WHERE_COND(uint32_t, float, u32, f32)
NUMR_WHERE_COND(uint32_t, double, u32, f64)
NUMR_WHERE_COND(__half, __half, f16, f16)
NUMR_WHERE_COND(__half, float, f16, f32)
NUMR_WHERE_COND(__half, double, f16, f64)
NUMR_WHERE_COND(__nv_bfloat16, __nv_bfloat16, bf16, bf16)
NUMR_WHERE_COND(__nv_bfloat16, float, bf16, f32)
NUMR_WHERE_COND(__nv_bfloat16, double, bf16, f64)
NUMR_WHERE_COND(numr_fp8_e4m3, numr_fp8_e4m3, fp8_e4m3, fp8_e4m3)
NUMR_WHERE_COND(numr_fp8_e5m2, numr_fp8_e5m2, fp8_e5m2, fp8_e5m2)

// ============================================================================
// Non-U8 condition, broadcast
// ============================================================================

NUMR_WHERE_COND_BROADCAST(float, float, f32, f32)
NUMR_WHERE_COND_BROADCAST(double, double, f64, f64)
NUMR_WHERE_COND_BROADCAST(float, double, f32, f64)
NUMR_WHERE_COND_BROADCAST(double, float, f64, f32)
NUMR_WHERE_COND_BROADCAST(int32_t, float, i32, f32)
NUMR_WHERE_COND_BROADCAST(int32_t, double, i32, f64)
NUMR_WHERE_COND_BROADCAST(int64_t, float, i64, f32)
NUMR_WHERE_COND_BROADCAST(int64_t, double, i64, f64)
NUMR_WHERE_COND_BROADCAST(uint32_t, float, u32, f32)
NUMR_WHERE_COND_BROADCAST(uint32_t, double, u32, f64)

} // extern "C"
