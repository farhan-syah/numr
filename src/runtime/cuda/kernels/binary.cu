// Binary element-wise CUDA kernels
//
// Operations: add, sub, mul, div, pow, max, min (every dtype below), plus
// atan2 (float dtypes), logical_and/or/xor (u8), and complex add/sub/mul/div.
//
// Dtypes: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
//         i64, i32, i16, i8, u64, u32, u16, u8
//
// Kernel naming, matching the names the Rust launchers build in
// src/runtime/cuda/kernels/binary.rs from dtype_suffix() in loader.rs:
//   {op}_{suffix}                          element-wise, same-shape operands
//   {op}_broadcast_{suffix}                broadcast, strides in device memory
//   {op}_broadcast_{suffix}_inline         broadcast, strides as scalar args
//   {op}_broadcast_fast_trailing_{suffix}  contiguous trailing broadcast
//
// The operation bodies, the four kernel-body templates, and the row macro all
// live in binary_ops.cuh, which also documents the integer wrapping,
// division-by-zero, and pow semantics.

#include "binary_ops.cuh"

// atan2 is elementwise-only and float-only, so it stays outside the row macro.
template<typename T> __device__ __forceinline__ T binop_atan2(T y, T x) {
    return atan2f(y, x);
}
template<> __device__ __forceinline__ double binop_atan2(double y, double x) {
    return atan2(y, x);
}
template<> __device__ __forceinline__ __half binop_atan2(__half y, __half x) {
    return __float2half(atan2f(__half2float(y), __half2float(x)));
}
template<> __device__ __forceinline__ __nv_bfloat16 binop_atan2(__nv_bfloat16 y, __nv_bfloat16 x) {
    return __float2bfloat16(atan2f(__bfloat162float(y), __bfloat162float(x)));
}
template<> __device__ __forceinline__ numr_fp8_e4m3 binop_atan2(numr_fp8_e4m3 y, numr_fp8_e4m3 x) {
    return numr_fp8_e4m3(f32_to_fp8_e4m3(atan2f(fp8_e4m3_to_f32(y.data), fp8_e4m3_to_f32(x.data))));
}
template<> __device__ __forceinline__ numr_fp8_e5m2 binop_atan2(numr_fp8_e5m2 y, numr_fp8_e5m2 x) {
    return numr_fp8_e5m2(f32_to_fp8_e5m2(atan2f(fp8_e5m2_to_f32(y.data), fp8_e5m2_to_f32(x.data))));
}

#define NUMR_ATAN2_KERNEL(T, S)                                                 \
    __global__ void atan2_##S(const T* y, const T* x, T* out, unsigned int n) { \
        binary_elementwise_impl<T>(y, x, out, n, binop_atan2<T>);               \
    }

extern "C" {

// ============================================================================
// Arithmetic: 7 operations x 4 kernel variants, one row per dtype
// ============================================================================

NUMR_BINARY_ROW(float, f32)
NUMR_BINARY_ROW(double, f64)
NUMR_BINARY_ROW(__half, f16)
NUMR_BINARY_ROW(__nv_bfloat16, bf16)
NUMR_BINARY_ROW(numr_fp8_e4m3, fp8_e4m3)
NUMR_BINARY_ROW(numr_fp8_e5m2, fp8_e5m2)
NUMR_BINARY_ROW(int64_t, i64)
NUMR_BINARY_ROW(int32_t, i32)
NUMR_BINARY_ROW(int16_t, i16)
NUMR_BINARY_ROW(int8_t, i8)
NUMR_BINARY_ROW(uint64_t, u64)
NUMR_BINARY_ROW(uint32_t, u32)
NUMR_BINARY_ROW(uint16_t, u16)
NUMR_BINARY_ROW(uint8_t, u8)

// ============================================================================
// atan2 (float dtypes only)
// ============================================================================

NUMR_ATAN2_KERNEL(float, f32)
NUMR_ATAN2_KERNEL(double, f64)
NUMR_ATAN2_KERNEL(__half, f16)
NUMR_ATAN2_KERNEL(__nv_bfloat16, bf16)
NUMR_ATAN2_KERNEL(numr_fp8_e4m3, fp8_e4m3)
NUMR_ATAN2_KERNEL(numr_fp8_e5m2, fp8_e5m2)

// ============================================================================
// Logical operations (input and output are u8, one byte per boolean)
// ============================================================================

__global__ void logical_and_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] && b[idx]) ? 1 : 0;
    }
}

__global__ void logical_or_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = (a[idx] || b[idx]) ? 1 : 0;
    }
}

__global__ void logical_xor_u8(const unsigned char* a, const unsigned char* b, unsigned char* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = ((a[idx] != 0) != (b[idx] != 0)) ? 1 : 0;
    }
}

// ============================================================================
// Complex operations
//
// Complex arithmetic is not the same expression as the real one — mul and div
// mix the components — so these keep their own helpers from dtype_traits.cuh
// instead of joining the row macro. There is no complex max/min/pow kernel.
// ============================================================================

// S is the kernel-name suffix (c64/c128); W is the width in the helper names
// (complex64_add / complex128_add).
#define NUMR_COMPLEX_BINARY(T, S, W, OP)                                        \
    __global__ void OP##_##S(const T* a, const T* b, T* out, unsigned int n) {  \
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (idx < n) {                                                          \
            out[idx] = complex##W##_##OP(a[idx], b[idx]);                       \
        }                                                                       \
    }

#define NUMR_COMPLEX_ROW(T, S, W)                                               \
    NUMR_COMPLEX_BINARY(T, S, W, add) NUMR_COMPLEX_BINARY(T, S, W, sub)         \
    NUMR_COMPLEX_BINARY(T, S, W, mul) NUMR_COMPLEX_BINARY(T, S, W, div)

NUMR_COMPLEX_ROW(float2, c64, 64)
NUMR_COMPLEX_ROW(double2, c128, 128)

} // extern "C"
