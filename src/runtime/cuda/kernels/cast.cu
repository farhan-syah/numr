// Type casting CUDA kernels
// Covers every ordered pair of: f32, f64, f16, bf16, fp8_e4m3, fp8_e5m2,
// i64, i32, i16, i8, u64, u32, u16, u8, bool
//
// Kernel naming: cast_{src_dtype}_{dst_dtype}
// Example: cast_f32_f16 converts from f32 to f16
//
// Identity pairs (cast_f32_f32 and friends) are NOT emitted: launch_cast()
// returns early when src_dtype == dst_dtype.
//
// Numerical semantics mirror the CPU reference in
// src/runtime/cpu/kernels/memory.rs, which funnels EVERY conversion through
// f64 and then applies a Rust `as` cast. Rust's `as` to an integer saturates:
// NaN becomes 0, out-of-range clamps to the nearest bound. Because the CPU path
// goes through f64, integer -> integer saturates as well (it does NOT wrap).

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "dtype_traits.cuh"
#include "narrow_f64.cuh"

// Bool is stored as one byte, exactly like U8. A distinct C++ type keeps the two
// apart in the templates below: U8 converts numerically, Bool collapses to 0/1.
struct numr_bool {
    unsigned char data;
};

// ============================================================================
// Source -> f64 (mirrors Element::to_f64 in src/dtype/element.rs)
// ============================================================================

__device__ __forceinline__ double cast_to_f64(float v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(double v) { return v; }
__device__ __forceinline__ double cast_to_f64(__half v) { return (double)__half2float(v); }
__device__ __forceinline__ double cast_to_f64(__nv_bfloat16 v) { return (double)__bfloat162float(v); }
__device__ __forceinline__ double cast_to_f64(numr_fp8_e4m3 v) { return (double)fp8_e4m3_to_f32(v.data); }
__device__ __forceinline__ double cast_to_f64(numr_fp8_e5m2 v) { return (double)fp8_e5m2_to_f32(v.data); }
__device__ __forceinline__ double cast_to_f64(signed char v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(short v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(int v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(long long v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(unsigned char v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(unsigned short v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(unsigned int v) { return (double)v; }
__device__ __forceinline__ double cast_to_f64(unsigned long long v) { return (double)v; }
// A Bool source reads the raw byte: the CPU path dispatches Bool through u8.
__device__ __forceinline__ double cast_to_f64(numr_bool v) { return (double)v.data; }

// ============================================================================
// Saturating f64 -> integer (mirrors Rust's `as` cast)
// ============================================================================
// hi_excl() is the smallest double strictly above every representable value, so
// the comparison is exact even where the maximum itself is not representable
// (i64, u64).

template <typename T> struct numr_int_range;

#define NUMR_INT_RANGE(T, LO_BOUND, HI_EXCL, LO_VALUE, HI_VALUE)               \
    template <> struct numr_int_range<T> {                                     \
        __device__ __forceinline__ static double lo_bound() { return LO_BOUND; } \
        __device__ __forceinline__ static double hi_excl() { return HI_EXCL; } \
        __device__ __forceinline__ static T lo_value() { return LO_VALUE; }    \
        __device__ __forceinline__ static T hi_value() { return HI_VALUE; }    \
    };

NUMR_INT_RANGE(signed char, -128.0, 128.0, (signed char)(-128), (signed char)127)
NUMR_INT_RANGE(short, -32768.0, 32768.0, (short)(-32768), (short)32767)
NUMR_INT_RANGE(int, -2147483648.0, 2147483648.0, (-2147483647 - 1), 2147483647)
NUMR_INT_RANGE(long long, -9223372036854775808.0, 9223372036854775808.0,
               (-9223372036854775807LL - 1), 9223372036854775807LL)
NUMR_INT_RANGE(unsigned char, 0.0, 256.0, (unsigned char)0, (unsigned char)255)
NUMR_INT_RANGE(unsigned short, 0.0, 65536.0, (unsigned short)0, (unsigned short)65535)
NUMR_INT_RANGE(unsigned int, 0.0, 4294967296.0, 0u, 4294967295u)
NUMR_INT_RANGE(unsigned long long, 0.0, 18446744073709551616.0, 0ull, 18446744073709551615ull)

#undef NUMR_INT_RANGE

template <typename T>
__device__ __forceinline__ T numr_sat_f64(double d) {
    if (d != d) return (T)0;                                    // Rust `as`: NaN -> 0
    if (d <= numr_int_range<T>::lo_bound()) return numr_int_range<T>::lo_value();
    if (d >= numr_int_range<T>::hi_excl()) return numr_int_range<T>::hi_value();
    return (T)d;                                                // in range: truncates toward zero
}

// ============================================================================
// f64 -> destination (mirrors the cast_from! macro in cpu/kernels/memory.rs)
// ============================================================================

template <typename Dst> __device__ __forceinline__ Dst cast_from_f64(double d);

template <> __device__ __forceinline__ double cast_from_f64<double>(double d) { return d; }
template <> __device__ __forceinline__ float cast_from_f64<float>(double d) { return (float)d; }
// f16/bf16 defer to narrow_f64.cuh: half::f16::from_f64 stages through f32 on
// x86-64 with F16C and half::bf16::from_f64 runs its own software rounding, so
// neither is __double2half/__double2bfloat16. Read that header before editing.
template <> __device__ __forceinline__ __half cast_from_f64<__half>(double d) { return numr_f64_to_f16(d); }
template <> __device__ __forceinline__ __nv_bfloat16 cast_from_f64<__nv_bfloat16>(double d) { return numr_f64_to_bf16(d); }
// FP8 rounds via f32, matching FP8E4M3::from_f64 / FP8E5M2::from_f64.
template <> __device__ __forceinline__ numr_fp8_e4m3 cast_from_f64<numr_fp8_e4m3>(double d) {
    return numr_fp8_e4m3(f32_to_fp8_e4m3((float)d));
}
template <> __device__ __forceinline__ numr_fp8_e5m2 cast_from_f64<numr_fp8_e5m2>(double d) {
    return numr_fp8_e5m2(f32_to_fp8_e5m2((float)d));
}
template <> __device__ __forceinline__ signed char cast_from_f64<signed char>(double d) { return numr_sat_f64<signed char>(d); }
template <> __device__ __forceinline__ short cast_from_f64<short>(double d) { return numr_sat_f64<short>(d); }
template <> __device__ __forceinline__ int cast_from_f64<int>(double d) { return numr_sat_f64<int>(d); }
template <> __device__ __forceinline__ long long cast_from_f64<long long>(double d) { return numr_sat_f64<long long>(d); }
template <> __device__ __forceinline__ unsigned char cast_from_f64<unsigned char>(double d) { return numr_sat_f64<unsigned char>(d); }
template <> __device__ __forceinline__ unsigned short cast_from_f64<unsigned short>(double d) { return numr_sat_f64<unsigned short>(d); }
template <> __device__ __forceinline__ unsigned int cast_from_f64<unsigned int>(double d) { return numr_sat_f64<unsigned int>(d); }
template <> __device__ __forceinline__ unsigned long long cast_from_f64<unsigned long long>(double d) { return numr_sat_f64<unsigned long long>(d); }
// Bool destination: nonzero (NaN included) is true, matching `to_f64() != 0.0`.
template <> __device__ __forceinline__ numr_bool cast_from_f64<numr_bool>(double d) {
    numr_bool out;
    out.data = (d != 0.0) ? 1 : 0;
    return out;
}

// ============================================================================
// Kernel template
// ============================================================================

template <typename Src, typename Dst>
__device__ __forceinline__ Dst cast_one(Src v) {
    return cast_from_f64<Dst>(cast_to_f64(v));
}

template <typename Src, typename Dst>
__device__ __forceinline__ void cast_impl(const Src* a, Dst* out, unsigned int n) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = cast_one<Src, Dst>(a[idx]);
    }
}

// ============================================================================
// Instantiation matrix
// ============================================================================
// One cell macro per destination dtype, one row macro invocation per source
// dtype. Each row temporarily blanks its own cell so the identity pair is not
// emitted.

#define NUMR_CAST_DEF(SRC_T, S, DST_T, D)                                       \
__global__ void cast_##S##_##D(const SRC_T* a, DST_T* out, unsigned int n) {    \
    cast_impl<SRC_T, DST_T>(a, out, n);                                         \
}

#define NUMR_CELL_f32(ST, S) NUMR_CAST_DEF(ST, S, float, f32)
#define NUMR_CELL_f64(ST, S) NUMR_CAST_DEF(ST, S, double, f64)
#define NUMR_CELL_f16(ST, S) NUMR_CAST_DEF(ST, S, __half, f16)
#define NUMR_CELL_bf16(ST, S) NUMR_CAST_DEF(ST, S, __nv_bfloat16, bf16)
#define NUMR_CELL_fp8_e4m3(ST, S) NUMR_CAST_DEF(ST, S, numr_fp8_e4m3, fp8_e4m3)
#define NUMR_CELL_fp8_e5m2(ST, S) NUMR_CAST_DEF(ST, S, numr_fp8_e5m2, fp8_e5m2)
#define NUMR_CELL_i64(ST, S) NUMR_CAST_DEF(ST, S, long long, i64)
#define NUMR_CELL_i32(ST, S) NUMR_CAST_DEF(ST, S, int, i32)
#define NUMR_CELL_i16(ST, S) NUMR_CAST_DEF(ST, S, short, i16)
#define NUMR_CELL_i8(ST, S) NUMR_CAST_DEF(ST, S, signed char, i8)
#define NUMR_CELL_u64(ST, S) NUMR_CAST_DEF(ST, S, unsigned long long, u64)
#define NUMR_CELL_u32(ST, S) NUMR_CAST_DEF(ST, S, unsigned int, u32)
#define NUMR_CELL_u16(ST, S) NUMR_CAST_DEF(ST, S, unsigned short, u16)
#define NUMR_CELL_u8(ST, S) NUMR_CAST_DEF(ST, S, unsigned char, u8)
#define NUMR_CELL_bool(ST, S) NUMR_CAST_DEF(ST, S, numr_bool, bool)

#define NUMR_CAST_ROW(ST, S)                                                    \
    NUMR_CELL_f32(ST, S) NUMR_CELL_f64(ST, S) NUMR_CELL_f16(ST, S)              \
    NUMR_CELL_bf16(ST, S) NUMR_CELL_fp8_e4m3(ST, S) NUMR_CELL_fp8_e5m2(ST, S)   \
    NUMR_CELL_i64(ST, S) NUMR_CELL_i32(ST, S) NUMR_CELL_i16(ST, S)              \
    NUMR_CELL_i8(ST, S) NUMR_CELL_u64(ST, S) NUMR_CELL_u32(ST, S)               \
    NUMR_CELL_u16(ST, S) NUMR_CELL_u8(ST, S) NUMR_CELL_bool(ST, S)

extern "C" {

#undef NUMR_CELL_f32
#define NUMR_CELL_f32(ST, S)
NUMR_CAST_ROW(float, f32)
#undef NUMR_CELL_f32
#define NUMR_CELL_f32(ST, S) NUMR_CAST_DEF(ST, S, float, f32)

#undef NUMR_CELL_f64
#define NUMR_CELL_f64(ST, S)
NUMR_CAST_ROW(double, f64)
#undef NUMR_CELL_f64
#define NUMR_CELL_f64(ST, S) NUMR_CAST_DEF(ST, S, double, f64)

#undef NUMR_CELL_f16
#define NUMR_CELL_f16(ST, S)
NUMR_CAST_ROW(__half, f16)
#undef NUMR_CELL_f16
#define NUMR_CELL_f16(ST, S) NUMR_CAST_DEF(ST, S, __half, f16)

#undef NUMR_CELL_bf16
#define NUMR_CELL_bf16(ST, S)
NUMR_CAST_ROW(__nv_bfloat16, bf16)
#undef NUMR_CELL_bf16
#define NUMR_CELL_bf16(ST, S) NUMR_CAST_DEF(ST, S, __nv_bfloat16, bf16)

#undef NUMR_CELL_fp8_e4m3
#define NUMR_CELL_fp8_e4m3(ST, S)
NUMR_CAST_ROW(numr_fp8_e4m3, fp8_e4m3)
#undef NUMR_CELL_fp8_e4m3
#define NUMR_CELL_fp8_e4m3(ST, S) NUMR_CAST_DEF(ST, S, numr_fp8_e4m3, fp8_e4m3)

#undef NUMR_CELL_fp8_e5m2
#define NUMR_CELL_fp8_e5m2(ST, S)
NUMR_CAST_ROW(numr_fp8_e5m2, fp8_e5m2)
#undef NUMR_CELL_fp8_e5m2
#define NUMR_CELL_fp8_e5m2(ST, S) NUMR_CAST_DEF(ST, S, numr_fp8_e5m2, fp8_e5m2)

#undef NUMR_CELL_i64
#define NUMR_CELL_i64(ST, S)
NUMR_CAST_ROW(long long, i64)
#undef NUMR_CELL_i64
#define NUMR_CELL_i64(ST, S) NUMR_CAST_DEF(ST, S, long long, i64)

#undef NUMR_CELL_i32
#define NUMR_CELL_i32(ST, S)
NUMR_CAST_ROW(int, i32)
#undef NUMR_CELL_i32
#define NUMR_CELL_i32(ST, S) NUMR_CAST_DEF(ST, S, int, i32)

#undef NUMR_CELL_i16
#define NUMR_CELL_i16(ST, S)
NUMR_CAST_ROW(short, i16)
#undef NUMR_CELL_i16
#define NUMR_CELL_i16(ST, S) NUMR_CAST_DEF(ST, S, short, i16)

#undef NUMR_CELL_i8
#define NUMR_CELL_i8(ST, S)
NUMR_CAST_ROW(signed char, i8)
#undef NUMR_CELL_i8
#define NUMR_CELL_i8(ST, S) NUMR_CAST_DEF(ST, S, signed char, i8)

#undef NUMR_CELL_u64
#define NUMR_CELL_u64(ST, S)
NUMR_CAST_ROW(unsigned long long, u64)
#undef NUMR_CELL_u64
#define NUMR_CELL_u64(ST, S) NUMR_CAST_DEF(ST, S, unsigned long long, u64)

#undef NUMR_CELL_u32
#define NUMR_CELL_u32(ST, S)
NUMR_CAST_ROW(unsigned int, u32)
#undef NUMR_CELL_u32
#define NUMR_CELL_u32(ST, S) NUMR_CAST_DEF(ST, S, unsigned int, u32)

#undef NUMR_CELL_u16
#define NUMR_CELL_u16(ST, S)
NUMR_CAST_ROW(unsigned short, u16)
#undef NUMR_CELL_u16
#define NUMR_CELL_u16(ST, S) NUMR_CAST_DEF(ST, S, unsigned short, u16)

#undef NUMR_CELL_u8
#define NUMR_CELL_u8(ST, S)
NUMR_CAST_ROW(unsigned char, u8)
#undef NUMR_CELL_u8
#define NUMR_CELL_u8(ST, S) NUMR_CAST_DEF(ST, S, unsigned char, u8)

#undef NUMR_CELL_bool
#define NUMR_CELL_bool(ST, S)
NUMR_CAST_ROW(numr_bool, bool)
#undef NUMR_CELL_bool
#define NUMR_CELL_bool(ST, S) NUMR_CAST_DEF(ST, S, numr_bool, bool)

} // extern "C"
