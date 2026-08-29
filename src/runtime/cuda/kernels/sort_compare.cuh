// Comparison and padding primitives shared by every sorting and searching
// kernel: the total order, the bitonic compare-and-swap, and the padding value
// that fills a sort buffer out to a power of two.
//
// Included by sort_bitonic.cuh and sort_scan.cuh, which sort.cu instantiates.

#ifndef NUMR_SORT_COMPARE_CUH
#define NUMR_SORT_COMPARE_CUH

#include <cuda_runtime.h>
#include <climits>
#include "dtype_traits.cuh"

// ============================================================================
// FP8 comparison operators
// ============================================================================
// FP8 is a byte struct with no arithmetic of its own, so the templates below
// need these before they can compare it.

#define NUMR_FP8_COMPARE_OPS(T, TO_F32)                                         \
    __device__ __forceinline__ bool operator<(T a, T b) {                       \
        return TO_F32(a.data) < TO_F32(b.data);                                 \
    }                                                                           \
    __device__ __forceinline__ bool operator>(T a, T b) {                       \
        return TO_F32(a.data) > TO_F32(b.data);                                 \
    }                                                                           \
    __device__ __forceinline__ bool operator==(T a, T b) {                      \
        return TO_F32(a.data) == TO_F32(b.data);                                \
    }                                                                           \
    __device__ __forceinline__ bool operator!=(T a, T b) {                      \
        return TO_F32(a.data) != TO_F32(b.data);                                \
    }

NUMR_FP8_COMPARE_OPS(numr_fp8_e4m3, fp8_e4m3_to_f32)
NUMR_FP8_COMPARE_OPS(numr_fp8_e5m2, fp8_e5m2_to_f32)

#undef NUMR_FP8_COMPARE_OPS

// ============================================================================
// Sort padding values
// ============================================================================

template<typename T> __device__ __forceinline__ T sort_pad_max();
template<typename T> __device__ __forceinline__ T sort_pad_min();

#define NUMR_SORT_PAD(T, MAX_EXPR, MIN_EXPR)                                    \
    template<> __device__ __forceinline__ T sort_pad_max<T>() { return MAX_EXPR; } \
    template<> __device__ __forceinline__ T sort_pad_min<T>() { return MIN_EXPR; }

NUMR_SORT_PAD(float, 1e38f, -1e38f)
NUMR_SORT_PAD(double, 1e308, -1e308)
NUMR_SORT_PAD(long long, LLONG_MAX, LLONG_MIN)
NUMR_SORT_PAD(int, INT_MAX, INT_MIN)
NUMR_SORT_PAD(short, SHRT_MAX, SHRT_MIN)
NUMR_SORT_PAD(signed char, SCHAR_MAX, SCHAR_MIN)
NUMR_SORT_PAD(unsigned long long, ULLONG_MAX, 0ull)
NUMR_SORT_PAD(unsigned int, UINT_MAX, 0u)
NUMR_SORT_PAD(unsigned short, USHRT_MAX, 0)
NUMR_SORT_PAD(unsigned char, UCHAR_MAX, 0)
NUMR_SORT_PAD(__half, __float2half(65504.0f), __float2half(-65504.0f))
NUMR_SORT_PAD(__nv_bfloat16, __float2bfloat16(1e38f), __float2bfloat16(-1e38f))
NUMR_SORT_PAD(numr_fp8_e4m3, numr_fp8_e4m3(f32_to_fp8_e4m3(FP8_E4M3_MAX)),
                             numr_fp8_e4m3(f32_to_fp8_e4m3(FP8_E4M3_MIN)))
NUMR_SORT_PAD(numr_fp8_e5m2, numr_fp8_e5m2(f32_to_fp8_e5m2(FP8_E5M2_MAX)),
                             numr_fp8_e5m2(f32_to_fp8_e5m2(FP8_E5M2_MIN)))

#undef NUMR_SORT_PAD

// ============================================================================
// Total order
// ============================================================================

// NaN detection. Integer types have no NaN, so the default is always false.
// __half/__nv_bfloat16 need __hisnan because their operator!= is ordered and
// returns false when either operand is NaN.
template<typename T> __device__ __forceinline__ bool sort_is_nan(T) { return false; }
template<> __device__ __forceinline__ bool sort_is_nan<float>(float v) { return isnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<double>(double v) { return isnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<__half>(__half v) { return __hisnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<__nv_bfloat16>(__nv_bfloat16 v) { return __hisnan(v); }
template<> __device__ __forceinline__ bool sort_is_nan<numr_fp8_e4m3>(numr_fp8_e4m3 v) { return isnan(fp8_e4m3_to_f32(v.data)); }
template<> __device__ __forceinline__ bool sort_is_nan<numr_fp8_e5m2>(numr_fp8_e5m2 v) { return isnan(fp8_e5m2_to_f32(v.data)); }

// Total order shared by every backend: NaN compares greater than all non-NaN
// values, NaNs tie with each other, and -0.0 ties with +0.0. Mirrors
// `Element::sort_cmp` on the CPU side.
template<typename T>
__device__ __forceinline__ int sort_cmp(T a, T b) {
    bool a_nan = sort_is_nan(a);
    bool b_nan = sort_is_nan(b);
    if (a_nan || b_nan) {
        if (a_nan && b_nan) return 0;
        return a_nan ? 1 : -1;
    }
    if (a < b) return -1;
    if (b < a) return 1;
    return 0;
}

// Padding value of maximum rank, so pad entries sort into the discarded tail.
// Ascending that is NaN for float types (NaN is the greatest value); descending
// it is -inf. Both must be beyond any real value, otherwise real infinities get
// pushed past the padding and dropped.
template<typename T> __device__ __forceinline__ T sort_pad_rank(bool descending) {
    return descending ? sort_pad_min<T>() : sort_pad_max<T>();
}
template<> __device__ __forceinline__ float sort_pad_rank<float>(bool descending) {
    return descending ? -INFINITY : NAN;
}
template<> __device__ __forceinline__ double sort_pad_rank<double>(bool descending) {
    return descending ? -INFINITY : NAN;
}
template<> __device__ __forceinline__ __half sort_pad_rank<__half>(bool descending) {
    return __float2half(descending ? -INFINITY : NAN);
}
template<> __device__ __forceinline__ __nv_bfloat16 sort_pad_rank<__nv_bfloat16>(bool descending) {
    return __float2bfloat16(descending ? -INFINITY : NAN);
}
template<> __device__ __forceinline__ numr_fp8_e4m3 sort_pad_rank<numr_fp8_e4m3>(bool descending) {
    return numr_fp8_e4m3(f32_to_fp8_e4m3(descending ? -INFINITY : NAN));
}
template<> __device__ __forceinline__ numr_fp8_e5m2 sort_pad_rank<numr_fp8_e5m2>(bool descending) {
    return numr_fp8_e5m2(f32_to_fp8_e5m2(descending ? -INFINITY : NAN));
}

// ============================================================================
// Bitonic compare-and-swap
// ============================================================================

// Rank order: the requested output order, ties broken by original index. The
// network always sorts ascending in this rank space, so `descending` flips the
// value comparison only and stability holds in both directions.
template<typename T>
__device__ __forceinline__ int sort_rank_cmp(T a_val, long long a_idx,
                                             T b_val, long long b_idx,
                                             bool descending) {
    int c = sort_cmp(a_val, b_val);
    if (descending) c = -c;
    if (c != 0) return c;
    if (a_idx == b_idx) return 0;
    return a_idx < b_idx ? -1 : 1;
}

template<typename T>
__device__ __forceinline__ void bitonic_cas_indexed(T& a_val, long long& a_idx,
                                                    T& b_val, long long& b_idx,
                                                    bool ascending_local,
                                                    bool descending) {
    int c = sort_rank_cmp(a_val, a_idx, b_val, b_idx, descending);
    bool swap = ascending_local ? (c > 0) : (c < 0);
    if (swap) {
        T tmp_val = a_val;
        a_val = b_val;
        b_val = tmp_val;
        long long tmp_idx = a_idx;
        a_idx = b_idx;
        b_idx = tmp_idx;
    }
}

#endif // NUMR_SORT_COMPARE_CUH
