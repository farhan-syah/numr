// A signed 128-bit accumulator built from two 64-bit halves.
//
// `long long` cannot widen further natively and `__int128` is not portable
// across the CUDA versions this project builds against, so the accumulator is
// two's-complement, held as `{lo, hi}` unsigned 64-bit halves - the same
// technique as `NumrI64` in `runtime/wgpu/shaders/int_saturate.wgsl`, one limb
// size up.
//
// This exists so integer kernels that build a running total match the CPU
// kernels' i128 accumulator (`WideAcc` in `runtime/cpu/kernels/wide_acc.rs`):
// the accumulator never saturates mid-computation, only the narrow-back store
// does. A total that leaves the output dtype's range and later returns to it
// therefore reports the true value instead of a clamped or wrapped one. Adding
// an intermediate clamp anywhere breaks that guarantee.
//
// 128 bits is wide enough for both consumers: a scan sums I64 elements, and a
// matmul sums products of two I64 elements (each at most 2^126) over a K that
// no allocation can push past 2^126 accumulated terms.

#ifndef NUMR_NUMR128_CUH
#define NUMR_NUMR128_CUH

#include <climits>

struct Numr128 {
    unsigned long long lo;
    unsigned long long hi;
};

__device__ __forceinline__ Numr128 numr128_from_i64(long long v) {
    unsigned long long lo = (unsigned long long)v;
    unsigned long long hi = (v < 0) ? 0xffffffffffffffffULL : 0ULL;
    Numr128 r;
    r.lo = lo;
    r.hi = hi;
    return r;
}

__device__ __forceinline__ Numr128 numr128_add(Numr128 a, Numr128 b) {
    unsigned long long lo = a.lo + b.lo;
    unsigned long long carry = (lo < a.lo) ? 1ULL : 0ULL;
    Numr128 r;
    r.lo = lo;
    r.hi = a.hi + b.hi + carry;
    return r;
}

// Two's-complement negation: invert both halves and add one, carrying into hi.
__device__ __forceinline__ Numr128 numr128_neg(Numr128 v) {
    Numr128 r;
    r.lo = ~v.lo + 1ULL;
    r.hi = ~v.hi + ((r.lo == 0ULL) ? 1ULL : 0ULL);
    return r;
}

// Exact signed 64x64 -> 128 multiply.
//
// The magnitudes are multiplied in the unsigned type and the sign is applied
// once at the end, so one routine covers all four sign combinations and no
// intermediate ever negates LLONG_MIN in the signed type. `0 - (unsigned)v` is
// the magnitude of any negative v including LLONG_MIN, where the signed negation
// would overflow.
//
// Each magnitude splits into 32-bit halves, giving four partial products:
//   p0 = a_lo*b_lo (weight 2^0), p1 = a_lo*b_hi and p2 = a_hi*b_lo (weight
//   2^32), p3 = a_hi*b_hi (weight 2^64). Every product fits in 64 bits.
// The two middle products are summed first; that sum can carry out of bit 63,
// and since it carries weight 2^32 the carry lands at 2^96, which is bit 32 of
// `hi`. Adding the middle sum's low half into `lo` can carry into `hi` as well.
__device__ __forceinline__ Numr128 numr128_mul_i64(long long a, long long b) {
    const bool negative = (a < 0) != (b < 0);

    unsigned long long ua = (a < 0) ? (0ULL - (unsigned long long)a) : (unsigned long long)a;
    unsigned long long ub = (b < 0) ? (0ULL - (unsigned long long)b) : (unsigned long long)b;

    const unsigned long long a_lo = ua & 0xffffffffULL;
    const unsigned long long a_hi = ua >> 32;
    const unsigned long long b_lo = ub & 0xffffffffULL;
    const unsigned long long b_hi = ub >> 32;

    const unsigned long long p0 = a_lo * b_lo;
    const unsigned long long p1 = a_lo * b_hi;
    const unsigned long long p2 = a_hi * b_lo;
    const unsigned long long p3 = a_hi * b_hi;

    const unsigned long long mid = p1 + p2;
    const unsigned long long mid_carry = (mid < p1) ? 1ULL : 0ULL;

    const unsigned long long lo = p0 + (mid << 32);
    const unsigned long long lo_carry = (lo < p0) ? 1ULL : 0ULL;

    Numr128 r;
    r.lo = lo;
    r.hi = p3 + (mid >> 32) + (mid_carry << 32) + lo_carry;

    // |a| * |b| <= 2^126, so the magnitude never reaches the 128-bit sign bit
    // and this negation is exact.
    return negative ? numr128_neg(r) : r;
}

// Narrow a 128-bit accumulator back to i64, saturating on overflow. `hi`'s
// sign bit gives the 128-bit value's sign; a value that fits in i64 has `hi`
// equal to the sign-extension of `lo`'s top bit.
__device__ __forceinline__ long long numr128_to_i64_sat(Numr128 v) {
    bool negative = (v.hi >> 63) != 0ULL;
    if (!negative) {
        if (v.hi == 0ULL && v.lo <= (unsigned long long)LLONG_MAX) {
            return (long long)v.lo;
        }
        return LLONG_MAX;
    }
    if (v.hi == 0xffffffffffffffffULL && v.lo >= (unsigned long long)LLONG_MIN) {
        return (long long)v.lo;
    }
    return LLONG_MIN;
}

// Narrow to i32 through the i64 rule so there is one saturation convention.
// i64 saturation preserves the sign of the overflow, so clamping the result
// again to i32 lands on the same bound the exact value would have.
__device__ __forceinline__ int numr128_to_i32_sat(Numr128 v) {
    long long w = numr128_to_i64_sat(v);
    if (w > (long long)INT_MAX) {
        return INT_MAX;
    }
    if (w < (long long)INT_MIN) {
        return INT_MIN;
    }
    return (int)w;
}

// Type-directed narrow-back, so a kernel templated over its element type picks
// the matching saturation rule. This selects between the two rules above; it
// does not add a third.
template<typename T> struct Numr128Narrow;

template<> struct Numr128Narrow<int> {
    static __device__ __forceinline__ int apply(Numr128 v) { return numr128_to_i32_sat(v); }
};

template<> struct Numr128Narrow<long long> {
    static __device__ __forceinline__ long long apply(Numr128 v) { return numr128_to_i64_sat(v); }
};

// Move a whole accumulator down the warp. The two halves shuffle independently
// because they are plain 64-bit values; the receiving lane reassembles the
// sender's exact accumulator, so a warp reduction built on this stays exact.
__device__ __forceinline__ Numr128 numr128_shfl_down(unsigned int mask, Numr128 v, int offset) {
    Numr128 r;
    r.lo = __shfl_down_sync(mask, v.lo, offset);
    r.hi = __shfl_down_sync(mask, v.hi, offset);
    return r;
}

#endif // NUMR_NUMR128_CUH
