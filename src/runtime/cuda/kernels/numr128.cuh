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

// Widen an unsigned 64-bit element. Unsigned values are zero-extended, never
// sign-extended: a u64 above LLONG_MAX is a large positive number, and routing
// it through `numr128_from_i64` would record it as negative.
__device__ __forceinline__ Numr128 numr128_from_u64(unsigned long long v) {
    Numr128 r;
    r.lo = v;
    r.hi = 0ULL;
    return r;
}

// The signed 128-bit bounds, used as the saturation targets below.
__device__ __forceinline__ Numr128 numr128_max() {
    Numr128 r;
    r.lo = 0xffffffffffffffffULL;
    r.hi = 0x7fffffffffffffffULL;
    return r;
}

__device__ __forceinline__ Numr128 numr128_min() {
    Numr128 r;
    r.lo = 0ULL;
    r.hi = 0x8000000000000000ULL;
    return r;
}

__device__ __forceinline__ bool numr128_is_negative(Numr128 v) {
    return (v.hi >> 63) != 0ULL;
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

// Saturating signed 128-bit add, matching the CPU kernels' `i128::saturating_add`.
//
// Two operands of the same sign whose sum reports the opposite sign is the only
// way a two's-complement add can leave the range, so that test is exact.
//
// Reaching this bound needs about 2^64 accumulated I64 elements, which no
// allocation can hold; it exists so the CUDA and CPU reductions agree on the
// pathological case rather than one wrapping and the other clamping.
__device__ __forceinline__ Numr128 numr128_add_sat(Numr128 a, Numr128 b) {
    Numr128 r = numr128_add(a, b);
    const bool a_neg = numr128_is_negative(a);
    const bool b_neg = numr128_is_negative(b);
    if (a_neg == b_neg && numr128_is_negative(r) != a_neg) {
        return a_neg ? numr128_min() : numr128_max();
    }
    return r;
}

// The magnitude of a signed 128-bit value, as raw 128-bit unsigned bits.
// Negating the negative bound yields its own bit pattern, which read as
// unsigned is exactly 2^127 - the correct magnitude.
__device__ __forceinline__ Numr128 numr128_magnitude(Numr128 v) {
    return numr128_is_negative(v) ? numr128_neg(v) : v;
}

// Exact unsigned 64x64 -> 128 multiply.
//
// Each operand splits into 32-bit halves, giving four partial products:
//   p0 = a_lo*b_lo (weight 2^0), p1 = a_lo*b_hi and p2 = a_hi*b_lo (weight
//   2^32), p3 = a_hi*b_hi (weight 2^64). Every product fits in 64 bits.
// The two middle products are summed first; that sum can carry out of bit 63,
// and since it carries weight 2^32 the carry lands at 2^96, which is bit 32 of
// `hi`. Adding the middle sum's low half into `lo` can carry into `hi` as well.
__device__ __forceinline__ Numr128 numr128_umul64(unsigned long long ua, unsigned long long ub) {
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
    return r;
}

// Exact signed 64x64 -> 128 multiply.
//
// The magnitudes are multiplied in the unsigned type and the sign is applied
// once at the end, so one routine covers all four sign combinations and no
// intermediate ever negates LLONG_MIN in the signed type. `0 - (unsigned)v` is
// the magnitude of any negative v including LLONG_MIN, where the signed negation
// would overflow.
__device__ __forceinline__ Numr128 numr128_mul_i64(long long a, long long b) {
    const bool negative = (a < 0) != (b < 0);

    unsigned long long ua = (a < 0) ? (0ULL - (unsigned long long)a) : (unsigned long long)a;
    unsigned long long ub = (b < 0) ? (0ULL - (unsigned long long)b) : (unsigned long long)b;

    Numr128 r = numr128_umul64(ua, ub);

    // |a| * |b| <= 2^126, so the magnitude never reaches the 128-bit sign bit
    // and this negation is exact.
    return negative ? numr128_neg(r) : r;
}

// Saturating signed 128x128 -> 128 multiply, matching `i128::saturating_mul`.
//
// `numr128_mul_i64` cannot serve a running product: the accumulator is already
// 128 bits wide by the second element. Magnitudes are multiplied unsigned and
// the sign applied once, the same shape as the 64-bit routine.
//
// Overflow of the 128-bit magnitude is detected structurally: if both operands
// need more than 64 bits the product needs more than 128, and otherwise at most
// one cross term is non-zero, so a carry out of it (or out of the final add
// into `hi`) is the only remaining way to leave the range.
//
// Saturating multiply is not associative, so a block-tree product that actually
// saturates can differ from the CPU's sequential one. That needs a partial
// product past 2^127, far beyond what any element dtype narrows back to, and
// the final saturating narrow lands on the same bound either way.
__device__ __forceinline__ Numr128 numr128_mul_sat(Numr128 a, Numr128 b) {
    const bool negative = numr128_is_negative(a) != numr128_is_negative(b);
    const Numr128 ua = numr128_magnitude(a);
    const Numr128 ub = numr128_magnitude(b);

    bool overflow = (ua.hi != 0ULL) && (ub.hi != 0ULL);
    Numr128 mag;
    mag.lo = 0ULL;
    mag.hi = 0ULL;

    if (!overflow) {
        const Numr128 lolo = numr128_umul64(ua.lo, ub.lo);
        // At most one of the two operands has a non-zero high half, so the two
        // cross terms cannot both contribute.
        const Numr128 cross = (ua.hi != 0ULL) ? numr128_umul64(ua.hi, ub.lo)
                                              : numr128_umul64(ua.lo, ub.hi);
        if (cross.hi != 0ULL) {
            overflow = true;
        } else {
            const unsigned long long hi = lolo.hi + cross.lo;
            if (hi < lolo.hi) {
                overflow = true;
            } else {
                mag.lo = lolo.lo;
                mag.hi = hi;
            }
        }
    }

    if (!overflow && (mag.hi >> 63) != 0ULL) {
        // The magnitude reached the sign bit. Only exactly 2^127 survives, and
        // only as the negative bound.
        overflow = !(negative && mag.hi == 0x8000000000000000ULL && mag.lo == 0ULL);
    }

    if (overflow) {
        return negative ? numr128_min() : numr128_max();
    }
    return negative ? numr128_neg(mag) : mag;
}

// Divide a signed 128-bit accumulator by an unsigned 64-bit count, truncating
// toward zero.
//
// This is the integer `mean` epilogue, and it must agree with the CPU's
// `sum / count` on i128 for negative sums too: Rust's `/` truncates toward
// zero, so the magnitude is divided and the sign reapplied afterwards rather
// than the division being allowed to floor.
//
// Restoring shift-subtract long division, 128 iterations. `rem` stays below
// `d`, so `2*rem + bit` needs one extra bit above 64 - carried in `rem_carry`
// - and a single conditional subtraction restores the invariant. This runs once
// per output element on one thread, so the bit loop costs nothing measurable.
__device__ __forceinline__ Numr128 numr128_div_u64_trunc(Numr128 a, unsigned long long d) {
    if (d == 0ULL) {
        d = 1ULL; // mirrors the CPU epilogue's `count.max(1)`
    }

    const bool negative = numr128_is_negative(a);
    const Numr128 u = numr128_magnitude(a);

    Numr128 q;
    q.lo = 0ULL;
    q.hi = 0ULL;
    unsigned long long rem = 0ULL;

    for (int i = 127; i >= 0; i--) {
        const unsigned long long bit =
            (i >= 64) ? ((u.hi >> (i - 64)) & 1ULL) : ((u.lo >> i) & 1ULL);
        const unsigned long long rem_carry = rem >> 63;
        rem = (rem << 1) | bit;
        if (rem_carry != 0ULL || rem >= d) {
            rem -= d;
            if (i >= 64) {
                q.hi |= (1ULL << (i - 64));
            } else {
                q.lo |= (1ULL << i);
            }
        }
    }

    return negative ? numr128_neg(q) : q;
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

// Narrow to u64, saturating. A negative accumulator clamps to 0 rather than
// wrapping to a huge positive count, which is what the CPU's
// `T::from_i128_saturating` does for unsigned element types.
__device__ __forceinline__ unsigned long long numr128_to_u64_sat(Numr128 v) {
    if (numr128_is_negative(v)) {
        return 0ULL;
    }
    if (v.hi != 0ULL) {
        return 0xffffffffffffffffULL;
    }
    return v.lo;
}

// Type-directed narrow-back, so a kernel templated over its element type picks
// the matching saturation rule. Every specialisation routes through the i64 or
// u64 rule above and then clamps to its own range, so there are exactly two
// saturation conventions no matter how many element types exist.
template<typename T> struct Numr128Narrow;

template<> struct Numr128Narrow<int> {
    static __device__ __forceinline__ int apply(Numr128 v) { return numr128_to_i32_sat(v); }
};

template<> struct Numr128Narrow<long long> {
    static __device__ __forceinline__ long long apply(Numr128 v) { return numr128_to_i64_sat(v); }
};

// Signed narrow widths: clamp the i64 result, whose sign already matches the
// exact value's, into the element's range.
#define NUMR128_NARROW_SIGNED(T, LO, HI) \
template<> struct Numr128Narrow<T> { \
    static __device__ __forceinline__ T apply(Numr128 v) { \
        long long w = numr128_to_i64_sat(v); \
        if (w > (long long)(HI)) return (T)(HI); \
        if (w < (long long)(LO)) return (T)(LO); \
        return (T)w; \
    } \
};

NUMR128_NARROW_SIGNED(short, SHRT_MIN, SHRT_MAX)
NUMR128_NARROW_SIGNED(signed char, SCHAR_MIN, SCHAR_MAX)

#undef NUMR128_NARROW_SIGNED

template<> struct Numr128Narrow<unsigned long long> {
    static __device__ __forceinline__ unsigned long long apply(Numr128 v) {
        return numr128_to_u64_sat(v);
    }
};

// Unsigned narrow widths: the u64 rule already clamped a negative value to 0,
// so only the upper bound is left to apply.
#define NUMR128_NARROW_UNSIGNED(T, HI) \
template<> struct Numr128Narrow<T> { \
    static __device__ __forceinline__ T apply(Numr128 v) { \
        unsigned long long w = numr128_to_u64_sat(v); \
        if (w > (unsigned long long)(HI)) return (T)(HI); \
        return (T)w; \
    } \
};

NUMR128_NARROW_UNSIGNED(unsigned int, UINT_MAX)
NUMR128_NARROW_UNSIGNED(unsigned short, USHRT_MAX)
NUMR128_NARROW_UNSIGNED(unsigned char, UCHAR_MAX)

#undef NUMR128_NARROW_UNSIGNED

// Type-directed widen-in, the inverse of Numr128Narrow. Signed types
// sign-extend, unsigned types zero-extend; picking the wrong one silently turns
// a large u64 into a negative accumulator.
template<typename T> struct Numr128From;

#define NUMR128_FROM_SIGNED(T) \
template<> struct Numr128From<T> { \
    static __device__ __forceinline__ Numr128 apply(T v) { return numr128_from_i64((long long)v); } \
};

#define NUMR128_FROM_UNSIGNED(T) \
template<> struct Numr128From<T> { \
    static __device__ __forceinline__ Numr128 apply(T v) { return numr128_from_u64((unsigned long long)v); } \
};

NUMR128_FROM_SIGNED(long long)
NUMR128_FROM_SIGNED(int)
NUMR128_FROM_SIGNED(short)
NUMR128_FROM_SIGNED(signed char)
NUMR128_FROM_UNSIGNED(unsigned long long)
NUMR128_FROM_UNSIGNED(unsigned int)
NUMR128_FROM_UNSIGNED(unsigned short)
NUMR128_FROM_UNSIGNED(unsigned char)

#undef NUMR128_FROM_SIGNED
#undef NUMR128_FROM_UNSIGNED

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
