// Exact, saturating integer exponentiation shared by the pow kernels.
//
// Integer pow is computed EXACTLY, by squaring — never through floating point.
// CUDA's `pow()` is accurate only to a few ULP, so a nearly-integer result
// truncates to the wrong integer: powf(5,3) yields 124.99999 -> 124, and even
// in double, pow(3.0, 1.0) yields 2.9999999999999996 -> 2. Widening the float
// does not fix it; removing the float does.
//
// Overflow saturates to the dtype's bound, matching what CPU's `as` cast has
// always produced. Multiplying in the element type would wrap instead, so the
// two backends would disagree on exactly the inputs that overflow.
//
// A negative exponent keeps the double path, matching CPU: the true value is a
// fraction, and CPU truncates it the same way (2^-1 -> 0.5 -> 0).
//
// __int128 is not portable across the CUDA versions this project builds
// against (see cumulative.cu), so overflow is detected in the native width,
// before each multiply, by dividing the bound rather than multiplying past it.

#ifndef NUMR_IPOW_CUH
#define NUMR_IPOW_CUH

#include <stdint.h>
#include "dtype_traits.cuh"

// Unsigned counterpart, saturation bounds, and a negativity test per integer
// dtype. The negativity test is a trait function rather than a bare `v < 0` so
// the unsigned instantiations never compare an unsigned value against zero.
template<typename I> struct numr_ipow_traits;

#define NUMR_IPOW_TRAITS_SIGNED(I, UT, IMAX, IMIN)                             \
    template<> struct numr_ipow_traits<I> {                                    \
        typedef UT U;                                                          \
        static __device__ __forceinline__ U imax() { return (U)(IMAX); }       \
        static __device__ __forceinline__ I sat_max() { return (IMAX); }       \
        static __device__ __forceinline__ I sat_min() { return (IMIN); }       \
        static __device__ __forceinline__ bool is_negative(I v) { return v < 0; } \
    };

#define NUMR_IPOW_TRAITS_UNSIGNED(UT, UMAX)                                    \
    template<> struct numr_ipow_traits<UT> {                                   \
        typedef UT U;                                                          \
        static __device__ __forceinline__ U imax() { return (UMAX); }          \
        static __device__ __forceinline__ UT sat_max() { return (UMAX); }      \
        static __device__ __forceinline__ UT sat_min() { return (UT)0; }       \
        static __device__ __forceinline__ bool is_negative(UT) { return false; } \
    };

NUMR_IPOW_TRAITS_SIGNED(int8_t, uint8_t, INT8_MAX, INT8_MIN)
NUMR_IPOW_TRAITS_SIGNED(int16_t, uint16_t, INT16_MAX, INT16_MIN)
NUMR_IPOW_TRAITS_SIGNED(int32_t, uint32_t, INT32_MAX, INT32_MIN)
NUMR_IPOW_TRAITS_SIGNED(int64_t, uint64_t, INT64_MAX, INT64_MIN)
NUMR_IPOW_TRAITS_UNSIGNED(uint8_t, UINT8_MAX)
NUMR_IPOW_TRAITS_UNSIGNED(uint16_t, UINT16_MAX)
NUMR_IPOW_TRAITS_UNSIGNED(uint32_t, UINT32_MAX)
NUMR_IPOW_TRAITS_UNSIGNED(uint64_t, UINT64_MAX)

#undef NUMR_IPOW_TRAITS_SIGNED
#undef NUMR_IPOW_TRAITS_UNSIGNED

// Exponentiation by squaring for an exponent already known to be non-negative.
//
// The exponent has its own type `E` because `pow_scalar` caps it at 1025 (see
// `numr_ipow_scalar`), which does not fit an i8 or u8 element type. Keeping the
// exponent wide lets one routine serve both the tensor-tensor pow (E == I) and
// the scalar pow (E == long long).
template<typename I, typename E>
__device__ __forceinline__ I numr_ipow_nonneg(I base, E exp) {
    typedef numr_ipow_traits<I> Tr;
    typedef typename Tr::U U;

    // The result is negative exactly when the base is negative and the exponent
    // is odd. Working on magnitudes in the unsigned type keeps one routine for
    // every instantiation and avoids negating INT_MIN in the signed types.
    const bool negative = Tr::is_negative(base) && ((exp & (E)1) != (E)0);
    const U bound = negative ? (U)(Tr::imax() + (U)1) : Tr::imax();
    const I saturated = negative ? Tr::sat_min() : Tr::sat_max();

    U acc = Tr::is_negative(base) ? (U)((U)0 - (U)base) : (U)base;
    U result = 1;
    E e = exp;
    while (e > (E)0) {
        if (e & (E)1) {
            if (acc != 0 && result > bound / acc) {
                return saturated;
            }
            result *= acc;
        }
        e >>= 1;
        if (e > (E)0) {
            // A squared acc is always consumed by a later multiply: the loop
            // only squares while bits remain in e, and e's highest remaining bit
            // is 1. Overflow here is therefore a definite overflow of the final
            // result, not a speculative one.
            if (acc != 0 && acc > bound / acc) {
                return saturated;
            }
            acc *= acc;
        }
    }

    // result <= bound, and bound is imax + 1 in the negative case, which the
    // signed type can only reach through this two-step negation.
    return negative ? (I)(-(I)(result - (U)1) - (I)1) : (I)result;
}

template<typename I>
__device__ __forceinline__ I numr_ipow(I base, I exp) {
    if (numr_ipow_traits<I>::is_negative(exp)) {
        return (I)numr_pow_safe((double)base, (double)exp);
    }
    return numr_ipow_nonneg<I, I>(base, exp);
}

// `base ** scalar` where the exponent arrives as a double.
//
// Only a non-negative whole exponent reaches this function. The host computes
// the output dtype from the input dtype and the exponent, and an integer raised
// to a negative or fractional power is a non-integer real, so those cases get an
// F64 output and never launch an integer kernel. The exponent stays a double
// through the launch because rounding it to the element type on the host would
// turn 2.5 into 2 and answer a different question.
template<typename I>
__device__ __forceinline__ I numr_ipow_scalar(I base, double exp) {
    // Above 1024 the outcome depends only on the base's magnitude and the
    // exponent's parity: magnitude 0 or 1 is already fixed, and anything larger
    // saturates. Capping there keeps the exponent small while preserving
    // parity, which is what a negative base needs. Device code cannot call the
    // shared Rust helper, so this cap is a hand-kept mirror of
    // `cap_ipow_exponent` in `runtime/common/helpers.rs` (also used by CPU and
    // WebGPU) — the two must keep agreeing.
    double capped = exp;
    if (capped > 1024.0) {
        capped = 1024.0 + (fmod(exp, 2.0) == 0.0 ? 0.0 : 1.0);
    }
    // The exponent stays a `long long` rather than narrowing to I: the cap of
    // 1025 does not fit an i8 or u8 element type.
    return numr_ipow_nonneg<I, long long>(base, (long long)capped);
}

#endif // NUMR_IPOW_CUH
