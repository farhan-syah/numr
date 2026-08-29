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

// Unsigned counterpart and range bounds per signed integer dtype.
template<typename I> struct numr_ipow_traits;

template<> struct numr_ipow_traits<int32_t> {
    typedef uint32_t U;
    static __device__ __forceinline__ U imax() { return (U)INT32_MAX; }
    static __device__ __forceinline__ int32_t sat_max() { return INT32_MAX; }
    static __device__ __forceinline__ int32_t sat_min() { return INT32_MIN; }
};

template<> struct numr_ipow_traits<int64_t> {
    typedef uint64_t U;
    static __device__ __forceinline__ U imax() { return (U)INT64_MAX; }
    static __device__ __forceinline__ int64_t sat_max() { return INT64_MAX; }
    static __device__ __forceinline__ int64_t sat_min() { return INT64_MIN; }
};

template<typename I>
__device__ __forceinline__ I numr_ipow(I base, I exp) {
    typedef numr_ipow_traits<I> Tr;
    typedef typename Tr::U U;

    if (exp < 0) {
        return (I)numr_pow_safe((double)base, (double)exp);
    }

    // The result is negative exactly when the base is negative and the exponent
    // is odd. Working on magnitudes in the unsigned type keeps one routine for
    // both instantiations and avoids negating INT_MIN in the signed type.
    const bool negative = (base < 0) && ((exp & 1) != 0);
    const U bound = negative ? (U)(Tr::imax() + (U)1) : Tr::imax();
    const I saturated = negative ? Tr::sat_min() : Tr::sat_max();

    U acc = (base < 0) ? ((U)0 - (U)base) : (U)base;
    U result = 1;
    I e = exp;
    while (e > 0) {
        if (e & 1) {
            if (acc != 0 && result > bound / acc) {
                return saturated;
            }
            result *= acc;
        }
        e >>= 1;
        if (e > 0) {
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
    // saturates. Capping there keeps the cast to I in range while preserving
    // parity, which is what a negative base needs. Device code cannot call the
    // shared Rust helper, so this cap is a hand-kept mirror of
    // `cap_ipow_exponent` in `runtime/common/helpers.rs` (also used by CPU and
    // WebGPU) — the two must keep agreeing.
    double capped = exp;
    if (capped > 1024.0) {
        capped = 1024.0 + (fmod(exp, 2.0) == 0.0 ? 0.0 : 1.0);
    }
    return numr_ipow<I>(base, (I)capped);
}

#endif // NUMR_IPOW_CUH
