// Exact, saturating integer exponentiation for the I32 `pow` shaders.
//
// This is the WGSL transliteration of `runtime/cuda/kernels/ipow.cuh` and
// `runtime/cpu/kernels/ipow.rs`. Integer pow is computed by squaring, never
// through floating point: `pow(5.0, 3.0)` returns 124.99999 on some hardware,
// which truncates to 124. Overflow saturates to i32's bound, because the CPU
// reference saturates and a wrapping multiply would disagree on exactly the
// inputs that overflow.
//
// Overflow is detected by `numr_u32_mul_exceeds` in ipow_common.wgsl, which is
// prepended to this module. Magnitudes live in u32, so the signed and unsigned
// helpers share one overflow rule.

const NUMR_I32_MAX: i32 = 2147483647;
const NUMR_I32_MIN: i32 = -2147483647 - 1;

fn numr_ipow_i32(base: i32, exp: i32) -> i32 {
    if (exp < 0) {
        // The true value is a fraction, which CPU truncates after computing it
        // in f64. Only four outcomes exist, and integer logic reaches each one
        // without a float.
        if (base == 0) {
            // 0 ** -n is infinity in f64, and the saturating cast gives i32 max.
            return NUMR_I32_MAX;
        }
        if (base == 1) {
            return 1;
        }
        if (base == -1) {
            if ((exp & 1) != 0) {
                return -1;
            }
            return 1;
        }
        // Every other base gives a magnitude below 1, which truncates to zero.
        return 0;
    }

    // The result is negative exactly when the base is negative and the exponent
    // is odd. Magnitudes live in u32 so that i32's minimum negates safely.
    let negative = base < 0 && (exp & 1) != 0;
    var bound: u32 = 2147483647u;
    if (negative) {
        bound = 2147483648u;
    }
    var saturated: i32 = NUMR_I32_MAX;
    if (negative) {
        saturated = NUMR_I32_MIN;
    }

    var acc: u32 = bitcast<u32>(base);
    if (base < 0) {
        acc = 0u - acc;
    }
    var result: u32 = 1u;
    var e: i32 = exp;
    while (e > 0) {
        if ((e & 1) != 0) {
            if (numr_u32_mul_exceeds(result, acc, bound)) {
                return saturated;
            }
            result = result * acc;
        }
        e = e >> 1u;
        if (e > 0) {
            // A squared acc is always consumed by a later multiply, so overflow
            // here is a definite overflow of the final result.
            if (numr_u32_mul_exceeds(acc, acc, bound)) {
                return saturated;
            }
            acc = acc * acc;
        }
    }

    if (negative) {
        // The negative bound's magnitude does not fit in i32, so negate in two
        // steps.
        return -i32(result - 1u) - 1;
    }
    return i32(result);
}
