// Exact, saturating integer exponentiation for the U32 `pow` shaders.
//
// This is the WGSL transliteration of `runtime/cuda/kernels/ipow.cuh` and
// `runtime/cpu/kernels/ipow.rs`, restricted to the unsigned case: the exponent
// is never negative, so there is no fractional branch. Overflow saturates to
// u32's bound, matching the CPU reference.
//
// Overflow is detected by `numr_u32_mul_exceeds` in int_saturate.wgsl, which is
// prepended to this module.

fn numr_ipow_u32(base: u32, exp: u32) -> u32 {
    var acc: u32 = base;
    var result: u32 = 1u;
    var e: u32 = exp;
    while (e > 0u) {
        if ((e & 1u) != 0u) {
            if (numr_u32_mul_exceeds(result, acc, NUMR_U32_MAX)) {
                return NUMR_U32_MAX;
            }
            result = result * acc;
        }
        e = e >> 1u;
        if (e > 0u) {
            // A squared acc is always consumed by a later multiply, so overflow
            // here is a definite overflow of the final result.
            if (numr_u32_mul_exceeds(acc, acc, NUMR_U32_MAX)) {
                return NUMR_U32_MAX;
            }
            acc = acc * acc;
        }
    }
    return result;
}
