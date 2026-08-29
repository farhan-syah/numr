// Overflow detection shared by the I32 and U32 `pow` shaders.
//
// Integer division is banned here. The NVIDIA shader compiler fails with
// "NVVM compilation failed" on a u32 divide inside the squaring loop, and
// division is the slowest integer operation on a GPU regardless. Both checks
// below use only shifts, masks, multiplies of 16-bit halves, and compares.

const NUMR_U32_MAX: u32 = 4294967295u;

// True when `a * b` leaves u32.
//
// Split both operands into 16-bit halves. The full product is
// (a_hi*b_hi << 32) + ((a_hi*b_lo + a_lo*b_hi) << 16) + a_lo*b_lo. The first
// term alone leaves u32, and the cross term leaves it once the cross exceeds
// 16 bits.
fn numr_u32_mul_overflows(a: u32, b: u32) -> bool {
    if (a == 0u || b == 0u) {
        return false;
    }
    let a_hi = a >> 16u;
    let a_lo = a & 0xffffu;
    let b_hi = b >> 16u;
    let b_lo = b & 0xffffu;
    if (a_hi != 0u && b_hi != 0u) {
        return true;
    }
    // At most one high half is non-zero here, so one cross term is zero and the
    // sum stays inside u32.
    let cross = a_hi * b_lo + a_lo * b_hi;
    if (cross > 0xffffu) {
        return true;
    }
    return (cross << 16u) > (NUMR_U32_MAX - a_lo * b_lo);
}

// True when `a * b` exceeds `bound`.
//
// The signed helper carries a bound below u32's maximum, because a negative
// result reaches one past i32's maximum magnitude while a positive one does not.
fn numr_u32_mul_exceeds(a: u32, b: u32, bound: u32) -> bool {
    if (numr_u32_mul_overflows(a, b)) {
        return true;
    }
    return a * b > bound;
}
