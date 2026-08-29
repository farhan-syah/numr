// Saturating integer arithmetic shared by the I32/U32 `pow` and `cumsum`
// shaders.
//
// Integer division is banned here. The NVIDIA shader compiler fails with
// "NVVM compilation failed" on a u32 divide inside a loop, and division is
// the slowest integer operation on a GPU regardless. Everything below uses
// only shifts, masks, multiplies of 16-bit halves, adds, and compares.

const NUMR_U32_MAX: u32 = 4294967295u;
const NUMR_I32_MAX: i32 = 2147483647;
const NUMR_I32_MIN: i32 = -2147483647 - 1;

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

// Saturating add, used by `cumsum_u32`. Element-wise: unsigned cumsum inputs
// never go negative, so the running total is monotonic and a per-step
// saturating add matches CPU's wide accumulator exactly - once clamped to
// u32::MAX it can never need to come back down.
fn numr_u32_sat_add(a: u32, b: u32) -> u32 {
    if (a > NUMR_U32_MAX - b) {
        return NUMR_U32_MAX;
    }
    return a + b;
}

// A 64-bit two's-complement accumulator built from two u32 halves, for
// `cumsum_i32`. WGSL has no i64. CPU accumulates cumsum in i128 (see
// `WideAcc` in `runtime/cpu/kernels/wide_acc.rs`) and narrows once per
// element written, saturating - so a running total that overflows i32 and
// later comes back into range must still report the in-range value. A
// per-step saturating add cannot do that (it would clamp the intermediate
// and never recover), so the accumulator must actually be wider than i32.
// 64 bits is far beyond any realistic scan length and needs no division to
// maintain: only a wraparound compare detects the add's carry.
struct NumrI64 {
    lo: u32,
    hi: u32,
}

fn numr_i64_from_i32(v: i32) -> NumrI64 {
    var hi = 0u;
    if (v < 0) {
        hi = 0xffffffffu;
    }
    return NumrI64(bitcast<u32>(v), hi);
}

fn numr_i64_add(a: NumrI64, b: NumrI64) -> NumrI64 {
    let lo = a.lo + b.lo;
    var carry = 0u;
    if (lo < a.lo) {
        carry = 1u;
    }
    return NumrI64(lo, a.hi + b.hi + carry);
}

// Narrow a 64-bit accumulator back to i32, saturating on overflow. The sign
// bit of `hi` gives the 64-bit value's sign; a value that fits in i32 has
// `hi` equal to the sign-extension of `lo`'s top bit.
fn numr_i64_to_i32_sat(v: NumrI64) -> i32 {
    let negative = (v.hi >> 31u) != 0u;
    if (!negative) {
        if (v.hi == 0u && v.lo <= 2147483647u) {
            return bitcast<i32>(v.lo);
        }
        return NUMR_I32_MAX;
    }
    if (v.hi == 0xffffffffu && v.lo >= 2147483648u) {
        return bitcast<i32>(v.lo);
    }
    return NUMR_I32_MIN;
}
