// Wide integer accumulator for the I32/U32 matmul shaders.
//
// Concatenated AFTER `int_saturate.wgsl`, whose `NumrI64`, `numr_i32_magnitude`,
// `numr_u32_sat_add`, `numr_u32_mul_overflows` and range constants everything
// below builds on. WGSL has no include and no forward declarations, so that
// order is load-bearing.
//
// Integer division is banned here for the same reason as in `int_saturate.wgsl`:
// the NVIDIA shader compiler fails with "NVVM compilation failed" on an integer
// divide inside a loop.
//
// `matmul` sums K products and narrows once, at the store: CPU keeps the running
// total in i128 (`matmul_scalar_acc` in `runtime/cpu/kernels/matmul/kernel.rs`)
// and CUDA in `Numr128` (`runtime/cuda/kernels/numr128.cuh`). A partial sum that
// leaves i32's range and comes back must therefore report the true value, so the
// accumulator has to be genuinely wider - a per-step saturating add is wrong for
// signed operands.
//
// 96 bits is wide enough that it can never overflow, so no intermediate
// saturation is needed at all: an i32 product needs at most 63 bits, and K of
// them need 63 + log2(K) bits, with K bounded by a storage buffer's element
// count. `NumrI64` is one limb short of that, which is why the accumulator below
// is a separate type rather than a reuse of it.

// Full 64-bit product of two u32 values, from 16-bit halves. Division-free like
// the rest of this file.
fn numr_u64_mul_u32(a: u32, b: u32) -> NumrI64 {
    let a_lo = a & 0xffffu;
    let a_hi = a >> 16u;
    let b_lo = b & 0xffffu;
    let b_hi = b >> 16u;
    var lo = a_lo * b_lo;
    var hi = a_hi * b_hi;
    // Each cross term contributes its low half to `lo` (shifted up 16 bits, the
    // shift discarding exactly the bits `hi` receives) and its high half to `hi`.
    let cross_a = a_hi * b_lo;
    let shifted_a = lo + (cross_a << 16u);
    if (shifted_a < lo) {
        hi = hi + 1u;
    }
    lo = shifted_a;
    hi = hi + (cross_a >> 16u);
    let cross_b = a_lo * b_hi;
    let shifted_b = lo + (cross_b << 16u);
    if (shifted_b < lo) {
        hi = hi + 1u;
    }
    lo = shifted_b;
    hi = hi + (cross_b >> 16u);
    return NumrI64(lo, hi);
}

// Two's-complement negation of a 64-bit value.
fn numr_i64_neg(v: NumrI64) -> NumrI64 {
    var hi = 0u - v.hi;
    if (v.lo != 0u) {
        hi = hi - 1u;
    }
    return NumrI64(0u - v.lo, hi);
}

// Exact i32 * i32 product. The magnitudes are multiplied unsigned because their
// product reaches 2^62, which no 32-bit intermediate holds, and because
// `numr_i32_magnitude` is what makes i32::MIN come out right.
fn numr_i64_mul_i32(a: i32, b: i32) -> NumrI64 {
    let mag = numr_u64_mul_u32(numr_i32_magnitude(a), numr_i32_magnitude(b));
    if ((a < 0) != (b < 0)) {
        return numr_i64_neg(mag);
    }
    return mag;
}

// A 96-bit two's-complement accumulator, three u32 limbs low to high.
struct NumrI96 {
    lo: u32,
    mid: u32,
    hi: u32,
}

fn numr_i96_from_i32(v: i32) -> NumrI96 {
    var high = 0u;
    if (v < 0) {
        high = 0xffffffffu;
    }
    return NumrI96(bitcast<u32>(v), high, high);
}

// Add a 64-bit value into the accumulator, sign-extending it to 96 bits first.
fn numr_i96_add_i64(a: NumrI96, b: NumrI64) -> NumrI96 {
    var b_hi = 0u;
    if ((b.hi >> 31u) != 0u) {
        b_hi = 0xffffffffu;
    }
    let lo = a.lo + b.lo;
    var carry_lo = 0u;
    if (lo < a.lo) {
        carry_lo = 1u;
    }
    // Two adds land in `mid`, so both can carry; `carry_mid` counts them.
    let mid_sum = a.mid + b.hi;
    var carry_mid = 0u;
    if (mid_sum < a.mid) {
        carry_mid = 1u;
    }
    let mid = mid_sum + carry_lo;
    if (mid < mid_sum) {
        carry_mid = carry_mid + 1u;
    }
    return NumrI96(lo, mid, a.hi + b_hi + carry_mid);
}

// Narrow the accumulator to i32, saturating. A value that fits has its upper
// limbs equal to the sign extension of `lo`.
fn numr_i96_to_i32_sat(v: NumrI96) -> i32 {
    let negative = (v.hi >> 31u) != 0u;
    if (!negative) {
        if (v.hi == 0u && v.mid == 0u && v.lo <= 2147483647u) {
            return bitcast<i32>(v.lo);
        }
        return NUMR_I32_MAX;
    }
    if (v.hi == 0xffffffffu && v.mid == 0xffffffffu && v.lo >= 2147483648u) {
        return bitcast<i32>(v.lo);
    }
    return NUMR_I32_MIN;
}

// One step of a u32 matmul accumulation: `acc + a * b`, saturating.
//
// No wide accumulator is needed here. Every term is non-negative, so the running
// total only grows: once it reaches u32::MAX the true total is at or past the
// clamp for good and can never come back down, which makes a per-step saturating
// add agree with CPU's i128 accumulator narrowed at the store. Same monotonicity
// argument as `numr_u32_sat_add` above. It does NOT extend to the product, which
// is why an overflowing product pins the total directly instead of being added.
fn numr_u32_mul_add_sat(acc: u32, a: u32, b: u32) -> u32 {
    if (numr_u32_mul_overflows(a, b)) {
        return NUMR_U32_MAX;
    }
    return numr_u32_sat_add(acc, a * b);
}
