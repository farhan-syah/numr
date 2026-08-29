// Truncating division of a wide integer by a 32-bit count.
//
// Concatenated AFTER `int_saturate.wgsl` and `int_matmul_acc.wgsl`, whose
// `NumrI64`, `numr_i64_neg`, `numr_i64_mul_i32` and range constants everything
// below builds on. WGSL has no include and no forward declarations, so that
// order is load-bearing.
//
// Integer `mean` and integer `linspace` both need one exact division of a
// 64-bit value by a 32-bit one, truncating toward zero, because that is what
// CPU does: `int_mean_from_sum` in `runtime/cpu/kernels/wide_acc.rs` divides an
// i128 accumulator by the reduced count exactly once, and `linspace_kernel` in
// `runtime/cpu/kernels/memory.rs` truncates `start + delta * i / divisor`.
//
// The `/` operator never appears in the loop below. The NVIDIA shader compiler
// fails with "NVVM compilation failed" on an integer divide inside a WGSL loop,
// which is why every helper in this family is built from shifts, masks and
// compares instead - see the header of `int_saturate.wgsl`.

// Bit `i` (0 = least significant) of a 64-bit value.
fn numr_u64_bit(v: NumrI64, i: u32) -> u32 {
    if (i >= 32u) {
        return (v.hi >> (i - 32u)) & 1u;
    }
    return (v.lo >> i) & 1u;
}

// Restoring long division of an unsigned 64-bit dividend by a 32-bit divisor.
//
// The remainder stays below the divisor, so it always fits in one u32; only the
// shift can push it to 33 bits, and `carry` carries that bit explicitly. The
// subtraction is then correct in wrapping u32 arithmetic: when `carry` is set,
// the true remainder is `2^32 + shifted`, and `shifted - d` modulo 2^32 is the
// true difference because that difference is itself below 2^32.
//
// A zero divisor returns zero, matching the CPU integer-division convention in
// `runtime/cpu/kernels/binary_int.rs`. Callers that have a meaningful answer for
// an empty count (`mean` returns the untouched accumulator) check first.
fn numr_u64_div_u32(n: NumrI64, d: u32) -> NumrI64 {
    if (d == 0u) {
        return NumrI64(0u, 0u);
    }
    var q_lo: u32 = 0u;
    var q_hi: u32 = 0u;
    var rem: u32 = 0u;
    for (var i: i32 = 63; i >= 0; i = i - 1) {
        let bit = numr_u64_bit(n, u32(i));
        let carry = rem >> 31u;
        let shifted = (rem << 1u) | bit;
        if (carry != 0u || shifted >= d) {
            rem = shifted - d;
            if (i >= 32) {
                q_hi = q_hi | (1u << u32(i - 32));
            } else {
                q_lo = q_lo | (1u << u32(i));
            }
        } else {
            rem = shifted;
        }
    }
    return NumrI64(q_lo, q_hi);
}

// Signed 64-bit divided by an unsigned 32-bit count, truncating toward zero.
//
// Truncation toward zero is division of the magnitudes with the sign reapplied,
// which is why the negative case negates on the way in and back out rather than
// dividing the two's-complement value directly.
fn numr_i64_div_u32_trunc(n: NumrI64, d: u32) -> NumrI64 {
    if ((n.hi >> 31u) != 0u) {
        return numr_i64_neg(numr_u64_div_u32(numr_i64_neg(n), d));
    }
    return numr_u64_div_u32(n, d);
}

// Narrow an UNSIGNED 64-bit value to u32, saturating at `u32::MAX`.
fn numr_u64_to_u32_sat(v: NumrI64) -> u32 {
    if (v.hi != 0u) {
        return NUMR_U32_MAX;
    }
    return v.lo;
}

// Narrow a SIGNED 64-bit value to u32, clamping to the u32 range at both ends.
// A negative value clamps to zero, matching `Element::from_f64` on CPU, where
// the float-to-unsigned `as` cast saturates rather than wrapping.
fn numr_i64_to_u32_sat(v: NumrI64) -> u32 {
    if ((v.hi >> 31u) != 0u) {
        return 0u;
    }
    return numr_u64_to_u32_sat(v);
}

// `a * b` where `a` is a signed 64-bit value and `b` an unsigned 32-bit one.
//
// Callers bound the operands so the true product fits in 64 bits; the high
// limb's own overflow is therefore unreachable, not silently dropped.
fn numr_i64_mul_u32(a: NumrI64, b: u32) -> NumrI64 {
    let negative = (a.hi >> 31u) != 0u;
    var mag = a;
    if (negative) {
        mag = numr_i64_neg(a);
    }
    var prod = numr_u64_mul_u32(mag.lo, b);
    prod = NumrI64(prod.lo, prod.hi + mag.hi * b);
    if (negative) {
        return numr_i64_neg(prod);
    }
    return prod;
}

// Widen a u32 into the 64-bit accumulator with no sign extension.
fn numr_u64_from_u32(v: u32) -> NumrI64 {
    return NumrI64(v, 0u);
}
