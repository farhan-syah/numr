// Deterministic float-to-integer conversion, matching CPU's `Element::from_f64`.
//
// Concatenated AFTER `int_saturate.wgsl`, whose range constants these build on.
// WGSL has no include and no forward declarations, so that order is
// load-bearing.
//
// WGSL leaves `u32(v)` and `i32(v)` implementation-defined once `v` is outside
// the destination type's range, and a NEGATIVE float converted to u32 is
// exactly that case: one driver can give 0, another the wrapped magnitude. CPU
// has no such freedom - `Element::from_f64` is Rust's `as`, which truncates
// toward zero, clamps to the type's bounds, and maps NaN to 0. Every integer
// shader that evaluates in f32 (`arange`, and `linspace` on fractional bounds)
// therefore routes its store through the guards below rather than through the
// bare cast.
//
// NaN needs no test of its own: every comparison against NaN is false, so a NaN
// falls past both bound checks into the 0 return.

// f32 -> u32, truncating toward zero, clamped to [0, u32::MAX].
//
// The bound is 2^32 rather than u32::MAX because u32::MAX has no f32
// representation: the nearest f32 above the largest representable u32 is
// exactly 2^32, and CPU's `as u32` saturates there too.
fn numr_f32_to_u32_sat(v: f32) -> u32 {
    if (v >= 4294967296.0) {
        return NUMR_U32_MAX;
    }
    if (v >= 1.0) {
        return u32(v);
    }
    // Negative, fractional-below-one, and NaN all truncate to zero.
    return 0u;
}

// f32 -> i32, truncating toward zero, clamped to [i32::MIN, i32::MAX].
//
// Both bounds are powers of two, so -2^31 is exactly representable and lands on
// i32::MIN under either rounding.
fn numr_f32_to_i32_sat(v: f32) -> i32 {
    if (v >= 2147483648.0) {
        return NUMR_I32_MAX;
    }
    if (v <= -2147483648.0) {
        return NUMR_I32_MIN;
    }
    if (v > -2147483648.0) {
        return i32(v);
    }
    // Only NaN reaches here.
    return 0;
}
