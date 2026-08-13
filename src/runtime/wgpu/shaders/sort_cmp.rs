//! Shared WGSL fragments for f32 sort ordering.
//!
//! Expand to WGSL string literals so they can be embedded into the sort, topk,
//! and searchsorted shaders via `concat!` (which only accepts literals, not
//! `const` items). The total order therefore exists in exactly one place.

/// NaN-aware comparison: `sort_is_nan_f32` and `sort_cmp_f32`.
///
/// Mirrors `Element::sort_cmp` — NaN compares greater than every non-NaN value,
/// NaNs tie with each other, and `-0.0` ties with `+0.0`.
macro_rules! sort_cmp_f32_wgsl {
    () => {
        r#"
// Tested on the bit pattern rather than `v != v`: shader compilers are free to
// fold a self-comparison to false, which would silently drop NaN handling.
fn sort_is_nan_f32(v: f32) -> bool {
    let bits = bitcast<u32>(v);
    return (bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0u;
}

fn sort_cmp_f32(a: f32, b: f32) -> i32 {
    let a_nan = sort_is_nan_f32(a);
    let b_nan = sort_is_nan_f32(b);
    if (a_nan || b_nan) {
        if (a_nan && b_nan) {
            return 0;
        }
        return select(-1, 1, a_nan);
    }
    if (a < b) {
        return -1;
    }
    if (a > b) {
        return 1;
    }
    return 0;
}
"#
    };
}

/// Bitonic-network ordering helpers: `sort_rank_less_f32` and `sort_pad_f32`.
///
/// Requires [`sort_cmp_f32_wgsl!`] to be concatenated ahead of it.
macro_rules! sort_rank_f32_wgsl {
    () => {
        r#"
// Rank order: the requested output order with ties broken by original index, so
// the network sorts by a single total order and is stable in both directions.
fn sort_rank_less_f32(a: f32, idx_a: i32, b: f32, idx_b: i32, descending: bool) -> bool {
    var c = sort_cmp_f32(a, b);
    if (descending) {
        c = -c;
    }
    if (c != 0) {
        return c < 0;
    }
    return idx_a < idx_b;
}

// Padding value of maximum rank, so pad entries sort into the discarded tail.
// Ascending that is NaN (the greatest value); descending it is -inf. Both must
// be beyond any real value, otherwise real infinities get dropped.
fn sort_pad_f32(descending: bool) -> f32 {
    return select(bitcast<f32>(0x7fc00000u), bitcast<f32>(0xff800000u), descending);
}
"#
    };
}

pub(crate) use {sort_cmp_f32_wgsl, sort_rank_f32_wgsl};
