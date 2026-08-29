// Linspace for U32.
//
// Concatenated after int_saturate.wgsl, int_matmul_acc.wgsl, int_wide_div.wgsl
// and int_from_float.wgsl, whose 64-bit and conversion helpers this file builds
// on.
//
// CPU computes `start + delta * i / divisor` in f64 and converts once, rounding
// toward zero (`linspace_kernel` in runtime/cpu/kernels/memory.rs). WGSL has no
// f64, and f32 carries only 24 mantissa bits, so a value whose exact answer is
// an integer above 2^24 would land just under it and truncate to the wrong
// element. So the integral case - the one an integer linspace is actually for -
// runs in exact 64-bit integer arithmetic instead:
//
//     trunc(start + delta * i / divisor) == trunc((start * divisor + delta * i) / divisor)
//
// The host precomputes `start * divisor` as the 64-bit `base`, and only takes
// this path when every intermediate is proven to fit. Fractional bounds fall
// back to the f32 evaluation, where truncation is far from any boundary - but
// that evaluation still goes through numr_f32_to_u32_sat, because a bare
// u32(negative) is implementation-defined in WGSL while CPU's `as u32` clamps
// to zero.

const WORKGROUP_SIZE: u32 = 256u;

struct LinspaceIntParams {
    steps: u32,
    exact: u32,
    divisor: u32,
    _pad0: u32,
    base_lo: u32,
    base_hi: u32,
    delta_lo: u32,
    delta_hi: u32,
    start_f32: f32,
    stop_f32: f32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> linspace_out: array<u32>;
@group(0) @binding(1) var<uniform> linspace_params: LinspaceIntParams;

@compute @workgroup_size(256)
fn linspace_u32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= linspace_params.steps) {
        return;
    }

    if (linspace_params.exact != 0u) {
        let base = NumrI64(linspace_params.base_lo, linspace_params.base_hi);
        let delta = NumrI64(linspace_params.delta_lo, linspace_params.delta_hi);
        let numer = numr_i64_add(base, numr_i64_mul_u32(delta, idx));
        linspace_out[idx] = numr_i64_to_u32_sat(numr_i64_div_u32_trunc(numer, linspace_params.divisor));
        return;
    }

    let t_val = f32(idx) / f32(linspace_params.steps - 1u);
    let start = linspace_params.start_f32;
    let value = start + (linspace_params.stop_f32 - start) * t_val;
    linspace_out[idx] = numr_f32_to_u32_sat(value);
}
