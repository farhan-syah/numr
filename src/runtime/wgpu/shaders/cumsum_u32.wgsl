// Cumulative sum shader for u32.
//
// Uses `numr_u32_sat_add` from int_saturate.wgsl (prepended to this module).
// Unlike i32, a per-step saturating add is exact here: u32 inputs never go
// negative, so the running total is monotonic and once it saturates to
// u32::MAX it can never need to come back down, unlike the signed case in
// cumsum_i32.wgsl.

struct CumsumParams {
    scan_size: u32,
    outer_size: u32,
}

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: CumsumParams;

@compute @workgroup_size(256)
fn cumsum_u32(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let outer_idx = global_id.x;
    if (outer_idx >= params.outer_size) {
        return;
    }

    let base = outer_idx * params.scan_size;
    var acc: u32 = 0u;
    for (var i: u32 = 0u; i < params.scan_size; i = i + 1u) {
        acc = numr_u32_sat_add(acc, input[base + i]);
        output[base + i] = acc;
    }
}
